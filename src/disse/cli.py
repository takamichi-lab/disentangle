"""Command-line interface for DISSE inference, download, and evaluation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .cache import load_embedding_cache, save_embedding_cache
from .metrics import evaluate_embedding_cache


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        if math.isnan(float(value)):
            return None
        if math.isinf(float(value)):
            return "Infinity" if float(value) > 0 else "-Infinity"
        return float(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def _write_json(data: Any, output: str | Path | None) -> None:
    encoded = json.dumps(_json_safe(data), indent=2, ensure_ascii=False)
    print(encoded)
    if output is not None:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded + "\n", encoding="utf-8")


def _demo_cache(output: str | Path, seed: int) -> Path:
    rng = np.random.default_rng(seed)
    source_ids, spatial_ids = np.meshgrid(np.arange(6), np.arange(6), indexing="ij")
    source_ids = source_ids.reshape(-1)
    spatial_ids = spatial_ids.reshape(-1)
    source_basis = np.eye(6)[source_ids]
    spatial_basis = np.eye(6)[spatial_ids]

    def noisy(array: np.ndarray) -> np.ndarray:
        return array + 0.01 * rng.standard_normal(array.shape)

    source = np.concatenate((2.0 * source_basis, 0.2 * spatial_basis), axis=1)
    spatial = np.concatenate((0.2 * source_basis, 2.0 * spatial_basis), axis=1)
    return save_embedding_cache(
        output,
        {
            "audio_source": noisy(source),
            "audio_spatial": noisy(spatial),
            "text_source": noisy(source),
            "text_spatial": noisy(spatial),
            "source_id": source_ids,
            "spatial_id": spatial_ids,
        },
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="disse")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download", help="Download a released artifact")
    download.add_argument("artifact", choices=("checkpoint", "evaluation-audio"))
    download.add_argument("--manifest", default="artifacts.json")
    download.add_argument("--audio-catalog", default="evaluation/audio_fixed.csv")
    download.add_argument("--output-dir", default="data/evaluation/dry")
    download.add_argument("--force", action="store_true")

    infer = subparsers.add_parser("infer", help="Extract DISSE embeddings")
    source = infer.add_mutually_exclusive_group()
    source.add_argument("--manifest", help="CSV manifest for batch inference")
    source.add_argument("--audio", help="One four-channel FOA WAV file")
    infer.add_argument("--text", help="One caption, optionally paired with --audio")
    infer.add_argument(
        "--modality",
        choices=("auto", "audio", "text", "both"),
        default="auto",
        help="Embedding branch to run (default: infer from inputs; manifests use both)",
    )
    infer.add_argument("--feature", help="Optional precomputed .pt for --audio")
    infer.add_argument("--source-id", default="single-source")
    infer.add_argument("--spatial-id", default="single-spatial")
    infer.add_argument("--data-root")
    infer.add_argument("--checkpoint", default="checkpoints/disse_epoch20.pt")
    infer.add_argument("--output", default="results/embeddings.npz")
    infer.add_argument("--batch-size", type=int, default=8)
    infer.add_argument("--device", default="auto")
    infer.add_argument("--amp", action="store_true")
    infer.add_argument("--no-strict", action="store_true")
    infer.add_argument("--model-cache-dir")

    evaluate = subparsers.add_parser("evaluate", help="Compute IIDR and retrieval")
    evaluate.add_argument("cache", help="Embedding cache (.npz)")
    evaluate.add_argument("--iidr-only", action="store_true")
    evaluate.add_argument("--ks", type=int, nargs="+", default=(1, 5, 10))
    evaluate.add_argument("--chunk-size", type=int, default=256)
    evaluate.add_argument("--output", default="results/metrics.json")

    validate = subparsers.add_parser("validate-manifest", help="Check evaluation paths")
    validate.add_argument("manifest")
    validate.add_argument("--data-root")

    demo = subparsers.add_parser("demo-cache", help="Create a synthetic IIDR cache")
    demo.add_argument("--output", default="results/demo_embeddings.npz")
    demo.add_argument("--seed", type=int, default=42)

    rirs = subparsers.add_parser(
        "generate-rirs", help="Generate the released 96 fixed A-format RIRs"
    )
    rirs.add_argument("--catalog", default="evaluation/rir_fixed.csv")
    rirs.add_argument("--output-dir", default="data/evaluation/rirs")
    rirs.add_argument("--force", action="store_true")

    manifest = subparsers.add_parser(
        "make-evaluation-manifest",
        help="Build the released 96 x 96 on-the-fly evaluation manifest",
    )
    manifest.add_argument("--dry-root", required=True)
    manifest.add_argument("--rir-root", default="data/evaluation/rirs")
    manifest.add_argument("--audio-catalog", default="evaluation/audio_fixed.csv")
    manifest.add_argument("--rir-catalog", default="evaluation/rir_fixed.csv")
    manifest.add_argument("--output", default="data/evaluation/manifest.csv")
    manifest.add_argument("--seed", type=int, default=42)
    manifest.add_argument("--no-check-files", action="store_true")
    return parser


def _infer_modalities(args: argparse.Namespace) -> tuple[str, ...]:
    if args.modality == "auto":
        if args.manifest:
            selected = ("audio", "text")
        else:
            selected = tuple(
                modality
                for modality, value in (("audio", args.audio), ("text", args.text))
                if value
            )
    elif args.modality == "both":
        selected = ("audio", "text")
    else:
        selected = (args.modality,)

    if not selected:
        raise SystemExit("infer requires --manifest, --audio, or --text")
    if args.manifest:
        if args.text:
            raise SystemExit("--text cannot be combined with --manifest")
        if args.feature:
            raise SystemExit("--feature can only be used with --audio")
        return selected

    if "audio" in selected and not args.audio:
        raise SystemExit("audio inference requires --audio")
    if "text" in selected and not args.text:
        raise SystemExit("text inference requires --text")
    if args.audio and "audio" not in selected:
        raise SystemExit("--audio was provided but audio inference is disabled")
    if args.text and "text" not in selected:
        raise SystemExit("--text was provided but text inference is disabled")
    if args.feature and not args.audio:
        raise SystemExit("--feature can only be used with --audio")
    return selected


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.command == "download":
        from .download import download_artifact, download_evaluation_audio

        if args.artifact == "evaluation-audio":
            download_evaluation_audio(
                manifest_path=args.manifest,
                catalog_path=args.audio_catalog,
                output_dir=args.output_dir,
                force=args.force,
            )
        else:
            download_artifact(
                args.artifact, manifest_path=args.manifest, force=args.force
            )
        return
    if args.command == "demo-cache":
        path = _demo_cache(args.output, args.seed)
        print(f"Wrote {path}")
        return
    if args.command == "generate-rirs":
        from .evaluation_data import generate_fixed_rirs

        outputs = generate_fixed_rirs(
            args.catalog, args.output_dir, force=args.force
        )
        print(f"Prepared {len(outputs)} RIRs in {args.output_dir}")
        return
    if args.command == "make-evaluation-manifest":
        from .evaluation_data import make_evaluation_manifest

        path = make_evaluation_manifest(
            args.audio_catalog,
            args.rir_catalog,
            args.dry_root,
            args.rir_root,
            args.output,
            seed=args.seed,
            check_files=not args.no_check_files,
        )
        print(f"Wrote {path} (9,216 rows)")
        return
    if args.command == "validate-manifest":
        from .embed import load_manifest, validate_manifest_files

        items = load_manifest(args.manifest, data_root=args.data_root)
        errors = validate_manifest_files(items)
        if errors:
            raise SystemExit("\n".join(errors))
        print(f"Validated {len(items)} manifest rows")
        return
    if args.command == "infer":
        from .embed import ManifestItem, encode_items, load_manifest

        modalities = _infer_modalities(args)
        if args.manifest:
            items = load_manifest(
                args.manifest,
                data_root=args.data_root,
                require_audio="audio" in modalities,
                require_text="text" in modalities,
            )
        else:
            items = [
                ManifestItem(
                    audio_path=Path(args.audio) if args.audio else None,
                    feature_path=Path(args.feature) if args.feature else None,
                    text=args.text,
                    source_id=args.source_id,
                    spatial_id=args.spatial_id,
                )
            ]
        output = encode_items(
            items,
            args.checkpoint,
            args.output,
            batch_size=args.batch_size,
            device=args.device,
            amp=args.amp,
            strict=not args.no_strict,
            model_cache_dir=args.model_cache_dir,
            modalities=modalities,
        )
        print(f"Wrote {output}")
        return
    if args.command == "evaluate":
        result = evaluate_embedding_cache(
            load_embedding_cache(args.cache),
            ks=args.ks,
            chunk_size=args.chunk_size,
            iidr_only=args.iidr_only,
        )
        _write_json(result, args.output)
        return
    raise AssertionError(args.command)
