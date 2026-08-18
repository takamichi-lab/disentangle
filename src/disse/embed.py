"""Manifest-driven DISSE embedding extraction."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .cache import save_embedding_cache


@dataclass(frozen=True)
class ManifestItem:
    text: str
    source_id: str
    spatial_id: str
    audio_path: Path | None = None
    feature_path: Path | None = None
    dry_path: Path | None = None
    rir_path: Path | None = None


def _first(row: dict[str, str], names: Iterable[str], *, required: bool = True) -> str:
    for name in names:
        value = str(row.get(name) or "").strip()
        if value:
            return value
    if required:
        raise ValueError(f"Manifest row is missing one of these columns: {', '.join(names)}")
    return ""


def _resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def load_manifest(
    path: str | Path, *, data_root: str | Path | None = None
) -> list[ManifestItem]:
    """Read public or legacy research manifests using documented aliases."""
    path = Path(path)
    root = Path(data_root) if data_root is not None else path.parent
    items: list[ManifestItem] = []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for line_number, row in enumerate(csv.DictReader(stream), start=2):
            try:
                audio = _first(
                    row, ("audio_path", "foa_path"), required=False
                )
                dry = _first(row, ("dry_path",), required=False)
                rir = _first(row, ("rir_path",), required=False)
                if not audio and not (dry and rir):
                    raise ValueError(
                        "each row needs audio_path/foa_path, or both dry_path and rir_path"
                    )
                feature = _first(
                    row, ("feature_path", "feat_path"), required=False
                )
                spatial_id = _first(
                    row, ("spatial_id", "space_id"), required=False
                )
                if not spatial_id:
                    spatial_id = rir
                if not spatial_id:
                    raise ValueError(
                        "Manifest row is missing spatial_id/space_id/rir_path"
                    )
                items.append(
                    ManifestItem(
                        audio_path=_resolve(root, audio) if audio else None,
                        feature_path=_resolve(root, feature) if feature else None,
                        dry_path=_resolve(root, dry) if dry else None,
                        rir_path=_resolve(root, rir) if rir else None,
                        text=_first(row, ("text", "caption")),
                        source_id=_first(row, ("source_id", "audiocap_id")),
                        spatial_id=spatial_id,
                    )
                )
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
    if not items:
        raise ValueError(f"Manifest contains no items: {path}")
    return items


def validate_manifest_files(items: Iterable[ManifestItem]) -> list[str]:
    errors: list[str] = []
    for index, item in enumerate(items):
        if item.audio_path is not None:
            if not item.audio_path.is_file():
                errors.append(f"row {index}: audio file not found: {item.audio_path}")
        else:
            if item.dry_path is None or not item.dry_path.is_file():
                errors.append(f"row {index}: dry file not found: {item.dry_path}")
            if item.rir_path is None or not item.rir_path.is_file():
                errors.append(f"row {index}: RIR file not found: {item.rir_path}")
        if item.feature_path is not None and not item.feature_path.is_file():
            errors.append(f"row {index}: feature file not found: {item.feature_path}")
    return errors


def encode_items(
    items: list[ManifestItem],
    checkpoint_path: str | Path,
    output_path: str | Path,
    *,
    batch_size: int = 8,
    device: str = "auto",
    amp: bool = False,
    strict: bool = True,
    model_cache_dir: str | None = None,
) -> Path:
    import torch
    import torch.nn.functional as functional
    from tqdm.auto import tqdm

    from .audio import prepare_audio, prepare_spatialized_audio
    from .checkpoint import load_checkpoint
    from .model import DISSE

    errors = validate_manifest_files(items)
    if errors:
        preview = "\n".join(errors[:20])
        suffix = f"\n... and {len(errors) - 20} more" if len(errors) > 20 else ""
        raise FileNotFoundError(preview + suffix)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)

    model = DISSE(
        audio_encoder_cfg={"cache_dir": model_cache_dir},
        text_encoder_cfg={"cache_dir": model_cache_dir},
    )
    metadata = load_checkpoint(model, checkpoint_path, strict=strict)
    print(
        f"Loaded checkpoint: {metadata['path']}"
        + (f" (epoch {metadata['epoch']})" if metadata["epoch"] is not None else "")
    )
    model.to(torch_device).eval()

    buffers: dict[str, list[np.ndarray]] = {
        "audio_source": [],
        "audio_spatial": [],
        "text_source": [],
        "text_spatial": [],
    }
    with torch.inference_mode():
        for start in tqdm(range(0, len(items), batch_size), desc="Embedding"):
            batch = items[start : start + batch_size]
            features = []
            for item in batch:
                if item.audio_path is not None:
                    features.append(prepare_audio(item.audio_path, item.feature_path))
                else:
                    assert item.dry_path is not None and item.rir_path is not None
                    features.append(
                        prepare_spatialized_audio(item.dry_path, item.rir_path)
                    )
            audio = {
                "i_act": torch.stack([item["i_act"] for item in features]).to(
                    torch_device
                ),
                "i_rea": torch.stack([item["i_rea"] for item in features]).to(
                    torch_device
                ),
                # HTSAT's processor consumes CPU waveforms internally.
                "omni_48k": torch.stack(
                    [item["omni_48k"] for item in features]
                ),
            }
            with torch.autocast(
                device_type=torch_device.type,
                enabled=amp and torch_device.type == "cuda",
            ):
                output = model(audio, [item.text for item in batch])
            mapping = {
                "audio_source": output["audio_source_emb"],
                "audio_spatial": output["audio_space_emb"],
                "text_source": output["text_source_emb"],
                "text_spatial": output["text_space_emb"],
            }
            for key, tensor in mapping.items():
                value = functional.normalize(tensor.float(), dim=-1)
                buffers[key].append(value.cpu().numpy())

    cache = {
        key: np.concatenate(chunks, axis=0) for key, chunks in buffers.items()
    }
    cache["source_id"] = np.asarray([item.source_id for item in items], dtype=str)
    cache["spatial_id"] = np.asarray([item.spatial_id for item in items], dtype=str)
    return save_embedding_cache(output_path, cache)
