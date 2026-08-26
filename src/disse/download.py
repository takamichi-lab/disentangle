"""Artifact downloads with atomic writes and SHA-256 verification."""

from __future__ import annotations

import csv
import json
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

from .checkpoint import sha256_file


def load_artifact_manifest(path: str | Path) -> dict[str, dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, dict):
        raise ValueError("Artifact manifest must be a JSON object")
    return manifest


def _is_google_drive(url: str) -> bool:
    return "drive.google.com" in url or "docs.google.com" in url


def _download_http(url: str, destination: Path) -> None:
    with (
        urllib.request.urlopen(url) as response,
        destination.open("wb") as stream,
    ):
        total = int(response.headers.get("Content-Length") or 0)
        try:
            from tqdm.auto import tqdm
        except ImportError:
            tqdm = None

        if tqdm is None:
            size = f" ({total / 1024**3:.2f} GiB)" if total else ""
            print(f"Downloading {url}{size}")
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
            return

        with tqdm(
            total=total or None, unit="B", unit_scale=True, desc="Downloading"
        ) as bar:
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
                bar.update(len(chunk))


def download_artifact(
    name: str,
    *,
    manifest_path: str | Path = "artifacts.json",
    force: bool = False,
) -> Path:
    manifest = load_artifact_manifest(manifest_path)
    if name not in manifest:
        raise KeyError(f"Unknown artifact {name!r}; choose from {sorted(manifest)}")
    entry = manifest[name]
    url = entry.get("url")
    if not url:
        raise RuntimeError(
            f"The download URL for {name!r} has not been published in {manifest_path}"
        )
    output = Path(entry["output"])
    expected = (entry.get("sha256") or "").lower()
    if output.is_file() and not force:
        if not expected or sha256_file(output) == expected:
            print(f"Already downloaded: {output}")
            return output
        raise RuntimeError(
            f"Existing file has the wrong SHA-256: {output}; pass --force to replace it"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".part")
    if temporary.exists():
        temporary.unlink()
    try:
        if _is_google_drive(url):
            try:
                import gdown
            except ImportError as error:
                raise RuntimeError(
                    "Google Drive downloads require `pip install gdown`"
                ) from error
            downloaded = gdown.download(
                url=url,
                output=str(temporary),
                quiet=False,
                fuzzy=True,
                use_cookies=False,
            )
            if downloaded is None:
                raise RuntimeError(f"Google Drive download failed: {url}")
        else:
            _download_http(url, temporary)
        actual = sha256_file(temporary)
        if expected and actual != expected:
            raise RuntimeError(
                f"SHA-256 mismatch for {name}: expected {expected}, received {actual}"
            )
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(f"Downloaded {name}: {output}")
    print(f"SHA-256: {actual}")
    return output


def _evaluation_source_ids(catalog_path: str | Path) -> list[str]:
    with Path(catalog_path).open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    source_ids = [str(row.get("audiocap_id") or "").strip() for row in rows]
    if not source_ids or any(not source_id for source_id in source_ids):
        raise ValueError(
            f"Audio catalog must contain a non-empty audiocap_id column: "
            f"{catalog_path}"
        )
    if len(source_ids) != len(set(source_ids)):
        raise ValueError(
            f"Audio catalog contains duplicate audiocap_id values: {catalog_path}"
        )
    return source_ids


def download_evaluation_audio(
    *,
    manifest_path: str | Path = "artifacts.json",
    catalog_path: str | Path = "evaluation/audio_fixed.csv",
    output_dir: str | Path = "data/evaluation/dry",
    force: bool = False,
) -> list[Path]:
    """Download the AudioCaps test archive and retain the selected dry clips."""
    source_ids = _evaluation_source_ids(catalog_path)
    output_dir = Path(output_dir)
    outputs = [output_dir / f"{source_id}.mp3" for source_id in source_ids]
    if not force and all(output.is_file() for output in outputs):
        print(f"Already prepared {len(outputs)} evaluation clips in {output_dir}")
        return outputs

    archive = download_artifact(
        "evaluation-audio", manifest_path=manifest_path, force=force
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(archive) as source_zip:
            members = set(source_zip.namelist())
            missing = [
                f"test/{source_id}.mp3"
                for source_id in source_ids
                if f"test/{source_id}.mp3" not in members
            ]
            if missing:
                preview = ", ".join(missing[:5])
                suffix = (
                    f", ... and {len(missing) - 5} more" if len(missing) > 5 else ""
                )
                raise FileNotFoundError(
                    f"Evaluation clips are missing from {archive}: {preview}{suffix}"
                )

            with tempfile.TemporaryDirectory(
                prefix=".evaluation-audio-", dir=output_dir.parent
            ) as temporary:
                temporary_dir = Path(temporary)
                prepared: dict[str, Path] = {}
                for source_id in source_ids:
                    member = f"test/{source_id}.mp3"
                    target = temporary_dir / f"{source_id}.mp3"
                    with (
                        source_zip.open(member) as source,
                        target.open("wb") as destination,
                    ):
                        shutil.copyfileobj(source, destination)
                    prepared[source_id] = target

                output_dir.mkdir(parents=True, exist_ok=True)
                for source_id, temporary_path in prepared.items():
                    output = output_dir / f"{source_id}.mp3"
                    if force or not output.is_file():
                        temporary_path.replace(output)
    finally:
        archive.unlink(missing_ok=True)

    print(f"Prepared {len(outputs)} evaluation clips in {output_dir}")
    return outputs
