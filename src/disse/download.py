"""Artifact downloads with atomic writes and optional SHA-256 verification."""

from __future__ import annotations

import json
import urllib.request
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
            with urllib.request.urlopen(url) as response, temporary.open("wb") as stream:
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
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
