"""Portable NumPy embedding-cache format used by DISSE evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np


EMBEDDING_KEYS = (
    "audio_source",
    "audio_spatial",
    "text_source",
    "text_spatial",
)
LABEL_KEYS = ("source_id", "spatial_id")
REQUIRED_KEYS = EMBEDDING_KEYS + LABEL_KEYS

_ALIASES = {
    "audio_space": "audio_spatial",
    "text_space": "text_spatial",
    "a_src": "audio_source",
    "a_spa": "audio_spatial",
    "t_src": "text_source",
    "t_spa": "text_spatial",
    "src_id": "source_id",
    "space_id": "spatial_id",
    "spa_id": "spatial_id",
}


def _canonicalize(data: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for key, value in data.items():
        result[_ALIASES.get(key, key)] = np.asarray(value)
    return result


def validate_embedding_cache(data: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Validate and return a canonical in-memory cache."""
    cache = _canonicalize(data)
    missing = [key for key in REQUIRED_KEYS if key not in cache]
    if missing:
        raise ValueError(f"Embedding cache is missing keys: {', '.join(missing)}")

    n_items = int(np.asarray(cache["source_id"]).reshape(-1).shape[0])
    if n_items == 0:
        raise ValueError("Embedding cache is empty")

    cache["source_id"] = np.asarray(cache["source_id"]).reshape(-1)
    cache["spatial_id"] = np.asarray(cache["spatial_id"]).reshape(-1)
    if cache["spatial_id"].shape[0] != n_items:
        raise ValueError("source_id and spatial_id have different lengths")

    for key in EMBEDDING_KEYS:
        emb = np.asarray(cache[key])
        if emb.ndim != 2:
            raise ValueError(f"{key} must be a 2-D array, got shape {emb.shape}")
        if emb.shape[0] != n_items:
            raise ValueError(
                f"{key} contains {emb.shape[0]} rows, expected {n_items}"
            )
        if not np.issubdtype(emb.dtype, np.number):
            raise ValueError(f"{key} must be numeric, got dtype {emb.dtype}")
        if not np.isfinite(emb).all():
            raise ValueError(f"{key} contains NaN or infinite values")
        cache[key] = emb
    return cache


def load_embedding_cache(path: str | Path) -> dict[str, np.ndarray]:
    """Load a compressed ``.npz`` cache without enabling pickle."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as loaded:
        return validate_embedding_cache({key: loaded[key] for key in loaded.files})


def save_embedding_cache(
    path: str | Path,
    data: Mapping[str, np.ndarray],
    *,
    compressed: bool = True,
) -> Path:
    """Validate and atomically save a portable ``.npz`` cache."""
    path = Path(path)
    cache = validate_embedding_cache(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    writer = np.savez_compressed if compressed else np.savez
    with temporary.open("wb") as stream:
        writer(stream, **{key: cache[key] for key in REQUIRED_KEYS})
    temporary.replace(path)
    return path
