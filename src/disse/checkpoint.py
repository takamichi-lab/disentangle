"""Safe loading and checksum helpers for released DISSE checkpoints."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _pick_state_dict(checkpoint: Any) -> dict:
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            candidate = checkpoint.get(key)
            if isinstance(candidate, dict):
                return candidate
    if isinstance(checkpoint, dict):
        return checkpoint
    raise TypeError("Checkpoint does not contain a state dictionary")


def load_checkpoint(
    model: Any,
    path: str | Path,
    *,
    map_location: str = "cpu",
    strict: bool = True,
    expected_epoch: int | None = None,
    allow_unused_checkpoint_keys: bool = False,
) -> dict[str, Any]:
    """Load a full or selected-modality DISSE model and return metadata."""
    import torch

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:  # PyTorch before weights_only was introduced.
        checkpoint = torch.load(path, map_location=map_location)

    state = _pick_state_dict(checkpoint)
    state = {
        (key.removeprefix("module.") if key.startswith("module.") else key): value
        for key, value in state.items()
    }
    ignored_keys: list[str] = []
    if allow_unused_checkpoint_keys:
        expected_keys = set(model.state_dict())
        ignored_keys = sorted(set(state) - expected_keys)
        state = {key: value for key, value in state.items() if key in expected_keys}
    incompatible = model.load_state_dict(state, strict=strict)

    epoch = checkpoint.get("epoch") if isinstance(checkpoint, dict) else None
    if expected_epoch is not None and epoch is not None and int(epoch) != expected_epoch:
        raise ValueError(f"Checkpoint epoch is {epoch}; expected {expected_epoch}")
    return {
        "path": str(path),
        "epoch": int(epoch) if epoch is not None else None,
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "ignored_keys": ignored_keys,
    }
