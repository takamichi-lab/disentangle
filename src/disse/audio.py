"""FOA loading and intensity-vector preprocessing for DISSE inference."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch
import torchaudio


FOA_SAMPLE_RATE = 48_000
IV_SAMPLE_RATE = 16_000
DURATION_SECONDS = 10.0
FOA_SAMPLES = int(FOA_SAMPLE_RATE * DURATION_SECONDS)


def _repeat_or_crop(waveform: torch.Tensor, length: int = FOA_SAMPLES) -> torch.Tensor:
    if waveform.shape[-1] == 0:
        raise ValueError("Audio waveform is empty")
    if waveform.shape[-1] < length:
        repeats = (length + waveform.shape[-1] - 1) // waveform.shape[-1]
        waveform = waveform.repeat(1, repeats)
    return waveform[..., :length]


def load_foa(path: str | Path) -> torch.Tensor:
    """Load FOA audio as ``[4, 480000]`` in W, Y, Z, X channel order."""
    waveform, sample_rate = torchaudio.load(str(path))
    if waveform.ndim != 2 or waveform.shape[0] != 4:
        raise ValueError(
            f"Expected a four-channel FOA file (W,Y,Z,X), got {waveform.shape}"
        )
    if sample_rate != FOA_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(
            waveform, sample_rate, FOA_SAMPLE_RATE
        )
    return _repeat_or_crop(waveform.float())


@lru_cache(maxsize=8)
def _load_dry_cached(path: str) -> torch.Tensor:
    """Load one local dry clip as mono, 48 kHz, and exactly 10 seconds."""
    waveform, sample_rate = torchaudio.load(path)
    if waveform.ndim != 2 or waveform.shape[0] == 0:
        raise ValueError(f"Invalid dry waveform shape: {waveform.shape}")
    waveform = waveform.float().mean(dim=0, keepdim=True)
    if sample_rate != FOA_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(
            waveform, sample_rate, FOA_SAMPLE_RATE
        )
    return _repeat_or_crop(waveform)


@lru_cache(maxsize=128)
def _load_a_format_rir_cached(path: str) -> torch.Tensor:
    """Load the four tetrahedral microphone responses used by DISSE."""
    rir, sample_rate = torchaudio.load(path)
    if rir.ndim != 2 or rir.shape[0] != 4:
        raise ValueError(f"Expected a four-channel A-format RIR, got {rir.shape}")
    if sample_rate != FOA_SAMPLE_RATE:
        rir = torchaudio.functional.resample(rir, sample_rate, FOA_SAMPLE_RATE)
    return rir.float()


def a_format_to_foa(wet: torch.Tensor) -> torch.Tensor:
    """Convert tetrahedral A-format channels to FOA channel order W,Y,Z,X."""
    if wet.ndim != 2 or wet.shape[0] != 4:
        raise ValueError(f"A-format tensor must have shape [4,T], got {wet.shape}")
    m0, m1, m2, m3 = wet
    w = (m0 + m1 + m2 + m3) / 2
    x = (m0 + m1 - m2 - m3) / 2
    y = (m0 - m1 + m2 - m3) / 2
    z = (m0 - m1 - m2 + m3) / 2
    return torch.stack((w, y, z, x))


def spatialize_dry_audio(
    dry_path: str | Path, rir_path: str | Path
) -> torch.Tensor:
    """Convolve a local dry clip with an A-format RIR and return 10-second FOA."""
    dry = _load_dry_cached(str(Path(dry_path).resolve()))
    rir = _load_a_format_rir_cached(str(Path(rir_path).resolve()))
    wet = torchaudio.functional.fftconvolve(dry, rir)
    wet = wet[..., :FOA_SAMPLES]
    return a_format_to_foa(wet)


@lru_cache(maxsize=8)
def _hann(n_fft: int, device: str, dtype: torch.dtype) -> torch.Tensor:
    return torch.hann_window(n_fft, device=torch.device(device), dtype=dtype)


def foa_to_intensity_vectors(
    foa_16k: torch.Tensor,
    *,
    n_fft: int = 400,
    hop_length: int = 100,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert ``[B,4,T]`` FOA to active/reactive intensity features."""
    if foa_16k.ndim != 3 or foa_16k.shape[1] != 4:
        raise ValueError(f"FOA tensor must have shape [B,4,T], got {foa_16k.shape}")
    batch, _, samples = foa_16k.shape
    window = _hann(n_fft, str(foa_16k.device), foa_16k.dtype)
    spectrum = torch.stft(
        foa_16k.reshape(-1, samples),
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        center=True,
        return_complex=True,
    ).reshape(batch, 4, n_fft // 2 + 1, -1)
    w, y, z, x = spectrum[:, 0], spectrum[:, 1], spectrum[:, 2], spectrum[:, 3]
    conjugate_w = w.conj()
    active = torch.stack(
        ((conjugate_w * y).real, (conjugate_w * z).real, (conjugate_w * x).real),
        dim=1,
    )
    reactive = torch.stack(
        ((conjugate_w * y).imag, (conjugate_w * z).imag, (conjugate_w * x).imag),
        dim=1,
    )
    norm = torch.linalg.vector_norm(active, dim=1, keepdim=True)
    active = torch.where(norm > epsilon, active / norm, active)
    reactive = torch.where(norm > epsilon, reactive / norm, reactive)
    return active.float(), reactive.float()


def features_from_foa(foa_48k: torch.Tensor) -> dict[str, torch.Tensor]:
    foa_48k = _repeat_or_crop(foa_48k.float())
    foa_16k = torchaudio.functional.resample(
        foa_48k, FOA_SAMPLE_RATE, IV_SAMPLE_RATE
    )
    active, reactive = foa_to_intensity_vectors(foa_16k.unsqueeze(0))
    return {
        "i_act": active.squeeze(0),
        "i_rea": reactive.squeeze(0),
        "omni_48k": foa_48k[0],
    }


def _load_feature_file(path: str | Path) -> dict[str, torch.Tensor]:
    try:
        feature = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        feature = torch.load(path, map_location="cpu")
    if not isinstance(feature, dict):
        raise ValueError(f"Feature file must contain a dictionary: {path}")
    missing = [key for key in ("i_act", "i_rea") if key not in feature]
    if missing:
        raise ValueError(f"Feature file {path} is missing: {', '.join(missing)}")
    return feature


def prepare_audio(
    audio_path: str | Path,
    feature_path: str | Path | None = None,
) -> dict[str, torch.Tensor]:
    """Load precomputed IV features when present, otherwise derive them."""
    foa = load_foa(audio_path)
    if feature_path is None:
        return features_from_foa(foa)
    feature = _load_feature_file(feature_path)
    active = feature["i_act"].float()
    reactive = feature["i_rea"].float()
    if active.ndim == 4 and active.shape[0] == 1:
        active = active.squeeze(0)
    if reactive.ndim == 4 and reactive.shape[0] == 1:
        reactive = reactive.squeeze(0)
    expected = (3, 201, 1601)
    if tuple(active.shape) != expected or tuple(reactive.shape) != expected:
        raise ValueError(
            f"Expected i_act/i_rea shape {expected}; got {active.shape}/{reactive.shape}"
        )
    omni = feature.get("omni_48k", foa[0]).float()
    if omni.ndim == 2 and omni.shape[0] == 1:
        omni = omni.squeeze(0)
    if omni.ndim != 1:
        raise ValueError(f"omni_48k must be one-dimensional, got {omni.shape}")
    omni = _repeat_or_crop(omni.unsqueeze(0)).squeeze(0)
    return {"i_act": active, "i_rea": reactive, "omni_48k": omni}


def prepare_spatialized_audio(
    dry_path: str | Path, rir_path: str | Path
) -> dict[str, torch.Tensor]:
    """Build model inputs directly from a local dry clip and a generated RIR."""
    return features_from_foa(spatialize_dry_audio(dry_path, rir_path))
