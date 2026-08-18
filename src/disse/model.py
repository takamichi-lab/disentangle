"""Checkpoint-compatible DISSE model definition.

Module and parameter names intentionally match the research implementation so
that the released full-model epoch-20 checkpoint can be loaded strictly. The
original workstation-specific spatial-encoder checkpoint path is not used:
all learned tensors are restored from the released DISSE checkpoint.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import nn
from transformers import (
    ClapAudioConfig,
    ClapAudioModel,
    ClapAudioModelWithProjection,
    ClapConfig,
    ClapProcessor,
    RobertaConfig,
    RobertaModel,
    RobertaTokenizer,
)


class HTSAT(nn.Module):
    def __init__(
        self,
        model_id: str = "laion/clap-htsat-fused",
        *,
        cache_dir: str | None = None,
        load_pretrained_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.processor = ClapProcessor.from_pretrained(model_id, cache_dir=cache_dir)
        if load_pretrained_backbone:
            full_model = ClapAudioModelWithProjection.from_pretrained(
                model_id, cache_dir=cache_dir
            )
            self.model = full_model.audio_model
        else:
            full_config = ClapConfig.from_pretrained(model_id, cache_dir=cache_dir)
            audio_config = full_config.audio_config
            if isinstance(audio_config, dict):
                audio_config = ClapAudioConfig(**audio_config)
            self.model = ClapAudioModel(audio_config)

    def forward(self, omni_wave: torch.Tensor) -> torch.Tensor:
        """Encode 48-kHz, 10-second W-channel waveforms of shape ``[B, T]``."""
        omni_cpu = omni_wave.detach().to("cpu")
        raw_audio = [omni_cpu[i].numpy() for i in range(omni_cpu.shape[0])]
        inputs = self.processor(
            audios=raw_audio,
            sampling_rate=48_000,
            return_tensors="pt",
            padding="repeatpad",
            truncation="max_length",
            max_length=480_000,
        )
        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        return self.model(**inputs).pooler_output


class AddCoords2D(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = x.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, height, device=x.device, dtype=x.dtype),
            torch.linspace(-1, 1, width, device=x.device, dtype=x.dtype),
            indexing="ij",
        )
        coordinates = torch.stack((yy, xx)).unsqueeze(0).repeat(batch, 1, 1, 1)
        return torch.cat((x, coordinates), dim=1)


class Branch(nn.Module):
    """Six-block coordinate-aware CNN used by the ELSA spatial encoder."""

    def __init__(self) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Sequential(
                AddCoords2D(),
                nn.Conv2d(5, 16, 3, padding=1),
                nn.BatchNorm2d(16),
                nn.MaxPool2d(2),
                nn.ELU(),
            )
        ]
        for _ in range(5):
            layers.append(
                nn.Sequential(
                    nn.Conv2d(16, 16, 3, padding=1),
                    nn.BatchNorm2d(16),
                    nn.MaxPool2d(2),
                    nn.ELU(),
                )
            )
        self.cnn = nn.Sequential(*layers)
        self.flat = nn.Sequential(nn.Flatten(), nn.Dropout(0.3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.flat(self.cnn(x))


class SpatialAttributesBranch(nn.Module):
    def __init__(self, hidden1: int = 128, hidden2: int = 32, out_dim: int = 44):
        super().__init__()
        self.act = Branch()
        self.rea = Branch()
        self.mlp = nn.Sequential(
            nn.Linear(2400, hidden1),
            nn.ELU(),
            nn.Linear(hidden1, hidden2),
            nn.ELU(),
            nn.Linear(hidden2, out_dim),
        )

    def forward(self, i_act: torch.Tensor, i_rea: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat((self.act(i_act), self.rea(i_rea)), dim=1))


class AudioEncoder(nn.Module):
    def __init__(
        self,
        hidden1: int = 768,
        out_dim: int = 512,
        *,
        clap_model_id: str = "laion/clap-htsat-fused",
        cache_dir: str | None = None,
        load_pretrained_backbones: bool = False,
    ) -> None:
        super().__init__()
        self.htsat = HTSAT(
            clap_model_id,
            cache_dir=cache_dir,
            load_pretrained_backbone=load_pretrained_backbones,
        )
        self.spatial_branch = SpatialAttributesBranch()
        self.spatial_to_elsa = nn.Sequential(nn.ELU(), nn.Linear(44, 192))
        self.audio_projection = nn.Sequential(
            nn.Linear(192 + 768, hidden1),
            nn.ELU(),
            nn.Linear(hidden1, out_dim),
        )

    def forward(
        self, i_act: torch.Tensor, i_rea: torch.Tensor, omni: torch.Tensor
    ) -> torch.Tensor:
        spatial = self.spatial_to_elsa(self.spatial_branch(i_act, i_rea))
        source = self.htsat(omni)
        return self.audio_projection(torch.cat((spatial, source), dim=1))


class TextEncoder(nn.Module):
    def __init__(
        self,
        pretrained_model_name: str = "roberta-base",
        mlp_hidden_size: int = 640,
        output_dim: int = 512,
        max_length: int = 512,
        *,
        cache_dir: str | None = None,
        load_pretrained_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.tokenizer = RobertaTokenizer.from_pretrained(
            pretrained_model_name, cache_dir=cache_dir
        )
        if load_pretrained_backbone:
            self.roberta = RobertaModel.from_pretrained(
                pretrained_model_name, cache_dir=cache_dir
            )
        else:
            config = RobertaConfig.from_pretrained(
                pretrained_model_name, cache_dir=cache_dir
            )
            self.roberta = RobertaModel(config)
        self.max_length = max_length
        self.proj1 = nn.Linear(self.roberta.config.hidden_size, mlp_hidden_size)
        self.reru = nn.ReLU()  # Name retained for checkpoint compatibility.
        self.proj2 = nn.Linear(mlp_hidden_size, output_dim)

    def forward(self, texts: Sequence[str]) -> torch.Tensor:
        inputs = self.tokenizer(
            list(texts),
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        ).to(self.roberta.device)
        pooled = self.roberta(**inputs).pooler_output
        return self.proj2(self.reru(self.proj1(pooled)))


class RegressionHead_for_physicalValue(nn.Module):
    """Name retained verbatim for compatibility with the training checkpoint."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))


class DISSE(nn.Module):
    def __init__(
        self,
        audio_encoder_cfg: dict | None = None,
        text_encoder_cfg: dict | None = None,
        *,
        modalities: Iterable[str] = ("audio", "text"),
    ) -> None:
        super().__init__()
        selected = frozenset(modalities)
        invalid = sorted(selected - {"audio", "text"})
        if invalid:
            raise ValueError(f"Unknown model modalities: {', '.join(invalid)}")
        if not selected:
            raise ValueError("At least one model modality is required")
        self.modalities = selected
        shared_dim = 512
        output_dim = 512
        if "audio" in selected:
            self.audio_encoder = AudioEncoder(**(audio_encoder_cfg or {}))
            self.audio_space_head = nn.Sequential(
                nn.ELU(), nn.Linear(shared_dim, output_dim)
            )
            self.audio_source_head = nn.Sequential(
                nn.ELU(), nn.Linear(shared_dim, output_dim)
            )
        else:
            self.audio_encoder = None
            self.audio_space_head = None
            self.audio_source_head = None
        if "text" in selected:
            self.text_encoder = TextEncoder(**(text_encoder_cfg or {}))
            self.text_space_head = nn.Sequential(
                nn.ELU(), nn.Linear(shared_dim, output_dim)
            )
            self.text_source_head = nn.Sequential(
                nn.ELU(), nn.Linear(shared_dim, output_dim)
            )
        else:
            self.text_encoder = None
            self.text_space_head = None
            self.text_source_head = None
        self.logit_scale = nn.Parameter(
            torch.tensor(np.log(1 / 0.07), dtype=torch.float32)
        )
        if "audio" in selected:
            self.direction_head = RegressionHead_for_physicalValue(output_dim, 2)
            self.area_head = RegressionHead_for_physicalValue(output_dim, 1)
            self.distance_head = RegressionHead_for_physicalValue(output_dim, 1)
            self.reverb_head = RegressionHead_for_physicalValue(output_dim, 1)
        else:
            self.direction_head = None
            self.area_head = None
            self.distance_head = None
            self.reverb_head = None

    def encode_audio(
        self, audio_data: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Encode audio without evaluating the text branch."""
        if self.audio_encoder is None:
            raise RuntimeError("This DISSE instance has no audio branch")
        assert self.audio_space_head is not None
        assert self.audio_source_head is not None
        assert self.direction_head is not None
        assert self.area_head is not None
        assert self.distance_head is not None
        assert self.reverb_head is not None
        audio_shared = self.audio_encoder(
            i_act=audio_data["i_act"],
            i_rea=audio_data["i_rea"],
            omni=audio_data["omni_48k"],
        )
        audio_spatial = self.audio_space_head(audio_shared)
        audio_source = self.audio_source_head(audio_shared)
        return {
            "audio_space_emb": audio_spatial,
            "audio_source_emb": audio_source,
            "direction": self.direction_head(audio_spatial),
            "area": self.area_head(audio_spatial),
            "distance": self.distance_head(audio_spatial),
            "reverb": self.reverb_head(audio_spatial),
        }

    def encode_text(self, text_data: Sequence[str]) -> dict[str, torch.Tensor]:
        """Encode text without evaluating the audio branch."""
        if self.text_encoder is None:
            raise RuntimeError("This DISSE instance has no text branch")
        assert self.text_space_head is not None
        assert self.text_source_head is not None
        text_shared = self.text_encoder(text_data)
        text_spatial = self.text_space_head(text_shared)
        text_source = self.text_source_head(text_shared)
        return {
            "text_space_emb": text_spatial,
            "text_source_emb": text_source,
        }

    def forward(
        self, audio_data: dict[str, torch.Tensor], text_data: Sequence[str]
    ) -> dict[str, torch.Tensor]:
        """Encode a paired batch for training and cross-modal evaluation."""
        scale = self.logit_scale.clamp(max=math.log(100.0)).exp()
        return {
            **self.encode_audio(audio_data),
            **self.encode_text(text_data),
            "logit_scale": scale,
            "tau": 1.0 / scale,
        }
