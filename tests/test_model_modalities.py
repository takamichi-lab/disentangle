import importlib.util
import unittest
from unittest.mock import patch


HAS_INFERENCE_DEPS = all(
    importlib.util.find_spec(name) is not None
    for name in ("torch", "transformers")
)

if HAS_INFERENCE_DEPS:
    import torch
    from torch import nn

    from disse.model import DISSE


@unittest.skipUnless(HAS_INFERENCE_DEPS, "inference dependencies are not installed")
class ModelModalityTests(unittest.TestCase):
    def _model(self):
        class AudioEncoder(nn.Module):
            def forward(self, i_act, i_rea, omni):
                del i_act, i_rea
                return omni

        class TextEncoder(nn.Module):
            def forward(self, texts):
                return torch.tensor(
                    [[len(text), text.count(" ")] for text in texts],
                    dtype=torch.float32,
                )

        model = DISSE.__new__(DISSE)
        nn.Module.__init__(model)
        model.audio_encoder = AudioEncoder()
        model.audio_space_head = nn.Identity()
        model.audio_source_head = nn.Identity()
        model.text_encoder = TextEncoder()
        model.text_space_head = nn.Identity()
        model.text_source_head = nn.Identity()
        model.direction_head = nn.Identity()
        model.area_head = nn.Identity()
        model.distance_head = nn.Identity()
        model.reverb_head = nn.Identity()
        model.logit_scale = nn.Parameter(torch.tensor(0.0))
        return model.eval()

    def test_single_modality_constructor_skips_unselected_encoder(self):
        with patch("disse.model.AudioEncoder", return_value=nn.Identity()) as audio, \
             patch("disse.model.TextEncoder", return_value=nn.Identity()) as text:
            audio_model = DISSE(modalities=("audio",))

        audio.assert_called_once()
        text.assert_not_called()
        self.assertIsNone(audio_model.text_encoder)
        self.assertFalse(
            any(key.startswith("text_") for key in audio_model.state_dict())
        )

        with patch("disse.model.AudioEncoder", return_value=nn.Identity()) as audio, \
             patch("disse.model.TextEncoder", return_value=nn.Identity()) as text:
            text_model = DISSE(modalities=("text",))

        audio.assert_not_called()
        text.assert_called_once()
        self.assertIsNone(text_model.audio_encoder)
        self.assertFalse(
            any(key.startswith("audio_") for key in text_model.state_dict())
        )

    def test_audio_embeddings_do_not_depend_on_text_input(self):
        model = self._model()
        audio = {
            "i_act": torch.zeros(1, 1),
            "i_rea": torch.zeros(1, 1),
            "omni_48k": torch.tensor([[1.0, 2.0]]),
        }

        first = model(audio, ["dog barking"])
        second = model(audio, ["completely different text"])

        torch.testing.assert_close(
            first["audio_source_emb"], second["audio_source_emb"]
        )
        torch.testing.assert_close(
            first["audio_space_emb"], second["audio_space_emb"]
        )

    def test_text_embeddings_do_not_depend_on_audio_input(self):
        model = self._model()
        first_audio = {
            "i_act": torch.zeros(1, 1),
            "i_rea": torch.zeros(1, 1),
            "omni_48k": torch.tensor([[1.0, 2.0]]),
        }
        second_audio = {
            "i_act": torch.ones(1, 1),
            "i_rea": torch.ones(1, 1),
            "omni_48k": torch.tensor([[9.0, 8.0]]),
        }

        first = model(first_audio, ["dog barking"])
        second = model(second_audio, ["dog barking"])

        torch.testing.assert_close(
            first["text_source_emb"], second["text_source_emb"]
        )
        torch.testing.assert_close(
            first["text_space_emb"], second["text_space_emb"]
        )


if __name__ == "__main__":
    unittest.main()
