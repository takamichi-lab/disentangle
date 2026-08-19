import unittest
from pathlib import Path
from unittest.mock import patch

from disse.cli import _build_parser, _infer_modalities, main


class InferenceCliTests(unittest.TestCase):
    def setUp(self):
        self.parser = _build_parser()

    def test_auto_detects_single_and_paired_modalities(self):
        audio = self.parser.parse_args(["infer", "--audio", "example.wav"])
        text = self.parser.parse_args(["infer", "--text", "dog barking"])
        paired = self.parser.parse_args(
            ["infer", "--audio", "example.wav", "--text", "dog barking"]
        )
        manifest = self.parser.parse_args(
            ["infer", "--manifest", "manifest.csv"]
        )

        self.assertEqual(_infer_modalities(audio), ("audio",))
        self.assertEqual(_infer_modalities(text), ("text",))
        self.assertEqual(_infer_modalities(paired), ("audio", "text"))
        self.assertEqual(_infer_modalities(manifest), ("audio", "text"))

    def test_audio_only_reaches_encoder_without_text(self):
        with patch("disse.embed.encode_items", return_value=Path("out.npz")) as run:
            main(["infer", "--audio", "example.wav"])

        items = run.call_args.args[0]
        self.assertEqual(len(items), 1)
        self.assertIsNone(items[0].text)
        self.assertEqual(run.call_args.kwargs["modalities"], ("audio",))

    def test_text_only_reaches_encoder_without_audio(self):
        with patch("disse.embed.encode_items", return_value=Path("out.npz")) as run:
            main(["infer", "--text", "dog barking"])

        items = run.call_args.args[0]
        self.assertEqual(len(items), 1)
        self.assertIsNone(items[0].audio_path)
        self.assertEqual(run.call_args.kwargs["modalities"], ("text",))

    def test_explicit_both_requires_both_inputs(self):
        args = self.parser.parse_args(
            ["infer", "--audio", "example.wav", "--modality", "both"]
        )
        with self.assertRaisesRegex(SystemExit, "text inference requires --text"):
            _infer_modalities(args)


if __name__ == "__main__":
    unittest.main()
