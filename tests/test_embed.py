import tempfile
import unittest
from pathlib import Path

from disse.embed import load_manifest


class ManifestModalityTests(unittest.TestCase):
    def test_audio_only_manifest_does_not_require_text(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "audio.csv"
            manifest.write_text(
                "audio_path,source_id,spatial_id\n"
                "example.wav,source-1,space-1\n",
                encoding="utf-8",
            )
            items = load_manifest(manifest, require_text=False)

        self.assertEqual(len(items), 1)
        self.assertIsNone(items[0].text)

    def test_text_only_manifest_does_not_require_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "text.csv"
            manifest.write_text(
                "text,source_id,spatial_id\n"
                "Dog barking,source-1,space-1\n",
                encoding="utf-8",
            )
            items = load_manifest(manifest, require_audio=False)

        self.assertEqual(len(items), 1)
        self.assertIsNone(items[0].audio_path)
        self.assertEqual(items[0].text, "Dog barking")


if __name__ == "__main__":
    unittest.main()
