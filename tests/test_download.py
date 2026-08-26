import csv
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from disse.checkpoint import sha256_file
from disse.download import download_evaluation_audio


class EvaluationAudioDownloadTests(unittest.TestCase):
    def _fixture(self, root: Path, *, include_second: bool = True) -> tuple[Path, Path]:
        catalog = root / "audio.csv"
        with catalog.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=("audiocap_id", "caption"))
            writer.writeheader()
            writer.writerow({"audiocap_id": "100", "caption": "first"})
            writer.writerow({"audiocap_id": "200", "caption": "second"})

        source_archive = root / "source.zip"
        with zipfile.ZipFile(source_archive, "w") as archive:
            archive.writestr("test/100.mp3", b"first-audio")
            if include_second:
                archive.writestr("test/200.mp3", b"second-audio")
            archive.writestr("test/not-selected.mp3", b"unused")

        downloaded_archive = root / "downloads" / "test.zip"
        artifact_manifest = root / "artifacts.json"
        artifact_manifest.write_text(
            json.dumps(
                {
                    "evaluation-audio": {
                        "url": source_archive.as_uri(),
                        "sha256": sha256_file(source_archive),
                        "output": str(downloaded_archive),
                    }
                }
            ),
            encoding="utf-8",
        )
        return catalog, artifact_manifest

    def test_extracts_only_catalog_clips_and_removes_archive(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog, manifest = self._fixture(root)
            output_dir = root / "dry"

            outputs = download_evaluation_audio(
                manifest_path=manifest,
                catalog_path=catalog,
                output_dir=output_dir,
            )

            self.assertEqual(outputs, [output_dir / "100.mp3", output_dir / "200.mp3"])
            self.assertEqual((output_dir / "100.mp3").read_bytes(), b"first-audio")
            self.assertEqual((output_dir / "200.mp3").read_bytes(), b"second-audio")
            self.assertFalse((output_dir / "not-selected.mp3").exists())
            self.assertFalse((root / "downloads" / "test.zip").exists())

            with patch("disse.download.download_artifact") as download:
                download_evaluation_audio(
                    manifest_path=manifest,
                    catalog_path=catalog,
                    output_dir=output_dir,
                )
            download.assert_not_called()

    def test_missing_catalog_clip_fails_without_partial_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog, manifest = self._fixture(root, include_second=False)
            output_dir = root / "dry"

            with self.assertRaisesRegex(FileNotFoundError, "test/200.mp3"):
                download_evaluation_audio(
                    manifest_path=manifest,
                    catalog_path=catalog,
                    output_dir=output_dir,
                )

            self.assertFalse(output_dir.exists())
            self.assertFalse((root / "downloads" / "test.zip").exists())


if __name__ == "__main__":
    unittest.main()
