import csv
import tempfile
import unittest
from pathlib import Path

from disse.evaluation_data import fixed_grid_summary, make_evaluation_manifest


ROOT = Path(__file__).resolve().parents[1]
AUDIO_CATALOG = ROOT / "evaluation/audio_fixed.csv"
RIR_CATALOG = ROOT / "evaluation/rir_fixed.csv"


class EvaluationDataTests(unittest.TestCase):
    def test_released_catalogs_define_96_by_96_grid(self):
        self.assertEqual(
            fixed_grid_summary(AUDIO_CATALOG, RIR_CATALOG),
            {"sources": 96, "spatial_conditions": 96, "pairs": 9216},
        )

    def test_manifest_is_source_major_and_seeded(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "manifest.csv"
            make_evaluation_manifest(
                AUDIO_CATALOG,
                RIR_CATALOG,
                root / "dry",
                root / "rirs",
                output,
                seed=42,
                check_files=False,
            )
            with output.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))

        self.assertEqual(len(rows), 9216)
        self.assertEqual(rows[0]["source_id"], "104274")
        self.assertEqual(rows[0]["spatial_id"], "auto_000000")
        self.assertEqual(rows[95]["spatial_id"], "auto_000095")
        self.assertEqual(rows[96]["source_id"], "107283")
        self.assertEqual(
            rows[0]["text"],
            "The sound: Rain is falling continuously is coming from the nearby "
            "down right of a spacious highly reverberant room.",
        )


if __name__ == "__main__":
    unittest.main()
