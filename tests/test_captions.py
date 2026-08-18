import random
import unittest

from disse.captions import augment_caption


class CaptionTests(unittest.TestCase):
    def test_large_room_threshold_matches_executable_training_code(self):
        caption = augment_caption(
            "A bell rings",
            {
                "source_distance_m": 1.5,
                "azimuth_deg": 0,
                "elevation_deg": 0,
                "area_m2": 101,
                "fullband_T30_ms": 500,
            },
            rng=random.Random(3),
        )
        self.assertTrue("large" in caption or "spacious" in caption)
        self.assertNotIn("mid-sized", caption)

    def test_seeded_generation_is_reproducible(self):
        metadata = {
            "source_distance_m": 0.5,
            "azimuth_deg": -90,
            "elevation_deg": 50,
            "area_m2": 40,
            "fullband_T30_ms": 1200,
        }
        first = augment_caption("Birdsong", metadata, rng=random.Random(42))
        second = augment_caption("Birdsong", metadata, rng=random.Random(42))
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
