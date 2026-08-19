import tempfile
import unittest
from pathlib import Path

import numpy as np

from disse.cache import (
    load_embedding_cache,
    save_embedding_cache,
    validate_embedding_cache,
)


class CacheTests(unittest.TestCase):
    def test_round_trip_without_pickle(self):
        rng = np.random.default_rng(2)
        data = {
            "audio_source": rng.normal(size=(6, 4)),
            "audio_spatial": rng.normal(size=(6, 4)),
            "text_source": rng.normal(size=(6, 4)),
            "text_spatial": rng.normal(size=(6, 4)),
            "source_id": np.array(["a", "a", "b", "b", "c", "c"]),
            "spatial_id": np.array([0, 1, 0, 1, 0, 1]),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.npz"
            save_embedding_cache(path, data)
            loaded = load_embedding_cache(path)
        for key, value in data.items():
            np.testing.assert_array_equal(loaded[key], value)

    def test_audio_only_round_trip(self):
        data = {
            "audio_source": np.eye(3),
            "audio_spatial": np.eye(3)[::-1],
            "source_id": np.array(["a", "b", "c"]),
            "spatial_id": np.array([0, 1, 2]),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audio-cache.npz"
            save_embedding_cache(path, data)
            loaded = load_embedding_cache(path)

        self.assertEqual(set(loaded), set(data))
        for key, value in data.items():
            np.testing.assert_array_equal(loaded[key], value)
        with self.assertRaisesRegex(ValueError, "text_source, text_spatial"):
            validate_embedding_cache(data, require_all_embeddings=True)


if __name__ == "__main__":
    unittest.main()
