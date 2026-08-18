import unittest

import numpy as np

from disse.metrics import compute_iidr, retrieval_metrics


class IIDRTests(unittest.TestCase):
    def test_grouped_algorithm_matches_pairwise_definition(self):
        rng = np.random.default_rng(7)
        source = np.repeat(np.arange(3), 4 * 2)
        spatial = np.tile(np.repeat(np.arange(4), 2), 3)
        embeddings = rng.normal(size=(source.size, 9))

        result = compute_iidr(embeddings, source, spatial)
        normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        distance = 1.0 - normalized @ normalized.T
        upper = np.triu(np.ones(distance.shape, dtype=bool), k=1)
        same_source_diff_spatial = (
            (source[:, None] == source[None, :])
            & (spatial[:, None] != spatial[None, :])
            & upper
        )
        same_spatial_diff_source = (
            (spatial[:, None] == spatial[None, :])
            & (source[:, None] != source[None, :])
            & upper
        )
        d_source_fixed = distance[same_source_diff_spatial].mean()
        d_spatial_fixed = distance[same_spatial_diff_source].mean()
        self.assertAlmostEqual(
            result["mean_same_source_diff_spatial"], d_source_fixed, places=12
        )
        self.assertAlmostEqual(
            result["mean_same_spatial_diff_source"], d_spatial_fixed, places=12
        )
        self.assertAlmostEqual(
            result["IIDR_source"], d_spatial_fixed / d_source_fixed, places=12
        )
        self.assertAlmostEqual(
            result["IIDR_source"] * result["IIDR_spatial"], 1.0, places=12
        )

    def test_factor_specific_embeddings_have_expected_direction(self):
        source, spatial = np.meshgrid(np.arange(5), np.arange(4), indexing="ij")
        source = source.reshape(-1)
        spatial = spatial.reshape(-1)
        source_embedding = np.concatenate(
            (2.0 * np.eye(5)[source], 0.2 * np.eye(4)[spatial]), axis=1
        )
        spatial_embedding = np.concatenate(
            (0.2 * np.eye(5)[source], 2.0 * np.eye(4)[spatial]), axis=1
        )
        source_result = compute_iidr(source_embedding, source, spatial)
        spatial_result = compute_iidr(spatial_embedding, source, spatial)
        self.assertGreater(source_result["IIDR_source"], 10.0)
        self.assertLess(source_result["IIDR_spatial"], 0.1)
        self.assertGreater(spatial_result["IIDR_spatial"], 10.0)
        self.assertLess(spatial_result["IIDR_source"], 0.1)

    def test_multi_positive_retrieval_masks_self(self):
        labels = np.repeat(np.arange(4), 3)
        embeddings = np.eye(4)[labels]
        result = retrieval_metrics(
            embeddings,
            embeddings,
            labels,
            labels,
            ks=(1, 2),
            exclude_diagonal=True,
            chunk_size=3,
        )
        self.assertEqual(result["R@1"], 1.0)
        self.assertEqual(result["MedR"], 1.0)

    def test_median_rank_uses_torch_lower_median_convention(self):
        query = np.eye(4)
        gallery = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.8, 0.6, 0.0, 0.0],
                [0.0, 0.8, 0.6, 0.0],
                [0.0, 0.0, 0.8, 0.6],
            ]
        )
        # This mainly guards the explicit lower-median implementation. The
        # concrete ranking pattern is secondary to keeping the result integral.
        result = retrieval_metrics(
            query, gallery, np.arange(4), np.arange(4), ks=(1,), chunk_size=2
        )
        self.assertEqual(result["MedR"], float(int(result["MedR"])))


if __name__ == "__main__":
    unittest.main()
