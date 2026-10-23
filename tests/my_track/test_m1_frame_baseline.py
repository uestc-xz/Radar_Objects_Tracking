import unittest

import numpy as np

from original.my_track.m1_frame_baseline import cluster_candidates, match_frame


class M1FrameBaselineTests(unittest.TestCase):
    def test_clusters_nearby_candidates_with_confidence_weighted_centroid(self):
        positions = np.array([[0.0, 1.0], [0.2, 1.0], [2.0, 4.0], [0.1, 1.0]])
        scores = np.array([0.5, 0.5, 0.99, 0.49])

        result = cluster_candidates(positions, scores, score_threshold=0.5, cluster_radius_m=0.25)

        np.testing.assert_allclose(result.positions, [[0.1, 1.0], [2.0, 4.0]], atol=1e-7)
        np.testing.assert_allclose(result.scores, [0.5, 0.99], atol=1e-7)
        np.testing.assert_array_equal(result.sizes, [2, 1])

    def test_candidates_exactly_at_cluster_radius_join_the_same_component(self):
        positions = np.array([[0.0, 0.0], [0.25, 0.0]])
        scores = np.array([0.8, 0.9])

        result = cluster_candidates(positions, scores, score_threshold=0.5, cluster_radius_m=0.25)

        self.assertEqual(len(result.positions), 1)
        self.assertEqual(result.sizes.tolist(), [2])

    def test_no_candidates_above_threshold_returns_empty_arrays(self):
        result = cluster_candidates(
            np.array([[1.0, 2.0]]), np.array([0.49]),
            score_threshold=0.5, cluster_radius_m=0.25,
        )

        self.assertEqual(result.positions.shape, (0, 2))
        self.assertEqual(result.scores.shape, (0,))
        self.assertEqual(result.sizes.shape, (0,))

    def test_invalid_measurements_and_parameters_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            cluster_candidates(
                np.array([[np.nan, 1.0]]), np.array([0.9]),
                score_threshold=0.5, cluster_radius_m=0.25,
            )
        with self.assertRaisesRegex(ValueError, "cluster_radius_m"):
            cluster_candidates(
                np.array([[0.0, 0.0]]), np.array([0.9]),
                score_threshold=0.5, cluster_radius_m=-0.1,
            )

    def test_frame_matching_ignores_empty_gt_and_rejects_pairs_outside_radius(self):
        result = match_frame(
            np.array([[0.0, 1.2], [2.2, 2.0], [8.0, 8.0]]),
            np.array([[0.0, 1.0], [2.0, 2.0], [0.0, 0.0]]),
            match_radius_m=0.5,
        )

        self.assertEqual(result.true_positives, 2)
        self.assertEqual(result.false_positives, 1)
        self.assertEqual(result.false_negatives, 0)
        self.assertAlmostEqual(result.error_sum_m, 0.4)

    def test_frame_matching_counts_empty_prediction_as_missed_targets(self):
        result = match_frame(np.empty((0, 2)), np.array([[1.0, 2.0], [0.0, 0.0]]))

        self.assertEqual(result.true_positives, 0)
        self.assertEqual(result.false_positives, 0)
        self.assertEqual(result.false_negatives, 1)
        self.assertEqual(result.error_sum_m, 0.0)

    def test_frame_matching_maximizes_in_radius_pairs_before_distance(self):
        result = match_frame(
            np.array([[0.0, 1.0], [1.0, 1.0]]),
            np.array([[0.0, 1.0], [-1.0, 1.0]]),
            match_radius_m=1.0,
        )

        self.assertEqual(result.true_positives, 2)
        self.assertEqual(result.false_positives, 0)
        self.assertEqual(result.false_negatives, 0)
        self.assertEqual(result.error_sum_m, 2.0)


if __name__ == "__main__":
    unittest.main()
