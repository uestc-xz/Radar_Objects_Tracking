import unittest

import numpy as np

from original.my_track.single_frame_evaluation import (
    compare_anchor_localization, evaluate_presence_thresholds,
)


class SingleFrameEvaluationTests(unittest.TestCase):
    def test_fixed_anchor_pairs_measure_regression_even_when_it_moves_outside_gate(self):
        anchors = np.array([[[1.0, 1.0], [8.0, 8.0], [0.0, 0.0]]])
        corrected = np.array([[[2.0, 1.0], [1.1, 1.0], [0.0, 0.0]]])
        targets = np.array([[[1.2, 1.0], [0.0, 0.0], [0.0, 0.0]]])
        result = compare_anchor_localization(
            anchors, corrected, targets, np.array([[True, True, False]]),
            match_radius_m=0.5,
        )
        self.assertEqual(result["matched_pairs"], 1)
        self.assertAlmostEqual(result["anchor_mae_m"], 0.2)
        self.assertAlmostEqual(result["corrected_mae_m"], 0.8)
        self.assertEqual(result["improved_fraction"], 0.0)

    def test_no_valid_proposals_still_counts_all_targets_as_misses(self):
        positions = np.zeros((1, 3, 2))
        targets = np.array([[[1.0, 1.0], [2.0, 2.0], [0.0, 0.0]]])
        mask = np.zeros((1, 3), dtype=bool)
        metrics = evaluate_presence_thresholds(positions, np.zeros((1, 3)), targets, (0.0,), prediction_mask=mask)
        self.assertEqual(metrics[0.0]["fn"], 2)
        self.assertEqual(metrics[0.0]["tp"], 0)
        comparison = compare_anchor_localization(positions, positions, targets, mask)
        self.assertEqual(comparison["matched_pairs"], 0)
        self.assertIsNone(comparison["corrected_mae_m"])

    def test_padding_is_excluded_even_at_zero_threshold(self):
        positions = np.array([[[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]]])
        result = evaluate_presence_thresholds(
            positions, np.array([[2.0, -20.0, -20.0]]), positions,
            thresholds=(0.0,), prediction_mask=np.array([[True, False, False]]),
        )
        self.assertEqual(result[0.0]["predicted_points"], 1)
        self.assertEqual(result[0.0]["fp"], 0)

    def test_large_finite_logits_do_not_overflow(self):
        positions = np.array([[[1.0, 1.0], [8.0, 8.0], [9.0, 9.0]]])
        targets = np.array([[[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]]])
        with np.errstate(over="raise"):
            result = evaluate_presence_thresholds(
                positions, np.array([[1000.0, -1000.0, -1000.0]]), targets, (0.5,),
            )
        self.assertEqual(result[0.5]["f1"], 1.0)

    def test_threshold_sweep_changes_detection_counts_without_changing_positions(self):
        positions = np.array([[[1.0, 1.0], [4.0, 4.0], [8.0, 8.0]]])
        presence_logits = np.array([[2.0, 2.0, -2.0]])
        targets = np.array([[[1.0, 1.0], [4.0, 4.0], [0.0, 0.0]]])

        result = evaluate_presence_thresholds(
            positions, presence_logits, targets, thresholds=(0.5, 0.9), match_radius_m=1.0
        )

        self.assertEqual(result[0.5]["tp"], 2)
        self.assertEqual(result[0.5]["fp"], 0)
        self.assertEqual(result[0.5]["fn"], 0)
        self.assertEqual(result[0.5]["f1"], 1.0)
        self.assertEqual(result[0.9]["tp"], 0)
        self.assertEqual(result[0.9]["fn"], 2)
        self.assertEqual(result[0.9]["predicted_points"], 0)
        np.testing.assert_array_equal(positions, [[[1.0, 1.0], [4.0, 4.0], [8.0, 8.0]]])

    def test_threshold_sweep_rejects_non_finite_or_out_of_range_thresholds(self):
        positions = np.zeros((0, 3, 2))
        logits = np.zeros((0, 3))
        targets = np.zeros((0, 3, 2))

        with self.assertRaisesRegex(ValueError, "threshold"):
            evaluate_presence_thresholds(positions, logits, targets, thresholds=(-0.1,))
        with self.assertRaisesRegex(ValueError, "threshold"):
            evaluate_presence_thresholds(positions, logits, targets, thresholds=(float("nan"),))


if __name__ == "__main__":
    unittest.main()
