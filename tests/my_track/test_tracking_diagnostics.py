import unittest

import numpy as np

from original.my_track.m1_temporal_baseline import TemporalFrameOutput
from original.my_track.tracking_diagnostics import tracking_diagnostics


def output(points, ids, observed=None):
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    seen = np.ones(len(ids), dtype=bool) if observed is None else np.asarray(observed)
    return TemporalFrameOutput(points, np.ones(len(ids)), np.asarray(ids), seen)


class TrackingDiagnosticsTests(unittest.TestCase):
    def test_perfect_track_has_perfect_identity_and_localization(self):
        targets = np.asarray([[[1, 2], [0, 0], [3, 2]]] * 3)
        frames = [output([[1, 2], [3, 2]], [4, 9])] * 3
        result = tracking_diagnostics(frames, targets)
        self.assertEqual(result['tp'], 6)
        self.assertEqual(result['identity_f1'], 1)
        self.assertEqual(result['matched_mae_m'], 0)

    def test_rebirth_counts_gt_id_change_even_when_owner_switches_are_zero(self):
        frames = [output([[1, 2]], [0]), output([[1, 2]], [1])]
        targets = np.asarray([[[1, 2]], [[1, 2]]])
        result = tracking_diagnostics(frames, targets)
        self.assertEqual(result['gt_id_switches'], 1)
        self.assertEqual(result['track_owner_switches'], 0)
        self.assertEqual(result['identity_f1'], .5)

    def test_empty_gt_slots_keep_original_identity(self):
        frames = [output([[1, 2]], [5]), output([[1, 2]], [5])]
        targets = np.asarray([[[1, 2], [0, 0]], [[0, 0], [1, 2]]])
        result = tracking_diagnostics(frames, targets)
        self.assertEqual(result['track_owner_switches'], 1)
        self.assertEqual(result['identity_f1'], .5)

    def test_missed_frame_then_return_counts_fragment_without_changing_id(self):
        frames = [output([[1, 2]], [0]), output([], []), output([[1, 2]], [0])]
        result = tracking_diagnostics(frames, np.asarray([[[1, 2]]] * 3))
        self.assertEqual(result['fragments'], 1)
        self.assertEqual(result['gt_id_switches'], 0)
        self.assertAlmostEqual(result['identity_f1'], .8)
        self.assertEqual(result['fn'], 1)

    def test_duplicate_prediction_is_false_positive_not_two_matches(self):
        result = tracking_diagnostics(
            [output([[1, 2], [1.1, 2]], [0, 1])], np.asarray([[[1, 2]]]),
        )
        self.assertEqual(result['tp'], 1)
        self.assertEqual(result['fp'], 1)
        self.assertEqual(result['false_emitted_tracks'], 1)

    def test_prediction_points_report_their_own_precision(self):
        result = tracking_diagnostics(
            [output([[1, 2], [4, 2]], [0, 1], [False, False])],
            np.asarray([[[1, 2]]]),
        )
        self.assertEqual(result['coasted_tp'], 1)
        self.assertEqual(result['coasted_fp'], 1)

    def test_empty_outputs_count_every_target_as_missed(self):
        result = tracking_diagnostics([output([], [])], np.asarray([[[1, 2]]]))
        self.assertEqual(result['fn'], 1)
        self.assertEqual(result['identity_f1'], 0)
        self.assertIsNone(result['matched_mae_m'])

    def test_matching_maximizes_cardinality_before_total_distance_at_gate_boundary(self):
        # 五个 1m 配对 vs 四个 0m 配对；必须选前者而非为了小误差丢弃一个 TP。
        points = [[x, 2] for x in range(5)]
        targets = np.asarray([[[x, 2] for x in range(1, 6)]])
        result = tracking_diagnostics([output(points, list(range(5)))], targets)
        self.assertEqual((result['tp'], result['fp'], result['fn']), (5, 0, 0))
        self.assertEqual(result['matched_mae_m'], 1.)

    def test_target_free_frame_counts_predictions_as_false_alarms(self):
        result = tracking_diagnostics([output([[1, 2]], [0])], np.zeros((1, 3, 2)))
        self.assertEqual((result['tp'], result['fp'], result['fn']), (0, 1, 0))
        self.assertEqual(result['identity_f1'], 0)

    def test_invalid_lengths_or_duplicate_ids_are_rejected(self):
        with self.assertRaises(ValueError):
            tracking_diagnostics([], np.asarray([[[1, 2]]]))
        with self.assertRaises(ValueError):
            tracking_diagnostics(
                [output([[1, 2], [2, 2]], [0, 0])], np.asarray([[[1, 2]]]),
            )


if __name__ == '__main__':
    unittest.main()
