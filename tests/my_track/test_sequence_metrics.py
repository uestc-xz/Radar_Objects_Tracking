import unittest

import torch

from original.my_track.metrics import sequence_tracking_metrics


def make_sequence():
    gt = torch.zeros(4, 3, 2)
    gt[:, 0] = torch.tensor([1.0, 2.0])
    gt[:, 1] = torch.tensor([4.0, 2.0])
    pred = gt.clone()
    active = torch.zeros(4, 3, dtype=torch.bool)
    active[:, :2] = True
    observed = active.clone()
    return pred, active, observed, gt


class SequenceMetricsTests(unittest.TestCase):
    def test_fixed_warmup_mapping_reports_stable_identity_and_zero_error(self):
        pred, active, observed, gt = make_sequence()
        metrics = sequence_tracking_metrics(pred, active, observed, gt, warmup=2, match_gate_m=1.0)

        self.assertEqual(metrics["idsw"], 0)
        self.assertEqual(metrics["slot_to_gt"], [0, 1, -1])
        self.assertAlmostEqual(metrics["mae_m"], 0.0, places=6)
        self.assertAlmostEqual(metrics["rmse_m"], 0.0, places=6)

    def test_persistent_slot_swap_after_warmup_counts_identity_switches(self):
        pred, active, observed, gt = make_sequence()
        pred[2:, 0] = gt[2:, 1]
        pred[2:, 1] = gt[2:, 0]
        metrics = sequence_tracking_metrics(pred, active, observed, gt, warmup=2, match_gate_m=1.0)

        self.assertEqual(metrics["idsw"], 2)
        self.assertGreater(metrics["mae_m"], 0.0)

    def test_unobserved_but_active_frames_are_counted_as_retained_predictions(self):
        pred, active, observed, gt = make_sequence()
        observed[2:, 0] = False
        metrics = sequence_tracking_metrics(pred, active, observed, gt, warmup=2, match_gate_m=1.0)

        self.assertEqual(metrics["predicted_count"], 2)
        self.assertEqual(metrics["miss_retention"], 1.0)
        self.assertEqual(metrics["observed_count"], 6)

    def test_untracked_targets_are_reported_instead_of_looking_like_zero_error_success(self):
        pred, active, observed, gt = make_sequence()
        active.zero_()
        observed.zero_()
        metrics = sequence_tracking_metrics(pred, active, observed, gt, warmup=2, match_gate_m=1.0)

        self.assertEqual(metrics["slot_to_gt"], [0, 1, -1])
        self.assertEqual(metrics["target_coverage"], 0.0)
        self.assertEqual(metrics["target_frame_coverage"], 0.0)
        self.assertEqual(metrics["untracked_target_frames"], 4)
        self.assertEqual(metrics["point_count"], 0)


if __name__ == "__main__":
    unittest.main()
