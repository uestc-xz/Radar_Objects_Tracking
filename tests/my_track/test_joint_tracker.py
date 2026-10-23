import unittest

import torch

from original.my_track.measurement import Measurements
from original.my_track.joint_tracker import JointTracker
from original.my_track.association import build_assignment_targets, tracker_losses


def make_state(batch=1, hist_len=4, initialized=False):
    hist = torch.zeros(batch, 3, hist_len, 4)
    if initialized:
        hist[..., -1, :2] = torch.tensor([2.0, 3.0])
        hist[..., -1, 2:] = torch.tensor([1.0, 0.0])
    return {
        "hist": hist,
        "feat": torch.zeros(batch, 3, 128),
        "confidence": torch.zeros(batch, 3),
        "confirmed": torch.zeros(batch, 3, dtype=torch.bool),
        "initialized": torch.full((batch, 3), initialized, dtype=torch.bool),
        "miss": torch.zeros(batch, 3),
        "age": torch.zeros(batch, 3),
    }


class JointTrackerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        self.model = JointTracker(hist_len=4)
        self.measurements = Measurements(
            pos=torch.randn(1, 50, 2),
            score=torch.rand(1, 50),
            feat=torch.randn(1, 50, 128),
        )

    def test_outputs_have_contract_shapes_and_finite_values(self):
        result = self.model(self.measurements, make_state(), dt=0.24)

        self.assertEqual(result.association_logits.shape, (1, 50, 4))
        self.assertEqual(result.assignments.shape, (1, 50))
        self.assertEqual(result.assignment_onehot.shape, (1, 50, 4))
        self.assertEqual(result.final_pos.shape, (1, 3, 2))
        self.assertEqual(result.observed.shape, (1, 3))
        self.assertTrue(torch.isfinite(result.final_pos).all())
        self.assertTrue(torch.isfinite(result.association_logits).all())

    def test_disabled_m1_features_do_not_change_tracker_outputs(self):
        model = JointTracker(hist_len=4, use_m1_features=False)
        state = make_state(initialized=True)
        state["feat"] = torch.randn_like(state["feat"])
        changed_measurements = self.measurements._replace(
            feat=torch.randn_like(self.measurements.feat) * 100.0
        )

        baseline = model(self.measurements, state, dt=0.24)
        changed = model(changed_measurements, state, dt=0.24)

        torch.testing.assert_close(baseline.association_logits, changed.association_logits)
        torch.testing.assert_close(baseline.fused_feat, torch.zeros_like(baseline.fused_feat))
        torch.testing.assert_close(baseline.final_pos, changed.final_pos)

    def test_one_candidate_has_one_hard_class_and_multiple_can_share_a_track(self):
        with torch.no_grad():
            self.model.pair_head[-1].weight.zero_()
            self.model.pair_head[-1].bias.zero_()
            self.model.clutter_head[-1].weight.zero_()
            self.model.clutter_head[-1].bias.fill_(-100.0)
            self.model.slot_embedding.zero_()
        result = self.model(self.measurements, make_state(), dt=0.24)

        self.assertTrue(torch.equal(result.assignment_onehot.sum(-1), torch.ones(1, 50)))
        self.assertTrue((result.assignments == 1).all())
        self.assertTrue(result.observed[0, 0])
        expected = (
            self.measurements.pos[0] * self.measurements.score[0, :, None]
        ).sum(0) / self.measurements.score[0].sum()
        torch.testing.assert_close(result.fused_pos[0, 0], expected, atol=1e-5, rtol=1e-5)

    def test_no_assigned_measurements_returns_history_prediction_without_correction(self):
        with torch.no_grad():
            self.model.pair_head[-1].weight.zero_()
            self.model.pair_head[-1].bias.fill_(-100.0)
            self.model.clutter_head[-1].weight.zero_()
            self.model.clutter_head[-1].bias.fill_(100.0)
        result = self.model(self.measurements, make_state(initialized=True), dt=0.24)

        self.assertFalse(result.observed.any())
        torch.testing.assert_close(result.final_pos, result.predicted_pos)
        torch.testing.assert_close(result.predicted_pos[0, 0], torch.tensor([2.24, 3.0]), atol=1e-5, rtol=0)

    def test_position_correction_can_leave_candidate_extent(self):
        with torch.no_grad():
            self.model.pair_head[-1].weight.zero_()
            self.model.pair_head[-1].bias.zero_()
            self.model.clutter_head[-1].weight.zero_()
            self.model.clutter_head[-1].bias.fill_(-100.0)
            self.model.slot_embedding.zero_()
            self.model.correction_head[-1].weight.zero_()
            self.model.correction_head[-1].bias.copy_(torch.tensor([0.5, 10.0, -10.0]))
        result = self.model(self.measurements, make_state(), dt=0.24)

        self.assertGreater(result.final_pos[0, 0, 0].item(), self.measurements.pos[..., 0].max().item())
        self.assertLess(result.final_pos[0, 0, 1].item(), self.measurements.pos[..., 1].min().item())

    def test_association_and_correction_heads_receive_gradients(self):
        result = self.model(self.measurements, make_state(), dt=0.24)
        loss = result.association_logits.square().mean() + result.final_pos.square().mean()
        loss.backward()

        self.assertIsNotNone(self.model.pair_head[-1].weight.grad)
        self.assertGreater(self.model.pair_head[-1].weight.grad.abs().sum().item(), 0.0)
        self.assertIsNotNone(self.model.correction_head[-1].weight.grad)
        self.assertGreater(self.model.correction_head[-1].weight.grad.abs().sum().item(), 0.0)

    def test_teacher_assignment_controls_fusion_at_full_ratio(self):
        with torch.no_grad():
            self.model.pair_head[-1].weight.zero_()
            self.model.pair_head[-1].bias.zero_()
            self.model.clutter_head[-1].weight.zero_()
            self.model.clutter_head[-1].bias.fill_(-100.0)
            self.model.slot_embedding.zero_()
        teacher_labels = torch.zeros(1, 50, dtype=torch.long)
        teacher_labels[0, 0] = 2

        result = self.model(
            self.measurements, make_state(), dt=0.24,
            teacher_assignment_labels=teacher_labels,
            teacher_assignment_ratio=1.0,
        )

        self.assertTrue((result.assignments == 1).all())
        self.assertFalse(result.observed[0, 0])
        self.assertTrue(result.observed[0, 1])
        torch.testing.assert_close(result.fused_pos[0, 1], self.measurements.pos[0, 0])

    def test_tiny_positive_score_preserves_normalized_fusion(self):
        measurements = Measurements(
            pos=torch.tensor([2.0, 4.0]).expand(1, 50, 2),
            score=torch.full((1, 50), 1e-6),
            feat=torch.ones(1, 50, 128),
        )
        teacher_labels = torch.zeros(1, 50, dtype=torch.long)
        teacher_labels[0, 0] = 1
        result = self.model(
            measurements, make_state(), teacher_assignment_labels=teacher_labels,
            teacher_assignment_ratio=1.0,
        )
        self.assertTrue(result.observed[0, 0])
        torch.testing.assert_close(result.fused_pos[0, 0], measurements.pos[0, 0])
        torch.testing.assert_close(result.fused_feat[0, 0], measurements.feat[0, 0])

    def test_zero_teacher_ratio_is_identical_to_predicted_fusion(self):
        baseline = self.model(self.measurements, make_state(), dt=0.24)
        teacher_labels = torch.full((1, 50), 3, dtype=torch.long)
        mixed = self.model(
            self.measurements, make_state(), dt=0.24,
            teacher_assignment_labels=teacher_labels,
            teacher_assignment_ratio=0.0,
        )

        torch.testing.assert_close(mixed.assignment_onehot, baseline.assignment_onehot)
        torch.testing.assert_close(mixed.fused_pos, baseline.fused_pos)
        torch.testing.assert_close(mixed.final_pos, baseline.final_pos)
        self.assertTrue(torch.equal(mixed.observed, baseline.observed))

    def test_teacher_ratio_requires_labels(self):
        with self.assertRaisesRegex(ValueError, "teacher_assignment_labels"):
            self.model(self.measurements, make_state(), teacher_assignment_ratio=0.5)

    def test_empty_slots_do_not_corrupt_adam_moments(self):
        with torch.no_grad():
            self.model.pair_head[-1].weight.zero_()
            self.model.pair_head[-1].bias.zero_()
            self.model.clutter_head[-1].weight.zero_()
            self.model.clutter_head[-1].bias.fill_(-5.0)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-5)
        gt = torch.tensor([[[1.0, 2.0], [0.0, 0.0], [0.0, 0.0]]])
        result = self.model(self.measurements, make_state())
        self.assertEqual(result.observed.tolist(), [[True, False, False]])
        losses = tracker_losses(
            result.association_logits, build_assignment_targets(self.measurements.pos, gt),
            result.final_pos, result.existence_prob, result.track_confidence, gt,
        )
        sum(losses.values()).backward()
        gradients = [p.grad for p in self.model.parameters() if p.grad is not None]
        self.assertTrue(all(torch.isfinite(g).all() and g.abs().max() < 1e6 for g in gradients))
        optimizer.step()
        self.assertTrue(all(torch.isfinite(state["exp_avg_sq"]).all()
                            for state in optimizer.state.values()))

    def test_zero_scores_have_zero_fusion_and_finite_backward(self):
        measurements = self.measurements._replace(score=torch.zeros(1, 50))
        result = self.model(measurements, make_state())
        self.assertFalse(result.observed.any())
        torch.testing.assert_close(result.fused_pos, torch.zeros(1, 3, 2))
        torch.testing.assert_close(result.fused_feat, torch.zeros(1, 3, 128))
        (result.existence_prob.sum() + result.track_confidence.sum()).backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in self.model.parameters()
                            if p.grad is not None))

    def test_new_track_starts_at_measurement_before_learning_correction(self):
        measurements = self.measurements._replace(
            pos=torch.tensor([2.0, 4.0]).expand(1, 50, 2))
        labels = torch.ones(1, 50, dtype=torch.long)
        result = self.model(measurements, make_state(),
                            teacher_assignment_labels=labels, teacher_assignment_ratio=1.0)
        torch.testing.assert_close(result.final_pos[0, 0], torch.tensor([2.0, 4.0]))
        loss = (result.final_pos[0, 0] - torch.tensor([2.1, 4.2])).square().mean()
        loss.backward()
        self.assertGreater(self.model.correction_head[-1].weight.grad.abs().sum(), 0)

    def test_initialized_track_keeps_learned_history_measurement_gate(self):
        measurements = self.measurements._replace(
            pos=torch.tensor([2.0, 4.0]).expand(1, 50, 2))
        labels = torch.tensor([[1] * 25 + [2] * 25])
        state = make_state(initialized=True)
        state["initialized"][0, 1:] = False
        with torch.no_grad():
            self.model.correction_head[-1].weight.zero_()
            self.model.correction_head[-1].bias.zero_()
        result = self.model(measurements, state, dt=0.24,
                            teacher_assignment_labels=labels, teacher_assignment_ratio=1.0)
        torch.testing.assert_close(result.final_pos[0, 0], torch.tensor([2.12, 3.5]))
        torch.testing.assert_close(result.final_pos[0, 1], torch.tensor([2.0, 4.0]))


if __name__ == "__main__":
    unittest.main()
