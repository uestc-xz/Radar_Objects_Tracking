import unittest

import torch

from original.my_track.measurement import Measurements
from original.my_track.joint_tracker import JointTracker


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


if __name__ == "__main__":
    unittest.main()
