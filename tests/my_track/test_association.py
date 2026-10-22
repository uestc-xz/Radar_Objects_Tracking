import unittest

import torch

from original.my_track.association import (
    association_loss,
    build_assignment_targets,
    tracker_losses,
)


class AssociationTargetTests(unittest.TestCase):
    def test_fixed_slots_allow_many_candidates_per_track(self):
        candidates = torch.tensor([[[1.1, 1.0], [1.2, 1.0], [4.1, 4.0], [8.0, 8.0]]])
        gt = torch.tensor([[[1.0, 1.0], [0.0, 0.0], [4.0, 4.0]]])

        labels = build_assignment_targets(candidates, gt, radius_m=0.5)

        self.assertEqual(labels.tolist(), [[1, 1, 3, 0]])

    def test_empty_ground_truth_labels_all_candidates_as_clutter(self):
        labels = build_assignment_targets(
            torch.ones(2, 5, 2), torch.zeros(2, 3, 2), radius_m=1.0
        )
        self.assertEqual(torch.count_nonzero(labels).item(), 0)

    def test_non_finite_coordinates_are_rejected(self):
        candidates = torch.zeros(1, 2, 2)
        candidates[0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            build_assignment_targets(candidates, torch.zeros(1, 3, 2))

    def test_association_loss_is_finite_and_differentiable_without_positives(self):
        logits = torch.randn(2, 5, 4, requires_grad=True)
        labels = torch.zeros(2, 5, dtype=torch.long)

        loss = association_loss(logits, labels)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_tracker_losses_cover_all_terms_and_backpropagate(self):
        logits = torch.randn(1, 4, 4, requires_grad=True)
        final_pos = torch.randn(1, 3, 2, requires_grad=True)
        existence = torch.sigmoid(torch.randn(1, 3, requires_grad=True))
        confidence = torch.sigmoid(torch.randn(1, 3, requires_grad=True))
        gt = torch.tensor([[[1.0, 2.0], [0.0, 0.0], [3.0, 4.0]]])
        labels = torch.tensor([[1, 0, 3, 0]])

        losses = tracker_losses(logits, labels, final_pos, existence, confidence, gt)

        self.assertEqual(set(losses), {"association", "position", "existence", "confidence"})
        total = sum(losses.values())
        total.backward()
        self.assertIsNotNone(final_pos.grad)
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(total))

    def test_empty_tracks_have_zero_position_loss_and_finite_other_losses(self):
        logits = torch.zeros(1, 2, 4, requires_grad=True)
        final_pos = torch.randn(1, 3, 2, requires_grad=True)
        existence = torch.full((1, 3), 0.2, requires_grad=True)
        confidence = torch.full((1, 3), 0.1, requires_grad=True)
        gt = torch.zeros(1, 3, 2)

        losses = tracker_losses(
            logits, torch.zeros(1, 2, dtype=torch.long), final_pos,
            existence, confidence, gt,
        )

        self.assertEqual(losses["position"].item(), 0.0)
        self.assertTrue(all(torch.isfinite(loss) for loss in losses.values()))


if __name__ == "__main__":
    unittest.main()
