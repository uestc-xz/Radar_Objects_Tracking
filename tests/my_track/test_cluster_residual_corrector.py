import unittest

import torch

from original.my_track import cluster_residual_corrector as residual


class ClusterResidualCorrectorTests(unittest.TestCase):
    def test_explicit_cluster_evidence_has_expected_geometry_and_ignores_padding(self):
        model = residual.ClusterResidualCorrector(hidden_dim=16, use_cluster_evidence=True)
        relative = torch.tensor([[[[0.0, 0.0], [0.2, 0.0]],
                                 [[0.0, 0.0], [0.2, 0.0]],
                                 [[0.0, 0.0], [0.2, 0.0]]]])
        scores = torch.ones(1, 3, 2)
        local = torch.ones(1, 3, 2, dtype=torch.bool)
        anchors = torch.tensor([[[1.0, 2.0], [2.0, 2.0], [0.0, 0.0]]])
        mask = torch.tensor([[True, True, False]])
        evidence = model._cluster_evidence(relative, scores, local, anchors, mask)
        self.assertEqual(evidence.shape, (1, 3, 8))
        torch.testing.assert_close(evidence[0, 0, :4], torch.tensor([0.1, 0.0, 0.1, 0.0]) / 2.7)
        torch.testing.assert_close(evidence[0, 0, 4:7], torch.ones(3))
        self.assertAlmostEqual(evidence[0, 0, 7].item(), 1 / 2.7, places=6)
        self.assertEqual(evidence[0, 2].abs().sum().item(), 0)

    def test_evidence_variant_preserves_anchors_at_initialization_and_backpropagates(self):
        model = residual.ClusterResidualCorrector(hidden_dim=16, use_cluster_evidence=True)
        output = model(torch.tensor([[[1.0, 2.0], [1.2, 2.1]]]), torch.tensor([[0.99, 0.97]]))
        torch.testing.assert_close(output.positions, output.anchors)
        losses = residual.cluster_residual_loss(output, torch.tensor([[[1.2, 2.2]]]))
        losses["total"].backward()
        self.assertTrue(torch.isfinite(model.position_head.weight.grad).all())
        self.assertTrue(torch.isfinite(model.presence_head.weight.grad).all())

    def test_float32_threshold_boundary_cannot_disappear_between_groups(self):
        model = residual.ClusterResidualCorrector(hidden_dim=16)
        output = model(torch.tensor([[[1.0, 2.0]]]), torch.tensor([[0.96]]))
        self.assertEqual(output.valid_mask.sum().item(), 1)
        torch.testing.assert_close(output.anchors[0, 0], torch.tensor([1.0, 2.0]))

    def test_zero_initialization_preserves_high_confidence_cluster_and_score(self):
        model = residual.ClusterResidualCorrector(hidden_dim=16)
        positions = torch.tensor([[[1.0, 2.0], [1.2, 2.0], [4.0, 4.0], [9.0, 9.0]]])
        scores = torch.tensor([[0.99, 0.97, 0.9, 0.1]])

        output = model(positions, scores)

        centroid_x = (1.0 * 0.99 + 1.2 * 0.97) / 1.96
        torch.testing.assert_close(output.positions[0, 0], torch.tensor([centroid_x, 2.0]))
        torch.testing.assert_close(output.presence_logits[0, :2].sigmoid(), scores[0, [0, 2]])
        self.assertEqual(output.valid_mask.tolist(), [[True, True, False]])
        torch.testing.assert_close(output.positions, output.anchors)

    def test_low_confidence_neighbor_does_not_shift_primary_anchor(self):
        model = residual.ClusterResidualCorrector(hidden_dim=16)
        output = model(torch.tensor([[[1.0, 2.0], [1.6, 2.0]]]), torch.tensor([[0.99, 0.85]]))
        torch.testing.assert_close(output.positions[0, 0], torch.tensor([1.0, 2.0]))
        self.assertEqual(output.valid_mask.sum().item(), 1)

    def test_weak_bridge_cannot_create_a_third_high_confidence_proposal(self):
        model = residual.ClusterResidualCorrector(hidden_dim=16)
        positions = torch.tensor([[[1.0, 2.0], [3.0, 2.0], [1.7, 2.0], [2.3, 2.0]]])
        output = model(positions, torch.tensor([[0.99, 0.98, 0.85, 0.85]]))
        self.assertEqual(output.valid_mask.sum().item(), 2)

    def test_candidate_permutation_preserves_outputs_including_tied_scores(self):
        torch.manual_seed(42)
        model = residual.ClusterResidualCorrector(hidden_dim=16)
        positions = torch.tensor([[[1.0, 2.0], [1.2, 2.1], [-1.0, 4.0], [2.0, 5.0]]])
        scores = torch.tensor([[0.99, 0.97, 0.99, 0.85]])
        expected = model(positions, scores)
        actual = model(positions[:, [2, 1, 3, 0]], scores[:, [2, 1, 3, 0]])
        for first, second in zip(expected, actual):
            torch.testing.assert_close(first, second)

    def test_position_residual_is_bounded_even_with_extreme_head_parameters(self):
        model = residual.ClusterResidualCorrector(hidden_dim=16)
        with torch.no_grad():
            model.position_head.bias.fill_(1000.0)
        output = model(torch.tensor([[[1.0, 2.0]]]), torch.tensor([[0.99]]))
        self.assertTrue(((output.positions - output.anchors).abs() <= 0.5).all())

    def test_no_proposals_gives_masked_finite_outputs_and_zero_supervision(self):
        model = residual.ClusterResidualCorrector(hidden_dim=16)
        output = model(torch.zeros(2, 50, 2), torch.zeros(2, 50))
        losses = residual.cluster_residual_loss(output, torch.zeros(2, 3, 2))
        self.assertFalse(output.valid_mask.any())
        self.assertTrue(torch.isfinite(output.positions).all())
        self.assertTrue((output.presence_logits.sigmoid() < 0.001).all())
        self.assertEqual(losses["total"].item(), 0.0)
        losses["total"].backward()

    def test_loss_ignores_far_gt_and_matches_using_fixed_anchor(self):
        positions = torch.tensor([[[1.1, 1.0], [9.0, 9.0], [0.0, 0.0]]], requires_grad=True)
        logits = torch.zeros(1, 3, requires_grad=True)
        anchors = torch.tensor([[[1.0, 1.0], [9.0, 9.0], [0.0, 0.0]]])
        output = residual.ClusterResidualOutput(positions, logits, anchors, torch.tensor([[True, True, False]]))
        targets = torch.tensor([[[1.2, 1.0], [3.0, 3.0], [0.0, 0.0]]])

        losses = residual.cluster_residual_loss(output, targets)

        self.assertAlmostEqual(losses["position"].item(), 0.0025, places=6)
        self.assertAlmostEqual(losses["presence"].item(), 0.693147, places=5)
        losses["total"].backward()
        self.assertLess(logits.grad[0, 0].item(), 0)
        self.assertGreater(logits.grad[0, 1].item(), 0)
        self.assertEqual(logits.grad[0, 2].item(), 0)
        self.assertEqual(positions.grad[0, 1].abs().sum().item(), 0)

    def test_loss_maximizes_in_radius_matching_before_distance(self):
        anchors = torch.tensor([[[1.0, 1.0], [1.9, 1.0], [0.0, 0.0]]])
        output = residual.ClusterResidualOutput(
            anchors.clone().requires_grad_(), torch.zeros(1, 3, requires_grad=True),
            anchors, torch.tensor([[True, True, False]]),
        )
        targets = torch.tensor([[[1.1, 1.0], [0.2, 1.0], [0.0, 0.0]]])
        losses = residual.cluster_residual_loss(output, targets)
        losses["total"].backward()
        self.assertTrue((output.presence_logits.grad[0, :2] < 0).all())

    def test_invalid_input_is_rejected(self):
        model = residual.ClusterResidualCorrector(hidden_dim=16)
        for score in (float("nan"), float("inf"), -0.1, 1.1):
            with self.subTest(score=score), self.assertRaises(ValueError):
                model(torch.zeros(1, 1, 2), torch.tensor([[score]]))


if __name__ == "__main__":
    unittest.main()
