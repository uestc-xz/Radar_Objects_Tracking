import unittest

import numpy as np
import torch

from original.my_track.m1_feature_projection import (
    FeatureProjection,
    build_identity_triplets,
    match_candidate_features,
    triplet_cosine_loss,
)


class M1FeatureProjectionTests(unittest.TestCase):
    def test_projection_returns_normalized_features_and_preserves_batch_order(self):
        torch.manual_seed(3)
        model = FeatureProjection(input_dim=4, output_dim=2)
        inputs = torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.]])

        outputs = model(inputs)

        self.assertEqual(tuple(outputs.shape), (2, 2))
        torch.testing.assert_close(outputs.norm(dim=-1), torch.ones(2))
        torch.testing.assert_close(model(inputs.flip(0)), outputs.flip(0))

    def test_projection_rejects_invalid_shape_and_nonfinite_values(self):
        model = FeatureProjection(input_dim=4, output_dim=2)
        with self.assertRaises(ValueError):
            model(torch.ones(2, 3))
        with self.assertRaises(ValueError):
            model(torch.tensor([[1., 0., float('nan'), 0.]]))

    def test_candidate_matching_uses_distance_gate_score_threshold_and_gt_slot(self):
        positions = np.asarray([[0.1, 2.], [0.2, 2.], [2.1, 2.]])
        scores = np.asarray([.99, .95, .98])
        features = np.eye(3, dtype=np.float32)
        targets = np.asarray([[0., 2.], [2., 2.], [0., 0.]])

        matched_features, identities = match_candidate_features(
            positions, scores, features, targets,
            gate_m=.5, score_threshold=.96,
        )

        np.testing.assert_array_equal(matched_features, features[[0, 2]])
        np.testing.assert_array_equal(identities, [0, 1])

    def test_candidate_matching_rejects_invalid_or_ambiguous_shapes(self):
        with self.assertRaises(ValueError):
            match_candidate_features(
                np.zeros((2, 2)), np.ones(2), np.zeros((2, 4)), np.zeros((2, 2)),
            )
        with self.assertRaises(ValueError):
            match_candidate_features(
                np.zeros((2, 2)), np.asarray([1., np.nan]), np.zeros((2, 4)),
                np.zeros((3, 2)),
            )

    def test_triplets_are_local_identity_consistent_and_temporally_bounded(self):
        identities = np.asarray([0, 1, 0, 1, 0])
        frames = np.asarray([0, 0, 2, 2, 20])

        triplets = build_identity_triplets(identities, frames, max_positive_gap=3)

        self.assertGreater(len(triplets), 0)
        for anchor, positive, negative in triplets:
            self.assertEqual(identities[anchor], identities[positive])
            self.assertNotEqual(identities[anchor], identities[negative])
            self.assertLessEqual(abs(int(frames[anchor]) - int(frames[positive])), 3)
            self.assertLessEqual(abs(int(frames[anchor]) - int(frames[negative])), 3)

    def test_triplet_builder_returns_empty_without_positive_and_negative_pairs(self):
        self.assertEqual(
            build_identity_triplets([0, 0], [1, 2], max_positive_gap=2).shape,
            (0, 3),
        )
        self.assertEqual(
            build_identity_triplets([0, 1], [1, 2], max_positive_gap=2).shape,
            (0, 3),
        )

    def test_triplet_cosine_loss_rewards_correct_identity_ordering(self):
        anchor = torch.tensor([[1., 0.]])
        positive = torch.tensor([[1., 0.]])
        negative = torch.tensor([[0., 1.]])

        loss = triplet_cosine_loss(anchor, positive, negative, margin=.2)

        self.assertAlmostEqual(float(loss), 0., places=6)
        self.assertTrue(torch.isfinite(loss))


if __name__ == '__main__':
    unittest.main()
