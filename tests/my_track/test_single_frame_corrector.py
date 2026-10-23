import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from original.my_track.single_frame_corrector import (
    SingleFrameSetPredictor,
    single_frame_set_loss,
)
from scripts.train_single_frame_corrector import CachedM1Frames


class SingleFrameCorrectorTests(unittest.TestCase):
    def test_predictor_returns_three_unordered_points_and_presence_logits(self):
        model = SingleFrameSetPredictor(hidden_dim=32, num_heads=4)
        positions = torch.zeros(2, 50, 2)
        scores = torch.zeros(2, 50)

        predicted_positions, presence_logits = model(positions, scores)

        self.assertEqual(tuple(predicted_positions.shape), (2, 3, 2))
        self.assertEqual(tuple(presence_logits.shape), (2, 3))
        self.assertTrue(torch.isfinite(predicted_positions).all())
        self.assertTrue(torch.isfinite(presence_logits).all())

    def test_candidate_order_does_not_change_set_prediction(self):
        torch.manual_seed(7)
        model = SingleFrameSetPredictor(hidden_dim=32, num_heads=4).eval()
        positions = torch.randn(1, 50, 2)
        scores = torch.rand(1, 50)
        permutation = torch.randperm(50)

        expected = model(positions, scores)
        actual = model(positions[:, permutation], scores[:, permutation])

        torch.testing.assert_close(actual[0], expected[0], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(actual[1], expected[1], atol=1e-6, rtol=1e-6)

    def test_set_loss_is_invariant_to_ground_truth_slot_permutation(self):
        predicted_positions = torch.tensor([[[1.1, 1.0], [4.1, 4.0], [9.0, 9.0]]], requires_grad=True)
        presence_logits = torch.tensor([[4.0, 4.0, -4.0]], requires_grad=True)
        targets = torch.tensor([[[1.0, 1.0], [4.0, 4.0], [0.0, 0.0]]])
        permuted_targets = targets[:, [1, 2, 0]]

        first = single_frame_set_loss(predicted_positions, presence_logits, targets)
        second = single_frame_set_loss(predicted_positions, presence_logits, permuted_targets)

        torch.testing.assert_close(first["total"], second["total"])
        first["total"].backward()
        self.assertTrue(torch.isfinite(predicted_positions.grad).all())
        self.assertTrue(torch.isfinite(presence_logits.grad).all())

    def test_empty_frame_trains_all_predictions_as_absent_with_zero_position_loss(self):
        predicted_positions = torch.tensor([[[3.0, 2.0], [-1.0, 0.5], [2.0, 4.0]]], requires_grad=True)
        presence_logits = torch.zeros(1, 3, requires_grad=True)
        targets = torch.zeros(1, 3, 2)

        losses = single_frame_set_loss(predicted_positions, presence_logits, targets)

        self.assertEqual(losses["position"].item(), 0.0)
        self.assertAlmostEqual(losses["presence"].item(), 0.693147, places=5)
        losses["total"].backward()
        self.assertTrue(torch.isfinite(predicted_positions.grad).all())
        self.assertTrue(torch.isfinite(presence_logits.grad).all())

    def test_cached_frame_adapter_reads_cold_cache_paths_and_mean_gt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root = root / "data"
            cache_root = root / "cache"
            (data_root / "record" / "signals").mkdir(parents=True)
            labels_dir = data_root / "record" / "labels"
            labels_dir.mkdir()
            cache_dir = cache_root / "record" / "m1_cache"
            cache_dir.mkdir(parents=True)
            np.save(data_root / "record" / "signals" / "1.npy", np.zeros((8, 89, 10), np.complex64))
            labels = np.zeros((10, 6), dtype=np.float32)
            labels[:, 0:2] = (1.25, 2.5)
            np.save(labels_dir / "1.npy", labels)
            output = np.zeros((50, 4), dtype=np.float32)
            logits = np.zeros((50, 2), dtype=np.float32)
            logits[0] = (0.0, np.log(3.0))
            features = np.zeros((50, 128), dtype=np.float32)
            np.savez(cache_dir / "1.npz", out=output, conf=logits, feat=features)

            dataset = CachedM1Frames(str(data_root), str(cache_root))
            positions, scores, targets = dataset[0]

        self.assertEqual(tuple(positions.shape), (50, 2))
        self.assertAlmostEqual(scores[0].item(), 0.75, places=6)
        torch.testing.assert_close(targets[0], torch.tensor([1.25, 2.5]))
        self.assertEqual(torch.count_nonzero(targets[1:]).item(), 0)


if __name__ == "__main__":
    unittest.main()
