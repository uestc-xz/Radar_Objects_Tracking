import tempfile
import unittest
from pathlib import Path

import torch

from original.my_track.cluster_residual_corrector import ClusterResidualCorrector
from original.my_track.single_frame_corrector import SingleFrameSetPredictor, single_frame_set_loss
from scripts import train_single_frame_corrector as runner


class SingleFrameRunnerTests(unittest.TestCase):
    def test_model_factory_retains_legacy_default(self):
        self.assertIsInstance(runner.build_single_frame_model(), SingleFrameSetPredictor)
        self.assertIsInstance(runner.build_single_frame_model("cluster-residual"), ClusterResidualCorrector)
        self.assertTrue(runner.build_single_frame_model("cluster-evidence").use_cluster_evidence)
        with self.assertRaises(ValueError):
            runner.build_single_frame_model("unknown")

    def test_legacy_loss_path_is_unchanged(self):
        output = (torch.ones(1, 3, 2), torch.zeros(1, 3))
        targets = torch.tensor([[[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]]])
        expected = single_frame_set_loss(*output, targets)
        actual = runner.single_frame_training_loss(output, targets)
        for name in expected:
            torch.testing.assert_close(actual[name], expected[name])

    def test_anchored_runner_uses_gated_loss_for_far_gt(self):
        model = runner.build_single_frame_model("cluster-residual")
        output = model(torch.tensor([[[1.0, 1.0]]]), torch.tensor([[0.99]]))
        targets = torch.tensor([[[8.0, 8.0], [0.0, 0.0], [0.0, 0.0]]])
        losses = runner.single_frame_training_loss(output, targets)
        self.assertEqual(losses["position"].item(), 0.0)
        self.assertGreater(losses["presence"].item(), 4.0)

    def test_best_f1_and_best_loss_are_saved_independently_with_last(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            first = {"epoch": 1, "validation": {"total": 0.1, "f1": 0.8}}
            best_loss, best_f1 = runner.save_single_frame_checkpoints(output, first, float("inf"), -1.0)
            second = {"epoch": 2, "validation": {"total": 0.2, "f1": 0.9}}
            runner.save_single_frame_checkpoints(output, second, best_loss, best_f1)
            for filename, expected_epoch in (
                ("best_single_frame_corrector.pth", 1),
                ("best_f1_single_frame_corrector.pth", 2),
                ("last_single_frame_corrector.pth", 2),
            ):
                checkpoint = torch.load(output / filename, weights_only=False)
                self.assertEqual(checkpoint["epoch"], expected_epoch)


if __name__ == "__main__":
    unittest.main()
