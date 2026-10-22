import unittest
import tempfile
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from original.my_track.joint_tracker import JointTracker
from original.my_track.track_state import TrackBank
from original.my_dataset.track_sequence_dataset import TrackSequenceColdDataset
from original.my_track import config
from original.my_track.train import evaluate_full_sequence_dataset, run_training
from original.my_track.train_tracker import forward_batch, forward_batch_cold


class StubM1(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.seen_queries = []

    def forward(self, signals):
        batch = signals.shape[0]
        self.seen_queries.append(batch)
        out = torch.zeros(batch, 50, 4, device=signals.device)
        out[..., 0] = self.scale
        out[..., 1] = 2.0
        conf = torch.zeros(batch, 50, 2, device=signals.device)
        feat = self.scale.expand(batch, 50, 128)
        return out, conf, feat


def make_inputs(batch=1, frames=3):
    out = torch.zeros(batch, frames, 50, 4)
    out[..., 0] = 1.0
    out[..., 1] = 2.0
    conf = torch.zeros(batch, frames, 50, 2)
    feat = torch.randn(batch, frames, 50, 128)
    gt = torch.zeros(batch, frames, 3, 2)
    gt[..., 0, :] = torch.tensor([1.0, 2.0])
    return out, conf, feat, gt


class TrackerTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)
        self.tracker = JointTracker(hist_len=2)
        self.banks = [TrackBank(hist_len=2, feat_dim=128)]

    def test_cold_path_uses_cached_tensors_and_supervises_every_frame(self):
        out, conf, feat, gt = make_inputs(frames=3)

        loss, n_frames, assign_hist, trajectory, loss_parts = forward_batch_cold(
            out, conf, feat, gt, self.banks, self.tracker, return_details=True
        )

        self.assertEqual(n_frames, 3)
        self.assertEqual(sum(assign_hist), 3 * 50)
        self.assertEqual(trajectory["final_pos"].shape, (1, 3, 3, 2))
        self.assertEqual(trajectory["observed"].shape, (1, 3, 3))
        self.assertEqual(set(loss_parts), {"association", "position", "existence", "confidence"})
        loss.backward()
        self.assertIsNotNone(self.tracker.correction_head[-1].weight.grad)

    def test_warm_path_freezes_m1_but_keeps_tracker_trainable(self):
        model = StubM1()
        signals = torch.zeros(1, 3, 8, 89, 10, dtype=torch.complex64)
        gt = torch.zeros(1, 3, 3, 2)
        gt[..., 0, :] = torch.tensor([1.0, 2.0])

        loss, n_frames, assign_hist = forward_batch(
            signals, gt, model, self.banks, self.tracker, m1_batch_size=2
        )
        loss.backward()

        self.assertEqual(n_frames, 3)
        self.assertEqual(sum(assign_hist), 3 * 50)
        self.assertFalse(model.training)
        self.assertFalse(model.scale.requires_grad)
        self.assertIsNone(model.scale.grad)
        self.assertEqual(model.seen_queries, [2, 1])
        self.assertIsNotNone(self.tracker.pair_head[-1].weight.grad)

    def test_window_entry_resets_previous_track_state(self):
        out, conf, feat, gt = make_inputs(frames=1)
        self.banks[0].confirmed[:] = True
        self.banks[0].frames_seen = 99

        forward_batch_cold(out, conf, feat, gt, self.banks, self.tracker)

        self.assertEqual(self.banks[0].frames_seen, 1)
        self.assertFalse(self.banks[0].confirmed.any())

    def test_full_sequence_validation_resets_once_and_reads_frames_lazily(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "data"
            cache_root = Path(temp_dir) / "cache"
            folder = root / "record" 
            signal_dir = folder / "signals"
            label_dir = folder / "labels"
            cache_dir = cache_root / "record" / "m1_cache"
            signal_dir.mkdir(parents=True)
            label_dir.mkdir()
            cache_dir.mkdir(parents=True)
            for frame_idx in range(3):
                np.save(signal_dir / f"{frame_idx}.npy", np.zeros((8, 89, 10), dtype=np.complex64))
                labels = np.zeros((10, 6), dtype=np.float32)
                labels[:, :2] = (1.0, 2.0)
                np.save(label_dir / f"{frame_idx}.npy", labels)
                out = np.zeros((50, 4), dtype=np.float32)
                out[:, :2] = (1.0, 2.0)
                conf = np.zeros((50, 2), dtype=np.float32)
                conf[:, 1] = 3.0
                feat = np.zeros((50, 128), dtype=np.float32)
                np.savez(cache_dir / f"{frame_idx}.npz", out=out, conf=conf, feat=feat)

            dataset = TrackSequenceColdDataset(
                str(root), str(cache_root), window_size=20, stride=10, full_sequence=True
            )
            banks = [TrackBank(hist_len=2, feat_dim=128)]
            banks[0].frames_seen = 99
            metrics = evaluate_full_sequence_dataset(
                dataset, None, self.tracker, banks, torch.device("cpu"), is_cold=True
            )

        self.assertEqual(metrics["sequences"], 1)
        self.assertEqual(metrics["frames"], 3)
        self.assertEqual(banks[0].frames_seen, 3)

    def test_training_reports_batch_and_validation_progress(self):
        out, conf, feat, gt = make_inputs(frames=1)
        tracker = JointTracker(hist_len=2)
        optimizer = torch.optim.Adam(tracker.parameters(), lr=1e-3)
        banks = [TrackBank(hist_len=2, feat_dim=128)]
        progress_output = StringIO()

        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = str(Path(temp_dir) / "tracker_train_log.csv")
            with (
                patch.object(config, "EPOCHS", 1),
                patch.object(config, "IS_COLD", True),
                patch.object(config, "OUT_DIR", temp_dir),
                redirect_stderr(progress_output),
            ):
                run_training(
                    torch.device("cpu"), None, tracker, optimizer,
                    [(out, conf, feat, gt)], [], banks, csv_path,
                )

        rendered = progress_output.getvalue()
        self.assertIn("Epoch 1/1 [train]", rendered)
        self.assertIn("Epoch 1/1 [val]", rendered)
        self.assertIn("loss=", rendered)


if __name__ == "__main__":
    unittest.main()
