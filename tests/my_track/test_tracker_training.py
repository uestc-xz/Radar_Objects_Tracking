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
from original.my_track.train import (
    evaluate_full_sequence_dataset,
    gradient_l2_norm,
    run_training,
)
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

    def test_teacher_assignment_schedule_warmup_and_linear_anneal(self):
        self.assertEqual(config.teacher_assignment_ratio(1, warmup_epochs=2, anneal_epochs=4), 1.0)
        self.assertEqual(config.teacher_assignment_ratio(2, warmup_epochs=2, anneal_epochs=4), 1.0)
        self.assertEqual(config.teacher_assignment_ratio(3, warmup_epochs=2, anneal_epochs=4), 0.75)
        self.assertEqual(config.teacher_assignment_ratio(5, warmup_epochs=2, anneal_epochs=4), 0.25)
        self.assertEqual(config.teacher_assignment_ratio(6, warmup_epochs=2, anneal_epochs=4), 0.0)
        self.assertEqual(config.teacher_assignment_ratio(7, warmup_epochs=2, anneal_epochs=4), 0.0)

    def test_teacher_curriculum_uses_one_training_forward_per_frame(self):
        out, conf, feat, gt = make_inputs(frames=2)
        optimizer = torch.optim.Adam(self.tracker.parameters(), lr=1e-3)
        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.object(config, "EPOCHS", 1),
                patch.object(config, "IS_COLD", True),
                patch.object(config, "OUT_DIR", temp_dir),
                patch.object(self.tracker, "forward", wraps=self.tracker.forward) as forward,
            ):
                run_training(
                    torch.device("cpu"), None, self.tracker, optimizer,
                    [(out, conf, feat, gt)], [], self.banks,
                    str(Path(temp_dir) / "tracker_train_log.csv"), show_progress=False,
                )
            self.assertEqual(forward.call_count, 2)
            for call in forward.call_args_list:
                self.assertEqual(call.kwargs["teacher_assignment_ratio"], 1.0)
                self.assertIsNotNone(call.kwargs["teacher_assignment_labels"])
        self.assertEqual(self.banks[0].frames_seen, 2)

    def test_diagnostic_learning_rate_and_gradient_norm_are_calibrated(self):
        parameter = nn.Parameter(torch.tensor([0.0, 0.0]))
        parameter.grad = torch.tensor([3.0, 4.0])
        unused = nn.Parameter(torch.tensor([1.0]))

        self.assertEqual(config.LR, 3e-4)
        self.assertEqual(gradient_l2_norm([parameter, unused]), 5.0)

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
        self.assertIn("association_accuracy", metrics)
        self.assertIn("association_foreground_accuracy", metrics)
        self.assertIn("association_true_probability", metrics)
        self.assertIn("active_position_out_of_bounds_ratio", metrics)
        self.assertGreaterEqual(metrics["association_accuracy"], 0.0)
        self.assertLessEqual(metrics["association_accuracy"], 1.0)

    def test_full_sequence_validation_batches_independent_sequences(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "data"
            cache_root = Path(temp_dir) / "cache"
            for sequence_name in ("record_a", "record_b"):
                signal_dir = root / sequence_name / "signals"
                label_dir = root / sequence_name / "labels"
                cache_dir = cache_root / sequence_name / "m1_cache"
                signal_dir.mkdir(parents=True)
                label_dir.mkdir()
                cache_dir.mkdir(parents=True)
                for frame_idx in range(2):
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
            banks = [TrackBank(hist_len=2, feat_dim=128) for _ in range(2)]
            with patch("original.my_track.train.forward_batch_cold", wraps=forward_batch_cold) as forward:
                with patch.object(config, "VAL_SEQUENCE_BATCH_SIZE", 2):
                    metrics = evaluate_full_sequence_dataset(
                        dataset, None, self.tracker, banks, torch.device("cpu"), is_cold=True
                    )

            self.assertEqual(metrics["sequences"], 2)
            self.assertEqual(metrics["frames"], 4)
            self.assertEqual(max(call.args[0].shape[0] for call in forward.call_args_list), 2)

    def test_parallel_validation_matches_serial_for_mixed_sequences(self):
        with torch.no_grad():
            self.tracker.pair_head[-1].weight.zero_()
            self.tracker.pair_head[-1].bias.fill_(2.0)
            self.tracker.clutter_head[-1].weight.zero_()
            self.tracker.clutter_head[-1].bias.fill_(-2.0)
        with tempfile.TemporaryDirectory() as temp_dir:
            cold_records, warm_records = [], []
            for seq_idx, (length, people) in enumerate(((4, 1), (2, 3), (3, 0), (1, 2))):
                cold_frames, warm_frames = [], []
                for frame_idx in range(length):
                    stem = Path(temp_dir) / f"{seq_idx}_{frame_idx}"
                    signal_path = str(stem) + "_signal.npy"
                    label_path = str(stem) + "_label.npy"
                    cache_path = str(stem) + ".npz"
                    np.save(signal_path, np.zeros((8, 89, 10), dtype=np.complex64))
                    label = np.zeros((10, 6), dtype=np.float32)
                    for slot in range(people):
                        label[:, slot * 2:slot * 2 + 2] = (
                            0.4 * slot + 0.02 * frame_idx, 2.0 + seq_idx * 0.3)
                    np.save(label_path, label)
                    out = np.zeros((50, 4), dtype=np.float32)
                    out[:, :2] = (0.1 * seq_idx, 2.0 + 0.02 * frame_idx)
                    conf = np.zeros((50, 2), dtype=np.float32)
                    conf[:, 1] = 3.0
                    feat = np.full((50, 128), seq_idx + 0.1 * frame_idx, dtype=np.float32)
                    np.savez(cache_path, out=out, conf=conf, feat=feat)
                    cold_frames.append((cache_path, label_path, frame_idx))
                    warm_frames.append((signal_path, label_path, frame_idx))
                cold_records.append({"frames": cold_frames})
                warm_records.append({"frames": warm_frames})

            for is_cold, records in ((True, cold_records), (False, warm_records)):
                with self.subTest(is_cold=is_cold):
                    model = None if is_cold else StubM1()
                    banks = [TrackBank(hist_len=2, feat_dim=128, confirm_threshold=0.0)
                             for _ in range(3)]
                    with patch.object(config, "VAL_SEQUENCE_BATCH_SIZE", 1):
                        serial = evaluate_full_sequence_dataset(
                            records, model, self.tracker, banks, torch.device("cpu"),
                            is_cold=is_cold)
                    with patch.object(config, "VAL_SEQUENCE_BATCH_SIZE", 3):
                        parallel = evaluate_full_sequence_dataset(
                            records, model, self.tracker, banks, torch.device("cpu"),
                            is_cold=is_cold)
                    self.assertEqual(parallel["frames"], 10)
                    self.assertEqual(parallel["sequences"], 4)
                    self.assertGreater(parallel["active_count"], 0)
                    for name in serial:
                        self.assertAlmostEqual(serial[name], parallel[name], delta=1e-5, msg=name)
                    if model is not None:
                        self.assertIn(3, model.seen_queries)

    def test_validation_handles_empty_input_limits_and_missing_banks(self):
        device = torch.device("cpu")
        metrics = evaluate_full_sequence_dataset([], None, self.tracker, [], device)
        self.assertTrue(all(value == 0 for value in metrics.values()))
        with self.assertRaisesRegex(ValueError, "TrackBank"):
            evaluate_full_sequence_dataset([{"frames": []}], None, self.tracker, [], device)
        metrics = evaluate_full_sequence_dataset(
            [{"frames": []}, {"frames": []}], None, self.tracker, self.banks, device,
            max_sequences=1)
        self.assertEqual(metrics["sequences"], 1)
        self.assertEqual(metrics["frames"], 0)

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
            self.assertTrue((Path(temp_dir) / "joint_tracker_last.pth").exists())
            self.assertFalse((Path(temp_dir) / "joint_tracker_best.pth").exists())

        rendered = progress_output.getvalue()
        self.assertIn("Epoch 1/1 [train]", rendered)
        self.assertIn("Epoch 1/1 [val]", rendered)
        self.assertIn("loss=", rendered)


if __name__ == "__main__":
    unittest.main()
