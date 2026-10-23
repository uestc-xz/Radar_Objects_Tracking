"""Bounded cold-cache training screen, not a replacement for full evaluation.

Example (requires an unused output directory):
    python scripts/screen_tracker.py --output test_results_tracker_screen/run1

Uses production training, losses and recursive validation unchanged. Only the
number of selected windows and validation frames is limited. No M1 is loaded.
"""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from original.my_dataset.track_sequence_dataset import TrackSequenceColdDataset
from original.my_track import config
from original.my_track.joint_tracker import JointTracker
from original.my_track.track_state import TrackBank
from original.my_track.train import run_training


def select_window_indices(windows, per_sequence):
    """Select evenly spaced windows within every independent recording."""
    if per_sequence < 1:
        raise ValueError("per_sequence must be positive")
    groups = {}
    for idx, (folder_idx, _start) in enumerate(windows):
        groups.setdefault(folder_idx, []).append(idx)
    selected = []
    for indices in groups.values():
        positions = np.linspace(0, len(indices) - 1, min(per_sequence, len(indices)), dtype=int)
        selected.extend(indices[int(position)] for position in positions)
    return selected


def create_output_directory(path):
    """Refuse reuse even when the previous directory is empty."""
    Path(path).mkdir(parents=True, exist_ok=False)


def make_teacher_schedule(base_schedule, warmup_epochs, anneal_epochs):
    """Bind explicit curriculum values instead of mutating captured defaults."""
    return lambda epoch: base_schedule(epoch, warmup_epochs, anneal_epochs)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--windows-per-sequence", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--val-frames", type=int, default=200, help="Prefix per recording; 0=full.")
    parser.add_argument("--lr", type=float, default=config.LR)
    parser.add_argument("--teacher-warmup-epochs", type=int, default=None)
    parser.add_argument("--teacher-anneal-epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--disable-m1-features", action="store_true",
                        help="Train this screen with the M1 embedding ablated.")
    args = parser.parse_args(argv)
    if min(args.epochs, args.windows_per_sequence, args.batch_size) < 1:
        parser.error("epochs, windows-per-sequence and batch-size must be positive")
    if args.val_frames < 0 or 0 < args.val_frames <= config.HIST_LEN:
        parser.error("val-frames must be 0 or greater than HIST_LEN")
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("lr must be finite and positive")
    if (args.teacher_warmup_epochs is None) != (args.teacher_anneal_epochs is None):
        parser.error("teacher warmup and anneal overrides must be provided together")
    if args.teacher_warmup_epochs is not None and min(
        args.teacher_warmup_epochs, args.teacher_anneal_epochs
    ) < 0:
        parser.error("teacher warmup and anneal epochs must be non-negative")
    if args.teacher_warmup_epochs is not None:
        config.TEACHER_ASSOC_WARMUP_EPOCHS = args.teacher_warmup_epochs
        config.TEACHER_ASSOC_ANNEAL_EPOCHS = args.teacher_anneal_epochs
        config.teacher_assignment_ratio = make_teacher_schedule(
            config.teacher_assignment_ratio,
            args.teacher_warmup_epochs,
            args.teacher_anneal_epochs,
        )
    output = args.output.resolve()
    if output.exists():
        parser.error(f"output already exists; refusing to overwrite: {output}")

    device_name = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_dataset = TrackSequenceColdDataset(
        config.TRAIN_ROOT, str(Path(config.CACHE_ROOT) / "fast_dataset_multi_train"),
        config.WINDOW_SIZE, config.STRIDE)
    val_dataset = TrackSequenceColdDataset(
        config.VAL_ROOT, str(Path(config.CACHE_ROOT) / "fast_dataset_multi_test"),
        config.WINDOW_SIZE, config.STRIDE, full_sequence=True)
    indices = select_window_indices(train_dataset.windows, args.windows_per_sequence)
    if not indices or not len(val_dataset):
        parser.error("screen requires nonempty training and validation sequences")
    print(f"SCREEN ONLY: {len(indices)} windows from {len(train_dataset.folders)} training records; "
          f"{len(val_dataset)} validation records, prefix={args.val_frames or 'full'}; "
          f"epochs={args.epochs}, batch={args.batch_size}, lr={args.lr:g}, device={device}", flush=True)
    train_samples = [train_dataset[idx] for idx in indices]
    records = [val_dataset[idx] for idx in range(len(val_dataset))]
    if args.val_frames:
        records = [dict(record, frames=record["frames"][:args.val_frames]) for record in records]
    window_manifest = []
    for idx in indices:
        folder_idx, start = train_dataset.windows[idx]
        folder = train_dataset.folders[folder_idx]
        frames = folder["frames"][start:start + config.WINDOW_SIZE]
        window_manifest.append({"folder": folder["folder"], "sample_ids": [row[2] for row in frames]})
    manifest = {
        "arguments": {**vars(args), "output": str(output)},
        "screen_only": True,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], cwd=REPO_ROOT, text=True),
        "train_windows": window_manifest,
        "validation": [{"folder": record["folder"], "sample_ids": [row[2] for row in record["frames"]]}
                       for record in records],
        "loss_weights": config.TRACKER_TRAIN_LOSS_WEIGHTS,
        "gt_slot_canonicalization": "per-window/record, first-valid x then y, stable for sequence",
        "validation_loss_weights": config.TRACKER_LOSS_WEIGHTS,
        "gradient_clip": config.TRACKER_MAX_GRAD_NORM,
        "teacher_curriculum": [config.teacher_assignment_ratio(epoch) for epoch in range(1, args.epochs + 1)],
        "teacher_warmup_epochs": config.TEACHER_ASSOC_WARMUP_EPOCHS,
        "teacher_anneal_epochs": config.TEACHER_ASSOC_ANNEAL_EPOCHS,
        "use_m1_features": not args.disable_m1_features,
        "dt": config.DT,
    }
    create_output_directory(output)
    with (output / "screen_manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=True, indent=2)
    # Process-local overrides only; the standard entry and config.py stay unchanged.
    config.EPOCHS = args.epochs
    config.IS_COLD = True
    config.OUT_DIR = str(output)
    tracker = JointTracker(use_m1_features=not args.disable_m1_features).to(device)
    optimizer = torch.optim.Adam(tracker.parameters(), lr=args.lr)
    loader = DataLoader(train_samples, batch_size=args.batch_size, shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(args.seed))
    banks = [TrackBank() for _ in range(max(args.batch_size, config.VAL_SEQUENCE_BATCH_SIZE))]
    run_training(device, None, tracker, optimizer, loader, records, banks,
                 str(output / "tracker_train_log.csv"))
    print(f"Screen complete: {output}. Truncated metrics are not full-set results.", flush=True)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
