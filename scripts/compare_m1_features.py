"""Compare an existing tracker checkpoint with M1 embeddings enabled/disabled.

This is an inference-only ablation. Both runs use the same weights, cached M1
coordinates/confidence, validation prefix, and production sequence evaluator.
"""
import argparse
import json
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from original.my_dataset.track_sequence_dataset import TrackSequenceColdDataset
from original.my_track import config
from original.my_track.joint_tracker import JointTracker
from original.my_track.track_state import TrackBank
from original.my_track.train import evaluate_full_sequence_dataset


class PrefixDataset:
    """Expose a bounded prefix of each full validation record."""

    def __init__(self, dataset, max_sequences=None, max_frames=None):
        count = len(dataset) if max_sequences is None else min(len(dataset), max_sequences)
        self.records = []
        for index in range(count):
            record = dataset[index]
            frames = record["frames"]
            if max_frames is not None:
                frames = frames[:max_frames]
            self.records.append({**record, "frames": frames})

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]


def load_tracker(checkpoint_path, device, use_m1_features):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    tracker = JointTracker(use_m1_features=use_m1_features).to(device)
    tracker.load_state_dict(checkpoint["tracker_state_dict"], strict=True)
    return tracker


def run_mode(checkpoint_path, dataset, device, use_m1_features, show_progress):
    tracker = load_tracker(checkpoint_path, device, use_m1_features)
    banks = [TrackBank() for _ in range(max(config.VAL_SEQUENCE_BATCH_SIZE, len(dataset)))]
    return evaluate_full_sequence_dataset(
        dataset, None, tracker, banks, device, is_cold=True,
        show_progress=show_progress,
        progress_desc=f"Validation (M1 features {'on' if use_m1_features else 'off'})",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=200,
                        help="Prefix per sequence; 0 evaluates complete sequences.")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--show-progress", action="store_true")
    parser.add_argument("--output", type=Path, default=None,
                        help="Optional JSON destination; defaults to stdout only.")
    args = parser.parse_args(argv)
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")
    if args.max_sequences is not None and args.max_sequences < 1:
        parser.error("max-sequences must be positive")
    if args.max_frames < 0:
        parser.error("max-frames must be non-negative")
    device_name = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device_name == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    device = torch.device(device_name)
    torch.set_num_threads(2)

    dataset = TrackSequenceColdDataset(
        config.VAL_ROOT,
        str(Path(config.CACHE_ROOT) / "fast_dataset_multi_test"),
        config.WINDOW_SIZE,
        config.STRIDE,
        full_sequence=True,
    )
    dataset = PrefixDataset(
        dataset,
        max_sequences=args.max_sequences,
        max_frames=args.max_frames or None,
    )
    if not len(dataset):
        parser.error("validation dataset is empty")

    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "device": str(device),
        "sequences": len(dataset),
        "frames_per_sequence": [len(record["frames"]) for record in dataset.records],
        "m1_features_on": run_mode(args.checkpoint, dataset, device, True, args.show_progress),
        "m1_features_off": run_mode(args.checkpoint, dataset, device, False, args.show_progress),
    }
    rendered = json.dumps(result, ensure_ascii=True, indent=2)
    if args.output is None:
        print(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"Wrote comparison: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
