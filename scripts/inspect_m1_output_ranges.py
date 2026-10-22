"""Read-only statistics for cached M1 outputs and matching GT labels.

The script does not load the tracker, alter cache files, or write any output.
It answers two diagnostic questions:

1. What coordinate range does M1 produce for all queries and for high-score
   queries?
2. How does that range compare with the valid ground-truth center range?

Run from the repository root, for example:

    E:\\Miniconda3\\python.exe scripts/inspect_m1_output_ranges.py
    E:\\Miniconda3\\python.exe scripts/inspect_m1_output_ranges.py --split test

The defaults match the paths used by ``save_m1_cache.py`` and the tracking
dataset.  Override them with ``--cache-root`` and ``--data-root`` when moving
the dataset to another machine.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np


DEFAULT_CACHE_ROOT = Path(r"E:\Jupyter\NewMOTR数据集")
DEFAULT_DATA_ROOT = Path(r"E:\Jupyter\track")
SPLIT_DIRS = {
    "train": "fast_dataset_multi_train",
    "test": "fast_dataset_multi_test",
}
DEFAULT_THRESHOLDS = (0.50, 0.90, 0.99)
QUANTILES = (0, 1, 5, 25, 50, 75, 95, 99, 100)


@dataclass
class SplitStats:
    """Streaming counters and small arrays needed for exact quantiles."""

    split: str
    files: int = 0
    candidates: int = 0
    valid_gt_points: int = 0
    invalid_files: int = 0
    nonfinite_values: int = 0
    shape_errors: int = 0
    score_parts: list[np.ndarray] = field(default_factory=list)
    xy_parts: list[np.ndarray] = field(default_factory=list)
    gt_parts: list[np.ndarray] = field(default_factory=list)
    threshold_xy_parts: dict[float, list[np.ndarray]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.threshold_xy_parts = {threshold: [] for threshold in DEFAULT_THRESHOLDS}

    def add(self, out: np.ndarray, conf: np.ndarray, gt: np.ndarray | None) -> None:
        """Accumulate one frame while retaining only values used in the report."""
        self.files += 1
        if out.ndim != 2 or out.shape[1] < 2 or conf.shape != (out.shape[0], 2):
            self.shape_errors += 1
            return

        xy = np.asarray(out[:, :2], dtype=np.float64)
        logits = np.asarray(conf, dtype=np.float64)
        if not np.isfinite(xy).all() or not np.isfinite(logits).all():
            self.nonfinite_values += 1

        # Stable two-class softmax for the foreground (target) probability.
        shifted = logits - logits.max(axis=-1, keepdims=True)
        exp_logits = np.exp(shifted)
        scores = exp_logits[:, 1] / exp_logits.sum(axis=-1)
        finite = np.isfinite(xy).all(axis=1) & np.isfinite(scores)
        xy = xy[finite]
        scores = scores[finite]
        self.candidates += int(xy.shape[0])
        self.xy_parts.append(xy)
        self.score_parts.append(scores)
        for threshold in self.threshold_xy_parts:
            selected = scores >= threshold
            if selected.any():
                self.threshold_xy_parts[threshold].append(xy[selected])

        if gt is not None:
            gt_array = np.asarray(gt, dtype=np.float64)
            # Match TrackSequenceDataset: average the 10 sub-steps first,
            # then reshape the six values into three target centers.
            if gt_array.ndim == 2 and gt_array.shape[1] == 6:
                gt_xy = gt_array.mean(axis=0).reshape(-1, 2)
            else:
                gt_xy = gt_array.reshape(-1, 2)
            valid_gt = gt_xy[np.isfinite(gt_xy).all(axis=1) & (np.abs(gt_xy).sum(axis=1) > 0)]
            self.valid_gt_points += int(valid_gt.shape[0])
            if valid_gt.size:
                self.gt_parts.append(valid_gt)


def numeric_npz_files(cache_dir: Path) -> Iterable[Path]:
    """Yield numeric cache files in frame order, ignoring unrelated files."""
    files = []
    for path in cache_dir.glob("*.npz"):
        try:
            frame_id = int(path.stem)
        except ValueError:
            continue
        files.append((frame_id, path))
    for _frame_id, path in sorted(files):
        yield path


def quantile_table(values: np.ndarray) -> str:
    """Format x/y quantiles in meters without hiding empty input."""
    if values.size == 0:
        return "(no finite values)"
    q = np.percentile(values, QUANTILES, axis=0)
    labels = ", ".join(
        f"p{int(percentile):02d}=({row[0]:.4f},{row[1]:.4f})"
        for percentile, row in zip(QUANTILES, q)
    )
    return labels


def scalar_quantile_table(values: np.ndarray) -> str:
    """Format score quantiles."""
    if values.size == 0:
        return "(no finite values)"
    q = np.percentile(values, QUANTILES)
    return ", ".join(f"p{int(percentile):02d}={value:.6f}" for percentile, value in zip(QUANTILES, q))


def scan_split(
    split: str,
    cache_root: Path,
    data_root: Path,
    max_files: int | None,
    progress: bool,
) -> SplitStats:
    """Scan one split and match each cache frame to its label file when present."""
    split_dir = SPLIT_DIRS[split]
    cache_split_root = cache_root / split_dir
    data_split_root = data_root / split_dir
    stats = SplitStats(split)
    if not cache_split_root.is_dir():
        raise FileNotFoundError(f"Cache split does not exist: {cache_split_root}")

    folder_names = sorted(
        path.name
        for path in cache_split_root.iterdir()
        if path.is_dir() and (path / "m1_cache").is_dir()
    )
    scan_started = time.perf_counter()
    for folder_name in folder_names:
        cache_dir = cache_split_root / folder_name / "m1_cache"
        files = list(numeric_npz_files(cache_dir))
        if max_files is not None:
            remaining = max_files - stats.files
            if remaining <= 0:
                break
            files = files[:remaining]

        for cache_path in files:
            label_path = data_split_root / folder_name / "labels" / f"{cache_path.stem}.npy"
            gt = None
            if label_path.is_file():
                gt = np.load(label_path, allow_pickle=False)
            try:
                with np.load(cache_path, allow_pickle=False) as cached:
                    out = cached["out"]
                    conf = cached["conf"]
            except (KeyError, OSError, ValueError):
                stats.invalid_files += 1
                continue
            stats.add(out, conf, gt)

        if progress:
            print(
                f"[{split}] {folder_name}: {len(files):,} frames scanned "
                f"({stats.files:,} total, {time.perf_counter() - scan_started:.1f}s)",
                flush=True,
            )
        if max_files is not None and stats.files >= max_files:
            break

    return stats


def print_split_report(stats: SplitStats) -> None:
    """Print a compact, comparable report for one split."""
    xy = np.concatenate(stats.xy_parts) if stats.xy_parts else np.empty((0, 2))
    scores = np.concatenate(stats.score_parts) if stats.score_parts else np.empty((0,))
    gt = np.concatenate(stats.gt_parts) if stats.gt_parts else np.empty((0, 2))

    print(f"\n[{stats.split}]")
    print(
        f"files={stats.files:,} candidates={stats.candidates:,} "
        f"valid_gt_points={stats.valid_gt_points:,} "
        f"invalid_files={stats.invalid_files} shape_errors={stats.shape_errors} "
        f"nonfinite_files={stats.nonfinite_values}"
    )
    print(f"M1 out[:, :2] quantiles (meters): {quantile_table(xy)}")
    if xy.size:
        print(
            f"M1 out[:, :2] min=({xy[:, 0].min():.4f},{xy[:, 1].min():.4f}) "
            f"max=({xy[:, 0].max():.4f},{xy[:, 1].max():.4f}) "
            f"max_abs={np.abs(xy).max():.4f}"
        )
    print(f"foreground score quantiles: {scalar_quantile_table(scores)}")
    if scores.size:
        for threshold, threshold_parts in stats.threshold_xy_parts.items():
            selected_xy = (
                np.concatenate(threshold_parts)
                if threshold_parts
                else np.empty((0, 2))
            )
            ratio = selected_xy.shape[0] / scores.size
            print(
                f"score>={threshold:.2f}: {selected_xy.shape[0]:,} "
                f"({ratio:.4%}) candidates; range/quantiles: {quantile_table(selected_xy)}"
            )
    print(f"valid GT center quantiles (meters): {quantile_table(gt)}")
    if gt.size:
        print(
            f"GT min=({gt[:, 0].min():.4f},{gt[:, 1].min():.4f}) "
            f"max=({gt[:, 0].max():.4f},{gt[:, 1].max():.4f}) "
            f"max_abs={np.abs(gt).max():.4f}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "test", "both"), default="both")
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional per-split frame limit for a quick smoke scan.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-sequence progress output.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    splits = list(SPLIT_DIRS) if args.split == "both" else [args.split]
    print(f"cache_root={args.cache_root}")
    print(f"data_root={args.data_root}")
    for split in splits:
        try:
            stats = scan_split(
                split,
                args.cache_root,
                args.data_root,
                args.max_files,
                progress=not args.quiet,
            )
        except FileNotFoundError as exc:
            parser.error(str(exc))
        print_split_report(stats)
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    raise SystemExit(main())
