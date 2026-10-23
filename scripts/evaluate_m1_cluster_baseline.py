"""Evaluate a training-free M1 candidate clustering baseline on cached validation data."""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from original.my_dataset.track_sequence_dataset import TrackSequenceColdDataset
from original.my_track import config
from original.my_track.m1_frame_baseline import cluster_candidates, match_frame


def _new_counts():
    return {"tp": 0, "fp": 0, "fn": 0, "error_sum_m": 0.0,
            "predicted_points": 0, "candidate_count": 0, "frames": 0}


def _add_frame(counts, match, predicted_count, candidate_count):
    counts["tp"] += match.true_positives
    counts["fp"] += match.false_positives
    counts["fn"] += match.false_negatives
    counts["error_sum_m"] += match.error_sum_m
    counts["predicted_points"] += int(predicted_count)
    counts["candidate_count"] += int(candidate_count)
    counts["frames"] += 1


def _summarize(counts):
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        **counts,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matched_mae_m": counts["error_sum_m"] / tp if tp else 0.0,
    }


def evaluate(dataset, thresholds, radii, match_radius_m):
    """Aggregate frame-local detections; GT enters only the matching metric."""
    raw_counts = {threshold: _new_counts() for threshold in thresholds}
    clustered_counts = {
        (threshold, radius): _new_counts()
        for threshold in thresholds for radius in radii
    }
    sequence_lengths = []

    for record in dataset:
        frames = record["frames"]
        sequence_lengths.append(len(frames))
        for cache_path, label_path, _sample_id in frames:
            with np.load(cache_path) as cached:
                positions = np.asarray(cached["out"][:, :2], dtype=np.float64)
                logits = np.asarray(cached["conf"], dtype=np.float64)
            logits = logits - logits.max(axis=-1, keepdims=True)
            exp_logits = np.exp(logits)
            scores = exp_logits[:, 1] / exp_logits.sum(axis=-1)
            labels = np.load(label_path).mean(axis=0).reshape(3, 2).astype(np.float64)
            valid_gt = labels[np.abs(labels).sum(axis=1) > 0]

            for threshold in thresholds:
                selected = scores >= threshold
                raw_match = match_frame(positions[selected], valid_gt, match_radius_m)
                _add_frame(raw_counts[threshold], raw_match, int(selected.sum()), int(selected.sum()))
                for radius in radii:
                    clustered = cluster_candidates(positions, scores, threshold, radius)
                    cluster_match = match_frame(clustered.positions, valid_gt, match_radius_m)
                    _add_frame(
                        clustered_counts[(threshold, radius)], cluster_match,
                        len(clustered.positions), int(selected.sum()),
                    )

    return {
        "sequence_frames": sequence_lengths,
        "raw_candidates": [
            {"score_threshold": threshold, **_summarize(raw_counts[threshold])}
            for threshold in thresholds
        ],
        "clustered": [
            {"score_threshold": threshold, "cluster_radius_m": radius,
             **_summarize(clustered_counts[(threshold, radius)])}
            for threshold in thresholds for radius in radii
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-sequences", type=int, default=None,
                        help="Maximum number of validation recordings; default uses all.")
    parser.add_argument("--max-frames", type=int, default=200,
                        help="Prefix per recording; 0 evaluates complete recordings.")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.8, 0.9, 0.96])
    parser.add_argument("--cluster-radii", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    parser.add_argument("--match-radius", type=float, default=1.0,
                        help="Evaluation TP radius in meters; not used to make predictions.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Optional JSON path. Results otherwise print to stdout.")
    args = parser.parse_args(argv)
    if args.max_sequences is not None and args.max_sequences < 1:
        parser.error("max-sequences must be positive")
    if args.max_frames < 0:
        parser.error("max-frames must be non-negative")
    if not args.thresholds or any(
        not math.isfinite(value) or not 0 <= value <= 1 for value in args.thresholds
    ):
        parser.error("all confidence thresholds must be in [0,1]")
    if not args.cluster_radii or any(
        not math.isfinite(value) or value < 0 for value in args.cluster_radii
    ):
        parser.error("cluster radii must be non-negative")
    if not math.isfinite(args.match_radius) or args.match_radius < 0:
        parser.error("match-radius must be non-negative")

    dataset = TrackSequenceColdDataset(
        config.VAL_ROOT,
        str(Path(config.CACHE_ROOT) / "fast_dataset_multi_test"),
        config.WINDOW_SIZE,
        config.STRIDE,
        full_sequence=True,
    )
    limit = len(dataset) if args.max_sequences is None else min(len(dataset), args.max_sequences)
    records = []
    for index in range(limit):
        record = dataset[index]
        frames = record["frames"]
        if args.max_frames:
            frames = frames[:args.max_frames]
        records.append({**record, "frames": frames})
    if not records:
        parser.error("validation dataset is empty")

    results = evaluate(records, args.thresholds, args.cluster_radii, args.match_radius)
    payload = {
        "mode": "frame-local, training-free; no identity or temporal state",
        "data_root": config.VAL_ROOT,
        "cache_root": str(Path(config.CACHE_ROOT) / "fast_dataset_multi_test"),
        "sequences": len(records),
        "sequence_names": [record["folder"] for record in records],
        "score_thresholds": args.thresholds,
        "cluster_radii_m": args.cluster_radii,
        "match_radius_m": args.match_radius,
        **results,
    }
    rendered = json.dumps(payload, ensure_ascii=True, indent=2)
    if args.output is None:
        print(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"Wrote baseline metrics: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
