"""Evaluate a training-free temporal association baseline on frozen M1 data."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from original.my_dataset.track_sequence_dataset import TrackSequenceColdDataset
from original.my_track import config
from original.my_track.m1_frame_baseline import cluster_candidates, match_frame
from original.my_track.m1_temporal_baseline import TemporalCandidateTracker


def known_target_count(sequence_name):
    """Read the fixed target count encoded in a validation folder name.

    Folder names currently follow ``<date>_<target_count>_<variant>``.  This
    helper is intentionally limited to sequence metadata: it does not inspect
    labels and it never returns target coordinates or identities.
    """
    parts = str(sequence_name).split("_")
    if len(parts) < 2:
        raise ValueError(
            f"sequence metadata has no target count: {sequence_name!r}"
        )
    try:
        count = int(parts[1])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"sequence metadata has invalid target count: {sequence_name!r}"
        ) from exc
    if not 1 <= count <= 3:
        raise ValueError(
            f"sequence metadata target count must be in [1, 3]: {sequence_name!r}"
        )
    return count


def _target(label_path):
    labels = np.load(label_path).mean(axis=0).reshape(3, 2).astype(np.float64)
    return labels[np.abs(labels).sum(axis=1) > 0]


def _new_counts():
    return {"tp": 0, "fp": 0, "fn": 0, "error_sum_m": 0.0, "predicted_points": 0}


def _add(counts, result, predicted_count):
    counts["tp"] += result.true_positives
    counts["fp"] += result.false_positives
    counts["fn"] += result.false_negatives
    counts["error_sum_m"] += result.error_sum_m
    counts["predicted_points"] += int(predicted_count)


def _summary(counts, frames):
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
        "frames": frames,
    }


def _identity_switches(predictions, targets, track_ids, gate_m=1.0):
    """Count owner changes for persistent track IDs; metric only, no GT in tracking."""
    owners = {}
    switches = 0
    for points, gt, ids in zip(predictions, targets, track_ids):
        valid_gt = gt[np.abs(gt).sum(axis=1) > 0]
        if not len(points) or not len(valid_gt):
            continue
        distances = np.linalg.norm(points[:, None] - valid_gt[None, :, :], axis=-1)
        for point_index, track_id in enumerate(ids):
            owner = int(distances[point_index].argmin())
            if distances[point_index, owner] > gate_m:
                continue
            track_id = int(track_id)
            previous = owners.get(track_id)
            if previous is not None and previous != owner:
                switches += 1
            owners[track_id] = owner
    return switches


def evaluate_sequence(
    record,
    score_threshold,
    cluster_radius_m,
    association_radius_m,
    max_missed,
    max_tracks=3,
):
    tracker = TemporalCandidateTracker(
        score_threshold=score_threshold,
        cluster_radius_m=cluster_radius_m,
        association_radius_m=association_radius_m,
        max_missed=max_missed,
        max_tracks=max_tracks,
    )
    counts = _new_counts()
    observed_counts = _new_counts()
    all_predictions, all_targets, all_ids = [], [], []
    retained_misses = 0
    for cache_path, label_path, _sample_id in record["frames"]:
        with np.load(cache_path) as cached:
            positions = np.asarray(cached["out"][:, :2], dtype=np.float64)
            logits = np.asarray(cached["conf"], dtype=np.float64)
        logits -= logits.max(axis=-1, keepdims=True)
        exp_logits = np.exp(logits)
        scores = exp_logits[:, 1] / exp_logits.sum(axis=-1)
        frame_target = _target(label_path)
        output = tracker.update(positions, scores)
        _add(counts, match_frame(output.positions, frame_target), len(output.positions))
        observed_positions = output.positions[output.observed]
        _add(observed_counts, match_frame(observed_positions, frame_target), len(observed_positions))
        retained_misses += int((~output.observed).sum())
        all_predictions.append(output.positions)
        all_targets.append(frame_target)
        all_ids.append(output.track_ids)
    return {
        "all": _summary(counts, len(record["frames"])),
        "observed_only": _summary(observed_counts, len(record["frames"])),
        "idsw": _identity_switches(all_predictions, all_targets, all_ids),
        "retained_miss_predictions": retained_misses,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--score-threshold", type=float, default=0.8)
    parser.add_argument("--cluster-radius", type=float, default=0.75)
    parser.add_argument("--association-radius", type=float, default=1.0)
    parser.add_argument("--max-missed", type=int, default=1)
    parser.add_argument(
        "--known-target-count",
        action="store_true",
        help=(
            "Limit new tracks to the target count encoded in each sequence "
            "name; this is an idealized metadata-only condition."
        ),
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.max_frames < 0 or args.max_missed < 0:
        parser.error("max-frames and max-missed must be non-negative")
    if not 0 <= args.score_threshold <= 1 or args.cluster_radius < 0:
        parser.error("invalid threshold or cluster radius")

    dataset = TrackSequenceColdDataset(
        config.VAL_ROOT,
        str(Path(config.CACHE_ROOT) / "fast_dataset_multi_test"),
        config.WINDOW_SIZE,
        config.STRIDE,
        full_sequence=True,
    )
    sequence_results = []
    for index in range(len(dataset)):
        record = dataset[index]
        frames = record["frames"][:args.max_frames] if args.max_frames else record["frames"]
        max_tracks = known_target_count(record["folder"]) if args.known_target_count else 3
        sequence_results.append(evaluate_sequence(
            {**record, "frames": frames}, args.score_threshold, args.cluster_radius,
            args.association_radius, args.max_missed, max_tracks=max_tracks,
        ))
    payload = {
        "mode": (
            "frame-local M1 clustering plus short temporal association; "
            "sequence-metadata target count; no training"
            if args.known_target_count
            else "frame-local M1 clustering plus short temporal association; no training"
        ),
        "target_count_mode": (
            "sequence_metadata" if args.known_target_count else "autonomous_max3"
        ),
        "score_threshold": args.score_threshold,
        "cluster_radius_m": args.cluster_radius,
        "association_radius_m": args.association_radius,
        "max_missed": args.max_missed,
        "sequences": len(sequence_results),
        "sequence_results": sequence_results,
    }
    rendered = json.dumps(payload, ensure_ascii=True, indent=2)
    if args.output is None:
        print(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"Wrote temporal baseline metrics: {args.output}")


if __name__ == "__main__":
    main()
