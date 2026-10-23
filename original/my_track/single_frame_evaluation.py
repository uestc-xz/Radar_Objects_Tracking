"""逐帧集合纠偏器的存在概率阈值扫描与检测指标。"""
from typing import Dict, Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import expit

from .m1_frame_baseline import match_frame


def compare_anchor_localization(
    anchors, corrected_positions, target_positions, prediction_mask, match_radius_m=1.0,
):
    """Compare coordinates on identical anchor/GT pairs, without score filtering.

    Pair identities are fixed before looking at corrected positions. A correction
    moving a point outside the gate stays in this diagnostic, so selective TP
    disappearance cannot create an apparent localization gain. This is an
    evaluation diagnostic only and never changes training assignments.
    """
    anchors = np.asarray(anchors, dtype=np.float64)
    corrected = np.asarray(corrected_positions, dtype=np.float64)
    targets = np.asarray(target_positions, dtype=np.float64)
    mask = np.asarray(prediction_mask)
    if anchors.ndim != 3 or anchors.shape[-1] != 2 or corrected.shape != anchors.shape:
        raise ValueError("anchors and corrected positions must have shape [N,K,2]")
    if targets.ndim != 3 or targets.shape[0] != len(anchors) or targets.shape[-1] != 2:
        raise ValueError("targets must have shape [N,G,2]")
    if mask.shape != anchors.shape[:2] or mask.dtype != np.bool_:
        raise ValueError("prediction_mask must be boolean with shape [N,K]")
    if any(not np.isfinite(value).all() for value in (anchors, corrected, targets)):
        raise ValueError("coordinate arrays must contain finite values")
    if not np.isfinite(match_radius_m) or match_radius_m < 0:
        raise ValueError("match_radius_m must be finite and non-negative")
    count = improved = 0
    anchor_sum = corrected_sum = residual_sum = 0.0
    for before, after, target, valid_slots in zip(anchors, corrected, targets, mask):
        before, after = before[valid_slots], after[valid_slots]
        target = target[np.abs(target).sum(axis=-1) > 0]
        if not len(before) or not len(target):
            continue
        distances = np.linalg.norm(before[:, None] - target[None], axis=-1)
        valid_pairs = distances <= match_radius_m
        penalty = (min(distances.shape) + 1) * (match_radius_m + 1)
        rows, columns = linear_sum_assignment(np.where(valid_pairs, distances, penalty))
        keep = valid_pairs[rows, columns]
        rows, columns = rows[keep], columns[keep]
        old_error = distances[rows, columns]
        new_error = np.linalg.norm(after[rows] - target[columns], axis=-1)
        count += len(rows)
        improved += int((new_error < old_error).sum())
        anchor_sum += float(old_error.sum())
        corrected_sum += float(new_error.sum())
        residual_sum += float(np.linalg.norm(after[rows] - before[rows], axis=-1).sum())
    return {
        "matched_pairs": count,
        "anchor_mae_m": anchor_sum / count if count else None,
        "corrected_mae_m": corrected_sum / count if count else None,
        "mean_residual_m": residual_sum / count if count else None,
        "improved_fraction": improved / count if count else None,
        "match_radius_m": float(match_radius_m),
    }


def evaluate_presence_thresholds(
    predicted_positions: np.ndarray,
    presence_logits: np.ndarray,
    target_positions: np.ndarray,
    thresholds: Iterable[float],
    match_radius_m: float = 1.0,
    prediction_mask: np.ndarray = None,
) -> Dict[float, dict]:
    """Compute frame-local P/R/F1 and matched MAE for each presence threshold.

    Inputs are arrays ``[N,3,2]``, ``[N,3]`` and ``[N,3,2]``. This function
    only filters predicted slots by sigmoid(logit); it never changes coordinates.
    """
    predicted_positions = np.asarray(predicted_positions, dtype=np.float64)
    presence_logits = np.asarray(presence_logits, dtype=np.float64)
    target_positions = np.asarray(target_positions, dtype=np.float64)
    if predicted_positions.ndim != 3 or predicted_positions.shape[-1] != 2:
        raise ValueError("predicted_positions must have shape [N,K,2]")
    if target_positions.ndim != 3 or target_positions.shape[0] != len(predicted_positions) or target_positions.shape[-1] != 2:
        raise ValueError("target_positions must have shape [N,G,2]")
    if presence_logits.shape != predicted_positions.shape[:2]:
        raise ValueError("presence_logits must have shape [N,K]")
    if prediction_mask is None:
        prediction_mask = np.ones_like(presence_logits, dtype=bool)
    prediction_mask = np.asarray(prediction_mask)
    if prediction_mask.shape != presence_logits.shape or prediction_mask.dtype != np.bool_:
        raise ValueError("prediction_mask must be boolean with shape [N,K]")
    if not (np.isfinite(predicted_positions).all()
            and np.isfinite(presence_logits).all()
            and np.isfinite(target_positions).all()):
        raise ValueError("evaluation arrays must contain only finite values")
    thresholds = tuple(float(value) for value in thresholds)
    if not thresholds or any(not np.isfinite(value) or not 0 <= value <= 1 for value in thresholds):
        raise ValueError("thresholds must be non-empty finite values in [0,1]")
    if not np.isfinite(match_radius_m) or match_radius_m < 0:
        raise ValueError("match_radius_m must be finite and non-negative")

    probabilities = expit(presence_logits)
    results = {}
    for threshold in thresholds:
        counts = {"tp": 0, "fp": 0, "fn": 0, "error_sum_m": 0.0, "predicted_points": 0}
        for frame_positions, frame_probabilities, frame_targets, frame_mask in zip(
            predicted_positions, probabilities, target_positions, prediction_mask
        ):
            active_positions = frame_positions[(frame_probabilities >= threshold) & frame_mask]
            matched = match_frame(active_positions, frame_targets, match_radius_m)
            counts["tp"] += matched.true_positives
            counts["fp"] += matched.false_positives
            counts["fn"] += matched.false_negatives
            counts["error_sum_m"] += matched.error_sum_m
            counts["predicted_points"] += len(active_positions)
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        results[threshold] = {
            **counts,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "matched_mae_m": counts["error_sum_m"] / tp if tp else 0.0,
            "frames": len(predicted_positions),
            "presence_threshold": threshold,
            "match_radius_m": float(match_radius_m),
        }
    return results
