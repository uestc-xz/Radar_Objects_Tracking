"""Training-free, frame-local clustering baseline for cached M1 measurements."""
from typing import NamedTuple

import numpy as np
from scipy.optimize import linear_sum_assignment


class ClusteredMeasurements(NamedTuple):
    """Cluster centroids, their maximum member scores, and member counts."""
    positions: np.ndarray
    scores: np.ndarray
    sizes: np.ndarray


class FrameMatchCounts(NamedTuple):
    """Micro-aggregatable matching counts for one frame."""
    true_positives: int
    false_positives: int
    false_negatives: int
    error_sum_m: float


def _validate_points(name, points):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"{name} must have shape [N,2]")
    if not np.isfinite(points).all():
        raise ValueError(f"{name} must contain only finite values")
    return points


def cluster_candidates(positions, scores, score_threshold=0.96, cluster_radius_m=0.5,
                       return_membership=False):
    """Threshold M1 candidates and merge radius-connected points by score-weighted mean.

    Connectivity is transitive: if A is within the radius of B and B is within
    the radius of C, all three belong to one component even when A-C exceeds it.
    Components are returned in order of their first input candidate.
    """
    positions = _validate_points("positions", positions)
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (len(positions),):
        raise ValueError("scores must have shape [N]")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("scores must be finite probabilities in [0,1]")
    if not np.isfinite(score_threshold) or not 0 <= score_threshold <= 1:
        raise ValueError("score_threshold must be finite and in [0,1]")
    if not np.isfinite(cluster_radius_m) or cluster_radius_m < 0:
        raise ValueError("cluster_radius_m must be finite and non-negative")

    selected = np.flatnonzero(scores >= score_threshold)
    if not len(selected):
        empty = ClusteredMeasurements(
            np.empty((0, 2), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
            np.empty((0,), dtype=np.int64),
        )
        return (empty, []) if return_membership else empty

    selected_positions = positions[selected]
    selected_scores = scores[selected]
    parent = np.arange(len(selected), dtype=np.int64)

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left, right):
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[max(root_left, root_right)] = min(root_left, root_right)

    delta = selected_positions[:, None, :] - selected_positions[None, :, :]
    distance_squared = np.einsum("ijk,ijk->ij", delta, delta)
    connected = np.triu(distance_squared <= float(cluster_radius_m) ** 2, k=1)
    for left, right in np.argwhere(connected):
        union(int(left), int(right))

    components = {}
    for local_index in range(len(selected)):
        components.setdefault(find(local_index), []).append(local_index)
    ordered_components = sorted(components.values(), key=lambda indices: indices[0])

    centroids = []
    component_scores = []
    sizes = []
    memberships = []
    for indices in ordered_components:
        member_scores = selected_scores[indices]
        weight_sum = float(member_scores.sum())
        if weight_sum > 0:
            centroid = np.average(selected_positions[indices], axis=0, weights=member_scores)
        else:
            centroid = selected_positions[indices].mean(axis=0)
        centroids.append(centroid)
        component_scores.append(float(member_scores.max()))
        sizes.append(len(indices))
        memberships.append(selected[indices].copy())

    result = ClusteredMeasurements(
        np.asarray(centroids, dtype=np.float64).reshape(-1, 2),
        np.asarray(component_scores, dtype=np.float64),
        np.asarray(sizes, dtype=np.int64),
    )
    return (result, memberships) if return_membership else result


def match_frame(predicted_positions, gt_positions, match_radius_m=1.0):
    """Hungarian-match one frame for evaluation only; pairs beyond radius are errors."""
    predicted_positions = _validate_points("predicted_positions", predicted_positions)
    gt_positions = _validate_points("gt_positions", gt_positions)
    if not np.isfinite(match_radius_m) or match_radius_m < 0:
        raise ValueError("match_radius_m must be finite and non-negative")
    gt_positions = gt_positions[np.abs(gt_positions).sum(axis=1) > 0]
    if not len(predicted_positions) or not len(gt_positions):
        tp = 0
        error_sum = 0.0
    else:
        delta = predicted_positions[:, None, :] - gt_positions[None, :, :]
        distances = np.sqrt(np.einsum("ijk,ijk->ij", delta, delta))
        valid = distances <= float(match_radius_m)
        prediction_count, target_count = distances.shape
        unmatched_cost = 1.0
        invalid_cost = 4.0
        scale = max(float(match_radius_m), np.finfo(np.float64).tiny)
        costs = np.full(
            (prediction_count + target_count, prediction_count + target_count),
            invalid_cost,
            dtype=np.float64,
        )
        costs[:prediction_count, :target_count] = np.where(valid, 0.5 * distances / scale, invalid_cost)
        costs[:prediction_count, target_count:] = unmatched_cost
        costs[prediction_count:, :target_count] = unmatched_cost
        costs[prediction_count:, target_count:] = 0.0
        rows, cols = linear_sum_assignment(costs)
        real_pairs = (rows < prediction_count) & (cols < target_count)
        rows, cols = rows[real_pairs], cols[real_pairs]
        matched = valid[rows, cols]
        tp = int(matched.sum())
        error_sum = float(distances[rows[matched], cols[matched]].sum())
    return FrameMatchCounts(
        true_positives=tp,
        false_positives=len(predicted_positions) - tp,
        false_negatives=len(gt_positions) - tp,
        error_sum_m=error_sum,
    )
