"""一对一距离门控：先最大化门内配对数，再最小化配对代价。"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def gated_pairs(left, right, gate_m, candidate_penalty=None, pairwise_penalty=None):
    """返回门限内的 (left_index, right_index)，惩罚只用于有效配对排序。"""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    for points in (left, right):
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            raise ValueError('points must be finite [N,2] arrays')
    if not np.isfinite(gate_m) or gate_m <= 0:
        raise ValueError('gate_m must be finite and positive')
    n, m = len(left), len(right)
    penalty = np.zeros(m) if candidate_penalty is None else np.asarray(candidate_penalty)
    if penalty.shape != (m,) or not np.isfinite(penalty).all() or (penalty < 0).any():
        raise ValueError('candidate_penalty must be finite non-negative [M]')
    pair_penalty = (np.zeros((n, m), dtype=np.float64) if pairwise_penalty is None
                    else np.asarray(pairwise_penalty, dtype=np.float64))
    if (pair_penalty.shape != (n, m) or not np.isfinite(pair_penalty).all() or
            (pair_penalty < 0).any()):
        raise ValueError('pairwise_penalty must be finite non-negative [N,M]')
    if not n or not m:
        return []
    distances = np.linalg.norm(left[:, None] - right[None], axis=-1)
    valid = distances <= gate_m
    pair_cost = distances / gate_m + penalty[None] + pair_penalty
    # 全部有效边代价之和小于 1，少一次配对则至少多付两个 dummy 代价。
    scale = 2 * (min(n, m) + 1) * (1 + float(penalty.max()) + float(pair_penalty.max()))
    costs = np.full((n + m, n + m), 4., dtype=np.float64)
    costs[:n, :m] = np.where(valid, pair_cost / scale, 4.)
    costs[:n, m:] = 1.
    costs[n:, :m] = 1.
    costs[n:, m:] = 0.
    rows, columns = linear_sum_assignment(costs)
    return [(int(i), int(j)) for i, j in zip(rows, columns)
            if i < n and j < m and valid[i, j]]
