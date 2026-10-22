"""整记录航迹身份与定位指标。

槽位到 GT 的置换仅在 warm-up 帧上按平均观测距离做一次 Hungarian 初始化，随后固定。
评估过程中不重排预测；ID switch 根据同一预测槽最近 GT 身份的所有权变化统计。
"""
import math

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def sequence_tracking_metrics(pred_pos, active, observed, gt_pos, warmup=20, match_gate_m=1.0):
    """计算固定槽映射下的序列级身份/位置/漏检保持指标。

    Args:
        pred_pos: 最终米制位置 `[T,3,2]`。
        active: 已确认航迹掩码 `[T,3]`。
        observed: 当前帧有候选支持掩码 `[T,3]`。
        gt_pos: 固定 GT 槽坐标 `[T,3,2]`；`(0,0)` 为空槽。
        warmup: 只用于一次性初始化槽位映射的前缀帧数。
        match_gate_m: ID 所有权判定最大距离和有效性门限，单位米。
    """
    if pred_pos.ndim != 3 or pred_pos.shape[1:] != (3, 2):
        raise ValueError("pred_pos must have shape [T,3,2]")
    frames = pred_pos.shape[0]
    if gt_pos.shape != pred_pos.shape:
        raise ValueError("gt_pos and pred_pos shapes must match")
    if active.shape != (frames, 3) or observed.shape != (frames, 3):
        raise ValueError("active and observed must have shape [T,3]")
    if warmup < 1 or not math.isfinite(float(match_gate_m)) or match_gate_m <= 0:
        raise ValueError("warmup and match_gate_m must be positive")
    for name, tensor in (("pred_pos", pred_pos), ("gt_pos", gt_pos)):
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} contains NaN or Inf")

    pred = pred_pos.detach().cpu().double()
    gt = gt_pos.detach().cpu().double()
    active = active.detach().cpu().bool()
    observed = observed.detach().cpu().bool()
    valid_gt = gt.abs().sum(dim=-1) > 0
    warm_end = min(int(warmup), frames)

    # 未观测槽对映射不提供证据；无证据的组合留作高代价而不参与重排。
    cost = np.full((3, 3), 1e6, dtype=np.float64)
    for track_idx in range(3):
        for gt_idx in range(3):
            mask = (active[:warm_end, track_idx] | observed[:warm_end, track_idx]) & valid_gt[:warm_end, gt_idx]
            if mask.any():
                errors = torch.linalg.vector_norm(
                    pred[:warm_end, track_idx][mask] - gt[:warm_end, gt_idx][mask], dim=-1
                )
                cost[track_idx, gt_idx] = float(errors.mean())
    rows, cols = linear_sum_assignment(cost)
    slot_to_gt = [-1, -1, -1]
    for row, col in zip(rows.tolist(), cols.tolist()):
        if cost[row, col] < 1e5:
            slot_to_gt[row] = col
    # Fixed-slot supervision provides a deterministic fallback when a target has no warm-up match.
    known_gt = valid_gt.any(dim=0).tolist()
    used_gt = {gt_idx for gt_idx in slot_to_gt if gt_idx >= 0}
    remaining_gt = [idx for idx, known in enumerate(known_gt) if known and idx not in used_gt]
    for track_idx in range(3):
        if slot_to_gt[track_idx] >= 0 or not remaining_gt:
            continue
        fallback = track_idx if track_idx in remaining_gt else remaining_gt[0]
        slot_to_gt[track_idx] = fallback
        remaining_gt.remove(fallback)

    # warm-up 的映射只用于初始身份标签；逐帧最近 GT 仅检测所有权是否变化。
    id_switches = 0
    last_owner = list(slot_to_gt)
    for time_idx in range(warm_end, frames):
        valid_indices = valid_gt[time_idx].nonzero(as_tuple=False).flatten()
        for track_idx in range(3):
            if not (active[time_idx, track_idx] or observed[time_idx, track_idx]):
                continue
            if valid_indices.numel() == 0:
                continue
            distances = torch.linalg.vector_norm(
                gt[time_idx, valid_indices] - pred[time_idx, track_idx], dim=-1
            )
            local_idx = int(distances.argmin().item())
            if float(distances[local_idx]) > match_gate_m:
                continue
            owner = int(valid_indices[local_idx].item())
            if last_owner[track_idx] >= 0 and owner != last_owner[track_idx]:
                id_switches += 1
            last_owner[track_idx] = owner

    point_errors = []
    missed_gt_frames = 0
    retained_misses = 0
    observed_count = 0
    predicted_count = 0
    active_count = 0
    covered_target_ids = set()
    covered_target_frames = 0
    valid_target_frames = int(valid_gt[warm_end:].sum().item())
    for track_idx, gt_idx in enumerate(slot_to_gt):
        if gt_idx < 0:
            continue
        gt_valid = valid_gt[:, gt_idx]
        tracked = active[:, track_idx] & gt_valid
        if tracked.any():
            point_errors.extend(torch.linalg.vector_norm(
                pred[tracked, track_idx] - gt[tracked, gt_idx], dim=-1
            ).tolist())
            active_count += int(tracked.sum().item())
        if tracked[warm_end:].any():
            covered_target_ids.add(gt_idx)
        covered_target_frames += int(tracked[warm_end:].sum().item())
        after_warmup = torch.arange(frames) >= warm_end
        missed = gt_valid & ~observed[:, track_idx] & after_warmup
        missed_gt_frames += int(missed.sum().item())
        retained_misses += int((missed & active[:, track_idx]).sum().item())
        observed_count += int((gt_valid & observed[:, track_idx]).sum().item())
        predicted_count += int((missed & active[:, track_idx]).sum().item())

    if point_errors:
        errors = torch.tensor(point_errors, dtype=torch.float64)
        mae = float(errors.mean().item())
        rmse = float(torch.sqrt(errors.square().mean()).item())
    else:
        mae = 0.0
        rmse = 0.0
    return {
        "idsw": id_switches,
        "mae_m": mae,
        "rmse_m": rmse,
        "miss_retention": retained_misses / missed_gt_frames if missed_gt_frames else 0.0,
        "observed_count": observed_count,
        "predicted_count": predicted_count,
        "active_count": active_count,
        "point_count": len(point_errors),
        "known_target_count": int(valid_gt.any(dim=0).sum().item()),
        "covered_target_count": len(covered_target_ids),
        "valid_target_frames": valid_target_frames,
        "tracked_target_frames": covered_target_frames,
        "target_coverage": len(covered_target_ids) / max(int(valid_gt.any(dim=0).sum().item()), 1),
        "target_frame_coverage": covered_target_frames / max(valid_target_frames, 1),
        "untracked_target_frames": max(valid_target_frames - covered_target_frames, 0),
        "miss_frames": missed_gt_frames,
        "slot_to_gt": slot_to_gt,
    }
