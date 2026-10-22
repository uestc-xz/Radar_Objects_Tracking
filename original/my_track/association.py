"""联合跟踪器的固定槽位监督与损失。

坐标均为米制 `(x, y)`。GT 的 3 个槽位在整条记录中固定；多个候选可监督到同一槽，
超出监督半径的候选标为杂波。本模块不做逐帧身份重排，也不负责网络状态更新。
"""
import math

import torch
import torch.nn.functional as F

from original.my_track.config import (
    ASSOC_LABEL_RADIUS_M,
    CLUTTER_WEIGHT_STEP,
    TRACK_QUALITY_SCALE_M,
)


def _require_finite(name, tensor):
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains NaN or Inf")


def _validate_position_pair(candidate_pos, gt_pos):
    if candidate_pos.ndim != 3 or candidate_pos.shape[-1] != 2:
        raise ValueError("candidate_pos must have shape [B,M,2]")
    if gt_pos.ndim != 3 or gt_pos.shape[1:] != (3, 2):
        raise ValueError("gt_pos must have shape [B,3,2]")
    if candidate_pos.shape[0] != gt_pos.shape[0]:
        raise ValueError("candidate_pos and gt_pos batch sizes must match")
    if candidate_pos.device != gt_pos.device:
        raise ValueError("candidate_pos and gt_pos must be on the same device")
    _require_finite("candidate_pos", candidate_pos)
    _require_finite("gt_pos", gt_pos)


def associate(logits):
    """按候选独立做杂波/固定槽 argmax，不对每槽施加一对一约束或 score 门控。"""
    if logits.ndim != 3 or logits.shape[-1] != 4:
        raise ValueError("logits must have shape [B,M,4]")
    _require_finite("logits", logits)
    return logits.argmax(dim=-1)


def build_assignment_targets(candidate_pos, gt_pos, radius_m=ASSOC_LABEL_RADIUS_M):
    """按固定 GT 槽位给每个候选标签，允许同槽拥有任意多个正候选。

    Args:
        candidate_pos: 候选米制坐标 `[B,M,2]`。
        gt_pos: 固定槽位 GT 坐标 `[B,3,2]`；`(0,0)` 表示空槽。
        radius_m: 最近 GT 距离不超过该半径才标成目标类；否则为杂波类 0。

    Returns:
        `long` 标签 `[B,M]`，GT 槽位 0..2 映射为类别 1..3。
    """
    _validate_position_pair(candidate_pos, gt_pos)
    if not math.isfinite(float(radius_m)) or radius_m < 0:
        raise ValueError("radius_m must be finite and non-negative")

    valid_gt = gt_pos.abs().sum(dim=-1) > 0
    distances = torch.cdist(candidate_pos, gt_pos, p=2)
    distances = distances.masked_fill(~valid_gt[:, None, :], float("inf"))
    nearest_dist, nearest_slot = distances.min(dim=-1)
    labels = torch.zeros(candidate_pos.shape[:2], dtype=torch.long, device=candidate_pos.device)
    positive = nearest_dist <= radius_m
    labels[positive] = nearest_slot[positive] + 1
    return labels


def association_loss(logits, labels, clutter_weight=None, epoch=0):
    """按样本分别归一化背景和正目标 CE，避免 50 个背景 query 主宰梯度。

    未传 `clutter_weight` 时背景/目标组等权。为兼容旧训练入口，可显式传背景组
    权重；若某样本没有正例，则返回其背景 CE，避免空集合均值造成 NaN。
    """
    if logits.ndim != 3 or logits.shape[-1] != 4:
        raise ValueError("logits must have shape [B,M,4]")
    if labels.shape != logits.shape[:2] or labels.dtype not in (torch.int32, torch.int64):
        raise ValueError("labels must be integer tensor with shape [B,M]")
    _require_finite("logits", logits)
    if labels.numel() and ((labels < 0).any() or (labels > 3).any()):
        raise ValueError("labels must be in [0,3]")
    if labels.device != logits.device:
        raise ValueError("labels and logits must be on the same device")

    bg_weight = None
    if clutter_weight is not None:
        bg_weight = float(clutter_weight) + max(0, int(epoch) - 1) * CLUTTER_WEIGHT_STEP
        if not math.isfinite(bg_weight) or bg_weight < 0:
            raise ValueError("clutter_weight must be finite and non-negative")

    losses = []
    for batch_idx in range(logits.shape[0]):
        item_loss = F.cross_entropy(logits[batch_idx], labels[batch_idx], reduction="none")
        bg = labels[batch_idx] == 0
        fg = ~bg
        bg_loss = item_loss[bg].mean() if bg.any() else None
        fg_loss = item_loss[fg].mean() if fg.any() else None
        if bg_loss is not None and fg_loss is not None:
            if bg_weight is None:
                item = 0.5 * (bg_loss + fg_loss)
            else:
                item = (bg_weight * bg_loss + fg_loss) / max(bg_weight + 1.0, 1e-12)
        elif bg_loss is not None:
            item = bg_loss
        elif fg_loss is not None:
            item = fg_loss
        else:
            item = logits[batch_idx].sum() * 0.0
        losses.append(item)
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def tracker_losses(association_logits, labels, final_pos, existence_prob,
                   track_confidence, gt_pos, quality_scale_m=TRACK_QUALITY_SCALE_M):
    """计算关联、最终位置、存在概率及航迹软质量四项标量损失。

    位置只在非空 GT 槽上监督。存在概率对三槽使用 BCE；质量目标为空槽取 0，
    有效槽取 `exp(-位置误差 / quality_scale_m)`，以保留坐标误差的软置信信息。
    """
    if final_pos.ndim != 3 or final_pos.shape[1:] != (3, 2):
        raise ValueError("final_pos must have shape [B,3,2]")
    expected = final_pos.shape[:2]
    if gt_pos.shape != final_pos.shape:
        raise ValueError("gt_pos and final_pos shapes must match")
    if existence_prob.shape != expected or track_confidence.shape != expected:
        raise ValueError("existence_prob and track_confidence must have shape [B,3]")
    for name, tensor in (("final_pos", final_pos), ("gt_pos", gt_pos),
                         ("existence_prob", existence_prob), ("track_confidence", track_confidence)):
        _require_finite(name, tensor)
    if not (0 <= existence_prob).all() or not (existence_prob <= 1).all():
        raise ValueError("existence_prob must be in [0,1]")
    if not (0 <= track_confidence).all() or not (track_confidence <= 1).all():
        raise ValueError("track_confidence must be in [0,1]")
    if not math.isfinite(float(quality_scale_m)) or quality_scale_m <= 0:
        raise ValueError("quality_scale_m must be finite and positive")

    valid_gt = gt_pos.abs().sum(dim=-1) > 0
    if valid_gt.any():
        position = F.smooth_l1_loss(final_pos[valid_gt], gt_pos[valid_gt])
        error = torch.linalg.vector_norm(final_pos.detach() - gt_pos, dim=-1)
        quality_target = torch.exp(-error / quality_scale_m) * valid_gt.to(final_pos.dtype)
    else:
        position = final_pos.sum() * 0.0
        quality_target = torch.zeros_like(track_confidence)

    existence_target = valid_gt.to(existence_prob.dtype)
    existence = F.binary_cross_entropy(existence_prob.clamp(1e-6, 1 - 1e-6), existence_target)
    confidence = F.smooth_l1_loss(track_confidence, quality_target)
    association = association_loss(association_logits, labels)
    return {
        "association": association,
        "position": position,
        "existence": existence,
        "confidence": confidence,
    }
