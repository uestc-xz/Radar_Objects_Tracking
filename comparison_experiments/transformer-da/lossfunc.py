"""MCD 损失：带掩码的多分类交叉熵与 Dice 损失。"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

import config  # noqa: E402
from original.my_dataset.track_sequence_dataset import (  # noqa: E402
    load_track_sequence_cold_dataset,
)


class MCDLoss(nn.Module):
    """计算 mask cross-entropy + Dice loss。

    Args:
        smooth: Dice 平滑项 gamma，默认 1。

    输入约定：
        probabilities: [B, Q, C]，经过 softmax 的类别概率。
        targets: [B, Q, C] one-hot，或 [B, Q] 类别索引。
        mask: [B, Q]，有效量测为 1，padding 为 0。
    返回总损失；可选返回各分量用于日志。
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self,probabilities: torch.Tensor,targets: torch.Tensor,mask: torch.Tensor,return_components: bool = False,):
        if targets.ndim == 2:
            one_hot_targets = F.one_hot(
                targets.to(torch.long), num_classes=probabilities.shape[-1]
            ).to(probabilities.dtype)
        else:
            one_hot_targets = targets.to(probabilities.dtype)

        valid = mask.to(probabilities.dtype).unsqueeze(-1)

        # 逐量测计算 CE；padding 槽乘 mask 后不产生梯度，再除以有效量测数。
        log_probabilities = probabilities.clamp_min(1e-8).log()
        per_measurement_ce = -(one_hot_targets * log_probabilities).sum(dim=-1)
        masked_ce = per_measurement_ce * mask.to(probabilities.dtype)
        loss_mask = masked_ce.sum() / mask.sum().clamp_min(1).to(probabilities.dtype)

        # 按类别汇总 batch/query 维度；先屏蔽 padding，再计算全局 Dice。
        masked_probabilities = probabilities * valid
        masked_targets = one_hot_targets * valid
        intersection = (masked_probabilities * masked_targets).sum(dim=(0, 1))
        squared_sum = (masked_probabilities.square() + masked_targets.square()).sum(dim=(0, 1))
        dice_per_class = (2.0 * intersection + self.smooth) / (squared_sum + self.smooth)
        # loss_dice = 1.0 - dice_per_class.sum()
        loss_dice = 1.0 - dice_per_class.mean()
        loss = loss_mask + loss_dice
        if return_components:
            return loss, {"loss_mask": loss_mask, "loss_dice": loss_dice}
        return loss


def velocity_regression_loss(
    predicted_velocities: torch.Tensor,
    velocity_targets: torch.Tensor,
    foreground_mask: torch.Tensor,
) -> torch.Tensor:
    """对前景量测计算速度回归损失（SmoothL1）。

    Args:
        predicted_velocities: [B, Q, 2]，网络回归头输出的逐量测速度位移。
        velocity_targets: [B, Q, 2]，每条 GT 航迹相邻帧位置差的逐量测复制。
        foreground_mask: [B, Q]，True 表示该量测属于某个目标；杂波不参与回归。
    """
    per_dimension = F.smooth_l1_loss(
        predicted_velocities, velocity_targets, reduction="none"
    )
    mask = foreground_mask.unsqueeze(-1).to(predicted_velocities.dtype)
    masked = per_dimension * mask
    return masked.sum() / mask.sum().clamp_min(1).to(predicted_velocities.dtype)


def _simple_test() -> None:
    """读取一个 M1 冻结缓存 batch 并计算一次损失，供最小连通性测试。"""
    if not config.IS_COLD:
        raise RuntimeError("此测试需要 config.IS_COLD=True")

    dataset = load_track_sequence_cold_dataset(
        data_root_folder=str(config.TRAIN_ROOT),
        cache_root_folder=str(config.CACHE_ROOT),
        window_size=config.WINDOW_SIZE,
        stride=config.STRIDE,
        full_sequence=False,
    )
    batch_size = min(2, len(dataset))
    batch = [dataset[index] for index in range(batch_size)]
    outputs, confidence_logits, _features, labels = zip(*batch)
    confidence_logits = torch.stack(confidence_logits)
    labels = torch.stack(labels)

    # 当前缓存只提供 M1 的二分类 logits，因此先将前景概率扩成 4 类，
    # 再由每个候选量测到 GT 标签槽的最近距离构造类别 0/1/2/3。
    # 数据集张量为 [B,T,Q,2]；将帧维 T 与 batch B 合并，按帧独立计算损失。
    batch_size, num_frames, num_queries, _ = confidence_logits.shape
    confidence_logits = confidence_logits.reshape(batch_size * num_frames, num_queries, 2)
    outputs = torch.stack(outputs).reshape(batch_size * num_frames, num_queries, -1)
    labels = labels.reshape(batch_size * num_frames, labels.shape[-2], 2)
    foreground_probability = confidence_logits.softmax(dim=-1)[..., 1].detach()
    probabilities = torch.cat(
        ((1.0 - foreground_probability).unsqueeze(-1),
         (foreground_probability / 3.0).unsqueeze(-1).expand(-1, -1, 3)),
        dim=-1,
    ).requires_grad_()
    # 量测槽位没有 padding：缓存固定保存 50 个 query，全部视为有效。
    mask = torch.ones(probabilities.shape[:-1], dtype=probabilities.dtype)

    # 标签中的第 0/1/2 个目标槽分别映射到类别 1/2/3；未匹配候选为杂波 0。
    # 此处仅为 smoke test，用最近 GT 距离分配类别，不作为正式标签构造方案。
    positions = outputs[..., :2]
    targets = torch.zeros(probabilities.shape[:-1], dtype=torch.long)
    valid_gt = labels.abs().sum(dim=-1) > 0
    for batch_index in range(batch_size * num_frames):
        if valid_gt[batch_index].any():
            distances = torch.cdist(positions[batch_index], labels[batch_index])
            distances[:, ~valid_gt[batch_index]] = float("inf")
            nearest_distance, nearest_slot = distances.min(dim=-1)
            matched = nearest_distance < 1.0
            targets[batch_index, matched] = nearest_slot[matched] + 1

    criterion = MCDLoss(smooth=1.0)
    loss, components = criterion(probabilities, targets, mask, return_components=True)
    loss.backward()

    print("M1 冻结缓存 batch size:", batch_size)
    print("probabilities.shape:", tuple(probabilities.shape))
    print("targets.shape:", tuple(targets.shape))
    print("mask.shape:", tuple(mask.shape))
    print("loss_mask:", components["loss_mask"].item())
    print("loss_dice:", components["loss_dice"].item())
    print("loss_mcd:", loss.item())
    print("gradient:", probabilities.grad)


if __name__ == "__main__":
    _simple_test()
