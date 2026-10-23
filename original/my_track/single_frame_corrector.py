"""冻结 M1 候选的逐帧集合定位纠偏器。

输入为米制候选坐标 [B,M,2] 和前景概率 [B,M]，不读取历史或 M1 特征；
输出为无序的最多 3 个米制坐标 [B,3,2] 与存在 logits [B,3]。
"""
from typing import Dict, Tuple

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn import functional as F


class SingleFrameSetPredictor(nn.Module):
    """用集合注意力将 M1 的候选集合压成至多三个目标点。"""

    def __init__(self, hidden_dim: int = 64, num_heads: int = 4):
        super().__init__()
        if hidden_dim <= 0 or num_heads <= 0 or hidden_dim % num_heads:
            raise ValueError("hidden_dim must be positive and divisible by num_heads")
        self.candidate_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.context_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.output_queries = nn.Parameter(torch.randn(1, 3, hidden_dim) * 0.02)
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=0.0, batch_first=True
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.position_head = nn.Linear(hidden_dim, 2)
        self.presence_head = nn.Linear(hidden_dim, 1)

    def forward(
        self, candidate_positions: torch.Tensor, candidate_scores: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(positions[B,3,2], presence_logits[B,3])`` in meters."""
        if candidate_positions.ndim != 3 or candidate_positions.shape[-1] != 2:
            raise ValueError("candidate_positions must have shape [B,M,2]")
        if candidate_scores.shape != candidate_positions.shape[:2]:
            raise ValueError("candidate_scores must have shape [B,M]")
        if candidate_positions.shape[1] == 0:
            raise ValueError("candidate set must contain at least one query")

        candidate_input = torch.cat(
            (candidate_positions, candidate_scores.unsqueeze(-1)), dim=-1
        )
        encoded = self.context_layer(self.candidate_encoder(candidate_input))
        queries = self.output_queries.expand(candidate_positions.shape[0], -1, -1)
        attended, _ = self.cross_attention(queries, encoded, encoded, need_weights=False)
        output = self.output_mlp(self.output_norm(queries + attended))
        return self.position_head(output), self.presence_head(output).squeeze(-1)


def _valid_target_positions(target_positions: torch.Tensor) -> torch.Tensor:
    """Use the project contract that exactly (0,0) denotes an empty GT slot."""
    return target_positions[target_positions.abs().sum(dim=-1) > 0]


def single_frame_set_loss(
    predicted_positions: torch.Tensor,
    presence_logits: torch.Tensor,
    target_positions: torch.Tensor,
    presence_weight: float = 0.5,
) -> Dict[str, torch.Tensor]:
    """Permutation-invariant coordinate and existence loss for one-frame sets.

    Each sample matches its valid GT points to prediction slots by minimum
    Euclidean distance. Position loss averages matched x/y Smooth-L1 terms;
    presence BCE averages all three prediction slots.
    """
    if predicted_positions.ndim != 3 or predicted_positions.shape[-1] != 2:
        raise ValueError("predicted_positions must have shape [B,K,2]")
    if target_positions.ndim != 3 or target_positions.shape[0] != predicted_positions.shape[0] or target_positions.shape[-1] != 2:
        raise ValueError("target_positions must have shape [B,G,2]")
    if presence_logits.shape != predicted_positions.shape[:2]:
        raise ValueError("presence_logits must have shape [B,K]")
    if presence_weight < 0 or not np.isfinite(presence_weight):
        raise ValueError("presence_weight must be finite and non-negative")

    batch_size, prediction_count, _ = predicted_positions.shape
    presence_targets = torch.zeros_like(presence_logits)
    matched_predictions = []
    matched_targets = []
    for batch_index in range(batch_size):
        targets = _valid_target_positions(target_positions[batch_index])
        if len(targets) == 0:
            continue
        distances = torch.cdist(
            predicted_positions[batch_index].detach(), targets.detach(), p=2
        ).cpu().numpy()
        prediction_indices, target_indices = linear_sum_assignment(distances)
        presence_targets[batch_index, prediction_indices] = 1.0
        matched_predictions.append(predicted_positions[batch_index, prediction_indices])
        matched_targets.append(targets[target_indices])

    if matched_predictions:
        position_loss = F.smooth_l1_loss(
            torch.cat(matched_predictions), torch.cat(matched_targets), reduction="mean"
        )
    else:
        position_loss = predicted_positions.sum() * 0.0
    presence_loss = F.binary_cross_entropy_with_logits(presence_logits, presence_targets)
    total = position_loss + presence_weight * presence_loss
    return {"total": total, "position": position_loss, "presence": presence_loss}
