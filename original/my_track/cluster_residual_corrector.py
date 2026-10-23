"""Local, measurement-anchored correction of frozen M1 frame proposals.

Unlike the learned-query experiment, the coordinate and confidence paths start
from M1 estimates. Proposal generation never receives labels or track history.
"""
from typing import Dict, NamedTuple

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn import functional as F

from .m1_frame_baseline import cluster_candidates


class ClusterResidualOutput(NamedTuple):
    positions: torch.Tensor
    presence_logits: torch.Tensor
    anchors: torch.Tensor
    valid_mask: torch.Tensor


class ClusterResidualCorrector(nn.Module):
    """Correct up to three M1 clusters using local candidate evidence.

    High-confidence clusters are constructed first so lower-confidence points
    cannot move their centroids. Supplemental clusters may recover weaker
    detections; the predicted confidence, not GT count, determines activation.
    """

    def __init__(self, hidden_dim: int = 32, use_cluster_evidence: bool = False):
        super().__init__()
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        self.use_cluster_evidence = bool(use_cluster_evidence)
        evidence_dim = 8 if self.use_cluster_evidence else 0
        self.neighbor_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
        )
        self.proposal_encoder = nn.Sequential(
            nn.Linear(hidden_dim + 4 + evidence_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
        )
        self.position_head = nn.Linear(hidden_dim, 2)
        self.presence_head = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.position_head.weight)
        nn.init.zeros_(self.position_head.bias)
        nn.init.zeros_(self.presence_head.weight)
        nn.init.zeros_(self.presence_head.bias)

    @staticmethod
    def _cluster_evidence(relative, scores, local, anchors, valid_mask):
        """Summarize spread, support, confidence and nearest competing cluster."""
        weights = scores * local
        weight_sum = weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        mean = (relative * weights.unsqueeze(-1)).sum(dim=-2) / weight_sum
        variance = ((relative - mean.unsqueeze(-2)).square() * weights.unsqueeze(-1)).sum(dim=-2) / weight_sum
        spread = variance.clamp_min(0).sqrt()
        high_count = ((scores >= 0.96) & local).sum(dim=-1, keepdim=True).float() / scores.shape[-1]
        available_count = local.sum(dim=-1, keepdim=True).float() / scores.shape[-1]
        max_score = scores.masked_fill(~local, 0).amax(dim=-1, keepdim=True)
        pairwise = torch.cdist(anchors, anchors)
        other = valid_mask[:, None, :] & valid_mask[:, :, None]
        eye = torch.eye(anchors.shape[1], dtype=torch.bool, device=anchors.device)[None]
        nearest = pairwise.masked_fill(~(other & ~eye), float("inf")).amin(dim=-1, keepdim=True)
        nearest = torch.where(torch.isfinite(nearest), nearest.clamp_max(2.7) / 2.7, 1.0)
        return torch.cat((mean / 2.7, spread / 2.7, high_count, available_count, max_score, nearest), dim=-1) * valid_mask.unsqueeze(-1)

    def _proposals(self, positions, scores):
        anchors = positions.new_zeros((len(positions), 3, 2))
        anchor_scores = scores.new_zeros((len(scores), 3))
        valid_mask = torch.zeros((len(positions), 3), dtype=torch.bool, device=positions.device)
        for batch_index, (points, probabilities) in enumerate(zip(
            positions.detach().cpu().numpy(), scores.detach().cpu().numpy()
        )):
            # Match the clustering helper's comparison dtype, so boundary scores
            # cannot fall into the gap between primary and supplemental groups.
            probabilities = probabilities.astype(np.float64, copy=False)
            primary = cluster_candidates(points, probabilities, 0.96, 0.75)
            eligible = (probabilities >= 0.8) & (probabilities < 0.96)
            if len(primary.positions):
                eligible &= np.all(np.linalg.norm(
                    points[:, None, :] - primary.positions[None, :, :], axis=-1
                ) > 0.75, axis=-1)
            supplemental = cluster_candidates(points[eligible], probabilities[eligible], 0.8, 0.75)
            proposals = list(zip(primary.positions, primary.scores))
            for point, score in zip(supplemental.positions, supplemental.scores):
                if not len(primary.positions) or np.all(
                    np.linalg.norm(primary.positions - point, axis=-1) > 0.75
                ):
                    proposals.append((point, score))
            proposals.sort(key=lambda pair: (-pair[1], pair[0][0], pair[0][1]))
            for slot, (point, score) in enumerate(proposals[:3]):
                anchors[batch_index, slot] = torch.as_tensor(
                    point, dtype=positions.dtype, device=positions.device
                )
                anchor_scores[batch_index, slot] = float(score)
                valid_mask[batch_index, slot] = True
        return anchors, anchor_scores, valid_mask

    def forward(self, candidate_positions, candidate_scores) -> ClusterResidualOutput:
        """Return meter coordinates, logits, fixed anchors and proposal mask."""
        if candidate_positions.ndim != 3 or candidate_positions.shape[-1] != 2:
            raise ValueError("candidate_positions must have shape [B,M,2]")
        if candidate_scores.shape != candidate_positions.shape[:2]:
            raise ValueError("candidate_scores must have shape [B,M]")
        if candidate_positions.shape[1] < 1:
            raise ValueError("candidate set must contain at least one query")
        if not torch.isfinite(candidate_positions).all() or not torch.isfinite(candidate_scores).all():
            raise ValueError("candidate inputs must be finite")
        if ((candidate_scores < 0) | (candidate_scores > 1)).any():
            raise ValueError("candidate_scores must be probabilities in [0,1]")

        anchors, anchor_scores, valid_mask = self._proposals(candidate_positions, candidate_scores)
        relative = candidate_positions[:, None, :, :] - anchors[:, :, None, :]
        scores = candidate_scores[:, None, :].expand(-1, 3, -1)
        local = relative.square().sum(dim=-1) <= 1.0
        features = self.neighbor_encoder(torch.cat((relative, scores.unsqueeze(-1)), dim=-1))
        weights = scores * local
        pooled = (features * weights.unsqueeze(-1)).sum(dim=2) / weights.sum(
            dim=2, keepdim=True
        ).clamp_min(1e-6)
        normalized_anchors = torch.stack(
            (anchors[..., 0] / 2.7, (anchors[..., 1] - 3.5) / 2.7), dim=-1
        )
        proposal_features = torch.cat((
            pooled, normalized_anchors, anchor_scores.unsqueeze(-1),
            local.float().mean(dim=-1, keepdim=True),
        ), dim=-1)
        if self.use_cluster_evidence:
            proposal_features = torch.cat((
                proposal_features,
                self._cluster_evidence(relative, scores, local, anchors, valid_mask),
            ), dim=-1)
        hidden = self.proposal_encoder(proposal_features)
        delta = 0.5 * torch.tanh(self.position_head(hidden))
        corrected = anchors + delta * valid_mask.unsqueeze(-1)
        logits = torch.logit(anchor_scores.clamp(1e-6, 1.0 - 1e-6))
        logits = logits + self.presence_head(hidden).squeeze(-1)
        logits = logits.masked_fill(~valid_mask, -20.0)
        return ClusterResidualOutput(corrected, logits, anchors, valid_mask)


def cluster_residual_loss(
    output: ClusterResidualOutput,
    target_positions: torch.Tensor,
    match_radius_m: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Supervise local corrections without inventing detections for distant GT.

    Assignments use fixed anchors, maximize the number of in-radius pairs, then
    minimize their distance. Unmatched GT remains an evaluation FN. Presence
    BCE averages real proposals only; padding never receives a target.
    """
    positions, logits, anchors, mask = output
    if positions.ndim != 3 or positions.shape[-1] != 2 or anchors.shape != positions.shape:
        raise ValueError("positions and anchors must have shape [B,K,2]")
    if logits.shape != positions.shape[:2] or mask.shape != logits.shape or mask.dtype != torch.bool:
        raise ValueError("logits and boolean mask must have shape [B,K]")
    if target_positions.ndim != 3 or target_positions.shape[0] != len(positions) or target_positions.shape[-1] != 2:
        raise ValueError("target_positions must have shape [B,G,2]")
    if not np.isfinite(match_radius_m) or match_radius_m < 0:
        raise ValueError("match_radius_m must be finite and non-negative")
    if any(not torch.isfinite(value).all() for value in (positions, logits, anchors, target_positions)):
        raise ValueError("loss inputs must be finite")

    labels = torch.zeros_like(logits)
    matched_positions, matched_targets = [], []
    for batch_index in range(len(positions)):
        targets = target_positions[batch_index]
        targets = targets[targets.abs().sum(dim=-1) > 0]
        slots = mask[batch_index].nonzero(as_tuple=True)[0]
        if not len(targets) or not len(slots):
            continue
        distances = torch.cdist(anchors[batch_index, slots].detach(), targets.detach()).cpu().numpy()
        valid = distances <= match_radius_m
        # A forbidden pair costs more than all possible valid distances combined.
        penalty = (min(distances.shape) + 1) * (match_radius_m + 1)
        rows, columns = linear_sum_assignment(np.where(valid, distances, penalty))
        keep = valid[rows, columns]
        rows, columns = rows[keep], columns[keep]
        selected_slots = slots[torch.as_tensor(rows, device=slots.device)]
        selected_targets = torch.as_tensor(columns, device=targets.device)
        labels[batch_index, selected_slots] = 1.0
        matched_positions.append(positions[batch_index, selected_slots])
        matched_targets.append(targets[selected_targets])

    if matched_positions and sum(len(value) for value in matched_positions):
        position_loss = F.smooth_l1_loss(torch.cat(matched_positions), torch.cat(matched_targets))
    else:
        position_loss = positions.sum() * 0.0
    presence_loss = F.binary_cross_entropy_with_logits(logits[mask], labels[mask]) if mask.any() else logits.sum() * 0.0
    return {"total": position_loss + 0.5 * presence_loss,
            "position": position_loss, "presence": presence_loss}
