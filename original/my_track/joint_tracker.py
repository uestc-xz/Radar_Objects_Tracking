"""候选与航迹联合建模、关联融合及最终坐标回归。

输入是完整 M1 候选集合 `[B,50]` 和 TrackBank 状态；输出是每候选的杂波/固定槽关联，
以及三个独立槽位的融合量测、历史预测、最终坐标、存在概率和航迹置信度。位置单位为米。
训练时关联采用 straight-through hard one-hot：前向保证候选唯一归属，反向保留 softmax 梯度。
"""
from typing import NamedTuple

import torch
from torch import nn
import torch.nn.functional as F

from original.my_track.config import (
    FEAT_DIM,
    HIST_LEN,
    NUM_TRACKS,
    TRACKER_ATTENTION_HEADS,
    TRACKER_D_MODEL,
    TRACKER_ENCODER_LAYERS,
    TRACKER_FFN_DIM,
)
from original.my_track.measurement import Measurements


class TrackerOutput(NamedTuple):
    """单帧逐槽结果；`observed` 与 ID 确认状态分开，预测不冒充量测。"""
    association_logits: torch.Tensor
    assignments: torch.Tensor
    assignment_onehot: torch.Tensor
    fused_pos: torch.Tensor
    fused_feat: torch.Tensor
    predicted_pos: torch.Tensor
    final_pos: torch.Tensor
    observed: torch.Tensor
    existence_prob: torch.Tensor
    track_confidence: torch.Tensor


class JointTracker(nn.Module):
    """轻量候选-航迹集合跟踪器。

    Args:
        num_tracks: 固定身份槽数，当前数据约定为 3。
        hist_len: TrackBank 历史长度，复用 `HIST_LEN`。
        feat_dim: M1 decoder feature 维度。
        d_model: 候选/航迹交互隐层维度。
        encoder_layers: 集合 Transformer 层数。
        attention_heads: 多头注意力头数。
        ffn_dim: Transformer 前馈层宽度。

    `measurements` 包含米制 `pos[B,50,2]`、概率 `score[B,50]` 和 `feat[B,50,D]`。
    `state` 由 TrackBank 提供，包含 `hist/feat/confidence/confirmed/initialized/miss/age`。
    `dt` 为秒。训练和推理使用相同的硬 argmax 前向决策。
    """

    def __init__(self, num_tracks=NUM_TRACKS, hist_len=HIST_LEN, feat_dim=FEAT_DIM,
                 d_model=TRACKER_D_MODEL, encoder_layers=TRACKER_ENCODER_LAYERS,
                 attention_heads=TRACKER_ATTENTION_HEADS, ffn_dim=TRACKER_FFN_DIM):
        super().__init__()
        if num_tracks != 3:
            raise ValueError("the current association contract requires exactly 3 track slots")
        if d_model % attention_heads:
            raise ValueError("d_model must be divisible by attention_heads")
        self.num_tracks = num_tracks
        self.hist_len = hist_len
        self.feat_dim = feat_dim
        self.d_model = d_model

        self.candidate_projection = nn.Sequential(
            nn.Linear(feat_dim + 3, d_model), nn.LayerNorm(d_model), nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.history_encoder = nn.GRU(input_size=4, hidden_size=d_model, batch_first=True)
        self.track_feature_projection = nn.Linear(feat_dim, d_model)
        self.track_aux_projection = nn.Sequential(nn.Linear(5, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.slot_embedding = nn.Parameter(torch.empty(num_tracks, d_model))
        nn.init.normal_(self.slot_embedding, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=attention_heads,
            dim_feedforward=ffn_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.context_encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_layers)

        pair_dim = d_model * 2 + 4  # context pair, relative xy, feature cosine, M1 score
        self.pair_head = nn.Sequential(
            nn.Linear(pair_dim, d_model), nn.GELU(), nn.Linear(d_model, 1),
        )
        self.clutter_head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 1))
        self.motion_head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 2))
        nn.init.zeros_(self.motion_head[-1].weight)
        nn.init.zeros_(self.motion_head[-1].bias)
        regression_dim = d_model * 2 + 3  # track/fused feature, fused-predicted xy, observed bit
        self.correction_head = nn.Sequential(
            nn.Linear(regression_dim, d_model), nn.GELU(), nn.Linear(d_model, 3),
        )
        self.existence_head = nn.Sequential(
            nn.Linear(regression_dim, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 1),
        )
        self.confidence_head = nn.Sequential(
            nn.Linear(regression_dim, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 1),
        )

    def _validate_inputs(self, measurements, state, dt):
        if not isinstance(measurements, Measurements):
            raise TypeError("measurements must be a Measurements tuple")
        pos, score, feat = measurements
        if pos.ndim != 3 or pos.shape[1:] != (50, 2):
            raise ValueError("measurements.pos must have shape [B,50,2]")
        batch = pos.shape[0]
        if score.shape != (batch, 50) or feat.shape != (batch, 50, self.feat_dim):
            raise ValueError("measurements score/feat shapes do not match the 50-query contract")
        required = ("hist", "feat", "confidence", "confirmed", "initialized", "miss", "age")
        if any(key not in state for key in required):
            raise ValueError(f"state must contain {required}")
        expected = {
            "hist": (batch, self.num_tracks, self.hist_len, 4),
            "feat": (batch, self.num_tracks, self.feat_dim),
            "confidence": (batch, self.num_tracks),
            "confirmed": (batch, self.num_tracks),
            "initialized": (batch, self.num_tracks),
            "miss": (batch, self.num_tracks),
            "age": (batch, self.num_tracks),
        }
        for key, shape in expected.items():
            if tuple(state[key].shape) != shape:
                raise ValueError(f"state[{key!r}] must have shape {shape}")
            if key in ("hist", "feat", "confidence") and state[key].dtype != pos.dtype:
                raise ValueError(f"state[{key!r}] dtype must match measurement positions")
            if key in ("hist", "feat", "confidence") and not torch.isfinite(state[key]).all():
                raise ValueError(f"state[{key!r}] contains NaN or Inf")
        if ((state["confidence"] < 0) | (state["confidence"] > 1)).any():
            raise ValueError("track confidence must be in [0,1]")
        if not torch.isfinite(torch.as_tensor(dt)) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        for name, tensor in (("position", pos), ("score", score), ("feature", feat)):
            if not torch.isfinite(tensor).all():
                raise ValueError(f"measurements {name} contain NaN or Inf")
        if ((score < 0) | (score > 1)).any():
            raise ValueError("measurement scores must be probabilities in [0,1]")
        if any(tensor.device != pos.device for tensor in (score, feat)):
            raise ValueError("measurement tensors must share a device")
        if any(state[key].device != pos.device for key in required):
            raise ValueError("measurement and state tensors must share a device")

    def forward(self, measurements, state, dt=0.24):
        self._validate_inputs(measurements, state, dt)
        pos, score, feat = measurements
        batch = pos.shape[0]

        candidate_input = torch.cat((pos, score.unsqueeze(-1), feat), dim=-1)
        candidate_tokens = self.candidate_projection(candidate_input)

        history = state["hist"].reshape(batch * self.num_tracks, self.hist_len, 4)
        _, history_hidden = self.history_encoder(history)
        history_tokens = history_hidden[-1].reshape(batch, self.num_tracks, self.d_model)
        confirmed = state["confirmed"].to(pos.dtype)
        initialized = state["initialized"].to(pos.dtype)
        aux = torch.stack((
            state["confidence"], confirmed, initialized,
            state["miss"].clamp(max=10 * self.hist_len).to(pos.dtype) / max(self.hist_len, 1),
            state["age"].clamp(max=self.hist_len).to(pos.dtype) / max(self.hist_len, 1),
        ), dim=-1)
        track_tokens = (
            history_tokens
            + self.track_feature_projection(state["feat"])
            + self.track_aux_projection(aux)
            + self.slot_embedding.unsqueeze(0)
        )

        contexts = self.context_encoder(torch.cat((candidate_tokens, track_tokens), dim=1))
        candidate_context = contexts[:, :50]
        track_context = contexts[:, 50:]

        last_state = state["hist"][:, :, -1]
        has_history = state["initialized"].bool()
        predicted_pos = last_state[..., :2] + last_state[..., 2:] * float(dt)
        predicted_pos = predicted_pos + self.motion_head(track_context) * float(dt) * has_history.unsqueeze(-1)
        predicted_pos = predicted_pos * has_history.unsqueeze(-1)

        relative_pos = (pos.unsqueeze(2) - predicted_pos.unsqueeze(1)) / 2.0
        cosine = F.cosine_similarity(
            feat.unsqueeze(2), state["feat"].unsqueeze(1), dim=-1, eps=1e-6
        ).unsqueeze(-1)
        pair_context = torch.cat((
            candidate_context.unsqueeze(2).expand(-1, -1, self.num_tracks, -1),
            track_context.unsqueeze(1).expand(-1, 50, -1, -1),
            relative_pos,
            cosine,
            score.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.num_tracks, -1),
        ), dim=-1)
        track_logits = self.pair_head(pair_context).squeeze(-1)
        clutter_logits = self.clutter_head(candidate_context)
        association_logits = torch.cat((clutter_logits, track_logits), dim=-1)

        assignment_prob = association_logits.softmax(dim=-1)
        hard_onehot = F.one_hot(association_logits.argmax(dim=-1), num_classes=4).to(pos.dtype)
        assignment_onehot = hard_onehot - assignment_prob.detach() + assignment_prob
        assignments = hard_onehot.argmax(dim=-1)

        track_weights = assignment_onehot[..., 1:] * score.unsqueeze(-1)
        weight_sum = track_weights.sum(dim=1)
        denominator = weight_sum.clamp_min(torch.finfo(pos.dtype).tiny)
        fused_pos = torch.einsum("bmk,bmd->bkd", track_weights, pos) / denominator.unsqueeze(-1)
        fused_feat = torch.einsum("bmk,bmd->bkd", track_weights, feat) / denominator.unsqueeze(-1)
        selected = assignments.unsqueeze(-1) == torch.arange(
            1, self.num_tracks + 1, device=pos.device
        ).view(1, 1, -1)
        observed = (selected & (score.unsqueeze(-1) > 0)).any(dim=1)

        regression_context = torch.cat((
            track_context,
            fused_feat,
            fused_pos - predicted_pos,
            observed.to(pos.dtype).unsqueeze(-1),
        ), dim=-1)
        correction = self.correction_head(regression_context)
        gate = correction[..., :1].sigmoid()
        corrected = predicted_pos + gate * (fused_pos - predicted_pos) + correction[..., 1:]
        final_pos = torch.where(observed.unsqueeze(-1), corrected, predicted_pos)
        existence_prob = self.existence_head(regression_context).squeeze(-1).sigmoid()
        track_confidence = self.confidence_head(regression_context).squeeze(-1).sigmoid()

        return TrackerOutput(
            association_logits=association_logits,
            assignments=assignments,
            assignment_onehot=assignment_onehot,
            fused_pos=fused_pos,
            fused_feat=fused_feat,
            predicted_pos=predicted_pos,
            final_pos=final_pos,
            observed=observed,
            existence_prob=existence_prob,
            track_confidence=track_confidence,
        )
