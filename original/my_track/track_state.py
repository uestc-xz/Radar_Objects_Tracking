"""固定槽位航迹状态库。

每个序列拥有独立 TrackBank。`initialized` 表示槽位已有状态，`confirmed` 表示 ID 已稳定，
`observed` 是本帧量测标记。漏检只更新预测历史和置信度，不会删除或复用已确认身份。
所有坐标/速度为米和米/秒；本对象不参与 autograd。
"""
import math

import torch

from original.my_track.config import (
    CONFIRM_THRESHOLD,
    FEAT_DIM,
    HIST_LEN,
    MISS_CONFIDENCE_DECAY,
    NUM_TRACKS,
)


class TrackBank:
    """保存单条记录的三槽状态及滚动确认统计。"""

    def __init__(self, num_tracks=NUM_TRACKS, hist_len=HIST_LEN, feat_dim=FEAT_DIM,
                 confirm_threshold=CONFIRM_THRESHOLD, miss_decay=MISS_CONFIDENCE_DECAY):
        if num_tracks != 3:
            raise ValueError("the current GT contract requires exactly 3 track slots")
        if hist_len < 1:
            raise ValueError("hist_len must be positive")
        if not math.isfinite(float(confirm_threshold)) or not 0 <= confirm_threshold <= 1:
            raise ValueError("confirm_threshold must be in [0,1]")
        if not math.isfinite(float(miss_decay)) or not 0 <= miss_decay <= 1:
            raise ValueError("miss_decay must be in [0,1]")
        self.num_tracks = num_tracks
        self.hist_len = hist_len
        self.feat_dim = feat_dim
        self.confirm_threshold = float(confirm_threshold)
        self.miss_decay = float(miss_decay)
        self.reset()

    def reset(self, device=None, dtype=torch.float32):
        """清空序列状态；新记录必须显式 reset，序列内不重置已确认槽。"""
        device = torch.device(device) if device is not None else torch.device("cpu")
        self.hist = torch.zeros(self.num_tracks, self.hist_len, 4, device=device, dtype=dtype)
        self.feat = torch.zeros(self.num_tracks, self.feat_dim, device=device, dtype=dtype)
        self.confidence = torch.zeros(self.num_tracks, device=device, dtype=dtype)
        self.existence_history = torch.zeros(self.num_tracks, self.hist_len, device=device, dtype=dtype)
        self.initialized = torch.zeros(self.num_tracks, device=device, dtype=torch.bool)
        self.confirmed = torch.zeros(self.num_tracks, device=device, dtype=torch.bool)
        self.miss = torch.zeros(self.num_tracks, device=device, dtype=torch.long)
        self.age = torch.zeros(self.num_tracks, device=device, dtype=torch.long)
        self.frames_seen = 0

    def as_tensors(self, device=None):
        """返回 `JointTracker` 所需批字段前的无梯度状态张量。"""
        return {
            "hist": self.hist.to(device=device),
            "feat": self.feat.to(device=device),
            "confidence": self.confidence.to(device=device),
            "confirmed": self.confirmed.to(device=device),
            "initialized": self.initialized.to(device=device),
            "miss": self.miss.to(device=device),
            "age": self.age.to(device=device),
        }

    def update(self, final_pos, fused_feat, observed, existence_prob,
               track_confidence, dt=0.24):
        """按帧提交模型输出并推进状态。

        `final_pos[3,2]` 在观测/预测两种情况下都写入历史；特征只在 observed 时更新。
        首次观测将速度设为零，后续速度用位置差分 EMA；confirmed 只允许 false->true。
        """
        expected = {
            "final_pos": (self.num_tracks, 2),
            "fused_feat": (self.num_tracks, self.feat_dim),
            "observed": (self.num_tracks,),
            "existence_prob": (self.num_tracks,),
            "track_confidence": (self.num_tracks,),
        }
        values = {
            "final_pos": final_pos,
            "fused_feat": fused_feat,
            "observed": observed,
            "existence_prob": existence_prob,
            "track_confidence": track_confidence,
        }
        for name, shape in expected.items():
            if tuple(values[name].shape) != shape:
                raise ValueError(f"{name} must have shape {shape}")
        if not math.isfinite(float(dt)) or dt <= 0:
            raise ValueError("dt must be finite and positive")

        device, dtype = self.hist.device, self.hist.dtype
        final_pos = final_pos.detach().to(device=device, dtype=dtype)
        fused_feat = fused_feat.detach().to(device=device, dtype=dtype)
        observed = observed.detach().to(device=device, dtype=torch.bool)
        existence_prob = existence_prob.detach().to(device=device, dtype=dtype)
        track_confidence = track_confidence.detach().to(device=device, dtype=dtype)
        for name, tensor in (("final_pos", final_pos), ("fused_feat", fused_feat),
                             ("existence_prob", existence_prob), ("track_confidence", track_confidence)):
            if not torch.isfinite(tensor).all():
                raise ValueError(f"{name} contains NaN or Inf")
        if ((existence_prob < 0) | (existence_prob > 1)).any():
            raise ValueError("existence_prob must be in [0,1]")
        if ((track_confidence < 0) | (track_confidence > 1)).any():
            raise ValueError("track_confidence must be in [0,1]")

        was_initialized = self.initialized
        initialized = was_initialized | observed
        previous_pos = self.hist[:, -1, :2]
        previous_vel = self.hist[:, -1, 2:]
        measured_velocity = (final_pos - previous_pos) / float(dt)
        velocity = torch.where(
            (was_initialized & observed).unsqueeze(-1),
            0.5 * previous_vel + 0.5 * measured_velocity,
            previous_vel,
        )
        velocity = torch.where((~was_initialized & observed).unsqueeze(-1), torch.zeros_like(velocity), velocity)
        state = torch.cat((final_pos, velocity), dim=-1)
        self.hist = torch.cat((self.hist[:, 1:], state.unsqueeze(1)), dim=1)

        update_feat = torch.where(
            (was_initialized & observed).unsqueeze(-1),
            0.5 * self.feat + 0.5 * fused_feat,
            fused_feat,
        )
        self.feat = torch.where(observed.unsqueeze(-1), update_feat, self.feat)
        decayed_confidence = self.confidence * self.miss_decay
        self.confidence = torch.where(observed, track_confidence, decayed_confidence)
        self.miss = torch.where(observed, torch.zeros_like(self.miss), self.miss + initialized.long())
        self.initialized = initialized
        self.age = torch.where(initialized, self.age + 1, torch.zeros_like(self.age))

        self.existence_history = torch.cat(
            (self.existence_history[:, 1:], existence_prob.unsqueeze(-1)), dim=-1
        )
        enough_history = self.age >= self.hist_len
        meets_presence = self.existence_history.mean(dim=-1) >= self.confirm_threshold
        self.confirmed = self.confirmed | (initialized & enough_history & meets_presence)
        self.frames_seen += 1


def batch_track_states(banks, device=None):
    """沿 batch 维堆叠同一时刻的多个独立 TrackBank。"""
    if not banks:
        raise ValueError("at least one TrackBank is required")
    states = [bank.as_tensors(device=device) for bank in banks]
    keys = states[0].keys()
    if any(state.keys() != keys for state in states[1:]):
        raise ValueError("all TrackBank instances must expose identical state fields")
    return {key: torch.stack([state[key] for state in states], dim=0) for key in keys}
