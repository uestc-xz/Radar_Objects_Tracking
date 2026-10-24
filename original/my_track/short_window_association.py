"""独立短窗候选排序；模型接口只接收量测，不接收 GT 或身份编号。"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .config import HIST_LEN


def encode_query(observations, current_step, positions, scores, primary, hist_len=HIST_LEN):
    """按真实 sample 年龄编码过去量测；相对最近一次观测，米制坐标。

    observations 为严格递增的 (sample_id, xy, score)。GT 配对属于离线调用方；
    此函数同样可编码未来闭环中实际消费的历史，不能包含当前或未来观测。
    """
    if isinstance(hist_len, bool) or not isinstance(hist_len, int) or hist_len < 2:
        raise ValueError('hist_len must be an integer >= 2')
    positions = np.asarray(positions, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    primary = np.asarray(primary, dtype=bool)
    if (positions.ndim != 2 or positions.shape[1] != 2 or
            scores.shape != (len(positions),) or primary.shape != scores.shape):
        raise ValueError('positions/scores/primary shapes must be [N,2]/[N]/[N]')
    if (not np.isfinite(positions).all() or not np.isfinite(scores).all() or
            ((scores < 0) | (scores > 1)).any()):
        raise ValueError('finite coordinates and scores in [0,1] required')
    recent, previous = [], None
    for step, xy, score in observations:
        xy = np.asarray(xy, dtype=np.float64)
        if (not isinstance(step, (int, np.integer)) or step >= current_step or
                (previous is not None and step <= previous) or xy.shape != (2,) or
                not np.isfinite(xy).all() or not np.isfinite(score) or not 0 <= score <= 1):
            raise ValueError('history must be finite, ordered and strictly before current_step')
        previous = step
        if current_step - step <= hist_len:
            recent.append((step, xy, score))
    if not recent:
        raise ValueError('at least one past observation inside hist_len is required')
    anchor = recent[-1][1]
    history = np.zeros((hist_len, 5), dtype=np.float32)
    for step, xy, score in recent:
        age = current_step - step
        history[hist_len - age] = [*(xy - anchor), score, age / hist_len, 1.]
    candidates = np.column_stack((positions - anchor, scores, primary)).astype(np.float32)
    return history, candidates


def masked_logits(logits, candidate_mask):
    """null 始终有效；padding 不参与 CE 的归一化分母或预测。"""
    if (logits.ndim != 2 or candidate_mask.shape != (len(logits), logits.shape[1] - 1) or
            candidate_mask.dtype != torch.bool or len(logits) == 0):
        raise ValueError('logits/mask must be nonempty [B,1+N]/bool[B,N]')
    if not torch.isfinite(logits).all():
        raise ValueError('logits must be finite')
    valid = torch.cat((torch.ones_like(candidate_mask[:, :1]), candidate_mask), dim=1)
    if candidate_mask.shape[1] == 0:
        valid = torch.ones_like(logits, dtype=torch.bool)
    return logits.masked_fill(~valid, torch.finfo(logits.dtype).min)


def ranking_loss(logits, labels, candidate_mask):
    """每查询一个正确候选（编号 1..N）或 null(0) 的平均交叉熵。"""
    logits = masked_logits(logits, candidate_mask)
    if (labels.shape != (len(logits),) or labels.dtype != torch.long or
            (labels < 0).any() or (labels >= logits.shape[1]).any()):
        raise ValueError('labels must be long[B] inside [0,N]')
    valid = torch.cat((torch.ones(len(logits), 1, device=logits.device, dtype=torch.bool),
                       candidate_mask), dim=1)
    if not valid.gather(1, labels[:, None]).all():
        raise ValueError('a target cannot refer to padding')
    return F.cross_entropy(logits, labels)


def baseline_logits(history, candidates, candidate_mask, use_velocity=False):
    """固定诊断：最近观测 / 最近两观测限速直线预测，1m null 门。

    速度单位 m/sample；不等价于 LifecycleTracker 的 EMA 和衰减预测。
    """
    offset = torch.zeros_like(history[:, 0, :2])
    if use_velocity:
        valid = history[:, :, 4] > .5
        indices = torch.arange(history.shape[1], device=history.device).expand(len(history), -1)
        ordered = indices.masked_fill(~valid, -1).topk(2, dim=1).values
        rows = torch.arange(len(history), device=history.device)
        last = history[rows, ordered[:, 0].clamp_min(0)]
        before = history[rows, ordered[:, 1].clamp_min(0)]
        gap = (before[:, 3] - last[:, 3]) * history.shape[1]
        velocity = (last[:, :2] - before[:, :2]) / gap.clamp_min(1)[:, None]
        speed = velocity.norm(dim=-1).clamp_min(1e-8)
        velocity = velocity * (.2 / speed).clamp_max(1)[:, None]
        offset = velocity * (last[:, 3] * history.shape[1])[:, None]
        offset = offset * (ordered[:, 1] >= 0)[:, None]
    distances = (candidates[:, :, :2] - offset[:, None]).norm(dim=-1)
    logits = torch.cat((distances.new_full((len(history), 1), -4.), -distances / .25), dim=1)
    return masked_logits(logits, candidate_mask)


class ShortWindowAssociation(nn.Module):
    """小型历史编码器 + 共享边残差；候选置换只改变对应输出顺序。"""

    def __init__(self, hist_len=HIST_LEN):
        super().__init__()
        if isinstance(hist_len, bool) or not isinstance(hist_len, int) or hist_len < 2:
            raise ValueError('hist_len must be an integer >= 2')
        self.hist_len = hist_len
        self.history_encoder = nn.Sequential(nn.Linear(hist_len * 5, 64), nn.GELU(),
                                             nn.Linear(64, 32), nn.GELU())
        self.edge = nn.Sequential(nn.Linear(36, 64), nn.GELU(), nn.Linear(64, 1))
        self.reject = nn.Sequential(nn.Linear(35, 32), nn.GELU(), nn.Linear(32, 1))
        for head in (self.edge[-1], self.reject[-1]):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, history, candidates, candidate_mask):
        """[B,H,5], [B,N,4], bool[B,N] -> logits [B,1+N]，首列 null。"""
        if (history.ndim != 3 or history.shape[1:] != (self.hist_len, 5) or
                candidates.ndim != 3 or candidates.shape[0] != len(history) or
                candidates.shape[2] != 4 or candidate_mask.shape != candidates.shape[:2] or
                candidate_mask.dtype != torch.bool):
            raise ValueError('invalid history/candidate/mask shape or dtype')
        if not torch.isfinite(history).all() or not torch.isfinite(candidates).all():
            raise ValueError('model inputs must be finite, including padding')
        base = baseline_logits(history, candidates, candidate_mask)
        encoded = self.history_encoder(history.flatten(1))
        count = candidate_mask.sum(dim=1)
        if candidates.shape[1]:
            distance = candidates[:, :, :2].norm(dim=-1)
            nearest = distance.masked_fill(~candidate_mask, torch.inf).amin(dim=1)
            nearest = torch.where(count > 0, nearest, 0.)
            best_score = candidates[:, :, 2].masked_fill(~candidate_mask, 0.).amax(dim=1)
        else:
            nearest = best_score = count.to(history.dtype)
        context = torch.stack((count.to(history.dtype) / 50., nearest, best_score), dim=1)
        reject = base[:, :1] + self.reject(torch.cat((encoded, context), dim=-1))
        edges = torch.cat((encoded[:, None].expand(-1, candidates.shape[1], -1), candidates), dim=-1)
        ranked = base[:, 1:] + self.edge(edges).squeeze(-1)
        return masked_logits(torch.cat((reject, ranked), dim=1), candidate_mask)
