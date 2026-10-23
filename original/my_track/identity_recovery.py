"""已确认航迹片段的在线身份恢复；不生成、删除或纠偏任何输出点。

只消费 LifecycleTracker 已确认的输出。历史端点和当前点来自推理，不接收 GT。
known_count 是显式固定人数/无消亡条件，禁止把此条件混作自主跟踪能力。
"""

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from .config import HIST_LEN
from .m1_temporal_baseline import TemporalFrameOutput


@dataclass
class _IdentityEndpoint:
    owner: int
    position: np.ndarray | None
    observed_step: int
    seen_step: int


class TrackletIdentityRecovery:
    """保留短片段原身份；新确认片段仅在空间和双向歧义检查通过时继承旧身份。"""

    def __init__(self, memory_steps=HIST_LEN * 50, gate_m=1., gate_growth_m=0.,
                 max_gate_m=7.64, ambiguity_margin_m=.5, known_count=None):
        for name, value in (('memory_steps', memory_steps), ('known_count', known_count)):
            if value is None and name == 'known_count':
                continue
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        for name, value in (('gate_m', gate_m), ('gate_growth_m', gate_growth_m),
                            ('max_gate_m', max_gate_m), ('ambiguity_margin_m', ambiguity_margin_m)):
            if not np.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and non-negative')
        if gate_m <= 0 or max_gate_m < gate_m:
            raise ValueError('0 < gate_m <= max_gate_m required')
        self.memory_steps = int(memory_steps)
        self.gate_m, self.gate_growth_m = float(gate_m), float(gate_growth_m)
        self.max_gate_m, self.ambiguity_margin_m = float(max_gate_m), float(ambiguity_margin_m)
        self.known_count = known_count
        self.reset()

    def reset(self):
        self._step = 0
        self._next_id = 0
        self._mapping = {}
        self._endpoints = {}
        self.statistics = {'recovered': 0, 'new_identities': 0, 'ambiguous_rejections': 0}

    def remap(self, frame):
        """保持坐标/分数/观测标记及行顺序，仅返回新的唯一整数 track_ids。"""
        points = np.asarray(frame.positions, dtype=np.float64)
        ids = np.asarray(frame.track_ids)
        scores, observed = np.asarray(frame.scores), np.asarray(frame.observed)
        if (points.ndim != 2 or points.shape[1] != 2 or ids.shape != (len(points),) or
                scores.shape != ids.shape or observed.shape != ids.shape or
                not np.isfinite(points).all() or not np.isfinite(scores).all() or
                not np.isfinite(ids).all() or not np.equal(ids, np.floor(ids)).all() or
                len(set(ids.tolist())) != len(ids)):
            raise ValueError('frame requires finite aligned fields and unique integral IDs')
        if not np.isin(observed, (True, False)).all():
            raise ValueError('observed must contain boolean values')
        self._step += 1
        raw_ids = [int(i) for i in ids]
        active = {self._mapping[i] for i in raw_ids if i in self._mapping}
        # 已知原始片段的映射不因记忆过期而删除；过期仅禁止陌生片段继承它。
        bank = [i for i, endpoint in self._endpoints.items()
                if i not in active and endpoint.position is not None and
                self._step - endpoint.seen_step <= self.memory_steps]
        new_rows = [row for row, raw_id in enumerate(raw_ids)
                    if raw_id not in self._mapping and observed[row]]
        inherited = {}
        if bank and new_rows:
            distances = np.linalg.norm(points[new_rows, None] -
                                       np.stack([self._endpoints[i].position for i in bank])[None],
                                       axis=-1)
            gaps = np.asarray([self._step - self._endpoints[i].observed_step for i in bank])
            gates = np.minimum(self.max_gate_m, self.gate_m + self.gate_growth_m * gaps)
            valid = distances <= gates[None]
            # 理想化固定人数：全部身份曾建立、当前只缺一人，才可排除身份歧义。
            unique_missing = (self.known_count is not None and len(new_rows) == len(bank) == 1
                              and len(self._endpoints) == self.known_count
                              and len(active) + 1 == self.known_count)
            if unique_missing:
                inherited[new_rows[0]] = bank[0]
            else:
                for row, candidate_row in enumerate(new_rows):
                    column = int(np.argmin(distances[row]))
                    if not valid[row, column] or int(np.argmin(distances[:, column])) != row:
                        continue
                    alternatives = np.concatenate((np.delete(distances[row], column),
                                                   np.delete(distances[:, column], row)))
                    # 同时排除一对多、多对一；不靠 greedy 的遍历顺序打破身份歧义。
                    if (len(alternatives) and
                            float(alternatives.min()) - distances[row, column] < self.ambiguity_margin_m):
                        continue
                    inherited[candidate_row] = bank[column]
                self.statistics['ambiguous_rejections'] += sum(
                    bool(valid[row].any()) and candidate_row not in inherited
                    for row, candidate_row in enumerate(new_rows))
        result_ids = []
        for row, raw_id in enumerate(raw_ids):
            if raw_id not in self._mapping:
                if row in inherited:
                    identity = inherited[row]
                    # 转让所有权，旧片段再次出现时不得重复使用已被继承的 ID。
                    self._mapping.pop(self._endpoints[identity].owner, None)
                    self.statistics['recovered'] += 1
                else:
                    identity = self._next_id
                    self._next_id += 1
                    self.statistics['new_identities'] += 1
                self._mapping[raw_id] = identity
            identity = self._mapping[raw_id]
            previous = self._endpoints.get(identity)
            position = points[row].copy() if observed[row] else (
                previous.position if previous is not None else None)
            observed_step = self._step if observed[row] else (
                previous.observed_step if previous is not None else self._step)
            self._endpoints[identity] = _IdentityEndpoint(raw_id, position, observed_step, self._step)
            result_ids.append(identity)
        return TemporalFrameOutput(frame.positions, frame.scores,
                                   np.asarray(result_ids, dtype=np.int64), frame.observed)


class IdentityRecoveryTracker:
    """组合适配器：底层继续独立决定检测，外层只负责身份别名。"""

    def __init__(self, tracker, **recovery_settings):
        self.tracker = tracker
        self.recovery = TrackletIdentityRecovery(**recovery_settings)

    @property
    def statistics(self):
        return {**self.tracker.statistics, **self.recovery.statistics}

    def reset(self):
        self.tracker.reset()
        self.recovery.reset()

    def update(self, positions, scores):
        return self.recovery.remap(self.tracker.update(positions, scores))
