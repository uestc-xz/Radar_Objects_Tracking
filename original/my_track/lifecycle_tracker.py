"""冻结 M1 的独立航迹管理器：分离出生、确认、身份保留和预测输出。

坐标为米，速度为每个输入步骤的位移；不依赖物理 DT、不读取 GT 或 M1 特征。
内部置信度是管理分数，不宣称是经过概率校准的目标存在概率。
"""

from collections import deque
from dataclasses import dataclass
from numbers import Integral

import numpy as np

from .config import HIST_LEN
from .gated_matching import gated_pairs
from .m1_frame_baseline import ClusteredMeasurements, cluster_candidates
from .m1_temporal_baseline import TemporalFrameOutput


@dataclass
class _ManagedTrack:
    track_id: int
    position: np.ndarray
    velocity: np.ndarray
    confidence: float
    hits: deque
    confirmed: bool = False
    missed: int = 0
    appearance: object = None
    last_feature_step: int = -1


class LifecycleTracker:
    """每个实例只处理一条按时间排序的序列；序列边界须 reset。"""

    def __init__(
        self, hist_len=HIST_LEN, max_tracks=3, score_threshold=.8,
        birth_threshold=.96, cluster_radius_m=.75, association_radius_m=1.,
        confirm_fraction=.6, tentative_max_missed=2, max_missed=None,
        output_max_missed=2, confidence_decay=.95, output_confidence=.5,
        velocity_alpha=.2, max_step_m=.2, prediction_decay=.8,
        no_death=False, weak_updates=True, association_policy='cascade',
    ):
        max_missed = hist_len if max_missed is None else max_missed
        for name, value, lower in (
            ('hist_len', hist_len, 1), ('max_tracks', max_tracks, 1),
            ('tentative_max_missed', tentative_max_missed, 0),
            ('max_missed', max_missed, 0), ('output_max_missed', output_max_missed, 0),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < lower:
                raise ValueError(f'{name} must be an integer >= {lower}')
        for name, value in (
            ('score_threshold', score_threshold), ('birth_threshold', birth_threshold),
            ('confirm_fraction', confirm_fraction), ('confidence_decay', confidence_decay),
            ('output_confidence', output_confidence), ('velocity_alpha', velocity_alpha),
            ('prediction_decay', prediction_decay),
        ):
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f'{name} must be finite and in [0,1]')
        if birth_threshold < score_threshold or confirm_fraction <= 0:
            raise ValueError('birth_threshold >= score_threshold and confirm_fraction > 0 required')
        if association_policy not in ('joint', 'cascade'):
            raise ValueError('association_policy must be joint or cascade')
        for name, value, zero_allowed in (
            ('association_radius_m', association_radius_m, False),
            ('max_step_m', max_step_m, False), ('cluster_radius_m', cluster_radius_m, True),
        ):
            if not np.isfinite(value) or value < 0 or (value == 0 and not zero_allowed):
                raise ValueError(f'{name} has invalid range')
        self.hist_len, self.max_tracks = int(hist_len), int(max_tracks)
        self.score_threshold, self.birth_threshold = float(score_threshold), float(birth_threshold)
        self.cluster_radius_m, self.association_radius_m = float(cluster_radius_m), float(association_radius_m)
        self.confirm_fraction = float(confirm_fraction)
        self.tentative_max_missed, self.max_missed = int(tentative_max_missed), int(max_missed)
        self.output_max_missed = int(output_max_missed)
        self.confidence_decay, self.output_confidence = float(confidence_decay), float(output_confidence)
        self.velocity_alpha, self.max_step_m = float(velocity_alpha), float(max_step_m)
        self.prediction_decay = float(prediction_decay)
        self.no_death, self.weak_updates = bool(no_death), bool(weak_updates)
        self.association_policy = association_policy
        self.reset()

    def reset(self):
        self._tracks = []
        self._next_id = 0
        self._frame_index = 0
        self.statistics = {'created': 0, 'confirmed': 0, 'deleted': 0}

    def _predict(self, track, gap):
        decay = self.prediction_decay
        multiplier = gap if decay == 1 else (1 - decay ** gap) / (1 - decay)
        return track.position + multiplier * track.velocity

    def _proposals(self, positions, scores):
        # 一次验证全部候选后才允许状态改变；排序使相同集合的输入排列不影响出生 ID。
        positions = np.asarray(positions, dtype=np.float64)
        scores = np.asarray(scores, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1] != 2 or scores.shape != (len(positions),):
            raise ValueError('positions/scores must have shape [N,2]/[N]')
        if (not np.isfinite(positions).all() or not np.isfinite(scores).all() or
                ((scores < 0) | (scores > 1)).any()):
            raise ValueError('positions must be finite; scores must be finite in [0,1]')
        order = np.lexsort((positions[:, 1], positions[:, 0], -scores))
        positions, scores = positions[order], scores[order]
        primary = cluster_candidates(positions, scores, self.birth_threshold, self.cluster_radius_m)
        weak = (scores >= self.score_threshold) & (scores < self.birth_threshold)
        if len(primary.positions):
            distances = np.linalg.norm(positions[:, None] - primary.positions[None], axis=-1)
            weak &= (distances > self.cluster_radius_m).all(axis=1)
        supplemental = cluster_candidates(
            positions[weak], scores[weak], self.score_threshold, self.cluster_radius_m,
        )
        return primary, supplemental

    def _observe(self, track, position, score):
        gap = track.missed + 1
        delta = (position - track.position) / gap
        norm = float(np.linalg.norm(delta))
        if norm > self.max_step_m:
            delta = delta * self.max_step_m / norm
        track.velocity = (1 - self.velocity_alpha) * track.velocity + self.velocity_alpha * delta
        track.position = position.copy()
        track.confidence = .8 * track.confidence + .2 * float(score)
        track.missed = 0

    def _association_predictions(self, indices):
        """关联预测扩展点；默认公式保持原样，与最终输出位置的计算分开。"""
        return np.stack([self._predict(self._tracks[i], self._tracks[i].missed + 1)
                         for i in indices])

    def _confirmed_association_stages(self, indices):
        """默认所有确认状态一起配对；实验子类可增加可信内门的优先阶段。"""
        return [(indices, self.association_radius_m)]

    def _tentative_weak_association_stages(self, indices):
        """默认不允许暂定片段续接弱提议；独立实验子类可显式启用。"""
        return []

    def _proposal_features(self, positions, scores, features, primary, weak):
        """按聚类结果产生外观提议；默认路径不使用 M1 特征。"""
        return None, None

    def _association_pairwise_penalty(self, indices, proposal_features, gate_m):
        """额外的成对代价；基类保持 None，避免改变原关联结果。"""
        return None

    def _update_appearance(self, track, feature):
        """仅由显式实验子类维护外观状态。"""
        return None

    def _association_pairs(self, indices, points, scores, gate_m, pairwise_penalty):
        """默认配对公式不变；冻结模型实验可独立替换边排序。"""
        return gated_pairs(self._association_predictions(indices), points, gate_m,
                           .1 * (1 - scores), pairwise_penalty)

    def update(self, candidate_positions, candidate_scores, candidate_features=None):
        """输入本帧候选，输出已确认且当前仍允许报告的航迹点。"""
        primary, weak = self._proposals(candidate_positions, candidate_scores)
        primary_features, weak_features = self._proposal_features(
            candidate_positions, candidate_scores, candidate_features, primary, weak,
        )
        matched_tracks, used_primary, used_weak = set(), set(), set()

        def associate(indices, proposals, available, primary_count=None, radius_m=None,
                      feature_values=None):
            if not indices or not available:
                return
            gate = self.association_radius_m if radius_m is None else radius_m
            points = proposals.positions[available]
            feature_set = (primary_features if proposals is primary else weak_features
                           if proposals is weak else feature_values)
            self._current_proposal_positions = points
            pair_penalty = self._association_pairwise_penalty(
                indices, None if feature_set is None else feature_set[available], gate,
            )
            for row, col in self._association_pairs(
                    indices, points, proposals.scores[available], gate, pair_penalty):
                index, proposal_index = indices[row], available[col]
                self._observe(self._tracks[index], proposals.positions[proposal_index],
                              proposals.scores[proposal_index])
                self._update_appearance(
                    self._tracks[index], None if feature_set is None else feature_set[proposal_index],
                )
                matched_tracks.add(index)
                if proposals is primary or (primary_count is not None and proposal_index < primary_count):
                    used_primary.add(proposal_index)
                if proposals is weak:
                    used_weak.add(proposal_index)

        confirmed = [i for i, track in enumerate(self._tracks) if track.confirmed]
        tentative = [i for i, track in enumerate(self._tracks) if not track.confirmed]
        if self.association_policy == 'joint' and self.weak_updates:
            # M1 分数作为代价证据；不允许更远的高分点强制压制近处弱分真量测。
            combined = ClusteredMeasurements(
                np.concatenate((primary.positions, weak.positions)),
                np.concatenate((primary.scores, weak.scores)),
                np.concatenate((primary.sizes, weak.sizes)),
            )
            combined_features = (None if primary_features is None or weak_features is None else
                                 np.concatenate((primary_features, weak_features), axis=0))
            associate(confirmed, combined, list(range(len(combined.positions))),
                      primary_count=len(primary.positions), feature_values=combined_features)
        else:
            for indices, gate in self._confirmed_association_stages(confirmed):
                associate([i for i in indices if i not in matched_tracks], primary,
                          [i for i in range(len(primary.positions)) if i not in used_primary],
                          radius_m=gate)
        associate(tentative, primary, [i for i in range(len(primary.positions)) if i not in used_primary])
        if self.weak_updates and self.association_policy == 'cascade':
            for indices, gate in self._confirmed_association_stages(confirmed):
                associate([i for i in indices if i not in matched_tracks], weak,
                          [i for i in range(len(weak.positions)) if i not in used_weak],
                          radius_m=gate)
            for indices, gate in self._tentative_weak_association_stages(tentative):
                # 已确认片段先用弱提议，暂定片段只能取剩余且本步未被消费的量测。
                associate([i for i in indices if i not in matched_tracks], weak,
                          [i for i in range(len(weak.positions)) if i not in used_weak],
                          radius_m=gate)

        retained = []
        for index, track in enumerate(self._tracks):
            hit = index in matched_tracks
            track.hits.append(hit)
            if not hit:
                track.missed += 1
                track.confidence *= self.confidence_decay
            limit = self.max_missed if track.confirmed else self.tentative_max_missed
            if track.missed > limit and not (track.confirmed and self.no_death):
                self.statistics['deleted'] += 1
                continue
            self._confirm(track)
            retained.append(track)

        birth_order = np.lexsort((primary.positions[:, 1], primary.positions[:, 0], -primary.scores))
        for index in birth_order:
            if index in used_primary or len(retained) >= self.max_tracks:
                continue
            track = _ManagedTrack(
                self._next_id, primary.positions[index].copy(), np.zeros(2),
                float(primary.scores[index]), deque([True], maxlen=self.hist_len),
            )
            self._update_appearance(
                track, None if primary_features is None else primary_features[index],
            )
            self._next_id += 1
            self.statistics['created'] += 1
            self._confirm(track)
            retained.append(track)
        self._tracks = retained
        self._frame_index += 1
        emitted = sorted((track for track in retained if track.confirmed and
                          (track.missed == 0 or (track.missed <= self.output_max_missed and
                                                track.confidence >= self.output_confidence))),
                         key=lambda track: track.track_id)
        return TemporalFrameOutput(
            np.asarray([self._predict(t, t.missed) for t in emitted]).reshape(-1, 2),
            np.asarray([t.confidence for t in emitted]),
            np.asarray([t.track_id for t in emitted], dtype=np.int64),
            np.asarray([t.missed == 0 for t in emitted], dtype=bool),
        )

    def _confirm(self, track):
        if (not track.confirmed and len(track.hits) == self.hist_len and
                sum(track.hits) / self.hist_len >= self.confirm_fraction):
            track.confirmed = True
            self.statistics['confirmed'] += 1
