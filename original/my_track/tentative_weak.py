"""独立实验：暂定片段弱提议续接，以及可选的高分存在性确认。"""

from collections import deque

from .lifecycle_tracker import LifecycleTracker


class TentativeWeakLifecycleTracker(LifecycleTracker):
    """确认片段优先，弱提议只续接；沿用原 HIST_LEN 确认及输出契约。"""

    def __init__(self, primary_confirmation=False, **kwargs):
        if kwargs.get('association_policy', 'cascade') != 'cascade':
            raise ValueError('tentative weak updates require association_policy=cascade')
        self.primary_confirmation = bool(primary_confirmation)
        super().__init__(**kwargs)

    def reset(self):
        super().reset()
        self._primary_hits = {}
        self._primary_observed = set()

    def update(self, candidate_positions, candidate_scores, candidate_features=None):
        self._primary_observed = set()
        output = super().update(candidate_positions, candidate_scores, candidate_features)
        # 确认/删除后的片段不再需要暂定确认证据，避免长期记录积累无效历史。
        active = {track.track_id for track in self._tracks if not track.confirmed}
        self._primary_hits = {key: value for key, value in self._primary_hits.items()
                              if key in active}
        return output

    def _observe(self, track, position, score):
        super()._observe(track, position, score)
        if self.primary_confirmation and score >= self.birth_threshold:
            self._primary_observed.add(track.track_id)

    def _tentative_weak_association_stages(self, indices):
        return [(indices, self.association_radius_m)]

    def _confirm(self, track):
        if not self.primary_confirmation:
            return super()._confirm(track)
        if track.confirmed:
            return
        history = self._primary_hits.get(track.track_id)
        if history is None:
            # 新片段只由主高分提议出生；出生不经过 _observe，须单独计一次。
            history = deque([True], maxlen=self.hist_len)
            self._primary_hits[track.track_id] = history
        else:
            history.append(track.track_id in self._primary_observed)
        # 弱命中可维持片段，但不替代“目标确实存在”的高分历史支持。
        if len(history) == self.hist_len and sum(history) / self.hist_len >= self.confirm_fraction:
            super()._confirm(track)
