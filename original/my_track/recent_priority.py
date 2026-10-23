"""独立关联策略：近期活动轨迹仅在可信内门先匹配，剩余仍用原门限。"""

import numpy as np

from .lifecycle_tracker import LifecycleTracker


class RecentPriorityTracker(LifecycleTracker):
    """高分/弱分各自分阶段，门外不强行保护；不改变生命周期和输出公式。"""

    def __init__(self, *args, priority_gate_m=.5, require_stale=False, **kwargs):
        super().__init__(*args, **kwargs)
        if (not np.isfinite(priority_gate_m) or priority_gate_m <= 0 or
                priority_gate_m > self.association_radius_m):
            raise ValueError('priority_gate_m must be finite and in (0, association_radius_m]')
        if self.association_policy != 'cascade':
            raise ValueError('recent priority experiment requires cascade association')
        self.priority_gate_m = float(priority_gate_m)
        self.require_stale = bool(require_stale)

    def _confirmed_association_stages(self, indices):
        if self.require_stale and not any(self._tracks[i].missed > 0 for i in indices):
            return super()._confirmed_association_stages(indices)
        recent = [i for i in indices if self._tracks[i].missed == 0]
        return [(recent, self.priority_gate_m), (indices, self.association_radius_m)]
