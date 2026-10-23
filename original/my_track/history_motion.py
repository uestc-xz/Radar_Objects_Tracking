"""历史位置趋势只用于关联；保留原生命周期及已观测/漏检输出坐标规则。"""

from collections import deque
from numbers import Integral

import numpy as np

from .lifecycle_tracker import LifecycleTracker


def fit_history_prediction(positions, steps, gap, max_step_m=.2, prediction_decay=.8):
    """最小二乘趋势、速度范数截断及衰减预测；速度单位为 m/输入步。

    steps 必须严格递增，gap 是距最后观测的步数。截断速度后重新计算截距，
    使拟合线仍穿过历史均值；没有使用 GT、物理 DT 或预测伪观测。
    """
    positions = np.asarray(positions, dtype=np.float64)
    steps = np.asarray(steps, dtype=np.float64)
    if (positions.ndim != 2 or positions.shape[1] != 2 or len(positions) < 2 or
            steps.shape != (len(positions),) or not np.isfinite(positions).all() or
            not np.isfinite(steps).all() or not (np.diff(steps) > 0).all()):
        raise ValueError('at least two finite positions[N,2] and increasing steps[N] required')
    if isinstance(gap, bool) or not isinstance(gap, Integral) or gap < 0:
        raise ValueError('gap must be a non-negative integer')
    if not np.isfinite(max_step_m) or max_step_m <= 0:
        raise ValueError('max_step_m must be finite and positive')
    if not np.isfinite(prediction_decay) or not 0 <= prediction_decay <= 1:
        raise ValueError('prediction_decay must be in [0,1]')
    relative_steps = steps - steps[-1]
    centered_steps = relative_steps - relative_steps.mean()
    mean_position = positions.mean(axis=0)
    velocity = (centered_steps[:, None] * (positions - mean_position)).sum(axis=0)
    velocity /= float(np.dot(centered_steps, centered_steps))
    norm = float(np.linalg.norm(velocity))
    if norm > max_step_m:
        velocity *= max_step_m / norm
    intercept = mean_position - velocity * relative_steps.mean()
    multiplier = gap if prediction_decay == 1 else (1 - prediction_decay ** gap) / (1 - prediction_decay)
    return intercept + velocity * multiplier


class HistoryMotionTracker(LifecycleTracker):
    """每条记录独立实例；最近 HIST_LEN 个输入步骤内至少四次观测才启用趋势。"""

    def __init__(self, *args, min_motion_points=4, **kwargs):
        if (isinstance(min_motion_points, bool) or not isinstance(min_motion_points, Integral)
                or min_motion_points < 2):
            raise ValueError('min_motion_points must be an integer >= 2')
        self.min_motion_points = int(min_motion_points)
        super().__init__(*args, **kwargs)

    def reset(self):
        super().reset()
        self._step = 0
        self._motion_history = {}

    def _observe(self, track, position, score):
        super()._observe(track, position, score)
        history = self._motion_history.setdefault(track.track_id, deque(maxlen=self.hist_len))
        history.append((self._step + 1, position.copy()))

    def _association_predictions(self, indices):
        predictions = []
        current_step = self._step + 1
        for index in indices:
            track = self._tracks[index]
            history = [(t, p) for t, p in self._motion_history.get(track.track_id, ())
                       if t >= current_step - self.hist_len]
            if len(history) < self.min_motion_points:
                predictions.append(self._predict(track, track.missed + 1))
                continue
            steps, points = zip(*history)
            predictions.append(fit_history_prediction(
                points, steps, current_step - steps[-1], self.max_step_m, self.prediction_decay,
            ))
        return np.asarray(predictions).reshape(-1, 2)

    def update(self, positions, scores):
        # 原方法先验证候选；成功后才推进步号，非法输入不能消耗历史窗口。
        output = super().update(positions, scores)
        self._step += 1
        retained = {}
        for track in self._tracks:
            history = self._motion_history.get(track.track_id)
            if history is None:  # 新生首个观测不经过 _observe，需在这里记入。
                history = deque([(self._step, track.position.copy())], maxlen=self.hist_len)
            while history and history[0][0] < self._step - self.hist_len + 1:
                history.popleft()
            retained[track.track_id] = history
        self._motion_history = retained
        return output
