"""独立常速度滤波关联实验；滤波位置不替代原始量测输出。

时间单位为输入步。噪声必须显式提供并在验证前由训练集冻结；
本模块不读取 GT、不修改默认生命周期管理器。
"""

from numbers import Integral

import numpy as np

from .lifecycle_tracker import LifecycleTracker


NOISE_KEYS = ('measurement_var', 'acceleration_var', 'initial_velocity_var')


def _variance(value):
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (2,) or not np.isfinite(value).all() or (value <= 0).any():
        raise ValueError('axis variances must be finite positive [2]')
    return value


def validate_motion_noise(noise):
    """校验冻结噪声参数，返回可序列化副本，不静默补默认参数。"""
    if not isinstance(noise, dict) or set(noise) != set(NOISE_KEYS):
        raise ValueError(f'motion noise must contain exactly {NOISE_KEYS}')
    return {name: _variance(noise[name]).tolist() for name in NOISE_KEYS}


def _state_arrays(mean, covariance):
    mean = np.asarray(mean, dtype=np.float64)
    covariance = np.asarray(covariance, dtype=np.float64)
    if (mean.shape != (4,) or covariance.shape != (4, 4) or
            not np.isfinite(mean).all() or not np.isfinite(covariance).all() or
            not np.allclose(covariance, covariance.T, rtol=0, atol=1e-10)):
        raise ValueError('finite mean[4] and symmetric covariance[4,4] required')
    return mean, covariance


def predict_motion(mean, covariance, acceleration_var, gap):
    """从最后真实观测状态预测 gap 步，返回副本，不改变传入状态。"""
    mean, covariance = _state_arrays(mean, covariance)
    acceleration_var = _variance(acceleration_var)
    if isinstance(gap, bool) or not isinstance(gap, Integral) or gap < 0:
        raise ValueError('gap must be a non-negative integer')
    transition = np.eye(4)
    transition[:2, 2:] = np.eye(2)
    injection = np.vstack((.5 * np.eye(2), np.eye(2)))
    process_noise = (injection * acceleration_var) @ injection.T
    mean, covariance = mean.copy(), covariance.copy()
    for _ in range(gap):
        mean = transition @ mean
        covariance = transition @ covariance @ transition.T + process_noise
    return mean, .5 * (covariance + covariance.T)


def correct_motion(mean, covariance, measurement, measurement_var):
    """量测更新使用 Joseph 形式，降低浮点误差破坏协方差的风险。"""
    mean, covariance = _state_arrays(mean, covariance)
    measurement = np.asarray(measurement, dtype=np.float64)
    if measurement.shape != (2,) or not np.isfinite(measurement).all():
        raise ValueError('finite measurement[2] required')
    noise = np.diag(_variance(measurement_var))
    innovation_covariance = covariance[:2, :2] + noise
    gain = np.linalg.solve(innovation_covariance, covariance[:, :2].T).T
    corrected = mean + gain @ (measurement - mean[:2])
    residual = np.eye(4)
    residual[:, :2] -= gain
    posterior = residual @ covariance @ residual.T + gain @ noise @ gain.T
    return corrected, .5 * (posterior + posterior.T)


class KalmanMotionTracker(LifecycleTracker):
    """仅替换关联预测；生命周期位置/速度与最终输出仍由父类管理。"""

    def __init__(self, *args, measurement_var, acceleration_var, initial_velocity_var, **kwargs):
        noise = validate_motion_noise(dict(measurement_var=measurement_var,
                                          acceleration_var=acceleration_var,
                                          initial_velocity_var=initial_velocity_var))
        self.measurement_var = np.asarray(noise['measurement_var'])
        self.acceleration_var = np.asarray(noise['acceleration_var'])
        self.initial_velocity_var = np.asarray(noise['initial_velocity_var'])
        super().__init__(*args, **kwargs)

    def reset(self):
        super().reset()
        self._motion_states = {}

    def _prediction(self, track):
        mean, covariance = self._motion_states[track.track_id]
        return predict_motion(mean, covariance, self.acceleration_var, track.missed + 1)

    def _association_predictions(self, indices):
        return np.asarray([self._prediction(self._tracks[i])[0][:2] for i in indices]).reshape(-1, 2)

    def _observe(self, track, position, score):
        mean, covariance = self._prediction(track)
        corrected = correct_motion(mean, covariance, position, self.measurement_var)
        super()._observe(track, position, score)
        self._motion_states[track.track_id] = corrected

    def update(self, candidate_positions, candidate_scores):
        # 父类先验证输入；非法帧不能推进滤波时钟。漏检不写入伪量测。
        output = super().update(candidate_positions, candidate_scores)
        retained = {}
        for track in self._tracks:
            state = self._motion_states.get(track.track_id)
            if state is None:  # 新生不经过 _observe；出生量测只记一次。
                state = (np.concatenate((track.position.copy(), np.zeros(2))),
                         np.diag(np.concatenate((self.measurement_var, self.initial_velocity_var))))
            retained[track.track_id] = state
        self._motion_states = retained
        return output
