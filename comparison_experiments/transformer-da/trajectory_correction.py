"""用 DA 速度预测和 M1 位置量测做简单的逐 ID 位置融合。"""

from __future__ import annotations

import numpy as np


def _valid_position(points: np.ndarray) -> np.ndarray:
    return np.isfinite(points).all(axis=-1) & (np.abs(points).sum(axis=-1) > 0)


def derive_measurement_velocity(
    positions: np.ndarray,
    delta_t: float,
) -> np.ndarray:
    """从逐帧关联的 M1 位置差分速度，帧间缺失时不跨缺口差分。"""
    velocities = np.full_like(positions, np.nan, dtype=np.float32)
    valid = _valid_position(positions)
    consecutive = valid[1:] & valid[:-1]
    velocities[1:][consecutive] = (
        positions[1:][consecutive] - positions[:-1][consecutive]
    ) / delta_t
    return velocities


def fuse_positions_with_velocity(
    measurement_positions: np.ndarray,
    predicted_velocities: np.ndarray,
    delta_t: float,
    motion_weight: float,
) -> np.ndarray:
    """将 M1 位置与上一帧位置加 DA 速度得到的运动预测做凸组合。

    每条 ID 在连续量测段的起点用 M1 位置初始化。后续帧按
    ``(1-w) * M1 + w * (上一修正位置 + DA速度 * dt)`` 融合。
    漏检处保留 NaN，重新出现时从 M1 位置重新初始化，避免跨漏检间隔外推。
    """
    weight = float(motion_weight)
    if not 0.0 <= weight <= 1.0:
        raise ValueError("motion_weight 必须在 [0, 1] 内")

    corrected = np.full_like(measurement_positions, np.nan, dtype=np.float32)
    valid_measurement = _valid_position(measurement_positions)
    valid_velocity = np.isfinite(predicted_velocities).all(axis=-1)

    for track_index in range(measurement_positions.shape[1]):
        for frame_index in range(measurement_positions.shape[0]):
            if not valid_measurement[frame_index, track_index]:
                continue

            measurement = measurement_positions[frame_index, track_index]
            if (
                frame_index > 0
                and np.isfinite(corrected[frame_index - 1, track_index]).all()
                and valid_velocity[frame_index, track_index]
            ):
                prediction = (
                    corrected[frame_index - 1, track_index]
                    + predicted_velocities[frame_index, track_index] * delta_t
                )
                corrected[frame_index, track_index] = (
                    (1.0 - weight) * measurement + weight * prediction
                )
            else:
                corrected[frame_index, track_index] = measurement

    return corrected


def smooth_positions(
    positions: np.ndarray,
    window_size: int = 5,
) -> np.ndarray:
    """对每条 ID 的连续有效点段做居中移动平均，不跨漏检间隔平滑。"""
    if window_size < 1:
        raise ValueError("window_size 必须为正整数")
    if window_size == 1:
        return positions.copy()

    smoothed = np.full_like(positions, np.nan, dtype=np.float32)
    valid = _valid_position(positions)
    kernel = np.ones(window_size, dtype=np.float32) / window_size
    left_pad = window_size // 2
    right_pad = window_size - 1 - left_pad

    for track_index in range(positions.shape[1]):
        frame_index = 0
        while frame_index < positions.shape[0]:
            if not valid[frame_index, track_index]:
                frame_index += 1
                continue
            segment_start = frame_index
            while (
                frame_index < positions.shape[0]
                and valid[frame_index, track_index]
            ):
                frame_index += 1
            segment_end = frame_index
            segment = positions[segment_start:segment_end, track_index]
            padded = np.pad(
                segment,
                ((left_pad, right_pad), (0, 0)),
                mode="edge",
            )
            for coordinate in range(2):
                smoothed[segment_start:segment_end, track_index, coordinate] = np.convolve(
                    padded[:, coordinate], kernel, mode="valid"
                )

    return smoothed
