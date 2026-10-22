"""my_track 对外模块。

M1 由 `original.123ren.Mymodule` 提供；当前训练路径使用 JointTracker。旧式
FeatureMapper/ResidualGRU/ScoreFusion 仍导出以兼容历史脚本，但不参与新的联合跟踪训练。
"""
from .measurement import Measurements, generate_measurements
from .track_state import TrackBank, batch_track_states
from .association import (
    associate,
    association_loss,
    build_assignment_targets,
    tracker_losses,
)
from .joint_tracker import JointTracker, TrackerOutput
from .metrics import sequence_tracking_metrics

# Legacy modules remain import-compatible for old experiments.
from .feature_map import FeatureMapper
from .residual_gru import ResidualGRU
from .score_fusion import ScoreFusion
