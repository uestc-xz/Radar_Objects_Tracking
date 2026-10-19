# 跟踪关联网络子模块（M2~M8，本阶段范围）
# 说明：M1 是 original/123ren.py 中 Mymodule.forward 的返回值改动（返回 feat），
#       M2~M7 为新增模块；M8 航迹更新是 TrackBank.update 方法。对应 doc/跟踪任务书.md §3。
from .measurement import Measurements, generate_measurements   # M2 量测生成
from .track_state import TrackBank                              # M3 航迹状态库
from .feature_map import FeatureMapper                          # M4 特征映射分支
from .residual_gru import ResidualGRU                           # M5 几何残差 GRU
from .score_fusion import ScoreFusion                           # M6 融合打分
from .association import associate, build_assignment_targets, association_loss  # M7 关联决策与损失
