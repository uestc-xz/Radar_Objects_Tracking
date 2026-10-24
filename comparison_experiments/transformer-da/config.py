"""Transformer-DA 对比实验的最小数据配置。"""

from pathlib import Path

from original.my_track import config as tracker_config


# True：读取已经由 M1 生成的冻结缓存；False：读取原始雷达信号。
# 当前 main.py 在 False 时只负责加载原始信号，尚未执行 M1 前向。
IS_COLD = True

TRAIN_ROOT = Path(tracker_config.TRAIN_ROOT)
CACHE_ROOT = Path(tracker_config.CACHE_ROOT) / "fast_dataset_multi_train"
WINDOW_SIZE = tracker_config.WINDOW_SIZE
STRIDE = tracker_config.STRIDE
SAMPLE_INDEX = 420
