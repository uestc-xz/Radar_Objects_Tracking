"""Transformer-DA 对比实验的最小数据配置。"""

from pathlib import Path

from original.my_track import config as tracker_config


# True：读取已经由 M1 生成的冻结缓存；False：读取原始雷达信号。
# 当前 main.py 在 False 时只负责加载原始信号，尚未执行 M1 前向。
IS_COLD = True

TRAIN_ROOT = Path(tracker_config.TRAIN_ROOT)
VAL_ROOT = Path(tracker_config.VAL_ROOT)
CACHE_ROOT = Path(tracker_config.CACHE_ROOT) / "fast_dataset_multi_train"
VAL_CACHE_ROOT = Path(tracker_config.CACHE_ROOT) / "fast_dataset_multi_test"
WINDOW_SIZE = tracker_config.WINDOW_SIZE
STRIDE = tracker_config.STRIDE
SAMPLE_INDEX = 420

# Transformer-DA 网络参数。历史步数可直接修改，默认对应论文的 4 个历史时刻。
HISTORY_STEPS = 4
NUM_TRACKS = 3
NUM_CLASSES = 4  # 0=杂波，1~3=目标 ID
MEASUREMENT_DIM = 4  # [x, y, conf_bg, conf_fg]
HISTORY_STATE_DIM = 6  # [x, vx, y, vy, conf_bg, conf_fg]
MAX_MEASUREMENTS = 50
DELTA_T = tracker_config.DT
DA_D_MODEL = 512
DA_NUM_HEADS = 8
DA_ENCODER_LAYERS = 6
DA_DECODER_LAYERS = 6
DA_FEEDFORWARD_DIM = 512
DA_DROPOUT = 0.1
ASSIGNMENT_RADIUS_M = 1.0
DA_BATCH_SIZE = 8
DA_EPOCHS = 20
DA_LR = 1e-4
DA_MAX_WINDOWS = None  # 先用一个窗口验证训练闭环；设为 None 才遍历完整训练集。
DA_VAL_MAX_WINDOWS = None
DA_OUTPUT_DIR = Path(__file__).resolve().parent / "runs"
