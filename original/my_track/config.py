"""跟踪关联网络训练超参配置
============================
作用：
    把原 train_tracker.py 顶部的超参集中到一个文件，供「前向函数(train_tracker.py)」
    与「训练入口(train.py)」共用。逻辑与数值不变，仅做文件拆分。

约定：
    WEIGHT_PATH / OUT_DIR 用本文件(__file__)所在目录推导，即 original/my_track/，
    与原 train_tracker.py 里用 __file__ 推导的结果一致（两者同目录）。
"""
import os

# ==================== 路径配置 ====================
TRAIN_ROOT   = r'E:\Jupyter\track\fast_dataset_multi_train'
VAL_ROOT     = r'E:\Jupyter\track\fast_dataset_multi_test'
# 权重与输出目录都放在脚本同目录（original/my_track/ 下）
WEIGHT_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'best_model_multilayerloss128.pth')
# 每次受控训练使用独立目录，避免不同配置的 CSV 行混合。
OUT_DIR      = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs', 'teacher_lr3e4_diagnostics')

# 冷启动：True=跳过 M1 前向，直接从 save_m1_cache.py 的缓存读 out/conf/feat（不加载 M1 权重）
IS_COLD     = True
CACHE_ROOT  = r'E:\Jupyter\NewMOTR数据集'   # M1 缓存根目录；其下与原始同名 fast_dataset_multi_train/_test

WINDOW_SIZE  = 50      # 短序列长度（帧）
STRIDE       = 25      # 滑窗步长（overlap = 20 - 10 = 10）
HIST_LEN     = 20       # 短历史长度 H
NUM_TRACKS   = 3       # 航迹槽位数 N（= 最多人数）
FEAT_DIM     = 128     # dec_output 特征维度
BATCH_SIZE   = 32      # 每步并行的序列数（batch）；越大 GPU 占用越高，按显存调

EPOCHS         = 50
# 首轮课程训练中自主关联的验证 CE 持续发散，先只降低优化步长做对照。
LR             = 3e-5
CLUTTER_WEIGHT = 0.01   # 杂波类 CE 权重 λ（<1 = 压背景权重，让前景/航迹项主导损失；别设 0，否则不学拒绝杂波）
CLUTTER_WEIGHT_STEP = 0.03   # λ 每 epoch 的步进量：λ(epoch)=CLUTTER_WEIGHT+(epoch-1)*STEP；正=递增、负=递减、0=关闭
DT             = 0.24  # 帧间隔（秒）
CONF_THRESHOLD = 0.96  # 旧关联脚本遗留值；联合跟踪器不以 M1 score 做硬门控
ADD_NOISE      = True
MAX_VAL_WIN    = None  # 每 epoch 校验窗口数上限；None=全量（慢时可设 200）
VAL_SEQUENCE_BATCH_SIZE = 8  # 验证跨独立序列并行；序列内部仍按时间递归

LOG_EVERY   = 20       # 每 N 步把当前平均损失落一个点到 loss 曲线（滑动平均窗口长度）
PLOT_EVERY  = 50       # 每 N 步刷新一次 loss 曲线
SHOW_PROGRESS = True   # 显示每个 epoch 的训练 batch 与验证序列进度
SEED        = 0

# ==================== 联合航迹网络 ====================
ASSOC_LABEL_RADIUS_M = 1.0       # 仅定义候选关联正标签，不作推理门控
CONFIRM_THRESHOLD = 0.5           # HIST_LEN 预热后的平均存在概率确认初值
MISS_CONFIDENCE_DECAY = 0.95      # 每帧漏检时的航迹置信度衰减
TRACK_QUALITY_SCALE_M = 0.5       # exp(-位置误差/尺度) 的软质量标签尺度
# 数据集的物理探测边界；目前仅用于验证期跑飞航迹诊断，不改变模型输出。
TRACK_X_RANGE_M = (-2.7, 2.7)
TRACK_Y_RANGE_M = (0.8, 6.2)
TRACKER_D_MODEL = 128
TRACKER_ENCODER_LAYERS = 2
TRACKER_ATTENTION_HEADS = 4
TRACKER_FFN_DIM = 256
M1_BATCH_SIZE = 64                 # 暖路径分块前向，限制显存占用
# 保留原权重作为验证口径，训练调权不影响历史验证指标的可比性。
TRACKER_LOSS_WEIGHTS = {
    "association": 2.0,
    "position": 1.0,
    "existence": 0.25,
    "confidence": 0.1,
}
# 仅训练提高坐标回归的相对权重；四项原始子损失的日志数值不加权。
TRACKER_TRAIN_LOSS_WEIGHTS = {**TRACKER_LOSS_WEIGHTS, "position": 5.0}

# 训练期教师关联课程：先用固定 GT 槽位稳定融合/纠偏，再退火到自主关联。
TEACHER_ASSOC_WARMUP_EPOCHS = 5
TEACHER_ASSOC_ANNEAL_EPOCHS = 10


def teacher_assignment_ratio(epoch, warmup_epochs=TEACHER_ASSOC_WARMUP_EPOCHS,
                             anneal_epochs=TEACHER_ASSOC_ANNEAL_EPOCHS):
    """返回第 `epoch` 轮使用 GT 融合标签的比例，取值闭区间 `[0,1]`。"""
    if not isinstance(epoch, int) or epoch < 1:
        raise ValueError("epoch must be a positive integer")
    if not isinstance(warmup_epochs, int) or warmup_epochs < 0:
        raise ValueError("warmup_epochs must be a non-negative integer")
    if not isinstance(anneal_epochs, int) or anneal_epochs < 0:
        raise ValueError("anneal_epochs must be a non-negative integer")
    if epoch <= warmup_epochs:
        return 1.0
    if anneal_epochs == 0 or epoch >= warmup_epochs + anneal_epochs:
        return 0.0
    return 1.0 - (epoch - warmup_epochs) / anneal_epochs
