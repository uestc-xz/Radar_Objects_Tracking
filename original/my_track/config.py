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
OUT_DIR      = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs')

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
LR             = 1e-3
CLUTTER_WEIGHT = 0.01   # 杂波类 CE 权重 λ（<1 = 压背景权重，让前景/航迹项主导损失；别设 0，否则不学拒绝杂波）
CLUTTER_WEIGHT_STEP = 0.03   # λ 每 epoch 的步进量：λ(epoch)=CLUTTER_WEIGHT+(epoch-1)*STEP；正=递增、负=递减、0=关闭
DT             = 0.24  # 帧间隔（秒）
CONF_THRESHOLD = 0.96  # 量测置信度阈值 τ：s_i≥τ 才作为候选目标参与匹配（前端定位标定值）
ADD_NOISE      = True
MAX_VAL_WIN    = None  # 每 epoch 校验窗口数上限；None=全量（慢时可设 200）

LOG_EVERY   = 20       # 每 N 步把当前平均损失落一个点到 loss 曲线（滑动平均窗口长度）
PLOT_EVERY  = 50       # 每 N 步刷新一次 loss 曲线
SEED        = 0
