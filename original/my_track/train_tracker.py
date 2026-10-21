r"""
跟踪关联网络前向函数 + 训练入口（自回归 + 冻结 M1）
==================================================
本文件放：
    - 前向逻辑  _forward_sequences / forward_batch / forward_batch_cold
    - 训练入口  main()（组装数据/模型/优化器后，调用 train.run_training 跑训练循环）

训练方式（对应方案 A 冻结 M1 + 决策 D-ds2 自回归）：
    - 冻结 M1 定位前端：加载预训练权重 best_model_multilayerloss128.pth，
      全部参数 requires_grad=False 且 eval 模式；只训练跟踪侧
      M4(FeatureMapper) / M5(ResidualGRU) / M6(ScoreFusion)。
    - 自回归（放手学习）：每帧用模型自己的 M7 决策 associate(S) 更新航迹库
      TrackBank，航迹历史/特征来自模型自己的更新值（与推理一致，非 teacher forcing）。
    - 每窗口 = 20 帧短序列：首帧初始化航迹，后续 19 帧逐帧关联；B 条序列按 batch 并行，损失累加后反向。

实时查看结果：
    - 控制台每个 epoch 用一行进度条实时刷新（\r 覆盖同一行 + flush，PyCharm 运行
      面板可直接看），展示 loss / 平均 loss / 分配分布 / 用时；下一个 epoch 另起一行。
    - 每 PLOT_EVERY 步刷新一次 loss 曲线 PNG（runs/loss_curve.png），
      用图片查看器打开即可看到实时刷新。
    - 完整日志写 runs/train_log.csv；每 epoch 保存 last.pth / best.pth 到 runs/。

运行（仓库根目录 E:\Jupyter\NewMOTR）：
    E:\Miniconda3\python.exe original/my_track/train_tracker.py
建议 GPU；M1 冻结后 B 条序列的帧合并成一次 M1 前向（B*20 帧一个 batch）。

文件分工：
    超参在 config.py；训练函数（run_training / evaluate / _plot_loss）在 train.py；
    本文件放前向函数与入口 main()。
"""
import importlib
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# original/123ren.py 以数字开头，不能用普通 import，用 importlib 绕过
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO_ROOT)
_123ren = importlib.import_module('original.123ren')
Mymodule = _123ren.Mymodule

from original.my_dataset import (                                  # noqa: E402
    load_track_sequence_dataset,
    load_track_sequence_cold_dataset,
)
from original.my_track import (                                   # noqa: E402
    TrackBank, FeatureMapper, ResidualGRU, ScoreFusion,
    associate, build_assignment_targets, association_loss,
)
from original.my_track.config import *                            # noqa: E402,F401,F403


# ---------------------------------------------------------------------------
# 批量序列的逐帧跟踪前向（时间维自回归串行，batch 维 B 并行；训练/校验共用）
# ---------------------------------------------------------------------------
def _forward_sequences(coor, conf, feat, gt, banks, mapper, res_gru, scorer,
                       clutter_weight=CLUTTER_WEIGHT, dt=DT, conf_threshold=CONF_THRESHOLD, epoch=0):
    """对 B 条序列做逐帧 M2→M8 前向，返回 (损失和, 损失帧数, 分配统计)。

    与原单序列版逻辑完全一致，仅把「B 条序列各自 batch=1 前向」合并成
    「每帧一次 batch=B 前向」：时间维 t 因自回归仍逐帧串行，batch 维 B
    （序列间相互独立）拼进 M4/M5/M6 的 batch 维并行，把 B×(T-1) 次小 kernel
    启动压缩为 (T-1) 次 batched 前向。

    Args:
        coor  : [B, T, 50, 2]   每条序列每帧量测中心（已由 M1 前向得到）
        conf  : [B, T, 50, 2]   每条序列每帧置信 logits
        feat  : [B, T, 50, 128] 每条序列每帧量测特征
        gt    : [B, T, 3, 2]    真值中心（空槽 (0,0)）
        banks : list[TrackBank] 长度 B，每条序列一个（M3+M8，无参数）
        mapper/res_gru/scorer : M4/M5/M6（可训练，跨序列共享）

    Returns:
        total_loss : 标量张量（各帧损失累加，未除帧数；训练时带梯度）
        n_frames   : int  参与损失的帧数（= B*(T-1)）
        assign_hist: list[4] 类 0/1/2/3 的累计量测数（合计 = B*(T-1)*50，首帧不计）
    """
    B, T = coor.shape[0], coor.shape[1]
    total_loss = 0.0
    n_frames = 0
    assign_hist = [0, 0, 0, 0]

    for t in range(T):
        m_pos = coor[:, t]                              # [B,50,2] 本帧量测中心
        m_score = F.softmax(conf[:, t], dim=-1)[..., 1] # [B,50] 置信度 s_i
        m_feat = feat[:, t]                             # [B,50,128] 量测特征 f_i

        # 首帧只初始化航迹库，不参与损失（无历史可关联）
        if t == 0:
            for b in range(B):
                banks[b].init_from_measurements(m_pos[b], m_score[b], m_feat[b])
            continue

        # 拼接 B 条序列的航迹侧状态，让 M4/M5/M6 一次 batched 前向（batch 维 = B）
        bank_feat = torch.stack([banks[b].feat for b in range(B)], dim=0)  # [B,3,128]
        bank_hist = torch.stack([banks[b].hist for b in range(B)], dim=0)  # [B,3,H,4]

        # M4 特征映射：量测侧 φ(f_i) 与航迹侧 φ(f_k) 共用同一映射
        phi_i = mapper(m_feat)                          # [B,50,128]
        phi_k = mapper(bank_feat)                       # [B,3,128]

        # M5 几何残差 GRU：量测位置 × 航迹短历史 → 几何一致性特征
        g_geom = res_gru(m_pos, bank_hist)              # [B,50,3,64]

        # M6 融合打分：g_geom + 特征相似度 → 4 类打分（0 列杂波）
        S = scorer(g_geom, phi_i, phi_k)                # [B,50,4]

        # M7 监督 + 损失：置信度阈值先筛 + 匈牙利把 GT 槽位伪 ID 映射成量测类标签，加权 CE
        labels = build_assignment_targets(m_pos, gt[:, t], m_score,
                                          threshold=conf_threshold)  # [B,50]
        # 【口径】association_loss 的 CE 带 class_weights 时 mean 归约 = 除以「各元素
        # 权值和」而非元素数；若对整批 [B,50] 做一次 mean，会按整批权值和归一化，与
        # 原逐条版「每条先按自身权值和归一化再求和」量纲不同。故损失仍逐条算再求和
        # （仅为 B 次极小的 CE kernel，并行收益来自 M4/M5/M6 的 batched 前向，无影响）。
        for b in range(B):
            total_loss = total_loss + association_loss(S[b:b + 1], labels[b:b + 1], clutter_weight, epoch=epoch)
        n_frames += B

        # M8 自回归更新：用模型自己的决策（argmax + 1对1）更新航迹，detach 与推理一致
        with torch.no_grad():
            assignments = associate(S)                  # [B,50] ∈ {0,1,2,3}

        # 逐条更新航迹状态（bank 是每条序列独立对象，非网络前向，开销可忽略）+ 诊断统计
        for b in range(B):
            banks[b].update(assignments[b], m_pos[b], m_feat[b], dt=dt)
            for c in range(4):
                assign_hist[c] += (assignments[b] == c).sum().item()

    return total_loss, n_frames, assign_hist


# ---------------------------------------------------------------------------
# 批处理前向：M1 批量化一次 + 逐条序列跟踪（训练/校验共用）
# ---------------------------------------------------------------------------
def forward_batch(signals, gt, model, banks, mapper, res_gru, scorer,
                  clutter_weight=CLUTTER_WEIGHT, dt=DT, conf_threshold=CONF_THRESHOLD, epoch=0):
    """对 B 条序列做 M1→M8 前向，返回 (平均损失, 损失帧数, 分配统计)。

    逻辑与原实现完全一致，仅两处变化：
        - M1 前端把 B 条序列的帧拼成 [B*T, 8, 89, 10] 一次前向（批量化提速）；
        - 跟踪部分把 B 条序列拼进 batch 维，每帧一次 batched 前向（见 _forward_sequences）。

    Args:
        signals : [B, T, 8, 89, 10] complex64   B 条序列、各 T 帧信号
        gt      : [B, T, 3, 2]        float32   真值中心（空槽 (0,0)）
        model   : Mymodule（M1，冻结）
        banks   : list[TrackBank]，长度 ≥ B，每条序列一个
        mapper/res_gru/scorer : M4/M5/M6（跨序列共享）

    Returns:
        loss       : 标量张量（B*(T-1) 帧平均，训练时带梯度）
        n_frames   : int  参与损失的总帧数（= B*(T-1)）
        assign_hist: list[4] 类 0/1/2/3 的累计量测数（合计 = B*(T-1)*50，首帧不计）
    """
    B, T = signals.shape[0], signals.shape[1]
    # M1 前端冻结：B 条序列的所有帧一次性前向，再按 (B,T) 还原
    with torch.no_grad():
        out, conf, feat = model(signals.reshape(B * T, 8, 89, 10))  # [B*T,50,4]/[B*T,50,2]/[B*T,50,128]
    coor = out[:, :, :2].reshape(B, T, 50, 2)        # [B,T,50,2] 中心坐标
    conf = conf.reshape(B, T, 50, 2)                 # [B,T,50,2]
    feat = feat.reshape(B, T, 50, 128)               # [B,T,50,128]

    total_loss, n_frames, assign_hist = _forward_sequences(
        coor, conf, feat, gt, banks, mapper, res_gru, scorer,
        clutter_weight=clutter_weight, dt=dt, conf_threshold=conf_threshold, epoch=epoch)

    return total_loss / n_frames, n_frames, assign_hist


# ---------------------------------------------------------------------------
# 批处理前向（冷启动）：直接用缓存 M1 输出，跳过 M1 前向（训练/校验共用）
# ---------------------------------------------------------------------------
def forward_batch_cold(out, conf, feat, gt, banks, mapper, res_gru, scorer,
                       clutter_weight=CLUTTER_WEIGHT, dt=DT, conf_threshold=CONF_THRESHOLD, epoch=0):
    """冷启动：对 B 条序列的「缓存 M1 输出」做 M2→M8 前向，返回 (平均损失, 损失帧数, 分配统计)。

    与 forward_batch 的区别：省掉 M1 前向（out/conf/feat 已由缓存加载好），
    跟踪部分复用 _forward_sequences（batch 维并行），逻辑完全一致。

    Args:
        out  : [B, T, 50, 4]   缓存 M1 最终输出（前 2 列 = 中心坐标，后 2 列冗余未用）
        conf : [B, T, 50, 2]   缓存 M1 置信 logits
        feat : [B, T, 50, 128] 缓存 M1 特征
        gt   : [B, T, 3, 2]    真值中心（空槽 (0,0)）
        banks/mapper/res_gru/scorer : 同 forward_batch

    Returns:
        loss       : 标量张量（B*(T-1) 帧平均，训练时带梯度）
        n_frames   : int  参与损失的总帧数（= B*(T-1)）
        assign_hist: list[4] 类 0/1/2/3 的累计量测数
    """
    B, T = out.shape[0], out.shape[1]
    coor = out[..., :2]                               # [B,T,50,2] 中心坐标

    total_loss, n_frames, assign_hist = _forward_sequences(
        coor, conf, feat, gt, banks, mapper, res_gru, scorer,
        clutter_weight=clutter_weight, dt=dt, conf_threshold=conf_threshold, epoch=epoch)

    return total_loss / n_frames, n_frames, assign_hist


# ---------------------------------------------------------------------------
# 训练入口：组装数据/模型/优化器，再调 train.run_training 跑训练循环
# ---------------------------------------------------------------------------
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    print(f'设备: {device}', flush=True)

    # ---- 1) M1 定位前端：加载预训练权重并冻结（冷启动时跳过，直接用缓存输出）----
    if IS_COLD:
        model = None
        print('冷启动：跳过 M1 权重加载，直接用缓存 out/conf/feat', flush=True)
    else:
        model = Mymodule(input_channels=8, freq_bins=89, time_steps=10).to(device)
        ckpt = torch.load(WEIGHT_PATH, map_location=device)
        sd = ckpt['model_state_dict']
        # 剥掉 thop profile 注入的伪键（total_ops / total_params）
        for k in [k for k in sd if k.endswith('total_ops') or k.endswith('total_params')]:
            sd.pop(k)
        missing, unexpected = model.load_state_dict(sd, strict=True)
        assert not missing and not unexpected, (missing, unexpected)
        print(f'加载预训练 M1: epoch={ckpt.get("epoch")} 定位val_loss={ckpt.get("best_val_loss"):.4f}',
              flush=True)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        print('M1 已冻结（requires_grad=False + eval 模式）', flush=True)

    # ---- 2) 跟踪模块 M4/M5/M6（可训练）----
    mapper = FeatureMapper(in_dim=FEAT_DIM, out_dim=FEAT_DIM).to(device)
    res_gru = ResidualGRU(state_dim=4, hidden_dim=64).to(device)
    scorer = ScoreFusion(geom_dim=64).to(device)
    track_params = (list(mapper.parameters()) + list(res_gru.parameters())
                    + list(scorer.parameters()))
    n_param = sum(p.numel() for p in track_params)
    print(f'可训练参数（M4~M6）: {n_param}', flush=True)
    optimizer = torch.optim.Adam(track_params, lr=LR)

    # ---- 3) 数据 ----
    if IS_COLD:
        train_ds = load_track_sequence_cold_dataset(
            TRAIN_ROOT, os.path.join(CACHE_ROOT, 'fast_dataset_multi_train'), WINDOW_SIZE, STRIDE)
        val_ds = load_track_sequence_cold_dataset(
            VAL_ROOT, os.path.join(CACHE_ROOT, 'fast_dataset_multi_test'), WINDOW_SIZE, WINDOW_SIZE-1)
    else:
        train_ds = load_track_sequence_dataset(TRAIN_ROOT, WINDOW_SIZE, STRIDE, add_noise=ADD_NOISE)
        val_ds = load_track_sequence_dataset(VAL_ROOT, WINDOW_SIZE, WINDOW_SIZE - 1, add_noise=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              drop_last=True, num_workers=8)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            drop_last=False, num_workers=8)
    # 航迹库：每条序列一个，共 BATCH_SIZE 个；每窗口首帧 init_from_measurements 会全量重写字段
    banks = [TrackBank(num_tracks=NUM_TRACKS, hist_len=HIST_LEN, feat_dim=FEAT_DIM)
             for _ in range(BATCH_SIZE)]

    # ---- 4) 日志/输出 ----
    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, 'train_log.csv')
    plot_path = os.path.join(OUT_DIR, 'loss_curve.png')
    with open(csv_path, 'w') as f:
        f.write('step,epoch,val_loss\n')

    # 训练函数在 train.py；这里在函数体内才导入，是为了打破「train.py 顶部
    # 又 from ...train_tracker import forward_batch」造成的循环导入（本文件作脚本运行时是 __main__）。
    from original.my_track.train import run_training
    best_val = run_training(device, model, mapper, res_gru, scorer, optimizer,
                            train_loader, val_loader, banks, csv_path, plot_path)

    print(f'训练完成。最优 val_loss={best_val:.4f}，输出目录 {OUT_DIR}', flush=True)


if __name__ == '__main__':
    main()
