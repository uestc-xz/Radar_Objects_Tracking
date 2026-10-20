r"""
跟踪关联网络训练函数（训练循环 / 校验 / 绘图）
================================================
本文件放「训练相关函数」：
    - run_training   训练主循环：跑 EPOCHS 个 epoch，含校验/日志/checkpoint，返回最优 val_loss
    - evaluate       无梯度校验一遍 val_loader，返回平均损失
    - _plot_loss     训练/校验损失曲线 PNG

文件分工：
    超参在 config.py；前向函数（_forward_sequence / forward_batch / forward_batch_cold）
    与训练入口 main() 在 train_tracker.py。本文件只被 train_tracker.main 调用，不含入口。
"""
import os
import time

import torch

import matplotlib
matplotlib.use('Agg')  # 无界面后端，只 savefig 不弹窗，方便实时刷新
import matplotlib.pyplot as plt

from original.my_track.config import *                            # noqa: E402,F401,F403
from original.my_track.train_tracker import forward_batch, forward_batch_cold  # noqa: E402


# ---------------------------------------------------------------------------
# 校验（无梯度）
# ---------------------------------------------------------------------------
def evaluate(val_loader, model, mapper, res_gru, scorer, banks, is_cold=IS_COLD, max_win=MAX_VAL_WIN):
    """校验：无梯度跑一遍 val_loader，返回平均损失。

    is_cold=True 时 model=None，val_loader 产出 4 元组 (out,conf,feat,gt)，走冷启动前向；
    否则产出 2 元组 (signals,gt)，走 M1 前向。
    """
    mapper.eval(); res_gru.eval(); scorer.eval()
    device = next(mapper.parameters()).device    # 从跟踪模块推断设备（冷启动时 model=None，不能取 model 参数）
    total, cnt, seen = 0.0, 0, 0
    with torch.no_grad():
        for batch in val_loader:
            if max_win is not None and seen >= max_win:   # max_win 按「窗口数」计
                break
            if is_cold:
                out, conf, feat, gt = batch              # [B,T,50,4]/[B,T,50,2]/[B,T,50,128]/[B,T,3,2]
                out = out.to(device)
                conf = conf.to(device)
                feat = feat.to(device)
                gt = gt.to(device)
                loss, n, _ = forward_batch_cold(out, conf, feat, gt, banks, mapper, res_gru, scorer)
                seen += out.shape[0]
            else:
                signals, gt = batch                       # [B,T,8,89,10]/[B,T,3,2]
                signals = signals.to(device)
                gt = gt.to(device)
                loss, n, _ = forward_batch(signals, gt, model, banks, mapper, res_gru, scorer)
                seen += signals.shape[0]
            total += loss.item() * n
            cnt += n
    mapper.train(); res_gru.train(); scorer.train()
    return total / cnt if cnt else 0.0


# ---------------------------------------------------------------------------
# loss 曲线 PNG
# ---------------------------------------------------------------------------
def _plot_loss(train_hist, val_hist, path):
    """把训练/校验损失画成曲线，保存 PNG（供外部实时刷新查看）。"""
    plt.figure(figsize=(8, 5))
    if train_hist:
        xs, ys = zip(*train_hist)
        plt.plot(xs, ys, label='train loss', color='tab:blue')
    if val_hist:
        vxs, vys = zip(*val_hist)
        plt.plot(vxs, vys, label='val loss', marker='o', markersize=3, color='tab:red')
    plt.xlabel('step')
    plt.ylabel('loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()


# ---------------------------------------------------------------------------
# 训练主循环
# ---------------------------------------------------------------------------
def run_training(device, model, mapper, res_gru, scorer, optimizer,
                 train_loader, val_loader, banks, csv_path, plot_path):
    """训练主循环：跑 EPOCHS 个 epoch，每 epoch 校验/记日志/存 checkpoint，返回最优 val_loss。

    逻辑与原 train_tracker.py 的 main() 训练循环完全一致，仅从 main 抽离成函数。

    Args:
        device      : torch.device
        model       : Mymodule（M1，冻结；冷启动时为 None）
        mapper/res_gru/scorer : M4/M5/M6（可训练）
        optimizer   : Adam
        train_loader/val_loader : 数据加载器（冷启动产出 4 元组，否则 2 元组）
        banks       : list[TrackBank]
        csv_path    : train_log.csv 路径（header 已由 main 写好）
        plot_path   : loss_curve.png 路径

    Returns:
        best_val : float  历史最优 val_loss
    """
    global_step = 0
    best_val = float('inf')
    train_hist = []     # (step, 平均 train loss)
    val_hist = []       # (step, val loss)

    for epoch in range(1, EPOCHS + 1):
        mapper.train(); res_gru.train(); scorer.train()
        run_loss, run_n = 0.0, 0
        t0 = time.time()

        total_batches = len(train_loader)             # 进度条分母（本 epoch 的 batch 数）
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            if IS_COLD:
                out, conf, feat, gt = batch           # [B,T,50,4]/[B,T,50,2]/[B,T,50,128]/[B,T,3,2]
                out = out.to(device)
                conf = conf.to(device)
                feat = feat.to(device)
                gt = gt.to(device)
                loss, n, assign_hist = forward_batch_cold(
                    out, conf, feat, gt, banks, mapper, res_gru, scorer)
            else:
                signals, gt = batch                  # [B,T,8,89,10]/[B,T,3,2]
                signals = signals.to(device)
                gt = gt.to(device)
                loss, n, assign_hist = forward_batch(
                    signals, gt, model, banks, mapper, res_gru, scorer)
            loss.backward()
            optimizer.step()

            global_step += 1
            run_loss += loss.item() * n
            run_n += n

            # 当前滑动窗口的平均损失（自上次重置起），展示与落点共用同一值
            avg = run_loss / run_n

            # 每 LOG_EVERY 步落一个点到 loss 曲线，并重置滑动平均
            if global_step % LOG_EVERY == 0:
                train_hist.append((global_step, avg))
                run_loss, run_n = 0.0, 0

            # 实时刷新 loss 曲线
            if global_step % PLOT_EVERY == 0:
                _plot_loss(train_hist, val_hist, plot_path)

            # 进度条：\r 覆盖本 epoch 的同一行（end='' 不换行，flush 实时可见）
            frac = (i + 1) / total_batches
            n_bar = 30
            filled = int(round(n_bar * frac))
            bar = '#' * filled + '-' * (n_bar - filled)
            print(f'\r[epoch {epoch:2d}/{EPOCHS}] |{bar}| {i + 1:4d}/{total_batches}  '
                  f'loss={loss.item():.4f}  avg={avg:.4f}  分配={assign_hist}  '
                  f'{time.time() - t0:.0f}s',
                  end='', flush=True)

        # 本 epoch 进度条行结束，换行后再打印校验结果（下一个 epoch 另起一行）
        print()

        # ---- 每 epoch 校验 ----
        val_loss = evaluate(val_loader, model, mapper, res_gru, scorer, banks)
        val_hist.append((global_step, val_loss))
        print(f'=== epoch {epoch}/{EPOCHS} 结束: val_loss={val_loss:.4f}  '
              f'best={best_val:.4f} ===', flush=True)
        with open(csv_path, 'a') as f:
            f.write(f'{global_step},{epoch},{val_loss:.6f}\n')

        # ---- checkpoint ----
        ckpt = {
            'epoch': epoch,
            'mapper': mapper.state_dict(),
            'res_gru': res_gru.state_dict(),
            'scorer': scorer.state_dict(),
            'optimizer': optimizer.state_dict(),
            'val_loss': val_loss,
        }
        torch.save(ckpt, os.path.join(OUT_DIR, 'last.pth'))
        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, os.path.join(OUT_DIR, 'best.pth'))
            print(f'    >> 保存新最优 checkpoint (val_loss={best_val:.4f})', flush=True)

        _plot_loss(train_hist, val_hist, plot_path)

    return best_val
