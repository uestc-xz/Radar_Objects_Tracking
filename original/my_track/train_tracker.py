"""联合跟踪器训练前向与入口。

训练按窗口时间顺序迭代，每窗开始重置 TrackBank；冷路径消费已缓存的 M1 输出，暖路径
以 no-grad 分块调用冻结 M1。两条路径都通过同一 Measurements -> JointTracker 接口。
"""
import importlib
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from original.my_dataset import load_track_sequence_dataset, load_track_sequence_cold_dataset
from original.my_track.association import (
    association_diagnostics,
    build_assignment_targets,
    tracker_losses,
)
from original.my_track.config import (
    ADD_NOISE,
    BATCH_SIZE,
    CACHE_ROOT,
    DT,
    EPOCHS,
    FEAT_DIM,
    HIST_LEN,
    IS_COLD,
    LR,
    M1_BATCH_SIZE,
    NUM_TRACKS,
    OUT_DIR,
    SEED,
    TRACKER_LOSS_WEIGHTS,
    TRAIN_ROOT,
    VAL_ROOT,
    WEIGHT_PATH,
    WINDOW_SIZE,
    STRIDE,
)
from original.my_track.joint_tracker import JointTracker
from original.my_track.measurement import generate_measurements
from original.my_track.track_state import TrackBank, batch_track_states


def canonicalize_ground_truth_tracks(gt):
    """以每条序列中各目标首次出现的位置排序槽位，并在整个序列固定映射。

    接受 `[B,T,3,2]` 或 `[T,3,2]` 米制 GT；空槽排在有效目标之后。
    该排序只规范化序列局部的标签编号，不代表跨记录的全局身份。
    """
    unbatched = gt.ndim == 3
    if unbatched:
        gt = gt.unsqueeze(0)
    if gt.ndim != 4 or gt.shape[2:] != (NUM_TRACKS, 2):
        raise ValueError("gt must have shape [B,T,3,2] or [T,3,2]")
    if gt.shape[1] == 0:
        raise ValueError("gt sequence must contain at least one frame")

    canonical = []
    track_indices = torch.arange(NUM_TRACKS, device=gt.device)
    for sequence in gt:
        valid = sequence.abs().sum(dim=-1) > 0
        first_frame = valid.to(torch.int64).argmax(dim=0)
        first_position = sequence[first_frame, track_indices]
        present = valid.any(dim=0)
        x_key = torch.where(present, first_position[:, 0], torch.full_like(first_position[:, 0], float("inf")))
        y_key = torch.where(present, first_position[:, 1], torch.full_like(first_position[:, 1], float("inf")))

        # Stable two-key sort: x first, then y; original slot order breaks exact ties.
        order = torch.argsort(y_key, stable=True)
        order = order[torch.argsort(x_key[order], stable=True)]
        canonical.append(sequence.index_select(1, order))
    result = torch.stack(canonical, dim=0)
    return result[0] if unbatched else result


def _forward_sequences(coor, conf, feat, gt, banks, tracker, dt=DT,
                       return_details=False, reset_state=True, teacher_assignment_ratio=0.0,
                       loss_weights=None, framewise_loss=False):
    """对 `[B,T]` 缓存/M1 输出做逐帧联合跟踪并计算所有帧损失。

    每个滑窗开始时清空其状态，所有 `B*T` 帧均参与损失。状态库只接收 detach 后的
    模型输出，因此窗口内不跨帧保留 autograd 图；返回轨迹张量可直接用于指标。
    loss_weights 只控制总损失加权，默认沿用验证权重；loss_parts 返回当前路径的未加权值。
    训练使用单路径教师课程，按 teacher_assignment_ratio 从 GT 融合过渡到自主融合。
    framewise_loss=True 仅用于并行验证，先对每个样本计算损失再等权平均。
    """
    if coor.ndim != 4 or coor.shape[2:] != (50, 2):
        raise ValueError("coor must have shape [B,T,50,2]")
    batch, frames = coor.shape[:2]
    if conf.shape != (batch, frames, 50, 2):
        raise ValueError("conf must have shape [B,T,50,2]")
    if feat.shape != (batch, frames, 50, FEAT_DIM):
        raise ValueError(f"feat must have shape [B,T,50,{FEAT_DIM}]")
    if gt.shape != (batch, frames, NUM_TRACKS, 2):
        raise ValueError("gt must have shape [B,T,3,2]")
    # GT slot numbers are only sequence-local; use one stable geometric ordering per window.
    gt = canonicalize_ground_truth_tracks(gt)
    if len(banks) < batch:
        raise ValueError("one TrackBank is required for each batch item")
    if frames < 1:
        raise ValueError("a training window must contain at least one frame")
    if reset_state:
        for bank in banks[:batch]:
            bank.reset(device=coor.device, dtype=coor.dtype)

    weights = TRACKER_LOSS_WEIGHTS if loss_weights is None else loss_weights
    loss_sums = {name: [] for name in TRACKER_LOSS_WEIGHTS}
    frame_losses = []
    assign_hist = [0, 0, 0, 0]
    pos_steps, observed_steps, active_steps = [], [], []
    association_stat_sums = None

    for time_idx in range(frames):
        measurement = generate_measurements(coor[:, time_idx], conf[:, time_idx], feat[:, time_idx])
        state = batch_track_states(banks[:batch], device=coor.device)
        labels = build_assignment_targets(measurement.pos, gt[:, time_idx])
        output = tracker(
            measurement, state, dt=dt,
            teacher_assignment_labels=labels if teacher_assignment_ratio > 0 else None,
            teacher_assignment_ratio=teacher_assignment_ratio,
        )
        losses = tracker_losses(
            output.association_logits, labels, output.final_pos,
            output.existence_prob, output.track_confidence, gt[:, time_idx],
            framewise_position=framewise_loss,
        )
        if return_details:
            diagnostics = association_diagnostics(output.association_logits, labels)
            if association_stat_sums is None:
                association_stat_sums = {name: value.clone() for name, value in diagnostics.items()}
            else:
                for name, value in diagnostics.items():
                    association_stat_sums[name] = association_stat_sums[name] + value
        for name, value in losses.items():
            loss_sums[name].append(value)
        frame_losses.append(sum(weights[name] * losses[name] for name in losses))
        for class_id in range(4):
            assign_hist[class_id] += int((output.assignments == class_id).sum().item())

        for batch_idx, bank in enumerate(banks[:batch]):
            bank.update(
                output.final_pos[batch_idx],
                output.fused_feat[batch_idx],
                output.observed[batch_idx],
                output.existence_prob[batch_idx],
                output.track_confidence[batch_idx],
                dt=dt,
            )
        pos_steps.append(output.final_pos)
        observed_steps.append(output.observed)
        active_steps.append(torch.stack([bank.confirmed for bank in banks[:batch]], dim=0))

    mean_loss = torch.stack(frame_losses).mean()
    trajectory = {
        "final_pos": torch.stack(pos_steps, dim=1),
        "observed": torch.stack(observed_steps, dim=1),
        "active": torch.stack(active_steps, dim=1),
    }
    if return_details:
        trajectory["association_stats"] = association_stat_sums
    loss_parts = {name: torch.stack(values).mean() for name, values in loss_sums.items()}
    result = (mean_loss, batch * frames, assign_hist)
    return result + (trajectory, loss_parts) if return_details else result


def forward_batch_cold(out, conf, feat, gt, banks, tracker, dt=DT,
                       return_details=False, reset_state=True, teacher_assignment_ratio=0.0,
                       loss_weights=None, framewise_loss=False):
    """冷路径：直接消费缓存 `[B,T,50,{4,2,128}]`，不构造或调用 M1。"""
    if out.ndim != 4 or out.shape[-2:] != (50, 4):
        raise ValueError("out must have shape [B,T,50,4]")
    return _forward_sequences(
        out[..., :2], conf, feat, gt, banks, tracker, dt,
        return_details=return_details, reset_state=reset_state,
        teacher_assignment_ratio=teacher_assignment_ratio,
        loss_weights=loss_weights,
        framewise_loss=framewise_loss,
    )


def run_frozen_m1(signals, model, m1_batch_size=M1_BATCH_SIZE):
    """冻结模型并分块返回 M1 原始 out/conf/feat，不建立 M1 计算图。"""
    if signals.ndim < 2:
        raise ValueError("signals must have a batch dimension")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    outs, confs, feats = [], [], []
    with torch.no_grad():
        for start in range(0, signals.shape[0], m1_batch_size):
            out, conf, feat = model(signals[start:start + m1_batch_size])
            outs.append(out)
            confs.append(conf)
            feats.append(feat)
    return torch.cat(outs, dim=0), torch.cat(confs, dim=0), torch.cat(feats, dim=0)


def forward_batch(signals, gt, model, banks, tracker, dt=DT, m1_batch_size=M1_BATCH_SIZE,
                  return_details=False, reset_state=True, teacher_assignment_ratio=0.0,
                  loss_weights=None, framewise_loss=False):
    """暖路径：分块运行冻结 M1，再调用与冷路径相同的逐帧跟踪核心。"""
    if signals.ndim != 5:
        raise ValueError("signals must have shape [B,T,8,89,10]")
    batch, frames = signals.shape[:2]
    flat = signals.reshape(batch * frames, *signals.shape[2:])
    out, conf, feat = run_frozen_m1(flat, model, m1_batch_size)
    out = out.reshape(batch, frames, 50, 4)
    conf = conf.reshape(batch, frames, 50, 2)
    feat = feat.reshape(batch, frames, 50, FEAT_DIM)
    return _forward_sequences(
        out[..., :2], conf, feat, gt, banks, tracker, dt,
        return_details=return_details, reset_state=reset_state,
        teacher_assignment_ratio=teacher_assignment_ratio,
        loss_weights=loss_weights,
        framewise_loss=framewise_loss,
    )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    print(f"设备: {device}", flush=True)

    model = None
    if not IS_COLD:
        Mymodule = importlib.import_module("original.123ren").Mymodule
        model = Mymodule(input_channels=8, freq_bins=89, time_steps=10).to(device)
        checkpoint = torch.load(WEIGHT_PATH, map_location=device)
        state_dict = checkpoint["model_state_dict"]
        state_dict = {key: value for key, value in state_dict.items()
                      if not key.endswith(("total_ops", "total_params"))}
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    tracker = JointTracker().to(device)
    optimizer = torch.optim.Adam(tracker.parameters(), lr=LR)
    if IS_COLD:
        train_ds = load_track_sequence_cold_dataset(
            TRAIN_ROOT, os.path.join(CACHE_ROOT, "fast_dataset_multi_train"), WINDOW_SIZE, STRIDE)
        val_ds = load_track_sequence_cold_dataset(
            VAL_ROOT, os.path.join(CACHE_ROOT, "fast_dataset_multi_test"),
            WINDOW_SIZE, WINDOW_SIZE - 1, full_sequence=True)
    else:
        train_ds = load_track_sequence_dataset(TRAIN_ROOT, WINDOW_SIZE, STRIDE, add_noise=ADD_NOISE)
        val_ds = load_track_sequence_dataset(
            VAL_ROOT, WINDOW_SIZE, WINDOW_SIZE - 1, add_noise=False, full_sequence=True)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              drop_last=True, num_workers=8)
    banks = [TrackBank() for _ in range(BATCH_SIZE)]

    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, "tracker_train_log.csv")
    from original.my_track.train import run_training
    run_training(device, model, tracker, optimizer, train_loader, val_ds, banks, csv_path)


if __name__ == "__main__":
    main()
