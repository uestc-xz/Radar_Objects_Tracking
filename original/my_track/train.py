"""联合航迹网络训练循环与整记录验证。

训练仅优化 JointTracker 并在窗口边界清空状态；验证按文件夹完整回放、仅在记录边界
重置状态。新 checkpoint 与日志使用专属文件名，不覆盖旧实验产物。
"""
import csv
import os
import time

import numpy as np
import torch
from tqdm.auto import tqdm

from original.my_track import config
from original.my_track.metrics import sequence_tracking_metrics
from original.my_track.train_tracker import (
    forward_batch,
    forward_batch_cold,
    run_frozen_m1,
)


def _to_device(batch, device):
    return tuple(value.to(device) if torch.is_tensor(value) else value for value in batch)


@torch.no_grad()
def evaluate_full_sequence_dataset(dataset, model, tracker, banks, device,
                                   is_cold=config.IS_COLD, max_sequences=None,
                                   show_progress=False, progress_desc="Validation"):
    """逐条完整回放验证集；只在记录边界 reset，逐帧读取信号或 M1 缓存。"""
    tracker.eval()
    loss_total = 0.0
    frame_count = 0
    parts = {name: 0.0 for name in config.TRACKER_LOSS_WEIGHTS}
    idsw_total = 0
    active_total = 0
    observed_total = 0
    predicted_total = 0
    miss_frames_total = 0
    retained_misses_total = 0.0
    abs_error_total = 0.0
    squared_error_total = 0.0
    known_targets_total = 0
    covered_targets_total = 0
    valid_target_frames_total = 0
    tracked_target_frames_total = 0
    untracked_target_frames_total = 0
    sequence_count = min(len(dataset), max_sequences) if max_sequences is not None else len(dataset)

    sequence_progress = tqdm(
        range(sequence_count),
        desc=progress_desc,
        unit="seq",
        dynamic_ncols=True,
        leave=False,
        disable=not show_progress,
    )
    for sequence_idx in sequence_progress:
        record = dataset[sequence_idx]
        bank = banks[0]
        bank.reset(device=device, dtype=torch.float32)
        positions, active, observed, labels = [], [], [], []
        for frame in record["frames"]:
            if is_cold:
                cache_path, label_path, _sample_id = frame
                with np.load(cache_path) as cached:
                    out = torch.from_numpy(cached["out"]).to(device).view(1, 1, 50, 4)
                    conf = torch.from_numpy(cached["conf"]).to(device).view(1, 1, 50, 2)
                    feat = torch.from_numpy(cached["feat"]).to(device).view(1, 1, 50, config.FEAT_DIM)
                label_array = np.load(label_path)
                gt = torch.from_numpy(label_array).mean(dim=0).reshape(1, 1, 3, 2).to(device)
                loss, n_frames, _, trajectory, batch_parts = forward_batch_cold(
                    out, conf, feat, gt, [bank], tracker,
                    return_details=True, reset_state=False,
                )
            else:
                signal_path, label_path, _sample_id = frame
                signal = torch.from_numpy(np.load(signal_path)).unsqueeze(0).to(device)
                out, conf, feat = run_frozen_m1(signal, model)
                label_array = np.load(label_path)
                gt = torch.from_numpy(label_array).mean(dim=0).reshape(1, 1, 3, 2).to(device)
                loss, n_frames, _, trajectory, batch_parts = forward_batch_cold(
                    out.view(1, 1, 50, 4),
                    conf.view(1, 1, 50, 2),
                    feat.view(1, 1, 50, config.FEAT_DIM),
                    gt, [bank], tracker,
                    return_details=True, reset_state=False,
                )
            loss_total += float(loss.item()) * n_frames
            frame_count += n_frames
            for name, value in batch_parts.items():
                parts[name] += float(value.item()) * n_frames
            positions.append(trajectory["final_pos"][0, 0].cpu())
            active.append(trajectory["active"][0, 0].cpu())
            observed.append(trajectory["observed"][0, 0].cpu())
            labels.append(gt[0, 0].cpu())

        if not positions:
            continue
        sequence_metrics = sequence_tracking_metrics(
            torch.stack(positions), torch.stack(active), torch.stack(observed),
            torch.stack(labels), warmup=config.HIST_LEN,
        )
        idsw_total += sequence_metrics["idsw"]
        active_total += sequence_metrics["active_count"]
        known_targets_total += sequence_metrics["known_target_count"]
        covered_targets_total += sequence_metrics["covered_target_count"]
        valid_target_frames_total += sequence_metrics["valid_target_frames"]
        tracked_target_frames_total += sequence_metrics["tracked_target_frames"]
        untracked_target_frames_total += sequence_metrics["untracked_target_frames"]
        observed_total += sequence_metrics["observed_count"]
        predicted_total += sequence_metrics["predicted_count"]
        miss_frames_total += sequence_metrics["miss_frames"]
        retained_misses_total += sequence_metrics["miss_retention"] * sequence_metrics["miss_frames"]
        abs_error_total += sequence_metrics["mae_m"] * sequence_metrics["active_count"]
        squared_error_total += sequence_metrics["rmse_m"] ** 2 * sequence_metrics["active_count"]
        sequence_progress.set_postfix(
            loss=f"{loss_total / max(frame_count, 1):.5f}",
            frames=frame_count,
        )

    result = {name: value / max(frame_count, 1) for name, value in parts.items()}
    result["loss"] = loss_total / max(frame_count, 1)
    result["idsw"] = idsw_total
    result["mae_m"] = abs_error_total / max(active_total, 1)
    result["rmse_m"] = (squared_error_total / max(active_total, 1)) ** 0.5
    result["miss_retention"] = retained_misses_total / max(miss_frames_total, 1)
    result["observed_count"] = observed_total
    result["predicted_count"] = predicted_total
    result["active_count"] = active_total
    result["target_coverage"] = covered_targets_total / max(known_targets_total, 1)
    result["target_frame_coverage"] = tracked_target_frames_total / max(valid_target_frames_total, 1)
    result["untracked_target_frames"] = untracked_target_frames_total
    result["sequences"] = sequence_count
    result["frames"] = frame_count
    return result


def _append_csv(path, row):
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def run_training(device, model, tracker, optimizer, train_loader, val_dataset,
                 banks, csv_path, show_progress=config.SHOW_PROGRESS):
    """训练 tracker，并保存独立 checkpoint；返回最小验证损失。"""
    best_val = float("inf")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    for epoch in range(1, config.EPOCHS + 1):
        tracker.train()
        train_total = 0.0
        train_count = 0
        started = time.time()
        train_progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{config.EPOCHS} [train]",
            unit="batch",
            dynamic_ncols=True,
            leave=False,
            disable=not show_progress,
        )
        for batch in train_progress:
            batch = _to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            if config.IS_COLD:
                out, conf, feat, gt = batch
                loss, n_frames, _ = forward_batch_cold(out, conf, feat, gt, banks, tracker)
            else:
                signals, gt = batch
                loss, n_frames, _ = forward_batch(signals, gt, model, banks, tracker)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite tracker training loss")
            loss.backward()
            optimizer.step()
            train_total += float(loss.detach().item()) * n_frames
            train_count += n_frames
            train_progress.set_postfix(
                loss=f"{train_total / max(train_count, 1):.5f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

        train_loss = train_total / max(train_count, 1)
        val_metrics = evaluate_full_sequence_dataset(
            val_dataset,
            model,
            tracker,
            banks,
            device,
            show_progress=show_progress,
            progress_desc=f"Epoch {epoch}/{config.EPOCHS} [val]",
        )
        _append_csv(csv_path, {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_association": val_metrics["association"],
            "val_position": val_metrics["position"],
            "val_existence": val_metrics["existence"],
            "val_confidence": val_metrics["confidence"],
            "val_idsw": val_metrics["idsw"],
            "val_mae_m": val_metrics["mae_m"],
            "val_rmse_m": val_metrics["rmse_m"],
            "val_miss_retention": val_metrics["miss_retention"],
            "val_observed_count": val_metrics["observed_count"],
            "val_predicted_count": val_metrics["predicted_count"],
            "val_target_coverage": val_metrics["target_coverage"],
            "val_target_frame_coverage": val_metrics["target_frame_coverage"],
            "val_untracked_target_frames": val_metrics["untracked_target_frames"],
            "elapsed_seconds": round(time.time() - started, 2),
        })

        checkpoint = {
            "epoch": epoch,
            "tracker_state_dict": tracker.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_metrics": val_metrics,
            "tracker_loss_weights": dict(config.TRACKER_LOSS_WEIGHTS),
            "config": {
                "hist_len": config.HIST_LEN,
                "num_tracks": config.NUM_TRACKS,
                "feat_dim": config.FEAT_DIM,
                "dt": config.DT,
                "confirm_threshold": config.CONFIRM_THRESHOLD,
                "assoc_label_radius_m": config.ASSOC_LABEL_RADIUS_M,
            },
        }
        torch.save(checkpoint, os.path.join(config.OUT_DIR, "joint_tracker_last.pth"))
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(checkpoint, os.path.join(config.OUT_DIR, "joint_tracker_best.pth"))
        print(
            f"epoch {epoch}/{config.EPOCHS}: train={train_loss:.5f} "
            f"val={val_metrics['loss']:.5f} assoc={val_metrics['association']:.5f} "
            f"pos={val_metrics['position']:.5f} IDSW={val_metrics['idsw']} "
            f"MAE={val_metrics['mae_m']:.3f}m "
            f"coverage={val_metrics['target_coverage']:.2f} ({time.time() - started:.1f}s)",
            flush=True,
        )
    return best_val
