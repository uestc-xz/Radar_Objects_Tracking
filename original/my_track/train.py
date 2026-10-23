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


def gradient_l2_norm(parameters):
    """返回当前梯度的全局 L2 范数；无梯度参数不计入。"""
    squared_sum = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            squared_sum += float(parameter.grad.detach().square().sum().item())
    return squared_sum ** 0.5


def _safe_ratio(numerator, denominator):
    return numerator / denominator if denominator else 0.0


@torch.no_grad()
def evaluate_full_sequence_dataset(dataset, model, tracker, banks, device,
                                   is_cold=config.IS_COLD, max_sequences=None,
                                   show_progress=False, progress_desc="Validation"):
    """按时间步批量回放独立验证序列，序列内部仍保持递归状态。"""
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
    association_correct_total = 0
    association_count_total = 0
    association_true_probability_sum = 0.0
    association_foreground_correct_total = 0
    association_foreground_count_total = 0
    association_foreground_true_probability_sum = 0.0
    active_position_count = 0
    active_position_out_of_bounds_count = 0
    sequence_count = min(len(dataset), max_sequences) if max_sequences is not None else len(dataset)

    if not isinstance(config.VAL_SEQUENCE_BATCH_SIZE, int) or config.VAL_SEQUENCE_BATCH_SIZE < 1:
        raise ValueError("VAL_SEQUENCE_BATCH_SIZE must be a positive integer")
    if sequence_count and not banks:
        raise ValueError("validation requires at least one TrackBank")
    sequence_batch_size = max(1, min(
        int(config.VAL_SEQUENCE_BATCH_SIZE), len(banks), sequence_count,
    ))
    sequence_progress = tqdm(
        total=sequence_count,
        desc=progress_desc,
        unit="seq",
        dynamic_ncols=True,
        leave=False,
        disable=not show_progress,
    )
    for batch_start in range(0, sequence_count, sequence_batch_size):
        records = [dataset[idx] for idx in range(
            batch_start, min(batch_start + sequence_batch_size, sequence_count)
        )]
        batch_banks = banks[:len(records)]
        for bank in batch_banks:
            bank.reset(device=device, dtype=torch.float32)
        positions = [[] for _ in records]
        active = [[] for _ in records]
        observed = [[] for _ in records]
        labels = [[] for _ in records]
        max_frames = max(len(record["frames"]) for record in records)

        for time_idx in range(max_frames):
            active_indices = [
                idx for idx, record in enumerate(records)
                if time_idx < len(record["frames"])
            ]
            frames = [records[idx]["frames"][time_idx] for idx in active_indices]
            active_banks = [batch_banks[idx] for idx in active_indices]
            if is_cold:
                outs, confs, feats, gts = [], [], [], []
                for cache_path, label_path, _sample_id in frames:
                    with np.load(cache_path) as cached:
                        outs.append(torch.from_numpy(cached["out"].copy()))
                        confs.append(torch.from_numpy(cached["conf"].copy()))
                        feats.append(torch.from_numpy(cached["feat"].copy()))
                    label_array = np.load(label_path)
                    gts.append(torch.from_numpy(label_array).mean(dim=0).reshape(3, 2))
                out = torch.stack(outs, dim=0).to(device).unsqueeze(1)
                conf = torch.stack(confs, dim=0).to(device).unsqueeze(1)
                feat = torch.stack(feats, dim=0).to(device).unsqueeze(1)
                gt = torch.stack(gts, dim=0).to(device).unsqueeze(1)
            else:
                signals, gts = [], []
                for signal_path, label_path, _sample_id in frames:
                    signals.append(torch.from_numpy(np.load(signal_path)))
                    label_array = np.load(label_path)
                    gts.append(torch.from_numpy(label_array).mean(dim=0).reshape(3, 2))
                signal = torch.stack(signals, dim=0).to(device)
                out, conf, feat = run_frozen_m1(signal, model)
                out, conf, feat = (value.unsqueeze(1) for value in (out, conf, feat))
                gt = torch.stack(gts, dim=0).to(device).unsqueeze(1)

            loss, n_frames, _, trajectory, batch_parts = forward_batch_cold(
                out, conf, feat, gt, active_banks, tracker,
                return_details=True, reset_state=False, framewise_loss=True,
            )
            loss_total += float(loss.item()) * n_frames
            frame_count += n_frames
            for name, value in batch_parts.items():
                parts[name] += float(value.item()) * n_frames
            association_stats = trajectory["association_stats"]
            association_correct_total += int(association_stats["correct"].item())
            association_count_total += int(association_stats["count"].item())
            association_true_probability_sum += float(association_stats["true_probability_sum"].item())
            association_foreground_correct_total += int(association_stats["foreground_correct"].item())
            association_foreground_count_total += int(association_stats["foreground_count"].item())
            association_foreground_true_probability_sum += float(
                association_stats["foreground_true_probability_sum"].item()
            )
            # 按整个 batch 回传，避免每条序列分别发起 GPU -> CPU 复制。
            batch_positions = trajectory["final_pos"][:, 0].cpu()
            batch_active = trajectory["active"][:, 0].cpu()
            batch_observed = trajectory["observed"][:, 0].cpu()
            batch_labels = gt[:, 0].cpu()
            for local_idx, sequence_idx in enumerate(active_indices):
                positions[sequence_idx].append(batch_positions[local_idx])
                active[sequence_idx].append(batch_active[local_idx])
                observed[sequence_idx].append(batch_observed[local_idx])
                labels[sequence_idx].append(batch_labels[local_idx])
                active_positions = batch_positions[local_idx][batch_active[local_idx]]
                active_position_count += int(active_positions.shape[0])
                if active_positions.numel():
                    out_of_bounds = (
                        (active_positions[:, 0] < config.TRACK_X_RANGE_M[0])
                        | (active_positions[:, 0] > config.TRACK_X_RANGE_M[1])
                        | (active_positions[:, 1] < config.TRACK_Y_RANGE_M[0])
                        | (active_positions[:, 1] > config.TRACK_Y_RANGE_M[1])
                    )
                    active_position_out_of_bounds_count += int(out_of_bounds.sum().item())
            if show_progress and time_idx % 20 == 0:
                sequence_progress.set_postfix(
                    loss=f"{loss_total / max(frame_count, 1):.5f}",
                    frames=frame_count,
                    batch=len(active_indices),
                )

        for sequence_idx in range(len(records)):
            if not positions[sequence_idx]:
                continue
            sequence_metrics = sequence_tracking_metrics(
                torch.stack(positions[sequence_idx]),
                torch.stack(active[sequence_idx]),
                torch.stack(observed[sequence_idx]),
                torch.stack(labels[sequence_idx]),
                warmup=config.HIST_LEN,
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
        sequence_progress.update(len(records))
        sequence_progress.set_postfix(
            loss=f"{loss_total / max(frame_count, 1):.5f}",
            frames=frame_count,
        )
    sequence_progress.close()

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
    result["association_accuracy"] = _safe_ratio(association_correct_total, association_count_total)
    result["association_true_probability"] = _safe_ratio(
        association_true_probability_sum, association_count_total
    )
    result["association_foreground_accuracy"] = _safe_ratio(
        association_foreground_correct_total, association_foreground_count_total
    )
    result["association_foreground_true_probability"] = _safe_ratio(
        association_foreground_true_probability_sum, association_foreground_count_total
    )
    result["active_position_out_of_bounds_ratio"] = _safe_ratio(
        active_position_out_of_bounds_count, active_position_count
    )
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
        teacher_ratio = config.teacher_assignment_ratio(epoch)
        train_total = 0.0
        train_count = 0
        train_parts_total = {name: 0.0 for name in config.TRACKER_LOSS_WEIGHTS}
        gradient_norm_total = 0.0
        gradient_norm_count = 0
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
                loss, n_frames, _, _trajectory, loss_parts = forward_batch_cold(
                    out, conf, feat, gt, banks, tracker,
                    return_details=True,
                    teacher_assignment_ratio=teacher_ratio,
                    loss_weights=config.TRACKER_TRAIN_LOSS_WEIGHTS,
                )
            else:
                signals, gt = batch
                loss, n_frames, _, _trajectory, loss_parts = forward_batch(
                    signals, gt, model, banks, tracker,
                    return_details=True,
                    teacher_assignment_ratio=teacher_ratio,
                    loss_weights=config.TRACKER_TRAIN_LOSS_WEIGHTS,
                )
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite tracker training loss")
            loss.backward()
            gradient_norm = gradient_l2_norm(tracker.parameters())
            optimizer.step()
            train_total += float(loss.detach().item()) * n_frames
            train_count += n_frames
            for name, value in loss_parts.items():
                train_parts_total[name] += float(value.detach().item()) * n_frames
            gradient_norm_total += gradient_norm
            gradient_norm_count += 1
            train_progress.set_postfix(
                loss=f"{train_total / max(train_count, 1):.5f}",
                assoc=f"{train_parts_total['association'] / max(train_count, 1):.4f}",
                pos=f"{train_parts_total['position'] / max(train_count, 1):.4f}",
                grad=f"{gradient_norm_total / max(gradient_norm_count, 1):.3f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                teacher=f"{teacher_ratio:.2f}",
            )

        train_loss = train_total / max(train_count, 1)
        train_parts = {
            name: value / max(train_count, 1) for name, value in train_parts_total.items()
        }
        train_gradient_norm = gradient_norm_total / max(gradient_norm_count, 1)
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
            "train_association": train_parts["association"],
            "train_position": train_parts["position"],
            "train_existence": train_parts["existence"],
            "train_confidence": train_parts["confidence"],
            "train_gradient_l2_norm": train_gradient_norm,
            "learning_rate": optimizer.param_groups[0]["lr"],
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
            "val_association_accuracy": val_metrics["association_accuracy"],
            "val_association_true_probability": val_metrics["association_true_probability"],
            "val_association_foreground_accuracy": val_metrics["association_foreground_accuracy"],
            "val_association_foreground_true_probability": val_metrics[
                "association_foreground_true_probability"
            ],
            "val_active_position_out_of_bounds_ratio": val_metrics[
                "active_position_out_of_bounds_ratio"
            ],
            "elapsed_seconds": round(time.time() - started, 2),
        })

        checkpoint = {
            "epoch": epoch,
            "tracker_state_dict": tracker.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "train_loss_parts": train_parts,
            "train_gradient_l2_norm": train_gradient_norm,
            "val_metrics": val_metrics,
            "tracker_loss_weights": dict(config.TRACKER_LOSS_WEIGHTS),
            "tracker_train_loss_weights": dict(config.TRACKER_TRAIN_LOSS_WEIGHTS),
            "config": {
                "hist_len": config.HIST_LEN,
                "num_tracks": config.NUM_TRACKS,
                "feat_dim": config.FEAT_DIM,
                "dt": config.DT,
                "confirm_threshold": config.CONFIRM_THRESHOLD,
                "assoc_label_radius_m": config.ASSOC_LABEL_RADIUS_M,
                "teacher_assoc_warmup_epochs": config.TEACHER_ASSOC_WARMUP_EPOCHS,
                "teacher_assoc_anneal_epochs": config.TEACHER_ASSOC_ANNEAL_EPOCHS,
                "teacher_assignment_ratio": teacher_ratio,
                "learning_rate": optimizer.param_groups[0]["lr"],
            },
        }
        torch.save(checkpoint, os.path.join(config.OUT_DIR, "joint_tracker_last.pth"))
        # 教师融合不参与验证；仅自主闭环阶段的验证结果可选择最终 best checkpoint。
        if teacher_ratio == 0.0 and val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(checkpoint, os.path.join(config.OUT_DIR, "joint_tracker_best.pth"))
        print(
            f"epoch {epoch}/{config.EPOCHS}: train={train_loss:.5f} "
            f"train_assoc={train_parts['association']:.5f} "
            f"train_pos={train_parts['position']:.5f} "
            f"grad={train_gradient_norm:.3f} "
            f"teacher={teacher_ratio:.2f} "
            f"val={val_metrics['loss']:.5f} assoc={val_metrics['association']:.5f} "
            f"pos={val_metrics['position']:.5f} IDSW={val_metrics['idsw']} "
            f"MAE={val_metrics['mae_m']:.3f}m "
            f"assoc_fg_acc={val_metrics['association_foreground_accuracy']:.3f} "
            f"oob={val_metrics['active_position_out_of_bounds_ratio']:.3f} "
            f"coverage={val_metrics['target_coverage']:.2f} ({time.time() - started:.1f}s)",
            flush=True,
        )
    return best_val
