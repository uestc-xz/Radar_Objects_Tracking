"""加载 Transformer-DA 权重并统计验证集指标。

默认评估 ``runs/best.pt``。脚本沿用训练阶段的窗口、历史状态和最近邻关联
标签构造方式，因此输出可以直接与训练日志中的 ``val_loss`` 对照。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from DANet import DANet  # noqa: E402
from lossfunc import MCDLoss  # noqa: E402
from train import build_history_states  # noqa: E402
from original.my_dataset.track_sequence_dataset import (  # noqa: E402
    load_track_sequence_cold_dataset,
)
from original.my_track.association import build_assignment_targets  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="统计 Transformer-DA 权重指标")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=config.DA_OUTPUT_DIR / "best.pt",
        help="要加载的 checkpoint，默认读取 runs/best.pt",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val"),
        default="val",
        help="评估数据划分，默认是验证集",
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help="最多评估多少个窗口；默认评估完整数据集",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.DA_BATCH_SIZE,
        help="评估 batch size",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker 数量；Windows 默认使用 0 以避免子进程权限问题",
    )
    parser.add_argument(
        "--match-radius",
        type=float,
        default=config.ASSIGNMENT_RADIUS_M,
        help="目标覆盖和坐标匹配的距离门限（米）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="统计结果 JSON 路径；省略时保存到 runs/metrics_<权重>_<划分>.json",
    )
    return parser.parse_args()


def build_model(checkpoint_config: dict, device: torch.device) -> DANet:
    """按 checkpoint 中保存的网络参数重建模型。"""
    model = DANet(
        num_tracks=checkpoint_config.get("num_tracks", config.NUM_TRACKS),
        history_steps=checkpoint_config.get("history_steps", config.HISTORY_STEPS),
        measurement_dim=checkpoint_config.get("measurement_dim", config.MEASUREMENT_DIM),
        num_classes=checkpoint_config.get("num_classes", config.NUM_CLASSES),
        d_model=checkpoint_config.get("d_model", config.DA_D_MODEL),
        num_heads=checkpoint_config.get("num_heads", config.DA_NUM_HEADS),
        encoder_layers=checkpoint_config.get("encoder_layers", config.DA_ENCODER_LAYERS),
        decoder_layers=checkpoint_config.get("decoder_layers", config.DA_DECODER_LAYERS),
        feedforward_dim=checkpoint_config.get("feedforward_dim", config.DA_FEEDFORWARD_DIM),
        dropout=checkpoint_config.get("dropout", config.DA_DROPOUT),
        max_measurements=config.MAX_MEASUREMENTS,
    ).to(device)
    return model


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator) / max(int(denominator), 1)


def _f1(precision: float, recall: float) -> float:
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def _update_target_matching(
    positions: torch.Tensor,
    predicted_classes: torch.Tensor,
    gt_positions: torch.Tensor,
    radius_m: float,
    totals: dict,
) -> None:
    """统计 M1 全候选和 DA 按类别筛选后的目标覆盖/距离。"""
    valid_gt = gt_positions.abs().sum(dim=-1) > 0
    for target_index in range(gt_positions.shape[0]):
        if not valid_gt[target_index]:
            continue
        target = gt_positions[target_index]

        all_distances = torch.linalg.vector_norm(positions - target, dim=-1)
        m1_distance = float(all_distances.min().item())
        totals["m1_target_count"] += 1
        totals["m1_error_sum_m"] += m1_distance
        if m1_distance <= radius_m:
            totals["m1_covered_target_count"] += 1

        target_class = target_index + 1
        selected = predicted_classes == target_class
        if not selected.any():
            continue
        da_distance = float(all_distances[selected].min().item())
        totals["da_predicted_target_count"] += 1
        totals["da_error_sum_m"] += da_distance
        if da_distance <= radius_m:
            totals["da_covered_target_count"] += 1


@torch.no_grad()
def evaluate(checkpoint_path: Path, split: str, max_windows: int | None,
             batch_size: int, num_workers: int, radius_m: float) -> dict:
    """运行完整评估并返回可序列化的指标字典。"""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(checkpoint.get("config", {}), device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    criterion = MCDLoss(smooth=1.0)

    if split == "train":
        data_root = config.TRAIN_ROOT
        cache_root = config.CACHE_ROOT
    else:
        data_root = config.VAL_ROOT
        cache_root = config.VAL_CACHE_ROOT
    dataset = load_track_sequence_cold_dataset(
        data_root_folder=str(data_root),
        cache_root_folder=str(cache_root),
        window_size=config.WINDOW_SIZE,
        stride=config.STRIDE,
        full_sequence=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    num_classes = config.NUM_CLASSES
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    totals = {
        "windows": 0,
        "frames": 0,
        "measurements": 0,
        "foreground_measurements": 0,
        "correct": 0,
        "foreground_correct": 0,
        "foreground_true_probability_sum": 0.0,
        "loss_sum": 0.0,
        "loss_mask_sum": 0.0,
        "loss_dice_sum": 0.0,
        "m1_target_count": 0,
        "m1_covered_target_count": 0,
        "m1_error_sum_m": 0.0,
        "da_predicted_target_count": 0,
        "da_covered_target_count": 0,
        "da_error_sum_m": 0.0,
    }

    progress = tqdm(loader, desc=f"evaluate {split}", unit="window")
    for window_index, batch in enumerate(progress):
        if max_windows is not None and window_index >= max_windows:
            break
        outputs, confidence_logits, _features, labels = [item.to(device) for item in batch]
        window_loss = 0.0
        window_frames = 0

        for frame_index in range(config.HISTORY_STEPS, outputs.shape[1]):
            history_states = build_history_states(
                outputs, confidence_logits, labels, frame_index
            )
            measurement_input = torch.cat(
                (outputs[:, frame_index, :, :2], confidence_logits[:, frame_index]),
                dim=-1,
            )
            network_output = model(history_states, measurement_input)
            target_classes = build_assignment_targets(
                outputs[:, frame_index, :, :2],
                labels[:, frame_index],
                radius_m=config.ASSIGNMENT_RADIUS_M,
            )
            mask = torch.ones_like(
                target_classes, dtype=network_output.probabilities.dtype
            )
            loss, components = criterion(
                network_output.probabilities,
                target_classes,
                mask,
                return_components=True,
            )
            window_loss += float(loss.item())
            totals["loss_mask_sum"] += float(components["loss_mask"].item())
            totals["loss_dice_sum"] += float(components["loss_dice"].item())
            window_frames += 1
            totals["frames"] += outputs.shape[0]

            predicted_classes = network_output.logits.argmax(dim=-1)
            probabilities = network_output.probabilities
            flat_target = target_classes.reshape(-1).cpu()
            flat_prediction = predicted_classes.reshape(-1).cpu()
            confusion += torch.bincount(
                flat_target * num_classes + flat_prediction,
                minlength=num_classes * num_classes,
            ).reshape(num_classes, num_classes)
            totals["measurements"] += int(flat_target.numel())
            foreground = target_classes.ne(0)
            totals["foreground_measurements"] += int(foreground.sum().item())
            totals["correct"] += int(predicted_classes.eq(target_classes).sum().item())
            totals["foreground_correct"] += int(
                (predicted_classes.eq(target_classes) & foreground).sum().item()
            )
            true_probability = probabilities.gather(
                -1, target_classes.unsqueeze(-1)
            ).squeeze(-1)
            totals["foreground_true_probability_sum"] += float(
                true_probability[foreground].sum().item()
            )

            for batch_index in range(outputs.shape[0]):
                _update_target_matching(
                    outputs[batch_index, frame_index, :, :2],
                    predicted_classes[batch_index],
                    labels[batch_index, frame_index],
                    radius_m,
                    totals,
                )

        totals["windows"] += 1
        totals["loss_sum"] += window_loss / max(window_frames, 1)
        progress.set_postfix(loss=f"{window_loss / max(window_frames, 1):.4f}")

    total_measurements = totals["measurements"]
    foreground_measurements = totals["foreground_measurements"]
    accuracy = _safe_ratio(totals["correct"], total_measurements)
    foreground_accuracy = _safe_ratio(
        totals["foreground_correct"], foreground_measurements
    )
    foreground_tp = int(confusion[1:, 1:].trace().item())
    foreground_fp = int(confusion[:, 1:].sum().item()) - foreground_tp
    foreground_fn = int(confusion[1:, :].sum().item()) - foreground_tp
    precision = _safe_ratio(foreground_tp, foreground_tp + foreground_fp)
    recall = _safe_ratio(foreground_tp, foreground_tp + foreground_fn)

    class_metrics = {}
    for class_index in range(num_classes):
        true_positive = int(confusion[class_index, class_index].item())
        predicted_count = int(confusion[:, class_index].sum().item())
        target_count = int(confusion[class_index, :].sum().item())
        class_precision = _safe_ratio(true_positive, predicted_count)
        class_recall = _safe_ratio(true_positive, target_count)
        class_metrics[str(class_index)] = {
            "precision": class_precision,
            "recall": class_recall,
            "f1": _f1(class_precision, class_recall),
            "support": target_count,
            "predicted": predicted_count,
        }

    frames = max(totals["frames"], 1)
    windows = max(totals["windows"], 1)
    result = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "split": split,
        "device": str(device),
        "assignment_radius_m": config.ASSIGNMENT_RADIUS_M,
        "matching_radius_m": radius_m,
        "windows": totals["windows"],
        "frames": totals["frames"],
        "measurements": total_measurements,
        "loss_mcd": totals["loss_sum"] / windows,
        "loss_mask": totals["loss_mask_sum"] / frames,
        "loss_dice": totals["loss_dice_sum"] / frames,
        "accuracy": accuracy,
        "foreground_accuracy": foreground_accuracy,
        "foreground_precision": precision,
        "foreground_recall": recall,
        "foreground_f1": _f1(precision, recall),
        "foreground_true_probability": _safe_ratio(
            int(round(totals["foreground_true_probability_sum"] * 1_000_000)),
            max(foreground_measurements, 1) * 1_000_000,
        ),
        "confusion_matrix": confusion.tolist(),
        "class_metrics": class_metrics,
        "m1_target_coverage": _safe_ratio(
            totals["m1_covered_target_count"], totals["m1_target_count"]
        ),
        "m1_nearest_mae_m": _safe_ratio(
            int(round(totals["m1_error_sum_m"] * 1_000_000)),
            max(totals["m1_target_count"], 1) * 1_000_000,
        ),
        "da_target_coverage": _safe_ratio(
            totals["da_covered_target_count"], totals["m1_target_count"]
        ),
        "da_classified_target_rate": _safe_ratio(
            totals["da_predicted_target_count"], totals["m1_target_count"]
        ),
        "da_classified_target_mae_m": _safe_ratio(
            int(round(totals["da_error_sum_m"] * 1_000_000)),
            max(totals["da_predicted_target_count"], 1) * 1_000_000,
        ),
    }
    return result


def main() -> None:
    args = parse_args()
    if not config.IS_COLD:
        raise RuntimeError("当前统计脚本只支持 IS_COLD=True 的 M1 冻结缓存")
    result = evaluate(
        checkpoint_path=args.checkpoint,
        split=args.split,
        max_windows=args.max_windows,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        radius_m=args.match_radius,
    )
    output = args.output
    if output is None:
        output = config.DA_OUTPUT_DIR / (
            f"metrics_{args.checkpoint.stem}_{args.split}.json"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"统计结果已保存：{output}")


if __name__ == "__main__":
    main()
