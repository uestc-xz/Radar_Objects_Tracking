"""最小数据集检查脚本。

只负责加载一条跟踪序列窗口，并打印输入信号与标签的基本信息。
不包含模型、训练或评估逻辑。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


# 允许从仓库根目录导入现有数据集实现。
REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

import config  # noqa: E402
from DANet import DANet  # noqa: E402
from train import train_from_config  # noqa: E402
from original.my_dataset.track_sequence_dataset import (  # noqa: E402
    load_track_sequence_cold_dataset,
    load_track_sequence_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查 Transformer-DA 对比实验的数据输入")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=config.TRAIN_ROOT,
        help="数据集根目录，默认读取当前项目配置中的 TRAIN_ROOT",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=config.WINDOW_SIZE,
        help="每个样本包含的连续帧数",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=config.STRIDE,
        help="滑窗步长",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=config.SAMPLE_INDEX,
        help="要打印的数据样本索引",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="调用 train.py 执行一次配置中的训练流程",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_printoptions(precision=6, linewidth=160, sci_mode=False)

    if args.train:
        train_from_config()
        return

    if config.IS_COLD:
        dataset = load_track_sequence_cold_dataset(
            data_root_folder=str(args.data_root),
            cache_root_folder=str(config.CACHE_ROOT),
            window_size=args.window_size,
            stride=args.stride,
            full_sequence=False,
        )
    else:
        dataset = load_track_sequence_dataset(
            data_root_folder=str(args.data_root),
            window_size=args.window_size,
            stride=args.stride,
            add_noise=False,
            delete_occlusion=False,
            full_sequence=False,
        )

    if len(dataset) == 0:
        raise RuntimeError(f"数据集为空：{args.data_root}")
    if not 0 <= args.index < len(dataset):
        raise IndexError(f"index={args.index} 超出范围，数据集长度为 {len(dataset)}")

    print("=== Transformer-DA 前向验证 ===")
    print(f"IS_COLD={config.IS_COLD}, sample_index={args.index}, dataset_size={len(dataset)}")
    if config.IS_COLD:
        outputs, confidences, features, labels = dataset[args.index]
        print(f"M1 outputs={tuple(outputs.shape)}, confidence={tuple(confidences.shape)}, labels={tuple(labels.shape)}")
        if outputs.shape[0] <= config.HISTORY_STEPS:
            raise RuntimeError("窗口长度不足以构造 Transformer-DA 历史输入")
        frame_index = config.HISTORY_STEPS
        history_positions = labels[:config.HISTORY_STEPS]
        previous = torch.cat((history_positions[:1], history_positions[:-1]), dim=0)
        velocity = (history_positions - previous) / config.DELTA_T
        history_confidence = []
        for history_index in range(config.HISTORY_STEPS):
            # 用当前位置最近的 M1 query 提取该目标的二分类置信度 logits。
            distances = torch.cdist(
                history_positions[history_index], outputs[history_index, :, :2]
            )
            nearest_query = distances.argmin(dim=-1)
            nearest_confidence = confidences[history_index].index_select(0, nearest_query)
            valid_target = history_positions[history_index].abs().sum(dim=-1) > 0
            history_confidence.append(nearest_confidence * valid_target.unsqueeze(-1))
        history_confidence = torch.stack(history_confidence, dim=0)
        history_states = torch.cat(
            (
                history_positions[..., 0:1], velocity[..., 0:1],
                history_positions[..., 1:2], velocity[..., 1:2],
                history_confidence,
            ),
            dim=-1,
        ).permute(1, 0, 2).unsqueeze(0)
        # 解码器不再使用 out 的后两个无用通道，改用 M1 的二分类置信度 logits。
        measurement_input = torch.cat(
            (outputs[frame_index, :, :2], confidences[frame_index]), dim=-1
        ).unsqueeze(0)
        da_net = DANet(
            num_tracks=config.NUM_TRACKS,
            history_steps=config.HISTORY_STEPS,
            measurement_dim=config.MEASUREMENT_DIM,
            num_classes=config.NUM_CLASSES,
            d_model=config.DA_D_MODEL,
            num_heads=config.DA_NUM_HEADS,
            encoder_layers=config.DA_ENCODER_LAYERS,
            decoder_layers=config.DA_DECODER_LAYERS,
            feedforward_dim=config.DA_FEEDFORWARD_DIM,
            dropout=config.DA_DROPOUT,
            max_measurements=config.MAX_MEASUREMENTS,
        ).eval()
        with torch.no_grad():
            da_output = da_net(history_states, measurement_input)
        print(f"history_states.shape: {tuple(history_states.shape)}")
        print(f"measurement_input.shape: {tuple(measurement_input.shape)}")
        print(f"association_logits.shape: {tuple(da_output.logits.shape)}")
        print(f"association_probabilities.shape: {tuple(da_output.probabilities.shape)}")
        print(f"probability_sum_error: {(da_output.probabilities.sum(-1) - 1).abs().max().item():.6e}")
    else:
        signals, labels = dataset[args.index]
        print(f"signals.shape: {tuple(signals.shape)}")
        print(f"signals.dtype: {signals.dtype}")
        print(f"signals.isfinite: {bool(torch.isfinite(signals.real).all() and torch.isfinite(signals.imag).all())}")
        print(f"signals.real.min/max: {signals.real.min().item():.6f} / {signals.real.max().item():.6f}")
        print(f"signals.imag.min/max: {signals.imag.min().item():.6f} / {signals.imag.max().item():.6f}")
        print("\n--- 第一帧原始信号数值片段 ---")
        print("signals[0, 0, 0, :5] =", signals[0, 0, 0, :5])

    print(f"labels={tuple(labels.shape)}, finite={bool(torch.isfinite(labels).all())}")


if __name__ == "__main__":
    main()
