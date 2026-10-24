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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_printoptions(precision=6, linewidth=160, sci_mode=False)

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

    print("=== Transformer-DA 数据集检查 ===")
    print(f"IS_COLD: {config.IS_COLD}")
    print(f"数据目录: {args.data_root}")
    print(f"样本总数: {len(dataset)}")
    print(f"样本索引: {args.index}")
    if config.IS_COLD:
        outputs, confidences, features, labels = dataset[args.index]
        print(f"M1 out.shape: {tuple(outputs.shape)}")
        print(f"M1 out.dtype: {outputs.dtype}")
        print(f"M1 conf.shape: {tuple(confidences.shape)}")
        print(f"M1 feat.shape: {tuple(features.shape)}")
        print(f"M1 out.isfinite: {bool(torch.isfinite(outputs).all())}")
        print(f"M1 conf.isfinite: {bool(torch.isfinite(confidences).all())}")
        print(f"M1 feat.isfinite: {bool(torch.isfinite(features).all())}")
        print("\n--- 第一帧 M1 输出数值片段 ---")
        print("out[0, :] =", outputs[0, :])
        print("conf[0, :] =", confidences[0, :])
        print("feat[0, 0, :8] =", features[0, 0, :8])
    else:
        signals, labels = dataset[args.index]
        print(f"signals.shape: {tuple(signals.shape)}")
        print(f"signals.dtype: {signals.dtype}")
        print(f"signals.isfinite: {bool(torch.isfinite(signals.real).all() and torch.isfinite(signals.imag).all())}")
        print(f"signals.real.min/max: {signals.real.min().item():.6f} / {signals.real.max().item():.6f}")
        print(f"signals.imag.min/max: {signals.imag.min().item():.6f} / {signals.imag.max().item():.6f}")
        print("\n--- 第一帧原始信号数值片段 ---")
        print("signals[0, 0, 0, :5] =", signals[0, 0, 0, :5])

    print(f"labels.shape: {tuple(labels.shape)}")
    print(f"labels.dtype: {labels.dtype}")
    print(f"labels.isfinite: {bool(torch.isfinite(labels).all())}")
    print("\n--- 全部窗口标签（每帧 3 个目标的 x,y）---")
    print(labels)
    print("\n--- 标签数值范围 ---")
    print(f"labels.min/max: {labels.min().item():.6f} / {labels.max().item():.6f}")


if __name__ == "__main__":
    main()
