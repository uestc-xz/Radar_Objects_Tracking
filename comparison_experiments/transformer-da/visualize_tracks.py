"""用 matplotlib 逐帧查看 Transformer-DA 的 ID 关联结果和 GT 航迹。

运行后使用键盘控制：
    Right / D：下一帧
    Left / A：上一帧
    Space：自动播放/暂停
    Home / End：跳到首帧/末帧
    Q / Esc：关闭窗口

图中灰色点是杂波候选，蓝/橙/绿色点分别是 DA 判定的 ID1/ID2/ID3。
每个 ID 只保留最近 15 帧的点迹尾迹；彩色叉号和虚线是对应 GT。
脚本只做数据可视化，不修改数据和模型权重。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

# PyCharm 的 backend_interagg 在部分版本中会让 plt.show() 立即返回，
# 因而脚本刚画出第一帧就结束。使用 TkAgg 让图像保持为独立交互窗口。
import matplotlib

if os.name == "nt" and not os.environ.get("MPLBACKEND"):
    try:
        matplotlib.use("TkAgg", force=True)
    except ImportError:
        pass

import matplotlib.pyplot as plt

plt.rcParams["axes.unicode_minus"] = False


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from DANet import DANet  # noqa: E402
from train import run_autoregressive  # noqa: E402
from original.my_dataset.track_sequence_dataset import (  # noqa: E402
    TrackSequenceColdDataset,
)


TARGET_COLORS = ("tab:blue", "tab:orange", "tab:green")
TRAIL_LENGTH = 15
X_LIMITS = (-3.5, 3.5)
Y_LIMITS = (0.0, 7.5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="逐帧查看 M1 候选和 GT 航迹")
    parser.add_argument(
        "--split",
        choices=("train", "val"),
        default="val",
        help="数据划分，默认读取验证集",
    )
    parser.add_argument(
        "--folder",
        default=None,
        help="要查看的序列文件夹名；省略时使用该划分的第一个文件夹",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="打开窗口时显示的起始帧下标",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.8,
        help="历史预热阶段高亮 M1 前景候选的置信度阈值",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=config.DA_OUTPUT_DIR / "best.pt",
        help="Transformer-DA 权重，默认读取 runs/best.pt",
    )
    parser.add_argument(
        "--inference-batch-size",
        type=int,
        default=16,
        help="批量计算逐帧关联结果时的 batch size",
    )
    parser.add_argument(
        "--play-interval",
        type=int,
        default=240,
        help="自动播放时每帧间隔，单位毫秒",
    )
    return parser.parse_args()


def load_sequence(split: str, folder_name: str | None):
    """读取指定文件夹的 M1 缓存、GT 坐标和样本编号。"""
    if split == "train":
        data_root = config.TRAIN_ROOT
        cache_root = config.CACHE_ROOT
    else:
        data_root = config.VAL_ROOT
        cache_root = config.VAL_CACHE_ROOT

    dataset = TrackSequenceColdDataset(
        data_root_folder=str(data_root),
        cache_root_folder=str(cache_root),
        window_size=config.WINDOW_SIZE,
        stride=config.STRIDE,
        full_sequence=True,
    )
    folder_names = [item["folder"] for item in dataset.folders]
    if not folder_names:
        raise RuntimeError(f"没有找到可视化序列：{data_root}")
    if folder_name is None:
        print("可选择的可视化序列：")
        for index, item in enumerate(dataset.folders):
            print(f"  {index}: {item['folder']}，帧数：{len(item['frames'])}")
        try:
            choice = input(f"请输入序列编号（0-{len(folder_names) - 1}）：").strip()
        except EOFError:
            print("未检测到终端输入，默认选择序列 0")
            choice = "0"
        try:
            folder_index = int(choice)
        except ValueError as error:
            raise ValueError("序列编号必须是整数") from error
        if not 0 <= folder_index < len(folder_names):
            raise ValueError(f"序列编号必须在 0 到 {len(folder_names) - 1} 之间")
        folder_name = folder_names[folder_index]
    if folder_name not in folder_names:
        available = ", ".join(folder_names[:10])
        raise ValueError(f"找不到文件夹 {folder_name!r}，可选示例：{available}")

    record = dataset[folder_names.index(folder_name)]
    outputs = []
    confidence_logits = []
    positions = []
    foreground_probability = []
    labels = []
    sample_ids = []
    for cache_path, label_path, sample_id in record["frames"]:
        cache = np.load(cache_path)
        output = cache["out"].astype(np.float32)
        conf = cache["conf"].astype(np.float32)
        outputs.append(output)
        confidence_logits.append(conf)
        positions.append(output[:, :2])
        # conf 是 M1 的 [背景, 前景] logits，转换为前景概率用于显示筛选。
        foreground_probability.append(torch.from_numpy(conf).softmax(dim=-1)[:, 1].numpy())
        labels.append(np.load(label_path).mean(axis=0).reshape(-1, 2).astype(np.float32))
        sample_ids.append(sample_id)

    return {
        "folder": folder_name,
        "outputs": np.stack(outputs),
        "confidence_logits": np.stack(confidence_logits),
        "positions": np.stack(positions),
        "foreground_probability": np.stack(foreground_probability),
        "labels": np.stack(labels),
        "sample_ids": sample_ids,
    }


def _valid_points(points: np.ndarray) -> np.ndarray:
    return np.isfinite(points).all(axis=-1) & (np.abs(points).sum(axis=-1) > 0)


def _build_model(checkpoint: dict, device: torch.device) -> DANet:
    """按 checkpoint 保存的参数重建 Transformer-DA。"""
    model_config = checkpoint.get("config", {})
    model = DANet(
        num_tracks=model_config.get("num_tracks", config.NUM_TRACKS),
        history_steps=model_config.get("history_steps", config.HISTORY_STEPS),
        measurement_dim=model_config.get("measurement_dim", config.MEASUREMENT_DIM),
        num_classes=model_config.get("num_classes", config.NUM_CLASSES),
        d_model=model_config.get("d_model", config.DA_D_MODEL),
        num_heads=model_config.get("num_heads", config.DA_NUM_HEADS),
        encoder_layers=model_config.get("encoder_layers", config.DA_ENCODER_LAYERS),
        decoder_layers=model_config.get("decoder_layers", config.DA_DECODER_LAYERS),
        feedforward_dim=model_config.get("feedforward_dim", config.DA_FEEDFORWARD_DIM),
        dropout=model_config.get("dropout", config.DA_DROPOUT),
        max_measurements=config.MAX_MEASUREMENTS,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.eval()


@torch.no_grad()
def predict_associations(sequence: dict, checkpoint_path: Path, batch_size: int):
    """对整条序列计算 DA 类别和每个 ID 的代表点。"""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(checkpoint, device)

    outputs = torch.from_numpy(sequence["outputs"]).unsqueeze(0).to(device)
    confidence_logits = torch.from_numpy(sequence["confidence_logits"]).unsqueeze(0).to(device)
    frame_count, query_count = sequence["positions"].shape[:2]
    predicted_classes = np.full((frame_count, query_count), -1, dtype=np.int64)
    predicted_probabilities = np.zeros(
        (frame_count, query_count, config.NUM_CLASSES), dtype=np.float32
    )

    # 自回归滚动：历史由网络自己上一帧的输出维护，不依赖标签。
    # batch_size 保留仅为兼容命令行参数，滚动必须按序逐帧处理。
    frame_indices, network_outputs, _track_states = run_autoregressive(
        outputs, confidence_logits, model
    )
    for frame_index, network_output in zip(frame_indices, network_outputs):
        probabilities = network_output.probabilities[0]  # [50, 4]
        predicted_classes[frame_index] = probabilities.argmax(dim=-1).cpu().numpy()
        predicted_probabilities[frame_index] = probabilities.cpu().numpy()

    # 每帧每个 ID 可能有多个候选，这里选该 ID 概率最高的一个作为代表点。
    associated_positions = np.full((frame_count, config.NUM_TRACKS, 2), np.nan, dtype=np.float32)
    for frame_index in range(frame_count):
        for target_index in range(config.NUM_TRACKS):
            target_class = target_index + 1
            candidates = np.flatnonzero(predicted_classes[frame_index] == target_class)
            if len(candidates) == 0:
                continue
            best_query = candidates[
                predicted_probabilities[frame_index, candidates, target_class].argmax()
            ]
            associated_positions[frame_index, target_index] = sequence["positions"][
                frame_index, best_query
            ]

    return predicted_classes, predicted_probabilities, associated_positions, device


def show_sequence(sequence: dict, predicted_classes: np.ndarray,
                  predicted_probabilities: np.ndarray,
                  associated_positions: np.ndarray, start_frame: int,
                  confidence_threshold: float, play_interval: int) -> None:
    positions = sequence["positions"]
    foreground_probability = sequence["foreground_probability"]
    labels = sequence["labels"]
    frame_count = len(positions)
    if frame_count == 0:
        raise RuntimeError("序列没有可显示的帧")

    frame_index = max(0, min(int(start_frame), frame_count - 1))
    playing = False
    fig, (ax, table_ax) = plt.subplots(
        1,
        2,
        figsize=(16, 8),
        gridspec_kw={"width_ratios": (3.6, 2.4)},
    )
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.08, top=0.80, wspace=0.08)

    def draw_frame(index: int) -> None:
        ax.clear()
        table_ax.clear()
        table_ax.axis("off")
        current_positions = positions[index]
        current_probability = foreground_probability[index]
        current_labels = labels[index]
        current_classes = predicted_classes[index]
        valid_current = _valid_points(current_positions)

        if index < config.HISTORY_STEPS:
            high_confidence = valid_current & (current_probability >= confidence_threshold)
            ax.scatter(
                current_positions[high_confidence, 0],
                current_positions[high_confidence, 1],
                s=28,
                c="red",
                alpha=0.85,
                label=f"M1 前景候选 ≥ {confidence_threshold:.2f}",
            )

        for target_index, color in enumerate(TARGET_COLORS):
            target_class = target_index + 1
            associated = current_classes == target_class
            associated &= valid_current
            if associated.any():
                ax.scatter(
                    current_positions[associated, 0],
                    current_positions[associated, 1],
                    s=30,
                    color=color,
                    alpha=0.55,
                    label=f"DA ID{target_class} 候选",
                )

            trail_start = max(0, index - TRAIL_LENGTH + 1)
            target_trail = associated_positions[trail_start : index + 1, target_index]
            valid_target_trail = np.isfinite(target_trail).all(axis=-1)
            if valid_target_trail.any():
                ax.plot(
                    target_trail[valid_target_trail, 0],
                    target_trail[valid_target_trail, 1],
                    color=color,
                    linewidth=2.0,
                    alpha=0.9,
                    label=f"DA ID{target_class} 尾迹（{TRAIL_LENGTH}帧）",
                )
                current_track_point = associated_positions[index, target_index]
                if np.isfinite(current_track_point).all():
                    ax.scatter(
                        current_track_point[0],
                        current_track_point[1],
                        marker="o",
                        s=75,
                        facecolors="none",
                        edgecolors=color,
                        linewidths=2.0,
                    )
                    ax.text(
                        current_track_point[0],
                        current_track_point[1],
                        f" ID{target_class}",
                        color=color,
                        fontsize=10,
                        weight="bold",
                    )

            target_history = labels[trail_start : index + 1, target_index]
            valid_history = _valid_points(target_history)
            if valid_history.any():
                ax.plot(
                    target_history[valid_history, 0],
                    target_history[valid_history, 1],
                    color=color,
                    linewidth=1.5,
                    linestyle="--",
                    alpha=0.55,
                    label=f"GT{target_class}（{TRAIL_LENGTH}帧）",
                )
            if _valid_points(current_labels[target_index : target_index + 1])[0]:
                point = current_labels[target_index]
                ax.scatter(
                    point[0],
                    point[1],
                    marker="x",
                    s=90,
                    linewidths=2.0,
                    color=color,
                    label=f"GT{target_class}",
                )

        # 按需求固定显示范围，便于不同序列和不同帧直接比较。
        ax.set_xlim(*X_LIMITS)
        ax.set_ylim(*Y_LIMITS)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x / m")
        ax.set_ylabel("y / m")
        ax.grid(True, alpha=0.25)
        ax.set_title(
            f"{sequence['folder']}  frame {index + 1}/{frame_count} "
            f"(sample_id={sequence['sample_ids'][index]})\n"
            f"DA {'已输出关联' if index >= config.HISTORY_STEPS else '历史不足，暂未输出关联'}"
        )

        # 图注放在绘图区外，避免遮挡轨迹和候选点；不显示 DA0 杂波。
        handles, legend_labels = ax.get_legend_handles_labels()
        unique_handles = {}
        for handle, label in zip(handles, legend_labels):
            unique_handles.setdefault(label, handle)
        if unique_handles:
            ax.legend(
                unique_handles.values(),
                unique_handles.keys(),
                loc="lower center",
                bbox_to_anchor=(0.5, 1.04),
                ncol=3,
                fontsize=8,
                frameon=True,
            )

        # 右侧动态表格：固定 3 行、15 列，-- 表示该帧没有 DA 关联代表点。
        table_start = max(0, index - TRAIL_LENGTH + 1)
        visible_indices = list(range(table_start, index + 1))
        padding = TRAIL_LENGTH - len(visible_indices)
        column_labels = [""] * padding + [
            str(sequence["sample_ids"][frame_index]) for frame_index in visible_indices
        ]
        table_rows = []
        for target_index in range(config.NUM_TRACKS):
            row = ["--"] * padding
            for frame_index in visible_indices:
                point = associated_positions[frame_index, target_index]
                if np.isfinite(point).all():
                    row.append(f"{point[0]:.2f}\n{point[1]:.2f}")
                else:
                    row.append("--")
            table_rows.append(row)
        history_table = table_ax.table(
            cellText=table_rows,
            rowLabels=["ID1", "ID2", "ID3"],
            colLabels=column_labels,
            cellLoc="center",
            rowLoc="center",
            loc="center",
            colWidths=[0.061] * TRAIL_LENGTH,
        )
        history_table.auto_set_font_size(False)
        history_table.set_fontsize(7)
        history_table.scale(1.0, 2.2)
        for (row, column), cell in history_table.get_celld().items():
            if column == -1 and row > 0:
                cell.set_facecolor(TARGET_COLORS[row - 1])
                cell.get_text().set_color("white")
                cell.get_text().set_weight("bold")
            elif row == 0:
                cell.set_facecolor("#e8e8e8")
                cell.get_text().set_weight("bold")
            elif row > 0:
                cell.set_facecolor("#f8f8f8")
        table_ax.set_title(
            "最近 15 帧 DA 航迹\n单元格为 (x, y) / m；-- 表示未生成关联量测",
            fontsize=10,
            pad=12,
        )
        fig.canvas.draw_idle()

    def change_frame(step: int) -> None:
        nonlocal frame_index
        frame_index = (frame_index + step) % frame_count
        draw_frame(frame_index)

    def on_key(event) -> None:
        nonlocal playing
        if event.key in ("right", "d"):
            change_frame(1)
        elif event.key in ("left", "a"):
            change_frame(-1)
        elif event.key == "home":
            change_frame(-frame_count)
        elif event.key == "end":
            change_frame(frame_count)
        elif event.key == "space":
            playing = not playing
        elif event.key in ("q", "escape"):
            plt.close(fig)

    def on_timer() -> None:
        if playing:
            change_frame(1)

    fig.canvas.mpl_connect("key_press_event", on_key)
    timer = fig.canvas.new_timer(interval=max(int(play_interval), 20))
    timer.add_callback(on_timer)
    timer.start()
    draw_frame(frame_index)
    # 显式阻塞，确保窗口关闭前脚本不会结束。
    plt.show(block=True)


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.confidence_threshold <= 1.0:
        raise ValueError("confidence-threshold 必须在 [0, 1] 内")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"找不到 Transformer-DA 权重：{args.checkpoint}")
    sequence = load_sequence(args.split, args.folder)
    print(f"可视化序列：{sequence['folder']}，帧数：{len(sequence['positions'])}")
    print(f"加载权重：{args.checkpoint}")
    print("正在计算 DA 的逐帧 ID 关联结果……")
    predicted_classes, predicted_probabilities, associated_positions, device = (
        predict_associations(sequence, args.checkpoint, args.inference_batch_size)
    )
    print(f"关联计算完成，device={device}，尾迹长度={TRAIL_LENGTH} 帧")
    print("操作：左右键逐帧，空格播放/暂停，Home/End 跳转，Q/Esc 退出")
    show_sequence(
        sequence,
        predicted_classes=predicted_classes,
        predicted_probabilities=predicted_probabilities,
        associated_positions=associated_positions,
        start_frame=args.start_frame,
        confidence_threshold=args.confidence_threshold,
        play_interval=args.play_interval,
    )


if __name__ == "__main__":
    main()
