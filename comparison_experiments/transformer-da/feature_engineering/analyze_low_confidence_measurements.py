"""统计 M1 冻结缓存中不同置信度阈值以下量测的坐标分布。"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

import config  # noqa: E402


THRESHOLDS = [round(value, 2) for value in np.arange(0.80, 0.961, 0.01)]
OUTPUT_DIR = Path(__file__).resolve().parent


def discover_cache_files() -> list[Path]:
    if not config.IS_COLD:
        raise RuntimeError("config.IS_COLD 必须为 True，才能分析 M1 冻结缓存。")
    if not config.TRAIN_ROOT.is_dir():
        raise FileNotFoundError(f"找不到数据目录：{config.TRAIN_ROOT}")
    if not config.CACHE_ROOT.is_dir():
        raise FileNotFoundError(f"找不到 M1 冻结缓存目录：{config.CACHE_ROOT}")

    files = []
    for data_folder in sorted(path for path in config.TRAIN_ROOT.iterdir() if path.is_dir()):
        cache_folder = config.CACHE_ROOT / data_folder.name / "m1_cache"
        if not cache_folder.is_dir():
            continue
        files.extend(sorted(cache_folder.glob("*.npz"), key=lambda path: int(path.stem)))
    return files


def confidence_probability(conf_logits: np.ndarray) -> np.ndarray:
    """缓存保存的是 [背景, 前景] logits，转换为前景 softmax 概率。"""
    logits = np.asarray(conf_logits, dtype=np.float64)
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError(f"期望 conf.shape=[50,2]，实际为 {logits.shape}")
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits[:, 1] / exp_logits.sum(axis=-1)


def summarize(points: np.ndarray) -> dict[str, float | int | list[float]]:
    if points.size == 0:
        return {"count": 0}
    result: dict[str, float | int | list[float]] = {"count": int(len(points))}
    for axis, name in enumerate(("x", "y")):
        values = points[:, axis]
        result[f"{name}_min"] = float(np.min(values))
        result[f"{name}_max"] = float(np.max(values))
        result[f"{name}_mean"] = float(np.mean(values))
        result[f"{name}_std"] = float(np.std(values))
        result[f"{name}_quantiles_0_05_0_25_0_50_0_75_0_95"] = [
            float(value) for value in np.quantile(values, [0.05, 0.25, 0.5, 0.75, 0.95])
        ]
    return result


def main() -> None:
    files = discover_cache_files()
    if not files:
        raise RuntimeError(f"在 {config.CACHE_ROOT} 中没有发现任何 M1 缓存 .npz 文件")

    scores_by_frame: list[np.ndarray] = []
    coords_by_frame: list[np.ndarray] = []
    invalid_frames = 0
    for cache_path in files:
        with np.load(cache_path) as cache:
            if not {"out", "conf"}.issubset(cache.files):
                raise KeyError(f"缓存缺少 out/conf 字段：{cache_path}")
            output = np.asarray(cache["out"])
            if output.ndim != 2 or output.shape[1] < 2:
                raise ValueError(f"期望 out.shape=[50,>=2]，实际为 {output.shape}: {cache_path}")
            coords = output[:, :2].astype(np.float64, copy=False)
            scores = confidence_probability(cache["conf"])
        valid = np.isfinite(coords).all(axis=1) & np.isfinite(scores)
        invalid_frames += int((~valid).sum())
        coords_by_frame.append(coords[valid])
        scores_by_frame.append(scores[valid])

    all_coords = np.concatenate(coords_by_frame, axis=0)
    all_scores = np.concatenate(scores_by_frame, axis=0)
    results = []
    for threshold in THRESHOLDS:
        selected = all_scores < threshold
        stats = summarize(all_coords[selected])
        stats["threshold"] = threshold
        stats["fraction_of_all_measurements"] = float(np.mean(selected))
        results.append(stats)

    summary = {
        "dataset_split": "train",
        "is_cold": config.IS_COLD,
        "data_root": str(config.TRAIN_ROOT),
        "cache_root": str(config.CACHE_ROOT),
        "cache_frames": len(files),
        "measurements_per_frame_expected": 50,
        "valid_measurements": int(len(all_scores)),
        "invalid_measurements_skipped": invalid_frames,
        "confidence_definition": "softmax(conf_logits)[..., 1] (foreground probability)",
        "selection_definition": "score < threshold",
        "thresholds": results,
        "all_measurements_coordinate_summary": summarize(all_coords),
    }

    json_path = OUTPUT_DIR / "low_confidence_coordinate_distribution.json"
    csv_path = OUTPUT_DIR / "low_confidence_coordinate_distribution.csv"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        fieldnames = [
            "threshold", "count", "fraction_of_all_measurements",
            "x_min", "x_max", "x_mean", "x_std", "x_quantiles_0_05_0_25_0_50_0_75_0_95",
            "y_min", "y_max", "y_mean", "y_std", "y_quantiles_0_05_0_25_0_50_0_75_0_95",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"缓存帧数: {len(files):,}")
    print(f"有效量测数: {len(all_scores):,}；跳过非有限量测: {invalid_frames:,}")
    print("阈值表示前景置信度 score < threshold；坐标单位沿用 M1 输出（米）。")
    print("threshold | count | percent | x mean±std | y mean±std | x range | y range")
    for row in results:
        if row["count"] == 0:
            print(f"{row['threshold']:.2f} | 0 | 0.00% | - | - | - | -")
            continue
        print(
            f"{row['threshold']:.2f} | {row['count']:,} | "
            f"{row['fraction_of_all_measurements']:.2%} | "
            f"{row['x_mean']:.3f}±{row['x_std']:.3f} | "
            f"{row['y_mean']:.3f}±{row['y_std']:.3f} | "
            f"[{row['x_min']:.3f},{row['x_max']:.3f}] | "
            f"[{row['y_min']:.3f},{row['y_max']:.3f}]"
        )
    print(f"\nJSON: {json_path}")
    print(f"CSV:  {csv_path}")


if __name__ == "__main__":
    main()
