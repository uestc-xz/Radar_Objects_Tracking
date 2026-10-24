"""将低置信度量测坐标统计结果绘制为 PNG 表格。"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / "low_confidence_coordinate_distribution.csv"
JSON_PATH = HERE / "low_confidence_coordinate_distribution.json"


def load_rows() -> tuple[list[dict[str, str]], dict]:
    if not CSV_PATH.is_file() or not JSON_PATH.is_file():
        raise FileNotFoundError("请先运行 analyze_low_confidence_measurements.py 生成统计结果。")
    with CSV_PATH.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    with JSON_PATH.open(encoding="utf-8") as file:
        metadata = json.load(file)
    return rows, metadata


def make_table(title: str, columns: list[str], rows: list[list[str]], output: Path,
               widths: list[float] | None = None) -> None:
    fig, ax = plt.subplots(figsize=(max(10, len(columns) * 1.5), max(3.0, len(rows) * 0.34 + 1.5)))
    ax.axis("off")
    ax.set_title(title, fontsize=15, fontweight="bold", pad=18)
    table = ax.table(cellText=rows, colLabels=columns, cellLoc="center", loc="center",
                     colWidths=widths)
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    for (row, _col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#24476b")
            cell.set_text_props(color="white", weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#edf3f8")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def quantile_values(value: str) -> list[float]:
    return [float(item) for item in json.loads(value)]


def main() -> None:
    rows, metadata = load_rows()
    thresholds = [float(row["threshold"]) for row in rows]
    if not rows:
        raise RuntimeError("统计 CSV 为空。")

    key_rows = [rows[0], rows[len(rows) // 2], rows[-1]]
    key_table = []
    for row in key_rows:
        key_table.append([
            f"{float(row['threshold']):.2f}",
            f"{int(row['count']):,}",
            f"{float(row['fraction_of_all_measurements']):.2%}",
            f"{float(row['x_mean']):.4f} ± {float(row['x_std']):.4f}",
            f"{float(row['y_mean']):.4f} ± {float(row['y_std']):.4f}",
            f"[{float(row['x_min']):.3f}, {float(row['x_max']):.3f}]",
            f"[{float(row['y_min']):.3f}, {float(row['y_max']):.3f}]",
        ])
    make_table(
        "Low-confidence measurement coordinate summary (key thresholds)",
        ["Threshold", "Count", "All measurements", "X mean ± std (m)",
         "Y mean ± std (m)", "X range (m)", "Y range (m)"],
        key_table,
        HERE / "table_key_thresholds.png",
        [0.10, 0.13, 0.15, 0.18, 0.18, 0.14, 0.14],
    )

    quantile_table = []
    for row in key_rows:
        xq = quantile_values(row["x_quantiles_0_05_0_25_0_50_0_75_0_95"])
        yq = quantile_values(row["y_quantiles_0_05_0_25_0_50_0_75_0_95"])
        quantile_table.append([
            f"{float(row['threshold']):.2f}",
            *[f"{value:.4f}" for value in xq],
            *[f"{value:.4f}" for value in yq],
        ])
    make_table(
        "Coordinate quantiles of low-confidence measurements (meters)",
        ["Threshold", "X 5%", "X 25%", "X 50%", "X 75%", "X 95%",
         "Y 5%", "Y 25%", "Y 50%", "Y 75%", "Y 95%"],
        quantile_table,
        HERE / "table_coordinate_quantiles.png",
    )

    # 完整阈值序列表：保留每个 0.01 阈值，便于直接查看微小变化。
    full_rows = [[
        f"{float(row['threshold']):.2f}",
        f"{int(row['count']):,}",
        f"{float(row['fraction_of_all_measurements']):.2%}",
        f"{float(row['x_std']):.4f}",
        f"{float(row['y_std']):.4f}",
        f"{float(row['x_mean']):.4f}",
        f"{float(row['y_mean']):.4f}",
    ] for row in rows]
    make_table(
        f"All thresholds: {metadata['cache_frames']:,} cached frames, "
        f"{metadata['valid_measurements']:,} valid measurements",
        ["Threshold", "Count", "Share", "X std (m)", "Y std (m)", "X mean (m)", "Y mean (m)"],
        full_rows,
        HERE / "table_all_thresholds.png",
    )

    print("Generated tables:")
    for name in ("table_key_thresholds.png", "table_coordinate_quantiles.png", "table_all_thresholds.png"):
        print(HERE / name)


if __name__ == "__main__":
    main()
