r"""
M1 推理 vs 真值 对应情况审阅脚本
================================
作用：
    读取已保存的 M1 缓存（save_m1_cache.py 的产物），把每帧的「M1 推理量测」与
    「真值（label 中心）」做最近邻对应，输出一份审阅报告，供人工检查：
        - 每个真实目标，最近的 M1 量测是谁（coor / 置信度 / 距离）；
        - 每帧候选数（置信度 ≥ τ）与真实目标数是否一致；
        - 全数据集汇总统计（候选 vs 真值数量匹配、最近邻距离分布）。

对应关系口径：
    - 真值中心 = label(10,6) 对 10 个时间步平均后 reshape(3,2)，空槽 (0,0)。
    - 量测中心 = out[:, :2]（50 个 query 的中心坐标）。
    - 量测置信度 s_i = sigmoid(conf[:,1] - conf[:,0])（两分类 softmax 的目标通道）。
    - 候选 = s_i ≥ τ；τ 默认 0.96（与前端定位标定值一致）。

输出：
    E:\Jupyter\NewMOTR数据集\m1_vs_gt_review.txt  —— 逐帧明细（抽样）+ 全量汇总统计。

运行（仓库根目录 E:\Jupyter\NewMOTR）：
    E:\Miniconda3\python.exe original/my_track/review_m1_vs_gt.py
    参数：--threshold 0.96；--max-frames-per-folder N（每文件夹明细抽样帧数，默认 3）；
          --max-folders N（每 split 的文件夹数，None=全部）。
"""
import argparse
import os
import sys
import time

import numpy as np

# ==================== 路径配置（与 save_m1_cache.py 保持一致） ====================
SPLITS = {
    'train': ('fast_dataset_multi_train', r'E:\Jupyter\track\fast_dataset_multi_train'),
    'test':  ('fast_dataset_multi_test',  r'E:\Jupyter\track\fast_dataset_multi_test'),
}
CACHE_ROOT = r'E:\Jupyter\NewMOTR数据集'
REPORT_PATH = os.path.join(CACHE_ROOT, 'm1_vs_gt_review.txt')


def sigmoid_score(conf):
    """两分类 logits → 目标置信度 s_i = sigmoid(conf1 - conf0)，稳定数值。"""
    return 1.0 / (1.0 + np.exp(conf[:, 0] - conf[:, 1]))


def load_frame(cache_dir, label_dir, fid):
    """读取一帧的缓存与标签，返回 (coor[50,2], score[50], gt_valid[n,2])。"""
    d = np.load(os.path.join(cache_dir, f'{fid}.npz'))
    coor = d['out'][:, :2].astype(np.float32)
    score = sigmoid_score(d['conf'].astype(np.float32))
    lab = np.load(os.path.join(label_dir, f'{fid}.npy')).mean(0).reshape(3, 2)
    valid_mask = np.linalg.norm(lab, axis=1) > 1e-3          # 空槽 (0,0) 剔除
    gt_valid = lab[valid_mask]
    return coor, score, gt_valid


def nearest_gt_dist(coor, score, gt, threshold):
    """返回 (候选索引, 每个候选到最近 GT 的距离, 每个 GT 到最近候选的距离)。"""
    cand_idx = np.where(score >= threshold)[0]
    cand_dist = []          # 每个候选 → 最近 GT 距离
    for i in cand_idx:
        d = np.linalg.norm(gt - coor[i], axis=1)
        cand_dist.append(d.min())
    gt_dist = []            # 每个 GT → 最近候选距离（覆盖误差）
    for g in gt:
        if len(cand_idx) == 0:
            gt_dist.append(float('inf'))
        else:
            gt_dist.append(np.linalg.norm(coor[cand_idx] - g, axis=1).min())
    return cand_idx, cand_dist, gt_dist


def main():
    parser = argparse.ArgumentParser(description='M1 vs GT 对应审阅')
    parser.add_argument('--threshold', type=float, default=0.96)
    parser.add_argument('--max-frames-per-folder', type=int, default=3)
    parser.add_argument('--max-folders', type=int, default=None)
    args = parser.parse_args()

    # 控制台输出统一 UTF-8（避免 Windows GBK 控制台对 ✓/中文报 UnicodeEncodeError）
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

    t0 = time.time()
    lines = []
    lines.append('=' * 90)
    lines.append('M1 推理 vs 真值 对应情况审阅报告')
    lines.append(f'生成时间: {time.strftime("%Y-%m-%d %H:%M:%S")}   阈值 τ = {args.threshold}')
    lines.append('=' * 90)

    # 全量汇总统计累加器
    n_frames = 0
    n_gt = 0
    n_cand = 0
    cnt_eq = cnt_less = cnt_more = 0
    cand_dist_all = []      # 候选→最近GT 距离
    gt_dist_all = []        # GT→最近候选 距离（不含 inf）

    for split, (subdir, label_root) in SPLITS.items():
        cache_root = os.path.join(CACHE_ROOT, subdir)
        if not os.path.isdir(cache_root):
            lines.append(f'\n[split={split}] 缓存不存在，跳过: {cache_root}')
            continue
        folders = sorted(d for d in os.listdir(cache_root)
                         if os.path.isdir(os.path.join(cache_root, d)))
        if args.max_folders is not None:
            folders = folders[:args.max_folders]

        for folder in folders:
            cache_dir = os.path.join(cache_root, folder, 'm1_cache')
            label_dir = os.path.join(label_root, folder, 'labels')
            if not (os.path.isdir(cache_dir) and os.path.isdir(label_dir)):
                continue
            fids = sorted(int(f.split('.')[0]) for f in os.listdir(cache_dir) if f.endswith('.npz'))

            lines.append(f'\n{"-" * 90}')
            lines.append(f'[{split}/{folder}] 共 {len(fids)} 帧')
            lines.append(f'{"-" * 90}')

            detail_ids = fids[:args.max_frames_per_folder]   # 只对前 N 帧打印明细
            for fid in fids:
                coor, score, gt = load_frame(cache_dir, label_dir, fid)
                cand_idx, cand_dist, gt_dist = nearest_gt_dist(coor, score, gt, args.threshold)

                # 汇总统计
                n_frames += 1
                n_gt += len(gt)
                n_cand += len(cand_idx)
                if len(cand_idx) == len(gt):
                    cnt_eq += 1
                elif len(cand_idx) < len(gt):
                    cnt_less += 1
                else:
                    cnt_more += 1
                cand_dist_all += cand_dist
                gt_dist_all += [d for d in gt_dist if np.isfinite(d)]

                # 明细（抽样）
                if fid in detail_ids:
                    lines.append(f'\n  frame={fid}  GT数={len(gt)}  候选数(≥τ)={len(cand_idx)}')
                    for j, g in enumerate(gt):
                        if len(cand_idx) == 0:
                            lines.append(f'    GT#{j} {tuple(g.round(3))}  <- 无候选')
                            continue
                        di = np.linalg.norm(coor[cand_idx] - g, axis=1)
                        bi = cand_idx[int(np.argmin(di))]
                        mark = '✓' if score[bi] >= args.threshold else ' '
                        lines.append(f'    GT#{j} {tuple(g.round(3))}  <- q{int(bi):2d} '
                                     f'coor={tuple(coor[bi].round(3))} conf={score[bi]:.4f} '
                                     f'dist={di.min():.4f} {mark}')

    # ---- 汇总统计 ----
    lines.append('\n' + '=' * 90)
    lines.append('全量汇总统计')
    lines.append('=' * 90)
    lines.append(f'统计帧数        : {n_frames}')
    lines.append(f'真实目标总数    : {n_gt}')
    lines.append(f'候选量测总数(≥τ): {n_cand}')
    lines.append(f'候选数==GT数    : {cnt_eq} 帧  ({100.0 * cnt_eq / n_frames:.1f}%)' if n_frames else '候选数==GT数: 0')
    lines.append(f'候选数< GT数    : {cnt_less} 帧  ({100.0 * cnt_less / n_frames:.1f}%)' if n_frames else '候选数<GT数: 0')
    lines.append(f'候选数> GT数    : {cnt_more} 帧  ({100.0 * cnt_more / n_frames:.1f}%)' if n_frames else '候选数>GT数: 0')
    if cand_dist_all:
        a = np.array(cand_dist_all)
        lines.append(f'候选→最近GT 距离: 均值={a.mean():.4f}  中位={np.median(a):.4f}  最大={a.max():.4f}')
    if gt_dist_all:
        a = np.array(gt_dist_all)
        lines.append(f'GT→最近候选 距离(覆盖误差): 均值={a.mean():.4f}  中位={np.median(a):.4f}  最大={a.max():.4f}')

    report = '\n'.join(lines) + '\n'
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f'报告已保存: {REPORT_PATH}', flush=True)
    print(f'  统计帧数={n_frames}  真实目标={n_gt}  候选(≥τ)={n_cand}  '
          f'候选==GT:{cnt_eq}  <GT:{cnt_less}  >GT:{cnt_more}  (用时 {time.time() - t0:.0f}s)',
          flush=True)


if __name__ == '__main__':
    main()
