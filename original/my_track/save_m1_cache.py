r"""
M1 推理结果缓存脚本（冷启动预计算）
===================================
作用：
    用「冻结的 M1」（加载预训练权重 best_model_multilayerloss128.pth）对原始跟踪数据集
    （fast_dataset_multi_train / fast_dataset_multi_test）逐帧前向，把每个 query 的输出
        out  [50, 4]    = [coor(50,2), wh_out(50,2)]  最终回归输出（跟踪只用前 2 列作中心）
        conf [50, 2]    两分类 logits（[0]背景 / [1]有目标）
        feat [50, 128]  128 维身份特征（供跟踪 M4/M6 使用）
    保存为每个样本一个 .npz。之后训练/推理可用 is_cold 配置直接从缓存加载，跳过 M1 前向。

目录结构（与原始数据集同构：split → folder → 帧文件）：
    E:\Jupyter\NewMOTR数据集\
        fast_dataset_multi_train\
            20240913_1_3\
                m1_cache\
                    1.npz    2.npz    ...   （每个 npz 含 out/conf/feat 三个数组）
        fast_dataset_multi_test\ ...

运行（仓库根目录 E:\Jupyter\NewMOTR）：
    E:\Miniconda3\python.exe original/my_track/save_m1_cache.py                  # 全量
    E:\Miniconda3\python.exe original/my_track/save_m1_cache.py --split test --max-frames 20  # 试跑
    参数：--split train/test/both（默认 both）；--max-folders N；--max-frames N（每文件夹上限）；
          --batch N（默认 128）。已存在的 npz 会跳过（可断点续跑）。
"""
import argparse
import importlib
import os
import sys
import time

import numpy as np
import torch

# original/123ren.py 以数字开头，不能用普通 import，用 importlib 绕过
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO_ROOT)
_123ren = importlib.import_module('original.123ren')
Mymodule = _123ren.Mymodule

# ==================== 路径配置 ====================
# split 键 → (缓存子目录名, 原始数据根目录)。子目录名与原始数据集同名，便于 is_cold 映射。
SPLITS = {
    'train': ('fast_dataset_multi_train', r'E:\Jupyter\track\fast_dataset_multi_train'),
    'test':  ('fast_dataset_multi_test',  r'E:\Jupyter\track\fast_dataset_multi_test'),
}
OUT_ROOT    = r'E:\Jupyter\NewMOTR数据集'
WEIGHT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'best_model_multilayerloss128.pth')

BATCH = 128   # M1 批量化前向的帧数（显存不足可调小）


def load_frozen_m1(device):
    """加载预训练 M1 并冻结（与 train_tracker.py 同口径）。"""
    model = Mymodule(input_channels=8, freq_bins=89, time_steps=10).to(device)
    ckpt = torch.load(WEIGHT_PATH, map_location=device)
    sd = ckpt['model_state_dict']
    # 剥掉 thop profile 注入的伪键（total_ops / total_params）
    for k in [k for k in sd if k.endswith('total_ops') or k.endswith('total_params')]:
        sd.pop(k)
    missing, unexpected = model.load_state_dict(sd, strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def list_folder_frames(root, folder):
    """返回某个 folder 内按样本号排序的 (frame_id, signal_path) 列表。"""
    signals_dir = os.path.join(root, folder, 'signals')
    ids = sorted(int(f.split('.')[0]) for f in os.listdir(signals_dir) if f.endswith('.npy'))
    return [(fid, os.path.join(signals_dir, f'{fid}.npy')) for fid in ids]


def run_split(model, device, dir_name, root, max_folders, max_frames, batch):
    """对某个 split（train/test）下所有 folder 逐帧推理并缓存。"""
    out_root = os.path.join(OUT_ROOT, dir_name)
    # 只保留同时含 signals/ 与 labels/ 的文件夹（跳过“新建文件夹”等空占位目录）
    folders = sorted(d for d in os.listdir(root)
                     if os.path.isdir(os.path.join(root, d))
                     and os.path.isdir(os.path.join(root, d, 'signals'))
                     and os.path.isdir(os.path.join(root, d, 'labels')))
    if max_folders is not None:
        folders = folders[:max_folders]

    for folder in folders:
        frames = list_folder_frames(root, folder)          # [(id, path)]
        if max_frames is not None:
            frames = frames[:max_frames]

        cache_dir = os.path.join(out_root, folder, 'm1_cache')
        os.makedirs(cache_dir, exist_ok=True)

        # 断点续跑：跳过已存在的 npz
        todo = [(fid, sp) for fid, sp in frames
                if not os.path.exists(os.path.join(cache_dir, f'{fid}.npz'))]
        skipped = len(frames) - len(todo)
        if not todo:
            print(f'[{dir_name}/{folder}] 已全部缓存（{len(frames)} 帧），跳过', flush=True)
            continue

        t0 = time.time()
        done = 0
        # 批量化：一次前向处理 batch 帧，充分利用 GPU
        for i in range(0, len(todo), batch):
            chunk = todo[i:i + batch]
            signals = np.stack([np.load(sp) for _fid, sp in chunk])          # [b,8,89,10] complex64
            x = torch.from_numpy(signals).to(device)                          # complex64 直接入 M1
            with torch.no_grad():
                out, conf, feat = model(x)                                    # [b,50,4]/[b,50,2]/[b,50,128]
            out_np = out.detach().cpu().numpy().astype(np.float32)
            conf_np = conf.detach().cpu().numpy().astype(np.float32)
            feat_np = feat.detach().cpu().numpy().astype(np.float32)
            for j, (fid, _sp) in enumerate(chunk):
                np.savez(os.path.join(cache_dir, f'{fid}.npz'),
                         out=out_np[j], conf=conf_np[j], feat=feat_np[j])
            done += len(chunk)
            print(f'[{dir_name}/{folder}] {done}/{len(todo)} 帧  '
                  f'(跳过 {skipped}, 用时 {time.time() - t0:.0f}s)', flush=True)


def main():
    parser = argparse.ArgumentParser(description='M1 推理结果缓存')
    parser.add_argument('--split', default='both', choices=['train', 'test', 'both'])
    parser.add_argument('--max-folders', type=int, default=None)
    parser.add_argument('--max-frames', type=int, default=None)
    parser.add_argument('--batch', type=int, default=BATCH)
    args = parser.parse_args()

    # 控制台输出统一 UTF-8（避免 Windows GBK 控制台对中文进度报错/乱码）
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'设备: {device}', flush=True)
    model = load_frozen_m1(device)
    print('M1 已加载并冻结（eval 模式）', flush=True)

    splits = list(SPLITS.keys()) if args.split == 'both' else [args.split]
    for split in splits:
        dir_name, root = SPLITS[split]
        print(f'== 处理 split={split}  root={root} ==', flush=True)
        run_split(model, device, dir_name, root, args.max_folders, args.max_frames, args.batch)

    print(f'完成。缓存根目录: {OUT_ROOT}', flush=True)


if __name__ == '__main__':
    main()
