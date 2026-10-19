"""
跟踪序列数据集构建器 (TrackSequenceDataset)
============================================
作用：
    把每个文件夹（= 一条长序列，通常上千样本）按文件名数字排序，切成「滑窗」短序列：
    窗口长度 window_size=20、步长 stride=10（相邻窗口 overlap = window_size - stride = 10）。
    每个窗口 = 一个训练样本，供跟踪网络（M1~M8）逐帧自回归训练。

与定位数据集 FastNpyDataset1 的区别：
    - 定位：单帧打乱，返回 (signal(8,89,10), label(10,6))。
    - 跟踪：窗口内帧保持时间顺序、不 shuffle，返回
            (signals[T,8,89,10] complex64, labels[T,3,2] float32)，
      labels 已对 10 个时间步平均成 (3,2) 中心坐标（跟踪只用中心，不加 w/h）。

遮挡：
    - 跟踪任务的核心就是「短时遮挡下维持 ID」，故默认 delete_occlusion=False（保留遮挡帧）。
    - 若 delete_occlusion=True 会剔除遮挡帧，导致窗口内时间不连续（破坏 Δt=0.24s 的
      速度假设），仅用于对照实验，正常跟踪训练不要开。

约定：
    - 帧间隔 Δt = 0.24s（相邻 .npy 之间），速度由相邻帧中心差 / Δt 得到。
    - 每个文件夹内 GT 槽位顺序固定 = 序列内伪 ID。
"""
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset


class TrackSequenceDataset(Dataset):
    """滑窗式跟踪序列数据集（窗口内有序，窗口间可 shuffle）。"""

    def __init__(self, data_root_folder, window_size=20, stride=10,
                 add_noise=True, delete_occlusion=False):
        self.data_root_folder = data_root_folder
        self.window_size = window_size
        self.stride = stride
        self.add_noise = add_noise
        self.delete_occlusion = delete_occlusion

        # 每个文件夹 → 有序帧列表 [(signal_path, label_path, sample_id), ...]
        self.folders = self._build_folder_frame_lists()
        # 滑窗切分 → [(folder_idx, start_frame_idx), ...]
        self.windows = self._build_windows()

        print(f"跟踪序列数据集加载完成：{len(self.folders)} 个文件夹，{len(self.windows)} 个窗口")
        print(f"  窗口长度={self.window_size}，步长={self.stride}"
              f"（overlap={self.window_size - self.stride}）")
        print(f"  数据根目录: {data_root_folder}")

    # ------------------------------------------------------------------
    def _build_folder_frame_lists(self):
        """遍历每个文件夹，返回每个文件夹内按样本号排序的帧列表。"""
        folders = []
        for file_folder in sorted(os.listdir(self.data_root_folder)):
            folder_path = os.path.join(self.data_root_folder, file_folder)
            if not os.path.isdir(folder_path):
                continue
            signals_dir = os.path.join(folder_path, "signals")
            labels_dir = os.path.join(folder_path, "labels")
            if not (os.path.exists(signals_dir) and os.path.exists(labels_dir)):
                continue

            # 按文件名中的数字排序（= 时间顺序）
            signal_files = sorted(
                [f for f in os.listdir(signals_dir) if f.endswith('.npy')],
                key=lambda x: int(x.split('.')[0]))
            label_files = sorted(
                [f for f in os.listdir(labels_dir) if f.endswith('.npy')],
                key=lambda x: int(x.split('.')[0]))
            if len(signal_files) != len(label_files):
                print(f"警告: {file_folder} 信号/标签数量不一致，跳过")
                continue

            frames = []
            for signal_file in signal_files:
                signal_path = os.path.join(signals_dir, signal_file)
                label_path = os.path.join(labels_dir, signal_file)
                if not os.path.exists(label_path):
                    print(f"警告: 缺标签 {label_path}，跳过该帧")
                    continue
                sample_id = int(signal_file.split('.')[0])
                frames.append((signal_path, label_path, sample_id))

            # 可选：剔除遮挡帧（默认 False，跟踪需保留遮挡）
            if self.delete_occlusion:
                frames = self._drop_occluded_frames(file_folder, frames)

            if len(frames) >= self.window_size:
                folders.append({'folder': file_folder, 'frames': frames})
            else:
                print(f"跳过 {file_folder}：帧数 {len(frames)} < 窗口长度 {self.window_size}")

        return folders

    def _drop_occluded_frames(self, file_folder, frames):
        """按遮挡样本统计文件剔除遮挡帧（与 FastNpyDataset1 同口径）。"""
        occlusion_file = os.path.join(self.data_root_folder, file_folder, "遮挡样本id统计.txt")
        occluded_ids = set()
        if os.path.exists(occlusion_file):
            with open(occlusion_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            for line in lines[1:]:  # 跳过标题行
                if line.strip():
                    try:
                        occluded_ids.add(int(line.strip()))
                    except ValueError:
                        continue
        else:
            print(f"警告: 未找到遮挡统计文件 {occlusion_file}，保留全部帧")
        return [fr for fr in frames if fr[2] not in occluded_ids]

    def _build_windows(self):
        """对每个文件夹做滑窗切分，返回 [(folder_idx, start_frame_idx), ...]。"""
        windows = []
        for folder_idx, folder in enumerate(self.folders):
            n = len(folder['frames'])
            # start = 0, stride, 2*stride, ... 直到放不下一个完整窗口
            for start in range(0, n - self.window_size + 1, self.stride):
                windows.append((folder_idx, start))
        return windows

    # ------------------------------------------------------------------
    def _add_gaussian_noise(self, signal):
        """复数信号加高斯噪声（与定位数据集一致：50% 概率、std=0.03、实/虚分别加）。"""
        if random.random() > 0.5:
            return signal
        noise_std = 0.03
        real_noise = torch.randn_like(signal.real) * noise_std
        imag_noise = torch.randn_like(signal.imag) * noise_std
        return signal + torch.complex(real_noise, imag_noise)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        """返回一个窗口：signals [T,8,89,10] complex64, labels [T,3,2] float32。"""
        folder_idx, start = self.windows[idx]
        frames = self.folders[folder_idx]['frames'][start: start + self.window_size]

        signals, labels = [], []
        for signal_path, label_path, _sample_id in frames:
            signal = torch.from_numpy(np.load(signal_path))   # (8,89,10) complex64
            label = torch.from_numpy(np.load(label_path))     # (10,6) float32

            if self.add_noise:
                signal = self._add_gaussian_noise(signal)

            # 对 10 个时间步平均 -> (6,) -> (3,2) 中心坐标
            center = label.mean(dim=0).reshape(-1, 2)         # (3,2)
            signals.append(signal)
            labels.append(center)

        return torch.stack(signals, dim=0), torch.stack(labels, dim=0)   # [T,8,89,10], [T,3,2]


def load_track_sequence_dataset(data_root_folder, window_size=20, stride=10,
                                add_noise=True, delete_occlusion=False):
    """加载跟踪序列数据集（接口与 load_fast_dataset1 保持一致）。"""
    print(f'加载跟踪序列数据集 (滑窗 {window_size}/{stride})')
    print(f'数据目录: {data_root_folder}')
    return TrackSequenceDataset(data_root_folder, window_size, stride, add_noise, delete_occlusion)
