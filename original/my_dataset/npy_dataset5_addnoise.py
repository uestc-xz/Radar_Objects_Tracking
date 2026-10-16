import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class NpyDataset5_addnoise(Dataset):
    def __init__(self, npy_folder, xlsx_folder, seq_length=50, overlap_step=5,
                 is_train=True, snr_range=(5, 30)):
        """
        带重叠窗口和噪声增强的数据集类（处理.npy格式信号数据）

        参数:
            npy_folder: .npy文件路径（信号数据）
            xlsx_folder: .xlsx文件路径（标签数据）
            seq_length: 时间序列长度（必须是5的倍数，如50=10×5）
            overlap_step: 相邻序列的重叠步长（必须小于seq_length）
            is_train: 是否为训练集（仅训练集添加噪声）
            snr_range: 信噪比范围（dB），格式为(最小值, 最大值)
        """
        assert seq_length % 5 == 0, "seq_length必须是5的倍数"
        assert overlap_step < seq_length, "重叠步长必须小于序列长度"
        assert snr_range[0] < snr_range[1], "信噪比范围最小值必须小于最大值"

        self.npy_folder = npy_folder
        self.xlsx_folder = xlsx_folder
        self.seq_length = seq_length
        self.overlap_step = overlap_step
        self.label_repeat = seq_length // 5  # 每个序列需要的标签行数
        self.is_train = is_train
        self.snr_range = snr_range
        self.file_pairs = self._collect_file_pairs()
        self.samples = self._prepare_samples()

    def _add_gaussian_noise(self, signal):
        """为复数信号添加高斯白噪声"""
        # 将复数信号分离为实部和虚部
        signal_real = signal.real
        signal_imag = signal.imag

        # 计算信号功率（实部和虚部分别计算）
        power_real = np.var(signal_real)
        power_imag = np.var(signal_imag)

        # 避免信号功率为0导致的数值问题
        if power_real < 1e-10 or power_imag < 1e-10:
            return signal

        # 随机生成当前样本的信噪比
        snr = np.random.uniform(self.snr_range[0], self.snr_range[1])

        # 根据信噪比计算噪声功率
        noise_power_real = power_real / (10 ** (snr / 10))
        noise_power_imag = power_imag / (10 ** (snr / 10))

        # 生成高斯噪声
        noise_real = np.random.normal(0, np.sqrt(noise_power_real), signal_real.shape)
        noise_imag = np.random.normal(0, np.sqrt(noise_power_imag), signal_imag.shape)

        # 添加噪声并合并为复数信号
        noisy_signal = (signal_real + noise_real) + 1j * (signal_imag + noise_imag)
        return noisy_signal

    def _collect_file_pairs(self):
        """收集匹配的文件对（.npy和.xlsx）"""
        npy_files = [f for f in os.listdir(self.npy_folder) if f.endswith('.npy')]
        return [
            (f, os.path.splitext(f)[0] + '.xlsx')
            for f in npy_files
            if os.path.exists(os.path.join(self.xlsx_folder, os.path.splitext(f)[0] + '.xlsx'))
        ]

    def _prepare_samples(self):
        """准备带重叠的样本数据（从.npy加载信号）"""
        samples = []

        for npy_file, xlsx_file in tqdm(self.file_pairs, desc="Processing files"):
            # 加载信号数据 (8x300xN) —— 直接从.npy读取
            signal = np.load(os.path.join(self.npy_folder, npy_file))

            # 加载标签数据 (Mx2)
            labels = pd.read_excel(
                os.path.join(self.xlsx_folder, xlsx_file),
                usecols=range(6),
                header=None
            ).values

            num_time_steps = signal.shape[2]
            num_labels = labels.shape[0]

            # 计算滑动窗口参数
            step = self.seq_length - self.overlap_step
            num_sequences = (num_time_steps - self.seq_length) // step + 1

            for seq_idx in range(num_sequences):
                # 信号窗口位置
                signal_start = seq_idx * step
                signal_end = signal_start + self.seq_length

                # 标签行范围（每5步对应1行标签）
                label_start = signal_start // 5
                label_end = label_start + self.label_repeat

                if label_end > num_labels:  # 跳过越界标签
                    continue

                # 提取信号 (8x300xseq_length)
                signal_seq = signal[:, :, signal_start:signal_end]

                # 训练集添加噪声
                if self.is_train:
                    signal_seq = self._add_gaussian_noise(signal_seq)

                # 提取并扩展标签 (seq_lengthx6)
                label_rows = labels[label_start:label_end]
                label_seq = np.repeat(label_rows, 5, axis=0)[:self.seq_length]

                samples.append((
                    torch.from_numpy(signal_seq).to(torch.complex64),  # 复数信号
                    torch.from_numpy(label_seq).to(torch.float32)
                ))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def load_dataset5_addnoise(npy_folder_path, xls_folder_path, seq_length, overlap_step, is_train=True):
    print('NpyDataset5_addnoise (支持.npy格式和噪声增强)')
    return NpyDataset5_addnoise(
        npy_folder_path,
        xls_folder_path,
        seq_length,
        overlap_step,
        is_train=is_train
    )
