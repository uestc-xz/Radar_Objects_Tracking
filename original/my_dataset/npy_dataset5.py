# import os
# import numpy as np
# import pandas as pd
# import torch
# from torch.utils.data import Dataset
# from tqdm import tqdm
#
#
# class NpyDataset5(Dataset):
#     def __init__(self, npy_folder, xlsx_folder, seq_length=50, overlap_step=5):
#         """
#         带重叠窗口的数据集类（处理.npy格式信号数据）
#
#         参数:
#             npy_folder: .npy文件路径（信号数据）
#             xlsx_folder: .xlsx文件路径（标签数据）
#             seq_length: 时间序列长度（必须是5的倍数，如50=10×5）
#             overlap_step: 相邻序列的重叠步长（必须小于seq_length）
#         """
#         assert seq_length % 5 == 0, "seq_length必须是5的倍数"
#         assert overlap_step < seq_length, "重叠步长必须小于序列长度"
#
#         self.npy_folder = npy_folder
#         self.xlsx_folder = xlsx_folder
#         self.seq_length = seq_length
#         self.overlap_step = overlap_step
#         self.label_repeat = seq_length // 5  # 每个序列需要的标签行数
#         self.file_pairs = self._collect_file_pairs()
#         self.samples = self._prepare_samples()
#
#     def _collect_file_pairs(self):
#         """收集匹配的文件对（.npy和.xlsx）"""
#         npy_files = [f for f in os.listdir(self.npy_folder) if f.endswith('.npy')]
#         return [
#             (f, os.path.splitext(f)[0] + '.xlsx')
#             for f in npy_files
#             if os.path.exists(os.path.join(self.xlsx_folder, os.path.splitext(f)[0] + '.xlsx'))
#         ]
#
#     def _prepare_samples(self):
#         """准备带重叠的样本数据（从.npy加载信号）"""
#         samples = []
#
#         for npy_file, xlsx_file in tqdm(self.file_pairs, desc="Processing files"):
#             # 加载信号数据 (8x300xN) —— 直接从.npy读取
#             signal = np.load(os.path.join(self.npy_folder, npy_file))
#             ## 这一步选择通道
#             # selected_channels = [0, 3,4, 7]
#             # signal = signal[selected_channels,:,:]
#             ##
#             # 加载标签数据 (Mx2)
#             labels = pd.read_excel(
#                 os.path.join(self.xlsx_folder, xlsx_file),
#                 usecols=range(2),
#                 header=None
#             ).values
#
#             num_time_steps = signal.shape[2]
#             num_labels = labels.shape[0]
#
#             # 计算滑动窗口参数
#             step = self.seq_length - self.overlap_step
#             num_sequences = (num_time_steps - self.seq_length) // step + 1
#
#             for seq_idx in range(num_sequences):
#                 # 信号窗口位置
#                 signal_start = seq_idx * step
#                 signal_end = signal_start + self.seq_length
#
#                 # 标签行范围（每5步对应1行标签）
#                 label_start = signal_start // 5
#                 label_end = label_start + self.label_repeat
#
#                 if label_end > num_labels:  # 跳过越界标签
#                     continue
#
#                 # 提取信号 (8x300xseq_length)
#                 signal_seq = signal[:, :, signal_start:signal_end]
#
#                 # 提取并扩展标签 (seq_lengthx2)
#                 label_rows = labels[label_start:label_end]
#                 label_seq = np.repeat(label_rows, 5, axis=0)[:self.seq_length]
#
#                 samples.append((
#                     torch.from_numpy(signal_seq).to(torch.complex64),  # 假设信号是复数
#                     torch.from_numpy(label_seq).to(torch.float32)
#                 ))
#
#         return samples
#
#     def __len__(self):
#         return len(self.samples)
#
#     def __getitem__(self, idx):
#         return self.samples[idx]
#
#
# def load_dataset5_npy(npy_folder_path, xls_folder_path, seq_length, overlap_step):
#     print('NpyDataset5_npy (支持.npy格式)')
#     return NpyDataset5(npy_folder_path, xls_folder_path, seq_length, overlap_step)

import os
import numpy as np
import pandas as pd
import torch
import random
from torch.utils.data import Dataset
from tqdm import tqdm


class NpyDataset5(Dataset):
    def __init__(self, npy_folder, xlsx_folder, seq_length=50, overlap_step=5,add_noise = True):
        assert seq_length % 5 == 0, "seq_length必须是5的倍数"
        assert overlap_step < seq_length, "重叠步长必须小于序列长度"

        self.npy_folder = npy_folder
        self.xlsx_folder = xlsx_folder
        self.seq_length = seq_length
        self.overlap_step = overlap_step
        self.label_repeat = seq_length // 5  # 每个序列需要的标签行数
        self.file_pairs = self._collect_file_pairs()
        self.samples = self._prepare_samples()
        self.add_noise = add_noise
    def _collect_file_pairs(self):
        npy_files = [f for f in os.listdir(self.npy_folder) if f.endswith('.npy')]
        # print(f"找到 {len(npy_files)} 个npy文件: {npy_files}")

        # 先计算匹配结果用于调试
        matched_pairs = [
            (f, os.path.splitext(f)[0] + '.xlsx')
            for f in npy_files
            if os.path.exists(os.path.join(self.xlsx_folder, os.path.splitext(f)[0] + '.xlsx'))
        ]

        return [
            (f, os.path.splitext(f)[0] + '.xlsx')
            for f in npy_files
            if os.path.exists(os.path.join(self.xlsx_folder, os.path.splitext(f)[0] + '.xlsx'))
        ]

    def _prepare_samples(self):
        samples = []

        for npy_file, xlsx_file in tqdm(self.file_pairs, desc="Processing files"):
            # 1. 加载信号数据 (8x300xN)
            # print(f"file:{npy_file}")
            parts = npy_file.split('_')
            num_people = int(parts[1])

            print(f"人数：{num_people}")
            signal = np.load(os.path.join(self.npy_folder, npy_file))
            print(signal.shape[2])
            num_time_steps = signal.shape[2]  # 信号总时间步数（N）
            # selected_channels = [0, 3,4, 7]
            # signal = signal[selected_channels,:,:]
            # 2. 加载标签数据并过滤全零行
            raw_labels = pd.read_excel(
                os.path.join(self.xlsx_folder, xlsx_file),
                usecols=range(6),
                header=None
            ).values  # 原始标签形状：(原始总行数, 2)

            # 关键修改1：过滤全零行（x=0且y=0的行）
            # 标记全零行：每行x和y都为0则视为无效
            is_zero_row = (raw_labels == 0).all(axis=1)  # 布尔数组：True表示全零行
            valid_labels = raw_labels[~is_zero_row]  # 过滤后有效标签：(有效行数, 2)
            zero_row_count = sum(is_zero_row)  # 全零行数量（需要跳过的行数）
            num_valid_labels = valid_labels.shape[0]  # 有效标签行数

            # 如果没有有效标签，跳过当前文件
            if num_valid_labels == 0:
                print(f"警告：文件 {xlsx_file} 没有有效标签（全为零行），已跳过")
                continue

            # 关键修改2：计算信号中需要跳过的时间步（全零行对应5个时间步/行）
            skipped_time_steps = zero_row_count * 5  # 前N个时间步对应全零标签，需跳过
            # 有效信号的起始时间步（从跳过无效部分后开始）
            valid_signal_start = skipped_time_steps
            # 有效信号的总时间步数（剩余可用于匹配有效标签的时间）
            valid_signal_length = num_time_steps - valid_signal_start

            # 3. 计算滑动窗口参数（基于有效信号长度）
            step = self.seq_length - self.overlap_step
            # 可生成的样本数量（基于有效信号长度计算）
            num_sequences = (valid_signal_length - self.seq_length) // step + 1 if valid_signal_length >= self.seq_length else 0

            if num_sequences <= 0:
                print(f"警告：文件 {npy_file} 有效信号长度不足，无法生成样本，已跳过")
                continue

            # 4. 切分信号序列与对应有效标签
            for seq_idx in range(num_sequences):
                # 信号窗口在原始信号中的位置（需加上跳过的时间步）
                signal_start = valid_signal_start + seq_idx * step  # 有效信号起始点 + 窗口偏移
                signal_end = signal_start + self.seq_length

                # 关键修改3：计算对应有效标签的行索引（跳过全零行后）
                # 原始逻辑：signal_start对应的标签行 = signal_start // 5
                # 修正后：减去全零行数量，得到有效标签中的行索引
                original_label_start = signal_start // 5  # 原始标签行索引（包含全零行）
                valid_label_start = original_label_start - zero_row_count  # 有效标签行索引
                valid_label_end = valid_label_start + self.label_repeat  # 有效标签结束行

                # 检查有效标签是否越界（避免超出有效标签行数）
                if valid_label_end > num_valid_labels:
                    continue

                # 提取信号序列（原始信号中切分）
                signal_seq = signal[:, :, signal_start:signal_end]

                # 提取有效标签行并扩展到信号时间步
                valid_label_rows = valid_labels[valid_label_start:valid_label_end]  # 从有效标签中取行
                label_seq = np.repeat(valid_label_rows, 5, axis=0)[:self.seq_length]  # 扩展到seq_length长度
                if (num_people==1):
                    label_seq[:,1] = label_seq[:,1] + 3.5
                    # label_seq[:, 3] = label_seq[:, 3]
                    # label_seq[:, 5] = label_seq[:, 5]
                    # label_seq[:,6] = num_people-1
                    # label_seq[:, 7] = 0
                    # label_seq[:, 8] = 0
                    # print(f"人数：{num_people}")
                if(num_people==2):
                    label_seq[:, 1] = label_seq[:, 1] + 3.5
                    label_seq[:, 3] = label_seq[:, 3] + 3.5
                    # label_seq[:, 5] = label_seq[:, 5]
                    # label_seq[:, 6] = num_people-1
                    # label_seq[:, 7] = num_people
                    # label_seq[:, 8] = 0
                    # print(f"人数：{num_people}")
                if(num_people==3):
                    label_seq[:, 1] = label_seq[:, 1] + 3.5
                    label_seq[:, 3] = label_seq[:, 3] + 3.5
                    label_seq[:, 5] = label_seq[:, 5] + 3.5
                    # label_seq[:, 6] = num_people-1
                    # label_seq[:, 7] = 0
                    # label_seq[:, 8] = num_people
                    # print(f"人数：{num_people}")
                samples.append((
                    torch.from_numpy(signal_seq).to(torch.complex64),
                    torch.from_numpy(label_seq).to(torch.float32)
                ))

        return samples

    def _add_gaussian_noise(self, signal):
        """为复数信号添加高斯噪声"""
        # 固定参数：50%概率添加噪声，标准差为0.01
        if random.random() > 0.5:
            return signal

        # 为实部和虚部分别添加独立的高斯噪声
        noise_std = 0.03
        real_noise = torch.randn_like(signal.real) * noise_std
        imag_noise = torch.randn_like(signal.imag) * noise_std
        noisy_signal = signal + torch.complex(real_noise, imag_noise)

        return noisy_signal


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        signal, label = self.samples[idx]
        signal_magnitude = torch.abs(signal)  # (8, 89, 10)

        # 沿着频率维度(第1维)找最大值，得到每个通道每个时间步的最大值 (8, 1, 10)
        max_per_channel_timestep = torch.max(signal_magnitude, dim=1, keepdim=True)[0]

        # 每个通道的每个时间步除以其对应的最大值
        signal = signal / (max_per_channel_timestep )
        # 根据add_noise参数决定是否添加噪声
        if self.add_noise:
            processed_signal = self._add_gaussian_noise(signal)
        else:
            processed_signal = signal  # 不加噪声，直接返回原信号

        return processed_signal, label



def load_dataset5_npy(npy_folder_path, xls_folder_path, seq_length, overlap_step,add_noise):
    print('NpyDataset5_npy (支持.npy格式，已处理全零标签行)')
    return NpyDataset5(npy_folder_path, xls_folder_path, seq_length, overlap_step,add_noise)