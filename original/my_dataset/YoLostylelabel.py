
import os
import numpy as np
import pandas as pd
import torch
import random
from torch.utils.data import Dataset
from tqdm import tqdm


class MyYoLoSytle(Dataset):
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

    def labels_to_yolo_format(self,num_people, avg_label, grid_size=20, area_range=(-2.6, 2.6, -2.6, 2.6), box_size=(0.3, 0.4)):
        """
        将原始标签数据转换为YOLO格式的20x20x6张量
        参数:
        - num_people: 当前帧中的人数 (整数)
        - coordinates: 目标中心坐标，形状为(2 * num_people)的一维数组 [x1, y1, x2, y2, ...]
        - grid_size: 网格大小，默认20x20
        - area_range: 探测区域范围 (x_min, x_max, y_min, y_max)
        - box_size: 固定边界框尺寸 (width, height)，单位：米
        返回:
        - yolo_output: 形状为(grid_size, grid_size, 6)的numpy数组
        """
        x_min, x_max, y_min, y_max = area_range
        width, height = box_size

        # 初始化输出张量 (x_offset, y_offset, w_norm, h_norm, confidence, class)
        yolo_output = np.zeros((grid_size, grid_size, 6), dtype=np.float32)

        # 计算每个网格的物理尺寸
        cell_width = (x_max - x_min) / grid_size
        cell_height = (y_max - y_min) / grid_size

        # 整个探测区域的宽度和高度（用于归一化w,h）
        area_width = x_max - x_min
        area_height = y_max - y_min
        if num_people == 1:
            coordinates = avg_label[0:2]  # 取前2个元素 [x1, y1]
        elif num_people == 2:
            coordinates = avg_label[0:4]  # 取前4个元素 [x1, y1, x2, y2]
        elif num_people == 3:
            coordinates = avg_label[0:6]  # 取前6个元素 [x1, y1, x2, y2, x3, y3]
        else:
            coordinates = np.array([])  # 无人情况
        # 处理每个目标
        for i in range(num_people):
            x = coordinates[2 * i]
            y = coordinates[2 * i + 1]

            # 检查坐标是否在有效范围内
            if not (x_min <= x <= x_max and y_min <= y <= y_max):
                print(f"警告: 坐标({x}, {y})超出探测区域，已跳过")
                continue

            # 计算目标所在的网格索引
            grid_i = int((x - x_min) / cell_width)
            grid_j = int((y - y_min) / cell_height)

            # 确保索引在有效范围内
            grid_i = max(0, min(grid_i, grid_size - 1))
            grid_j = max(0, min(grid_j, grid_size - 1))

            # 计算该网格的物理边界
            cell_x_min = x_min + grid_i * cell_width
            cell_y_min = y_min + grid_j * cell_height

            # 计算相对于网格左上角的偏移量 (归一化到0-1)
            x_offset = (x - cell_x_min) / cell_width
            y_offset = (y - cell_y_min) / cell_height

            # 归一化宽高 (相对于整个探测区域)
            w_norm = width / area_width
            h_norm = height / area_height

            # 填充YOLO输出
            yolo_output[grid_j, grid_i, 0] = x_offset  # 注意: 网格索引是 [行, 列]，即 [y, x]
            yolo_output[grid_j, grid_i, 1] = y_offset
            yolo_output[grid_j, grid_i, 2] = w_norm
            yolo_output[grid_j, grid_i, 3] = h_norm
            yolo_output[grid_j, grid_i, 4] = 1.0  # confidence
            # yolo_output[grid_j, grid_i, 5] = 1.0  # class (有目标)
            if num_people == 1:
                yolo_output[grid_j, grid_i, 5] = 0.0  # 1人 → 0
            elif num_people == 2:
                yolo_output[grid_j, grid_i, 5] = 1.0  # 2人 → 1
            elif num_people == 3:
                yolo_output[grid_j, grid_i, 5] = 2.0  # 3人 → 2
        return yolo_output
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
            # print(f"人数：{num_people}")
            signal = np.load(os.path.join(self.npy_folder, npy_file))
            num_time_steps = signal.shape[2]  # 信号总时间步数（N）
            # selected_channels = [0, 3,4, 7]
            # signal = signal[selected_channels,:,:]
            # 2. 加载标签数据并过滤全零行
            raw_labels = pd.read_excel(
                os.path.join(self.xlsx_folder, xlsx_file),
                usecols=range(7),
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
                avg_label = np.mean(valid_label_rows,axis=0)
                yolo_output_label = self.labels_to_yolo_format(num_people,avg_label)

                # label_seq[:, 6] = num_people - 1
                samples.append((
                    torch.from_numpy(signal_seq).to(torch.complex64),
                    # torch.from_numpy(label_seq).to(torch.float32)
                    torch.from_numpy(yolo_output_label).to(torch.float32)
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
        # 根据add_noise参数决定是否添加噪声
        if self.add_noise:
            processed_signal = self._add_gaussian_noise(signal)
        else:
            processed_signal = signal  # 不加噪声，直接返回原信号

        return processed_signal, label



def load_yolostyle_label(npy_folder_path, xls_folder_path, seq_length, overlap_step,add_noise):
    print('MyYoLoSytle (支持.npy格式，已处理全零标签行)')
    return MyYoLoSytle(npy_folder_path, xls_folder_path, seq_length, overlap_step,add_noise)




def add_wh_to_targets(targets, w=0.4, h=0.4, normalize_coords=True):
    """
    为有效目标添加固定的宽高，过滤全零目标

    Args:
        targets: (N, 2) 目标坐标张量
        w: 宽度值
        h: 高度值

    Returns:
        (N, 4) 张量，格式为 [x, y, w, h]，全零目标保持为零
    """
    # 找出非零目标（有效目标）
    non_zero_mask = (targets != 0).any(dim=1)
    valid_targets = targets[non_zero_mask]
    # 如果需要归一化坐标
    if normalize_coords and len(valid_targets) > 0:
        # 将坐标从 [-2.7, 2.7] 归一化到 [0, 1]
        x_min, x_max = -2.7, 2.7
        y_min, y_max = -2.7, 2.7

        # 归一化公式: (x - min) / (max - min)
        normalized_x = (valid_targets[:, 0] - x_min) / (x_max - x_min)
        normalized_y = (valid_targets[:, 1] - y_min) / (y_max - y_min)

        # 替换为归一化后的坐标
        valid_targets = torch.stack([normalized_x, normalized_y], dim=1)

    # 创建结果张量
    result = torch.zeros(targets.shape[0], 4)

    # 只为有效目标添加宽高
    if len(valid_targets) > 0:
        wh = torch.full((len(valid_targets), 2), 0.4)
        valid_result = torch.hstack([valid_targets, wh])
        result[non_zero_mask] = valid_result

    return result
    # # 创建结果张量
    # result = torch.zeros(targets.shape[0], 4)
    #
    # # 只为有效目标添加宽高
    # if len(valid_targets) > 0:
    #     wh = torch.full((len(valid_targets), 2), 0.4)
    #     valid_result = torch.hstack([valid_targets, wh])
    #     result[non_zero_mask] = valid_result
    #
    # return result