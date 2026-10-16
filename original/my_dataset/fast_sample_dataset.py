import os
import numpy as np
import torch
from torch.utils.data import Dataset
import random

## jia shangle tichu
# class FastNpyDataset(Dataset):
#     def __init__(self, data_root_folder, seq_length=10, add_noise=True, delete_walk=False):
#         """
#         Fast数据集加载器，支持剔除walk时间段样本
#
#         Args:
#             data_root_folder: fast数据集的根目录
#             seq_length: 序列长度（用于兼容性，实际从文件读取）
#             add_noise: 是否添加高斯噪声
#             delete_walk: 是否剔除walk时间段对应的样本
#         """
#         self.data_root_folder = data_root_folder
#         self.add_noise = add_noise
#         self.delete_walk = delete_walk
#
#         # 先构建完整的样本列表
#         self.all_samples = self._build_sample_list()
#
#         # 如果需要剔除walk样本，则过滤样本列表
#         if delete_walk:
#             self.samples = self._filter_walk_samples()
#         else:
#             self.samples = self.all_samples
#
#         print(f"Fast数据集加载完成，共 {len(self.samples)} 个样本")
#         print(f"数据根目录: {data_root_folder}")
#         if delete_walk:
#             print(f"已剔除walk样本，从 {len(self.all_samples)} 个减少到 {len(self.samples)} 个")
#
#     def _filter_walk_samples(self):
#         """过滤掉walk时间段对应的样本"""
#         filtered_samples = []
#         walk_samples_to_delete = self._get_all_walk_sample_ids()
#
#         for signal_path, label_path in self.all_samples:
#             # 提取样本ID (如: 1.npy -> 1)
#             sample_file = os.path.basename(signal_path)
#             sample_id = int(sample_file.split('.')[0])
#
#             # 如果样本ID不在要删除的范围内，则保留
#             if sample_id not in walk_samples_to_delete:
#                 filtered_samples.append((signal_path, label_path))
#
#         return filtered_samples
#
#     def _get_all_walk_sample_ids(self):
#         """从所有txt文件中获取需要剔除的walk样本ID"""
#         walk_sample_ids = set()
#
#         # 遍历所有子文件夹
#         for file_folder in sorted(os.listdir(self.data_root_folder)):
#             folder_path = os.path.join(self.data_root_folder, file_folder)
#
#             if os.path.isdir(folder_path):
#                 # 查找同名的txt文件
#                 txt_file = os.path.join(folder_path, f"{file_folder}.txt")
#
#                 if os.path.exists(txt_file):
#                     walk_sample_ids.update(self._parse_walk_txt_file(txt_file))
#
#         print(f"找到需要剔除的walk样本数量: {len(walk_sample_ids)}")
#         return walk_sample_ids
#
#     def _parse_walk_txt_file(self, txt_file):
#         """解析单个txt文件，提取要删除的样本ID"""
#         sample_ids = set()
#
#         try:
#             with open(txt_file, 'r', encoding='utf-8') as f:
#                 lines = f.readlines()
#
#             for line in lines:
#                 line = line.strip()
#                 # 查找样本ID范围行 (格式: 174.npy-227.npy)
#                 if '.npy-' in line and '\t' in line:
#                     parts = line.split('\t')
#                     sample_range = parts[-1]  # 最后一列是样本ID范围
#
#                     # 解析样本ID范围
#                     if '.npy-' in sample_range and '无对应样本' not in sample_range:
#                         start_end = sample_range.replace('.npy', '').split('-')
#                         if len(start_end) == 2:
#                             start_id = int(start_end[0])
#                             end_id = int(start_end[1])
#
#                             # 将该范围内的所有样本ID加入集合
#                             for sample_id in range(start_id, end_id + 1):
#                                 sample_ids.add(sample_id)
#
#         except Exception as e:
#             print(f"解析walk文件 {txt_file} 时出错: {e}")
#
#         return sample_ids
#
#     # 以下方法保持不变
#     def _build_sample_list(self):
#         """构建样本路径列表（原有逻辑）"""
#         samples = []
#
#         # 遍历所有子文件夹（对应原始xlsx文件）
#         for file_folder in sorted(os.listdir(self.data_root_folder)):
#             folder_path = os.path.join(self.data_root_folder, file_folder)
#
#             if os.path.isdir(folder_path):
#                 signals_dir = os.path.join(folder_path, "signals")
#                 labels_dir = os.path.join(folder_path, "labels")
#
#                 if os.path.exists(signals_dir) and os.path.exists(labels_dir):
#                     # 获取并排序信号文件
#                     signal_files = sorted(
#                         [f for f in os.listdir(signals_dir) if f.endswith('.npy')],
#                         key=lambda x: int(x.split('.')[0])
#                     )
#
#                     # 构建样本路径对
#                     for signal_file in signal_files:
#                         signal_path = os.path.join(signals_dir, signal_file)
#                         label_path = os.path.join(labels_dir, signal_file)  # 同名文件
#
#                         if not os.path.exists(label_path):
#                             print(f"警告: 找不到对应的标签文件 {label_path}")
#                             continue
#
#                         samples.append((signal_path, label_path))
#
#         return samples
#
#     def _add_gaussian_noise(self, signal):
#         """为复数信号添加高斯噪声（与原始代码一致）"""
#         # 固定参数：50%概率添加噪声，标准差为0.03
#         if random.random() > 0.5:
#             return signal
#
#         # 为实部和虚部分别添加独立的高斯噪声
#         noise_std = 0.03
#         real_noise = torch.randn_like(signal.real) * noise_std
#         imag_noise = torch.randn_like(signal.imag) * noise_std
#         noisy_signal = signal + torch.complex(real_noise, imag_noise)
#
#         return noisy_signal
#
#     def __len__(self):
#         return len(self.samples)
#
#     def __getitem__(self, idx):
#         """
#         返回格式与原始NpyDataset5完全一致：
#         - signal: torch.complex64, shape (8, 89, seq_length)
#         - label: torch.float32, shape (seq_length, 6)
#         """
#         signal_path, label_path = self.samples[idx]
#
#         # 加载信号和标签
#         signal_np = np.load(signal_path)  # 形状: (8, 89, seq_length), dtype: complex64
#         label_np = np.load(label_path)  # 形状: (seq_length, 6), dtype: float32
#
#         # 转换为torch tensor（保持原始数据类型）
#         signal = torch.from_numpy(signal_np)  # 自动保持complex64
#         label = torch.from_numpy(label_np)  # 自动保持float32
#
#         # 根据add_noise参数决定是否添加噪声
#         if self.add_noise:
#             processed_signal = self._add_gaussian_noise(signal)
#         else:
#             processed_signal = signal
#
#         # 返回值格式与原始NpyDataset5完全一致
#         return processed_signal, label

## yuanshi
class FastNpyDataset(Dataset):
    def __init__(self, data_root_folder, seq_length=10, add_noise=True):
        """
        Fast数据集加载器，返回值格式与原始NpyDataset5完全一致

        Args:
            data_root_folder: fast数据集的根目录
            seq_length: 序列长度（用于兼容性，实际从文件读取）
            add_noise: 是否添加高斯噪声
        """
        self.data_root_folder = data_root_folder
        self.add_noise = add_noise
        self.samples = self._build_sample_list()

        print(f"Fast数据集加载完成，共 {len(self.samples)} 个样本")
        print(f"数据根目录: {data_root_folder}")

    def _build_sample_list(self):
        """构建样本路径列表"""
        samples = []

        # 遍历所有子文件夹（对应原始xlsx文件）
        for file_folder in sorted(os.listdir(self.data_root_folder)):
            folder_path = os.path.join(self.data_root_folder, file_folder)

            if os.path.isdir(folder_path):
                signals_dir = os.path.join(folder_path, "signals")
                labels_dir = os.path.join(folder_path, "labels")

                if os.path.exists(signals_dir) and os.path.exists(labels_dir):
                    # 获取并排序信号文件
                    signal_files = sorted(
                        [f for f in os.listdir(signals_dir) if f.endswith('.npy')],
                        key=lambda x: int(x.split('.')[0])
                    )

                    # 验证文件数量一致性
                    label_files = sorted(
                        [f for f in os.listdir(labels_dir) if f.endswith('.npy')],
                        key=lambda x: int(x.split('.')[0])
                    )

                    if len(signal_files) != len(label_files):
                        print(f"警告: {file_folder} 中信号文件数({len(signal_files)})与标签文件数({len(label_files)})不一致")
                        continue

                    # 构建样本路径对
                    for signal_file in signal_files:
                        signal_path = os.path.join(signals_dir, signal_file)
                        label_path = os.path.join(labels_dir, signal_file)  # 同名文件

                        if not os.path.exists(label_path):
                            print(f"警告: 找不到对应的标签文件 {label_path}")
                            continue

                        samples.append((signal_path, label_path))

        return samples

    def _add_gaussian_noise(self, signal):
        """为复数信号添加高斯噪声（与原始代码一致）"""
        # 固定参数：50%概率添加噪声，标准差为0.03
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
        """
        返回格式与原始NpyDataset5完全一致：
        - signal: torch.complex64, shape (8, 89, seq_length)
        - label: torch.float32, shape (seq_length, 6)
        """
        signal_path, label_path = self.samples[idx]

        # 加载信号和标签
        signal_np = np.load(signal_path)  # 形状: (8, 89, seq_length), dtype: complex64
        label_np = np.load(label_path)  # 形状: (seq_length, 6), dtype: float32

        # 转换为torch tensor（保持原始数据类型）
        signal = torch.from_numpy(signal_np)  # 自动保持complex64
        label = torch.from_numpy(label_np)  # 自动保持float32

        # 根据add_noise参数决定是否添加噪声
        if self.add_noise:
            processed_signal = self._add_gaussian_noise(signal)
        else:
            processed_signal = signal

        # 返回值格式与原始NpyDataset5完全一致
        return processed_signal, label


def load_fast_dataset(data_root_folder, seq_length=10, add_noise=True):
    """
    加载fast数据集的函数，接口与原始load_dataset5_npy保持一致

    Args:
        data_root_folder: fast数据集的根目录
        seq_length: 序列长度（用于兼容性）
        add_noise: 是否添加噪声

    Returns:
        FastNpyDataset实例
    """
    print(f'加载Fast数据集 (预处理的.npy格式)')
    print(f'数据目录: {data_root_folder}')
    return FastNpyDataset(data_root_folder, seq_length, add_noise)






class FastNpyDataset1(Dataset):
    def __init__(self, data_root_folder, seq_length=10, add_noise=True, delete_occlusion=False):
        """
        Fast数据集加载器，返回值格式与原始NpyDataset5完全一致

        Args:
            data_root_folder: fast数据集的根目录
            seq_length: 序列长度（用于兼容性，实际从文件读取）
            add_noise: 是否添加高斯噪声
            delete_occlusion: 是否删除遮挡样本
        """
        self.data_root_folder = data_root_folder
        self.add_noise = add_noise
        self.delete_occlusion = delete_occlusion
        self.samples = self._build_sample_list()

        # 如果需要删除遮挡样本，构建有效样本索引
        if self.delete_occlusion:
            self.valid_indices = self._filter_occluded_samples()
            print(f"遮挡过滤后，剩余 {len(self.valid_indices)} 个有效样本")
        else:
            self.valid_indices = list(range(len(self.samples)))

        print(f"Fast数据集加载完成，共 {len(self.samples)} 个样本")
        print(f"数据根目录: {data_root_folder}")
        print(f"删除遮挡样本: {delete_occlusion}")

    def _build_sample_list(self):
        """构建样本路径列表"""
        samples = []

        # 遍历所有子文件夹（对应原始xlsx文件）
        for file_folder in sorted(os.listdir(self.data_root_folder)):
            folder_path = os.path.join(self.data_root_folder, file_folder)

            if os.path.isdir(folder_path):
                signals_dir = os.path.join(folder_path, "signals")
                labels_dir = os.path.join(folder_path, "labels")

                if os.path.exists(signals_dir) and os.path.exists(labels_dir):
                    # 获取并排序信号文件
                    signal_files = sorted(
                        [f for f in os.listdir(signals_dir) if f.endswith('.npy')],
                        key=lambda x: int(x.split('.')[0])
                    )

                    # 验证文件数量一致性
                    label_files = sorted(
                        [f for f in os.listdir(labels_dir) if f.endswith('.npy')],
                        key=lambda x: int(x.split('.')[0])
                    )

                    if len(signal_files) != len(label_files):
                        print(f"警告: {file_folder} 中信号文件数({len(signal_files)})与标签文件数({len(label_files)})不一致")
                        continue

                    # 构建样本路径对
                    for signal_file in signal_files:
                        signal_path = os.path.join(signals_dir, signal_file)
                        label_path = os.path.join(labels_dir, signal_file)  # 同名文件

                        if not os.path.exists(label_path):
                            print(f"警告: 找不到对应的标签文件 {label_path}")
                            continue

                        samples.append((signal_path, label_path, file_folder))

        return samples

    def _filter_occluded_samples(self):
        """
        读取遮挡样本统计文件，过滤掉被遮挡的样本
        返回有效样本的索引列表
        """
        valid_indices = []
        occlusion_stats = {}

        # 首先收集所有遮挡统计信息
        for idx, (signal_path, label_path, file_folder) in enumerate(self.samples):
            collection_folder = os.path.join(self.data_root_folder, file_folder)
            occlusion_file = os.path.join(collection_folder, "遮挡样本id统计.txt")

            if file_folder not in occlusion_stats:
                if os.path.exists(occlusion_file):
                    # 读取遮挡样本ID
                    with open(occlusion_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    occluded_ids = []
                    for line in lines[1:]:  # 跳过第一行标题
                        if line.strip():
                            try:
                                occluded_ids.append(int(line.strip()))
                            except ValueError:
                                continue
                    occlusion_stats[file_folder] = set(occluded_ids)
                    print(f"加载遮挡统计: {file_folder} -> {len(occluded_ids)} 个遮挡样本")
                else:
                    occlusion_stats[file_folder] = set()
                    print(f"警告: 未找到遮挡统计文件 {occlusion_file}，该文件夹所有样本将被保留")

        # 然后过滤样本
        occluded_count = 0
        for idx, (signal_path, label_path, file_folder) in enumerate(self.samples):
            # 提取样本ID（从文件名中提取数字）
            filename = os.path.basename(signal_path)
            sample_id = int(filename.split('.')[0])

            # 检查是否在遮挡列表中
            if sample_id in occlusion_stats[file_folder]:
                occluded_count += 1
            else:
                valid_indices.append(idx)

        print(f"遮挡过滤统计: 总共 {len(self.samples)} 个样本，过滤 {occluded_count} 个遮挡样本")
        return valid_indices

    def _add_gaussian_noise(self, signal):
        """为复数信号添加高斯噪声（与原始代码一致）"""
        # 固定参数：50%概率添加噪声，标准差为0.03
        if random.random() > 0.5:
            return signal

        # 为实部和虚部分别添加独立的高斯噪声
        noise_std = 0.03
        real_noise = torch.randn_like(signal.real) * noise_std
        imag_noise = torch.randn_like(signal.imag) * noise_std
        noisy_signal = signal + torch.complex(real_noise, imag_noise)

        return noisy_signal

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        """
        返回格式与原始NpyDataset5完全一致：
        - signal: torch.complex64, shape (8, 89, seq_length)
        - label: torch.float32, shape (seq_length, 6)
        """
        # 使用有效索引映射
        original_idx = self.valid_indices[idx]
        signal_path, label_path, _ = self.samples[original_idx]

        # 加载信号和标签
        signal_np = np.load(signal_path)  # 形状: (8, 89, seq_length), dtype: complex64
        label_np = np.load(label_path)  # 形状: (seq_length, 6), dtype: float32

        # 转换为torch tensor（保持原始数据类型）
        signal = torch.from_numpy(signal_np)  # 自动保持complex64
        label = torch.from_numpy(label_np)  # 自动保持float32

        # 根据add_noise参数决定是否添加噪声
        if self.add_noise:
            processed_signal = self._add_gaussian_noise(signal)
        else:
            processed_signal = signal

        # 返回值格式与原始NpyDataset5完全一致
        return processed_signal, label


def load_fast_dataset1(data_root_folder, seq_length=10, add_noise=True, delete_occlusion=False):
    """
    加载fast数据集的函数，接口与原始load_dataset5_npy保持一致

    Args:
        data_root_folder: fast数据集的根目录
        seq_length: 序列长度（用于兼容性）
        add_noise: 是否添加噪声
        delete_occlusion: 是否删除遮挡样本

    Returns:
        FastNpyDataset实例
    """
    print(f'加载Fast数据集 (预处理的.npy格式)')
    print(f'数据目录: {data_root_folder}')
    print(f'删除遮挡样本: {delete_occlusion}')
    return FastNpyDataset1(data_root_folder, seq_length, add_noise, delete_occlusion)