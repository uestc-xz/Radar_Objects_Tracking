"""
M4 特征映射分支 (Feature Mapper)
================================
作用：
    把 DETR 的 dec_output 特征（定位空间）映射到「关联 / 身份特征空间」。
    论文原方法只有低维几何量测，本分支用高维 dec_output 补充身份信息，
    是独立消融点（可去掉、或把输出维度 dφ 换成 64 做消融，见任务书 §7）。

结构：
    Linear(128 -> 128) + LayerNorm

输入：
    feat : [B, 50, 128]  dec_output 原始特征（M1 返回的 feat）

输出：
    mapped : [B, 50, 128]  φ(f) 映射后特征

约定：
    航迹侧特征 f_k 在 M3 中暂存为「原始 dec_output 特征」，M6 融合打分时
    对量测侧 φ(f_i) 与航迹侧 φ(f_k) 都应用本映射，避免 M3 耦合 M4 参数。
"""
import torch.nn as nn


class FeatureMapper(nn.Module):
    """dec_output 特征 → 关联/身份特征空间 的可学习映射。"""

    def __init__(self, in_dim=128, out_dim=128):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, feat):
        """前向映射。

        Args:
            feat : [B, 50, 128] 原始特征

        Returns:
            mapped : [B, 50, 128]  φ(f)
        """
        return self.norm(self.proj(feat))
