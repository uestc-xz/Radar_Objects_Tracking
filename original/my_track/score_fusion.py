"""
M6 融合打分 (Score Fusion)
==========================
作用：
    把 M5 的几何一致性特征 g_geom 与 M4 映射后的身份特征相似度 sim 融合，
    输出「量测 i 对各候选类别」的打分 S[i, :]，供 M7 关联决策做 softmax。
    类别 = 3 条航迹 + 1 个虚拟杂波类（共 4 类）。

融合方式（方案 A，晚融合，见 doc/跟踪任务书.md §3 M6）：
        sim(i,k) = cos( φ(f_i), φ(f_k) )                       # 特征相似度 ∈ [-1,1]
        S[i,k]   = MLP( concat([g_geom[i,k], sim(i,k)]) )      # k = 1..3 真实航迹
        S[i,0]   = clutter_logit                               # 虚拟杂波类，可学习标量

输入：
    g_geom  : [B, 50, 3, 64]   M5 输出的几何一致性特征
    phi_f_i : [B, 50, 128]     M4 映射后的量测特征 φ(f_i)
    phi_f_k : [B, 3, 128]      M4 映射后的航迹特征 φ(f_k)

输出：
    S : [B, 50, 4]   第 0 列=杂波分，第 1..3 列=与航迹 T_1..T_3 的匹配分

约定：
    本模块只做「融合打分」，特征映射由 M4(FeatureMapper) 在外部完成，
    量测侧与航迹侧共用同一个映射（M3 只存原始 dec_output 特征）。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScoreFusion(nn.Module):
    """晚融合打分：g_geom + 特征相似度 → 4 类打分。"""

    def __init__(self, geom_dim=64):
        super().__init__()
        self.geom_dim = geom_dim
        # 输入 = g_geom(64) + sim(1) = 65 维，输出 1 个标量分
        self.score_mlp = nn.Sequential(
            nn.Linear(geom_dim + 1, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        # 虚拟杂波类的可学习 logit（所有量测共享，等价于全局「无目标」阈值）
        self.clutter_logit = nn.Parameter(torch.zeros(1))

    def forward(self, g_geom, phi_f_i, phi_f_k):
        """前向：计算 (量测 i, 航迹 k) 的融合分 + 杂波分，拼成 [B,50,4]。

        Args:
            g_geom  : [B, 50, 3, 64] 几何一致性特征
            phi_f_i : [B, 50, 128]   量测映射特征 φ(f_i)
            phi_f_k : [B, 3, 128]    航迹映射特征 φ(f_k)

        Returns:
            S : [B, 50, 4]
        """
        B, M, N, _ = g_geom.shape   # M=50 量测，N=3 航迹

        # 1) 特征相似度：余弦相似度 = L2 归一化后的点积
        mi = F.normalize(phi_f_i, dim=-1)            # [B,50,128]
        mk = F.normalize(phi_f_k, dim=-1)            # [B,3,128]
        sim = torch.einsum('bmd,bkd->bmk', mi, mk)   # [B,50,3]

        # 2) 拼接 g_geom 与 sim，MLP 逐 (i,k) 打标量分
        feat = torch.cat([g_geom, sim.unsqueeze(-1)], dim=-1)     # [B,50,3,65]
        flat = feat.reshape(B * M * N, self.geom_dim + 1)         # [B*50*3, 65]
        track_score = self.score_mlp(flat).reshape(B, M, N)       # [B,50,3]

        # 3) 杂波列：可学习标量广播到所有量测，拼在最前（第 0 列）
        clutter = self.clutter_logit.reshape(1, 1, 1).expand(B, M, 1)  # [B,50,1]
        S = torch.cat([clutter, track_score], dim=-1)                  # [B,50,4]
        return S
