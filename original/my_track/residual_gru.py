"""
M5 几何残差 GRU (Residual GRU)
==============================
作用：
    对「量测位置」与「航迹短历史状态」逐历史步相减构造残差序列，用 GRU 提取
    短时动态一致性特征 g_geom。这是论文方法的主干，本模块是跟踪关联网络的
    几何分支，输出供 M6 融合打分。

残差构造（论文式）：量测无速度，速度通道只来自航迹侧，故对历史步 h：
        r_h = ( x_i - x_k^(h),  y_i - y_k^(h),  0 - vx_k^(h),  0 - vy_k^(h) )
    其中 h = 1..H，H 为短历史长度。

输入：
    meas_pos   : [B, 50, 2]    量测中心坐标 (x_i, y_i)
    track_hist : [B, 3, H, 4]  3 条航迹的短历史 (x,y,vx,vy)

输出：
    g_geom : [B, 50, 3, 64]    每对 (量测 i, 航迹 k) 的几何一致性特征
"""
import torch
import torch.nn as nn


class ResidualGRU(nn.Module):
    """量测-航迹残差序列 → GRU → 最后隐状态 g_geom。"""

    def __init__(self, state_dim=4, hidden_dim=64):
        super().__init__()
        self.state_dim = state_dim    # 单步残差维度 D=4 (x,y,vx,vy)
        self.hidden_dim = hidden_dim  # GRU 隐层维度 64
        # batch_first=True：输入按 [batch, seq_len(H), input_size(4)] 排列
        self.gru = nn.GRU(input_size=state_dim, hidden_size=hidden_dim, batch_first=True)

    def build_residual(self, meas_pos, track_hist):
        """构造残差序列 [B, 50, 3, H, 4]，供 GRU 逐 (i,k) 对处理。

        Args:
            meas_pos   : [B, 50, 2]   量测位置
            track_hist : [B, 3, H, 4] 航迹短历史

        Returns:
            residual : [B, 50, 3, H, 4]
        """
        # 量测位置扩维到 [B,50,1,1,2]；量测速度恒为 0（与航迹速度通道对齐后相减）
        meas_pos = meas_pos.unsqueeze(2).unsqueeze(3)          # [B,50,1,1,2]
        meas_vel = torch.zeros_like(meas_pos)                  # [B,50,1,1,2] = (0,0)

        # 航迹历史扩维到 [B,1,3,H,4]，拆分位置/速度通道
        track_hist = track_hist.unsqueeze(1)                   # [B,1,3,H,4]
        track_pos = track_hist[..., :2]                        # [B,1,3,H,2]
        track_vel = track_hist[..., 2:]                        # [B,1,3,H,2]

        # 广播相减：位置差 + 速度差（量测速度 = 0，故速度残差 = -v_k）
        d_pos = meas_pos - track_pos                           # [B,50,3,H,2]
        d_vel = meas_vel - track_vel                           # [B,50,3,H,2]
        residual = torch.cat([d_pos, d_vel], dim=-1)           # [B,50,3,H,4]
        return residual

    def forward(self, meas_pos, track_hist):
        """前向：残差序列 → GRU → 最后时刻隐状态 g_geom。

        Args:
            meas_pos   : [B, 50, 2]   量测位置
            track_hist : [B, 3, H, 4] 航迹短历史

        Returns:
            g_geom : [B, 50, 3, 64]
        """
        B, M = meas_pos.shape[0], meas_pos.shape[1]
        N = track_hist.shape[1]
        H = track_hist.shape[2]

        residual = self.build_residual(meas_pos, track_hist)   # [B,M,N,H,4]

        # 展平 (B,M,N) 为 batch，让 GRU 逐「量测-航迹」对处理 [H,4] 序列
        flat = residual.reshape(B * M * N, H, self.state_dim)  # [B*M*N, H, 4]
        _, h_n = self.gru(flat)                                # h_n: [1, B*M*N, 64]

        # 取最后时刻隐状态作为特征向量（论文口径），再还原 (B,M,N) 结构
        g_geom = h_n.squeeze(0)                                # [B*M*N, 64]
        g_geom = g_geom.reshape(B, M, N, self.hidden_dim)      # [B, M, N, 64]
        return g_geom
