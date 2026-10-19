"""
M3 航迹状态库 (Track State Bank)
================================
作用：
    维护固定 3 条航迹（对应最多 3 个目标）的状态，为 M5 的残差 GRU 提供每条航迹
    的短历史 (x,y,vx,vy) 序列，为 M6 提供航迹特征 f_k。

约定（见 doc/跟踪任务书.md §3 M3）：
    - 槽位数 N=3，与 GT 的 3 槽 (x,y)、空槽 (0,0) 对齐。
    - 每条航迹字段：位置 (x,y)、速度 (vx,vy)、特征 f_k、短历史 hist_k [H,4]、
      年龄 age、连续未匹配帧数 miss、存活标志 alive。
    - 每条序列人数固定 → 航迹主要在序列首帧初始化，帧内只做匹配 + coast。
    - 本阶段只实现【状态存储 + 首帧初始化】；匹配后的更新逻辑在 M8 实现。

输入（首帧量测，用于初始化航迹）：
    meas_pos   : [M, 2]   量测中心坐标（M=50）
    meas_score : [M]      量测置信度
    meas_feat  : [M, 128] 量测特征

输出（供 M5/M6 使用）：
    hist : [3, H, 4]   3 条航迹的短历史 (x,y,vx,vy)
    feat : [3, 128]    3 条航迹的特征 f_k
"""
import torch


class TrackBank:
    """固定 3 槽的航迹状态库。所有字段用张量批量存储，跟踪按 batch=1 逐帧进行。"""

    def __init__(self, num_tracks=3, hist_len=4, feat_dim=128):
        self.num_tracks = num_tracks   # 航迹槽位数（= 最多人数）
        self.hist_len = hist_len       # 短历史长度 H
        self.feat_dim = feat_dim       # 特征维度（= dec_output 维度 128）
        self.reset()

    def reset(self):
        """清空为未初始化状态（CPU 全零占位；init_from_measurements 会在量测设备上重建）。"""
        self.pos = torch.zeros(self.num_tracks, 2)                 # [3,2] 位置 (x,y)
        self.vel = torch.zeros(self.num_tracks, 2)                 # [3,2] 速度 (vx,vy)
        self.feat = torch.zeros(self.num_tracks, self.feat_dim)    # [3,128] 特征 f_k
        self.hist = torch.zeros(self.num_tracks, self.hist_len, 4) # [3,H,4] 短历史
        self.age = torch.zeros(self.num_tracks, dtype=torch.long)  # [3] 年龄
        self.miss = torch.zeros(self.num_tracks, dtype=torch.long) # [3] 连续未匹配帧数
        self.alive = torch.zeros(self.num_tracks, dtype=torch.bool)# [3] 存活标志

    def init_from_measurements(self, meas_pos, meas_score, meas_feat):
        """序列首帧初始化：取置信度最高的 num_tracks 个量测作为航迹起点。

        首帧无速度先验 → 速度置 0；短历史用 (x,y,0,0) 复制 hist_len 次填充。
        更细的出生/门控判定（用 s_i 阈值判 alive）留到 M7/M8 再实现。
        """
        device = meas_pos.device
        dtype = meas_pos.dtype
        topk = min(self.num_tracks, meas_pos.shape[0])

        # 按置信度取前 topk 个量测的索引
        _, idx = torch.topk(meas_score, k=topk)         # [topk]
        init_pos = meas_pos[idx]                        # [topk,2]
        init_vel = torch.zeros_like(init_pos)           # [topk,2] 首帧速度 0
        init_feat = meas_feat[idx]                      # [topk,128]

        # 在量测所在设备/精度上初始化所有字段；未占满的槽位保持全零（空槽，alive=False）
        self.pos = torch.zeros(self.num_tracks, 2, device=device, dtype=dtype)
        self.vel = torch.zeros(self.num_tracks, 2, device=device, dtype=dtype)
        self.feat = torch.zeros(self.num_tracks, self.feat_dim, device=device, dtype=dtype)
        self.hist = torch.zeros(self.num_tracks, self.hist_len, 4, device=device, dtype=dtype)
        self.age = torch.zeros(self.num_tracks, dtype=torch.long, device=device)
        self.miss = torch.zeros(self.num_tracks, dtype=torch.long, device=device)
        self.alive = torch.zeros(self.num_tracks, dtype=torch.bool, device=device)

        self.pos[:topk] = init_pos
        self.vel[:topk] = init_vel
        self.feat[:topk] = init_feat
        self.alive[:topk] = True
        self.age[:topk] = 1
        self.miss[:topk] = 0

        # 短历史 [topk,H,4]：每个历史步都填 (x,y,0,0)
        init_state = torch.cat([init_pos, init_vel], dim=-1)            # [topk,4]
        self.hist[:topk] = init_state.unsqueeze(1).repeat(1, self.hist_len, 1)

    def update(self, assignments, meas_pos, meas_feat, dt=0.24,
               pos_alpha=0.5, vel_alpha=0.5, feat_alpha=0.5):
        """M8 航迹更新：根据本帧关联结果 + 量测，更新 3 条航迹状态。

        更新规则（doc/跟踪任务书.md §3 M8，batch=1 逐帧调用）：
            匹配航迹 k（存在量测 i 使 assignments[i] == k+1）：
                速度  v_k ← (1-α_v) v_k + α_v (p_i - x_k)/dt     # 瞬时速度 EMA
                位置  x_k ← (1-α_p) x_k + α_p p_i                # 位置 EMA
                特征  f_k ← (1-α_f) f_k + α_f f_i                # 原始特征 EMA
                miss_k ← 0
            未匹配航迹 k（coast）：
                位置外推  x_k ← x_k + v_k * dt                   # 匀速外推
                速度/特征 不变，miss_k ← miss_k + 1
            两分支末尾都把 (x_k,y_k,vx_k,vy_k) 追加进 hist_k，截断保留最近 H 帧。

        Args:
            assignments : [M]       M7 关联结果，每量测归属 ∈ {0,1,2,3}
            meas_pos    : [M, 2]    量测中心坐标 (x, y)
            meas_feat   : [M, 128]  量测原始特征（dec_output）
            dt          : float     帧间隔，默认 0.24s
            pos_alpha / vel_alpha / feat_alpha : EMA 系数 ∈ (0,1]，越大越信当前量测

        说明：
            - 特征存「原始 dec_output」并做 EMA；φ 映射在 M6 打分时统一套用
              （见 feature_map.py 约定），故 M3/M8 不耦合 M4 参数。
            - 出生/死亡（alive 门控）留待完整跟踪流程（任务书 §3 M3/M7 第 6 步）。
            - 【重要】先 clone、在副本上更新、最后整体赋值替换，避免就地修改被计算图
              引用的旧张量：训练时 bank.feat/hist 会作为 mapper/GRU 的输入进入计算图，
              就地改会触发 autograd 的 inplace 版本冲突。航迹状态本就不参与反向传播
              （来自冻结 M1 + argmax 决策），clone 开销可忽略。
        """
        # 在副本上更新（不就地改 self.*，防止与计算图引用的旧张量冲突）
        pos = self.pos.clone()
        vel = self.vel.clone()
        feat = self.feat.clone()
        hist = self.hist.clone()
        miss = self.miss.clone()

        for k in range(self.num_tracks):
            # 归属航迹 k 的量测索引（M7 已做 1 对 1，至多 1 个）
            idxs = (assignments == (k + 1)).nonzero(as_tuple=False).squeeze(-1)

            if idxs.numel() > 0:
                i = idxs[0].item()                 # 匹配量测下标
                pi = meas_pos[i]                   # [2] 量测位置
                fi = meas_feat[i]                  # [128] 量测原始特征

                # 速度：由 (量测 - 当前位置)/dt 求瞬时速度，再 EMA 平滑
                vel_inst = (pi - pos[k]) / dt      # [2]
                vel[k] = (1 - vel_alpha) * vel[k] + vel_alpha * vel_inst

                # 位置：EMA
                pos[k] = (1 - pos_alpha) * pos[k] + pos_alpha * pi

                # 特征：原始特征 EMA
                feat[k] = (1 - feat_alpha) * feat[k] + feat_alpha * fi

                miss[k] = 0
            else:
                # coast：匀速外推，速度/特征不变
                pos[k] = pos[k] + vel[k] * dt
                miss[k] = miss[k] + 1

            # 无论匹配与否，都把当前状态追加进短历史，保证 hist 始终是最近 H 帧
            state = torch.cat([pos[k], vel[k]])            # [4]
            hist[k] = torch.cat([hist[k][1:], state.unsqueeze(0)], dim=0)

        # 存活航迹年龄 +1（alive 门控留待完整流程）
        age = self.age + self.alive.long()

        # 整体赋值替换旧张量（不就地修改）
        self.pos = pos
        self.vel = vel
        self.feat = feat
        self.hist = hist
        self.miss = miss
        self.age = age
