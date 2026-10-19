r"""
M1~M8 联调验证脚本 (demo)
=========================
作用：
    用随机权重 + 随机输入，跑通 M1(定位前端) → M2(量测生成) → M3(航迹状态库)
    → M4(特征映射) → M5(残差 GRU) → M6(融合打分) → M7(关联决策/损失)
    → M8(航迹更新) 的完整数据流，只验证【形状】是否对齐，不关心数值（随机权重无意义）。

运行（仓库根目录 E:\\Jupyter\\NewMOTR）：
    E:\\Miniconda3\\python.exe original/my_track/demo_m1_m8.py
"""
import importlib
import os
import sys

import torch

# 把仓库根目录加入 sys.path，保证以脚本方式运行(非 -m)时也能 import 到 original 包
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO_ROOT)

# original/123ren.py 以数字开头，不能用普通 import 语句，用 importlib 绕过
_123ren = importlib.import_module('original.123ren')
Mymodule = _123ren.Mymodule

from original.my_track import (            # noqa: E402
    generate_measurements,                 # M2 量测生成
    TrackBank,                             # M3 航迹状态库（含 M8 update 方法）
    FeatureMapper,                         # M4 特征映射分支
    ResidualGRU,                           # M5 几何残差 GRU
    ScoreFusion,                           # M6 融合打分
    associate,                             # M7 关联决策
    build_assignment_targets,              # M7 监督信号构造
    association_loss,                      # M7 关联损失
)


def main():
    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    H = 4  # 短历史长度（决策点 D2，先取论文值 4）

    # ---- M1 定位前端（随机权重，eval 模式）----
    model = Mymodule(input_channels=8, freq_bins=89, time_steps=10).to(device).eval()
    x = torch.randn(1, 8, 89, 10, dtype=torch.complex64, device=device)
    with torch.no_grad():
        out, conf, feat = model(x)                       # [1,50,4] [1,50,2] [1,50,128]
    coor = out[:, :, :2]                                 # [1,50,2] 中心坐标（out 后两维是冗余 classifier，跟踪不用）
    print(f'[M1] coor={tuple(coor.shape)}  conf={tuple(conf.shape)}  feat={tuple(feat.shape)}')

    # ---- M2 量测生成 ----
    meas = generate_measurements(coor, conf, feat)       # Measurements(pos, score, feat)
    print(f'[M2] pos={tuple(meas.pos.shape)}  score={tuple(meas.score.shape)}  feat={tuple(meas.feat.shape)}')

    # ---- M3 航迹状态库（首帧初始化，跟踪 batch=1 故取下标 0）----
    bank = TrackBank(num_tracks=3, hist_len=H, feat_dim=128)
    bank.init_from_measurements(meas.pos[0], meas.score[0], meas.feat[0])
    print(f'[M3] hist={tuple(bank.hist.shape)}  feat={tuple(bank.feat.shape)}  alive={bank.alive.tolist()}')

    # ---- M4 特征映射分支（量测侧与航迹侧共用同一映射）----
    mapper = FeatureMapper(in_dim=128, out_dim=128).to(device)
    phi_f_i = mapper(feat)                               # [1,50,128] 量测侧 φ(f_i)
    phi_f_k = mapper(bank.feat).unsqueeze(0)             # [1,3,128]  航迹侧 φ(f_k)
    print(f'[M4] phi(f_i)={tuple(phi_f_i.shape)}  phi(f_k)={tuple(phi_f_k.shape)}')

    # ---- M5 几何残差 GRU（量测位置 × 航迹短历史）----
    res_gru = ResidualGRU(state_dim=4, hidden_dim=64).to(device)
    g_geom = res_gru(coor, bank.hist.unsqueeze(0))       # [1,50,3,64]
    print(f'[M5] g_geom={tuple(g_geom.shape)}  (期望 [1,50,3,64])')

    # ---- M6 融合打分（g_geom + 特征相似度 → 4 类打分）----
    scorer = ScoreFusion(geom_dim=64).to(device)
    S = scorer(g_geom, phi_f_i, phi_f_k)                 # [1,50,4]
    print(f'[M6] S={tuple(S.shape)}  (期望 [1,50,4]，第 0 列=杂波，第 1..3 列=航迹)')

    # ---- M7 关联决策（softmax + 1对1约束）----
    assignments = associate(S)                           # [1,50] 每量测归属 ∈{0,1,2,3}
    dist = [(assignments[0] == c).sum().item() for c in range(4)]
    print(f'[M7] assignments={tuple(assignments.shape)}  分布(0/1/2/3)={dist}')

    # ---- M7 监督信号构造 + 损失（假 GT：2 人，槽 3 空）----
    fake_gt = torch.tensor([[[0.5, 1.0], [1.0, 2.0], [0.0, 0.0]]], device=device)
    labels = build_assignment_targets(coor, fake_gt)     # [1,50] ∈{0,1,2,3}
    label_dist = [(labels[0] == c).sum().item() for c in range(4)]
    loss = association_loss(S, labels, clutter_weight=2.0)
    print(f'[M7] labels 分布(0/1/2/3)={label_dist}  loss={loss.item():.4f}')

    # ---- M8 航迹更新（用本帧关联结果更新 3 条航迹）----
    pos_before = bank.pos.clone()
    bank.update(assignments[0], meas.pos[0], meas.feat[0], dt=0.24)
    print(f'[M8] pos 前={pos_before.detach().cpu().numpy().round(3)}')
    print(f'[M8] pos 后={bank.pos.detach().cpu().numpy().round(3)}')
    print(f'[M8] vel={bank.vel.detach().cpu().numpy().round(3)}  miss={bank.miss.tolist()}  age={bank.age.tolist()}')
    print(f'[M8] hist[0]={bank.hist[0].detach().cpu().numpy().round(3)}')

    print('\nM1~M8 数据流跑通，形状全部对齐。')


if __name__ == '__main__':
    main()
