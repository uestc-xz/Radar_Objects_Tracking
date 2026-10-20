"""
M7 关联决策与损失 (Association)
===============================
作用：
    1) 关联决策（推理）：把 M6 的打分矩阵 S 转为每个量测的归属 k_i ∈ {0,1,2,3}
       （0=杂波，1..3=航迹 T_1..T_3），流程 = softmax over 4 类 + 1 对 1 约束。
    2) 监督信号构造：用匈牙利匹配把 GT 槽位伪 ID 映射为每个量测的类标签。
    3) 关联损失：加权交叉熵（杂波类权重 λ）。

决策流程（见 doc/跟踪任务书.md §3 M7）：
    1. 可选门控：s_i < τ 的量测直接判杂波（决策 D4b，默认不门控）。
    2. 逐量测 softmax：p_i = softmax(S[i,:]) ∈ R^4，k_i = argmax p_i。
    3. 1 对 1 约束：同一航迹最多一个量测；冲突时只保留 p_i[k] 最大者，其余改判杂波。

约定：
    - 跟踪按 batch=1 逐帧进行；代码保留 B 维以兼容批量推理。
    - 本模块无学习参数，是纯函数。
    - 我们无「补零量测」：50 个 query 都是真实量测，故 CE 不做量测 mask；
      空槽（人数 < 3）通过「该槽从不作为标签」隐式处理，无需 mask。
"""
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------------------
# 1) 关联决策（推理）
# ---------------------------------------------------------------------------
def associate(S, score=None, gating_threshold=None):
    """把打分矩阵 S 转为每量测的归属（softmax + 1对1约束）。

    Args:
        S                : [B, 50, 4] 打分矩阵（第 0 列杂波，1..3 列航迹）
        score            : [B, 50]    量测置信度 s_i（仅当 gating_threshold 非 None 时用）
        gating_threshold : float|None 门控阈值 τ；None=不门控（默认）

    Returns:
        assignments : [B, 50] long，每量测归属 ∈ {0,1,2,3}
    """
    B, M, C = S.shape
    device = S.device

    p = torch.softmax(S, dim=-1)            # [B,50,4]
    assignments = torch.argmax(p, dim=-1)   # [B,50] 初步归属

    # 1) 可选门控：低置信度量测直接判杂波（D4b，默认关闭）
    if gating_threshold is not None and score is not None:
        assignments[score < gating_threshold] = 0

    # 3) 1 对 1 约束：同一航迹最多一个量测（tracking-by-detection 要求）
    for b in range(B):
        for k in range(1, C):
            # 属于航迹 k 的量测索引
            idxs = (assignments[b] == k).nonzero(as_tuple=False).squeeze(-1)
            if idxs.numel() > 1:
                probs = p[b, idxs, k]             # [num] 各冲突量测在类 k 上的概率
                best = idxs[torch.argmax(probs)]  # 只保留概率最大者
                assignments[b, idxs] = 0          # 其余改判杂波
                assignments[b, best] = k

    return assignments


# ---------------------------------------------------------------------------
# 2) 监督信号构造（训练）
# ---------------------------------------------------------------------------
def build_assignment_targets(pred_boxes, targets, scores, threshold=0.96):
    """用「置信度阈值先筛 + 匈牙利匹配」把 GT 槽位伪 ID 映射为每量测的类标签。

    规则（与定位前端置信度阈值的物理含义对齐）：
        1) s_i < τ 的量测直接判杂波（labels=0），不参与匹配；
        2) 候选量测（s_i ≥ τ）与真实目标做匈牙利 1 对 1 匹配，匹配上的候选
           赋对应槽位类（槽位 0..2 → 类 1..3）；
        3) 候选但未匹配上的，赋「距离它最近的真实目标」对应的槽位类。

    Args:
        pred_boxes : [B, 50, 2]  量测中心坐标 (x, y)
        targets    : [B, 3, 2]   GT 槽位中心（(0,0)=空槽，槽位顺序=伪 ID）
        scores     : [B, 50]     量测置信度 s_i = P(该 query 是目标) ∈ [0,1]
        threshold  : float       置信度阈值 τ，默认 0.96（前端定位标定值）

    Returns:
        labels : [B, 50] long，∈ {0,1,2,3}；0=杂波，j+1=GT 槽位 j
    """
    B, M, _ = pred_boxes.shape
    device = pred_boxes.device
    labels = torch.zeros(B, M, dtype=torch.long, device=device)

    for b in range(B):
        t = targets[b]                                    # [3,2]
        # 空槽判定：(x,y) 接近 (0,0) 即无目标
        is_invalid = torch.all(torch.isclose(t, torch.tensor([0.0, 0.0], device=device)), dim=1)
        valid_mask = ~is_invalid
        valid_targets = t[valid_mask]                     # [n,2]
        slot_indices = valid_mask.nonzero(as_tuple=False).squeeze(-1)  # [n] 原始槽位 0..2

        if len(valid_targets) == 0:
            continue                                      # 全空槽 → 所有量测判杂波（labels 保持 0）

        # 1) 置信度门控：s_i ≥ τ 才是候选目标，其余量测保持 0（杂波）
        cand_mask = scores[b] >= threshold                # [M] bool
        cand_idx = cand_mask.nonzero(as_tuple=False).squeeze(-1)   # [k] 候选量测下标
        if len(cand_idx) == 0:
            continue                                      # 无候选 → 全杂波（labels 保持 0）

        # 2) 中心欧氏距离做 1 对 1 匹配：valid_targets [n,2] × 候选 [k,2] → [k, n]
        dist_matrix = torch.cdist(valid_targets, pred_boxes[b][cand_idx], p=2).T   # [k, n]
        row_idx, col_idx = linear_sum_assignment(dist_matrix.detach().cpu().numpy())

        # row_idx=候选内部索引，col_idx=valid_targets 内部索引
        matched = [False] * len(cand_idx)
        for qi, ti in zip(row_idx, col_idx):
            labels[b, cand_idx[qi]] = slot_indices[ti].item() + 1   # 槽位 0..2 → 类 1..3
            matched[qi] = True

        # 3) 候选但未匹配上的：赋距离最近真实目标对应的槽位类
        for qi in range(len(cand_idx)):
            if matched[qi]:
                continue
            d = torch.norm(pred_boxes[b][cand_idx[qi]] - valid_targets, dim=-1)  # [n]
            ti = torch.argmin(d).item()
            labels[b, cand_idx[qi]] = slot_indices[ti].item() + 1

    return labels


# ---------------------------------------------------------------------------
# 3) 关联损失（训练）需要优化
# ---------------------------------------------------------------------------
def association_loss(S, labels, clutter_weight=1.0):
    """加权交叉熵损失（论文式）。

    Args:
        S              : [B, 50, 4] 打分矩阵（第 0 列杂波，1..3 列航迹）
        labels         : [B, 50]    真值类标签 ∈ {0,1,2,3}
        clutter_weight : float      杂波类权重 λ（>1 增强杂波识别惩罚，见任务书 §3 M7）

    Returns:
        loss : 标量
    """
    # 类权重：杂波类乘 λ，航迹类为 1
    class_weights = torch.tensor([clutter_weight, 1.0, 1.0, 1.0],
                                 device=S.device, dtype=S.dtype)
    return F.cross_entropy(S.reshape(-1, 4), labels.reshape(-1), weight=class_weights)
