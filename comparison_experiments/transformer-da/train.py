"""Transformer-DA 最小训练入口。"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from DANet import DANet  # noqa: E402
from lossfunc import MCDLoss, velocity_regression_loss  # noqa: E402
from original.my_dataset.track_sequence_dataset import (  # noqa: E402
    load_track_sequence_cold_dataset,
)
from original.my_track.association import build_assignment_targets  # noqa: E402


def pack_track_history(track_states):
    """把连续 HISTORY_STEPS 帧的逐帧航迹状态拼成编码器历史输入。

    Args:
        track_states: 长度 HISTORY_STEPS 的 list，每项 [B, num_tracks, 6]，
            6 维为 [x, vx, y, vy, conf_bg, conf_fg]。

    Returns:
        [B, num_tracks, HISTORY_STEPS, 6]。
    """
    stacked = torch.stack(track_states, dim=1)  # [B, H, 3, 6]
    return stacked.permute(0, 2, 1, 3)          # [B, 3, H, 6]


def cold_start_track_states(outputs, confidence_logits):
    """用 M1 量测冷启动前 HISTORY_STEPS 帧的航迹状态，不依赖任何标签。

    第 0 帧取前景置信度 top-3 的量测作为 3 条航迹种子（按置信度降序），
    之后每帧按最近邻追踪这 3 条航迹；速度为相邻帧位置差（位移，未除 Δt）。

    Returns:
        list[HISTORY_STEPS]，每项 [B, num_tracks, 6]。
    """
    batch_size = outputs.shape[0]
    device = outputs.device
    batch_index = torch.arange(batch_size, device=device)

    foreground_prob = confidence_logits[:, 0].softmax(dim=-1)[..., 1]  # [B,50]
    top3_query = foreground_prob.topk(config.NUM_TRACKS, dim=-1).indices  # [B,3]

    states = []
    prev_positions = None
    for frame_index in range(config.HISTORY_STEPS):
        positions_t = outputs[:, frame_index, :, :2]  # [B,50,2]
        if frame_index == 0:
            query = top3_query
            position = positions_t[batch_index[:, None], query]  # [B,3,2]
            velocity = torch.zeros_like(position)
        else:
            distances = torch.cdist(prev_positions, positions_t)  # [B,3,50]
            query = distances.argmin(dim=-1)  # [B,3]
            position = positions_t[batch_index[:, None], query]  # [B,3,2]
            velocity = position - prev_positions
        confidence = confidence_logits[:, frame_index][batch_index[:, None], query]  # [B,3,2]
        state = torch.cat(
            (
                position[..., 0:1], velocity[..., 0:1],
                position[..., 1:2], velocity[..., 1:2],
                confidence,
            ),
            dim=-1,
        )  # [B,3,6]
        states.append(state)
        prev_positions = position
    return states


# 冷启动参数实验记录（2026-10-27；25 条训练序列 2340 窗口 + 8 条验证序列 683 窗口）：
# 扫描了 首帧阈值 {0.90,0.92,0.94,0.96} × 续接阈值 {0.50,0.70,0.85} × 杂波半径 {0.3,0.5}
# × 匹配门限 {1.0,1.5}，GT 只用于离线统计，结论如下：
#   - 所有组合都没有出现 K=0；top-k 回退只在首帧阈值 >=0.94 时触发
#     （0.94：train 1 / val 2 个窗口；0.96：train 17 / val 7 个），且都还能建出 >=1 槽。
#   - 存在性覆盖率（train/val）随首帧阈值升高而下降：
#     0.90 .981/.965，0.92 .975/.954，0.94 .954/.932，0.96 .913/.903；
#     三目标从 0.973/.930 降到 0.873/.822——提高阈值等于用"漏建槽"换"少建槽"。
#   - 续接阈值 0.50~0.85、杂波半径 0.3~0.5 对 K 和覆盖率无可测影响（均值差 <=0.003 m）：
#     0.3~0.9 m 内的点前景分数约 0.01，任何阈值都选不到，半径只删杂波不删目标。
#   - 匹配门限 1.0 与 1.5 的均值差 <=0.003 m（1.0 在验证集略差，因为拒绝后只能空守位），
#     故保留 1.5；门槛收紧不会提升精度，只会让漂移的槽停止更新。
#   - 最终取值：首帧 0.90 / 续接 0.85 / 杂波 0.5 m / 门限 1.5 m。
# 注意：以上是验证集探索结果；K 仍可能落在 1~5，接入下游固定 3 槽前需先处理槽数不一致。
def cold_start_track_states_hungarian(outputs, confidence_logits):
    """阈值 + 匈牙利匹配的冷启动航迹初始化（不依赖 GT）。

    与 cold_start_track_states 的区别在种子选择与逐帧续接方式：
      - 第 1 帧：取前景置信度 > 0.90 的量测，有几条建几条槽；若一个都没有，
        退化为取前景置信度 top-NUM_TRACKS；
      - 第 2~HISTORY_STEPS 帧：取前景置信度 > 0.85 的量测，与各槽此前坐标的
        均值做匈牙利匹配（坐标为 M1 输出，非 GT），匹配上的续到对应槽后面，
        没匹配上的槽沿用上一帧位置（速度 0），多余的候选丢弃；
      - 距原点 0.5m 内的量测一律视为杂波剔除；匈牙利匹配距离超过 1.5m 视为
        未匹配；
      - 速度 = 相邻帧位置差（位移，未除 Δt），与回归头同单位。

    Returns:
        list[HISTORY_STEPS]，每项 [B, K, 6]；K 为本 batch 内最大槽数，不足 K
        的样本用 0 填充（0 表示空槽）。注意 K 可能与 NUM_TRACKS 不一致。
    """
    from scipy.optimize import linear_sum_assignment

    high_threshold = 0.90
    low_threshold = 0.85
    match_gate_m = 1.5               # 匹配距离门限（米），超过视为未匹配
    origin_clutter_radius_m = 0.5    # 距原点该半径内视为杂波
    batch_size = outputs.shape[0]
    device = outputs.device

    def _reject_origin_clutter(positions, indices):
        """剔除距原点 origin_clutter_radius_m 内的量测（视为杂波）。"""
        distance = torch.linalg.vector_norm(positions[indices], dim=-1)
        return indices[distance > origin_clutter_radius_m]

    foreground_prob = confidence_logits.softmax(dim=-1)[..., 1]  # [B, T, 50]

    # 逐样本维护槽；每槽记录各帧位置与置信度，位置列表用于算匹配锚点（历史均值）。
    slots_per_sample = [[] for _ in range(batch_size)]

    for frame_index in range(config.HISTORY_STEPS):
        positions_t = outputs[:, frame_index, :, :2]   # [B, 50, 2]
        conf_t = confidence_logits[:, frame_index]     # [B, 50, 2]
        threshold = high_threshold if frame_index == 0 else low_threshold

        for b in range(batch_size):
            cand_idx = (foreground_prob[b, frame_index] > threshold).nonzero(
                as_tuple=False
            ).reshape(-1)  # [m]
            cand_idx = _reject_origin_clutter(positions_t[b], cand_idx)

            if frame_index == 0:
                if cand_idx.numel() == 0:
                    # 首帧没有高置信候选：退化为前景置信度 top-NUM_TRACKS。
                    cand_idx = foreground_prob[b, frame_index].topk(
                        config.NUM_TRACKS
                    ).indices
                    cand_idx = _reject_origin_clutter(positions_t[b], cand_idx)
                # 每个候选建一个槽。
                for idx in cand_idx:
                    slots_per_sample[b].append(
                        {
                            "positions": [positions_t[b, idx]],
                            "confs": [conf_t[b, idx]],
                        }
                    )
                continue

            slots = slots_per_sample[b]
            if len(slots) == 0:
                continue
            if cand_idx.numel() == 0:
                # 本帧没有候选：所有槽沿用上一帧（速度 0）。
                for s in slots:
                    s["positions"].append(s["positions"][-1])
                    s["confs"].append(s["confs"][-1])
                continue

            cand_pos = positions_t[b, cand_idx]  # [m, 2]
            cand_conf = conf_t[b, cand_idx]      # [m, 2]
            anchor = torch.stack(
                [torch.stack(s["positions"]).mean(dim=0) for s in slots]
            )  # [k, 2]，槽历史位置的均值
            cost = torch.cdist(anchor, cand_pos)  # [k, m]
            row, col = linear_sum_assignment(cost.detach().cpu().numpy())
            matched = {
                int(i): int(j)
                for i, j in zip(row, col)
                if float(cost[i, j]) <= match_gate_m
            }
            for i, s in enumerate(slots):
                j = matched.get(i)
                if j is not None:
                    s["positions"].append(cand_pos[j])
                    s["confs"].append(cand_conf[j])
                else:
                    s["positions"].append(s["positions"][-1])
                    s["confs"].append(s["confs"][-1])

    max_slots = max((len(s) for s in slots_per_sample), default=0)
    states_per_frame = []
    for frame_index in range(config.HISTORY_STEPS):
        frame_states = torch.zeros(batch_size, max_slots, 6, device=device)
        for b in range(batch_size):
            for i, s in enumerate(slots_per_sample[b]):
                position = s["positions"][frame_index]
                confidence = s["confs"][frame_index]
                if frame_index == 0:
                    velocity = torch.zeros(2, device=device)
                else:
                    velocity = position - s["positions"][frame_index - 1]
                frame_states[b, i] = torch.cat(
                    (
                        position[0:1], velocity[0:1],
                        position[1:2], velocity[1:2],
                        confidence,
                    ),
                    dim=0,
                )
        states_per_frame.append(frame_states)
    return states_per_frame


def decode_track_states(network_output, outputs_t, confidence_logits_t, previous_states):
    """从网络逐量测输出解码出本帧 3 条航迹状态，不依赖标签。

    对每条航迹 k（类别 k+1）：
      - 取 argmax==k+1 且该类概率最高的量测；
      - 位置 = 该量测的 M1 位置，速度 = 该量测的网络速度头输出，置信度 = 该量测 M1 conf；
      - 本帧没有该类量测时沿用 previous_states 的对应槽。

    Returns:
        [B, num_tracks, 6]。
    """
    probabilities = network_output.probabilities  # [B,50,4]
    velocities = network_output.velocities        # [B,50,2]
    batch_size = outputs_t.shape[0]
    device = outputs_t.device
    batch_index = torch.arange(batch_size, device=device)

    decoded = []
    for track_index in range(config.NUM_TRACKS):
        target_class = track_index + 1
        is_class = probabilities.argmax(dim=-1) == target_class  # [B,50]
        class_prob = probabilities[..., target_class].masked_fill(~is_class, float("-inf"))
        best_query = class_prob.argmax(dim=-1)  # [B]
        has_class = is_class.any(dim=-1)        # [B]

        position = outputs_t[batch_index, best_query.clamp(min=0), :2]          # [B,2]
        velocity = velocities[batch_index, best_query.clamp(min=0)]             # [B,2]
        confidence = confidence_logits_t[batch_index, best_query.clamp(min=0)]  # [B,2]
        state = torch.cat(
            (
                position[:, 0:1], velocity[:, 0:1],
                position[:, 1:2], velocity[:, 1:2],
                confidence,
            ),
            dim=-1,
        )  # [B,6]
        state = torch.where(has_class[:, None], state, previous_states[:, track_index])
        decoded.append(state)
    return torch.stack(decoded, dim=1)  # [B,3,6]


def run_autoregressive(outputs, confidence_logits, model):
    """自回归滚动：用自生成航迹历史替代标签历史。

    前 HISTORY_STEPS 帧冷启动，之后每帧用 track_states[t-H:t] 拼历史喂网络，
    再用网络输出解码出本帧航迹状态。解码状态被 detach，梯度不穿越时间。

    Returns:
        frame_indices: list[int]，实际前向的帧下标（>= HISTORY_STEPS）。
        network_outputs: list[DANetOutput]，逐帧网络输出（训练时有梯度）。
        track_states: list[T]，逐帧航迹状态 [B, num_tracks, 6]。
    """
    track_states = cold_start_track_states(outputs, confidence_logits)
    frame_indices = []
    network_outputs = []
    for frame_index in range(config.HISTORY_STEPS, outputs.shape[1]):
        history = pack_track_history(
            track_states[frame_index - config.HISTORY_STEPS:frame_index]
        )
        measurement_input = torch.cat(
            (outputs[:, frame_index, :, :2], confidence_logits[:, frame_index]), dim=-1
        )
        network_output = model(history, measurement_input)
        frame_indices.append(frame_index)
        network_outputs.append(network_output)
        with torch.no_grad():
            new_state = decode_track_states(
                network_output,
                outputs[:, frame_index, :, :2],
                confidence_logits[:, frame_index],
                track_states[frame_index - 1],
            )
            track_states.append(new_state)
    return frame_indices, network_outputs, track_states


def _masked_cross_entropy(probabilities, targets, mask):
    """逐样本 masked CE，返回 [B]；只用于挑选槽位排列，不回传梯度。"""
    one_hot = F.one_hot(targets, num_classes=probabilities.shape[-1]).to(probabilities.dtype)
    log_probabilities = probabilities.clamp_min(1e-8).log()
    per_measurement = -(one_hot * log_probabilities).sum(dim=-1)
    masked = per_measurement * mask.to(probabilities.dtype)
    return masked.sum(dim=1) / mask.sum(dim=1).clamp_min(1).to(probabilities.dtype)


def _slot_permutation_maps(num_tracks, num_classes, device):
    """穷举槽位编号重排：0=杂波固定，类别 1..N 按排列重映射，返回 [P, C]。"""
    permutations = list(itertools.permutations(range(num_tracks)))
    maps = [[0] + [slot + 1 for slot in permutation] for permutation in permutations]
    tensor = torch.tensor(maps, dtype=torch.long, device=device)
    if tensor.shape[1] != num_classes:
        raise ValueError(
            f"槽位排列需要 {tensor.shape[1]} 个类别（杂波 + NUM_TRACKS），"
            f"实际 NUM_CLASSES={num_classes}"
        )
    return tensor


def compute_window_loss(model, batch, criterion, device):
    """窗口损失：逐帧 MCD + 速度回归。

    冷启动是"谁先过阈值谁占小槽位"，槽位编号与 GT 槽位编号没有固定对应，
    因此分类项按槽位编号做置换不变处理：每个样本在整个窗口里选一个固定排列
    （NUM_TRACKS! 种），使该样本整窗的 masked CE 之和最小，再按该排列计算
    MCD（含 Dice）。速度回归的目标由几何最近 GT 槽决定，与编号无关，不参与排列。
    """
    outputs, confidence_logits, _features, labels = [item.to(device) for item in batch]
    window_loss = outputs.new_zeros(())
    window_mcd = outputs.new_zeros(())
    window_regression = outputs.new_zeros(())

    frame_indices, network_outputs, _track_states = run_autoregressive(
        outputs, confidence_logits, model
    )
    permutation_maps = _slot_permutation_maps(
        config.NUM_TRACKS, config.NUM_CLASSES, outputs.device
    )

    frames = []
    permutation_totals = None
    for frame_index, network_output in zip(frame_indices, network_outputs):
        target_class = build_assignment_targets(
            outputs[:, frame_index, :, :2],
            labels[:, frame_index],
            radius_m=config.ASSIGNMENT_RADIUS_M,
        )
        mask = torch.ones_like(target_class, dtype=network_output.probabilities.dtype)

        # 速度回归：标签为相邻帧 GT 位置差（位移，未除以 Δt），
        # 只在被判为前景（属于某条航迹）的量测上监督，杂波量测不产生梯度。
        # 目标由几何最近 GT 槽决定，与类别编号无关，所以不随槽位置换变化。
        velocity_labels = labels[:, frame_index] - labels[:, frame_index - 1]
        foreground_mask = target_class > 0
        slot = (target_class - 1).clamp(min=0)
        velocity_target = velocity_labels.gather(
            1, slot.unsqueeze(-1).expand(-1, -1, velocity_labels.shape[-1])
        )
        frames.append((network_output, target_class, mask, velocity_target, foreground_mask))

        # 选择排列时只看 masked CE：Dice 是 batch 级统计量，无法逐样本拆分，
        # 而排列必须按样本选，因此用 CE 作为选择依据。
        with torch.no_grad():
            per_permutation = torch.stack(
                [
                    _masked_cross_entropy(
                        network_output.probabilities[..., mapping], target_class, mask
                    )
                    for mapping in permutation_maps
                ],
                dim=1,
            )  # [B, P]
        permutation_totals = (
            per_permutation
            if permutation_totals is None
            else permutation_totals + per_permutation
        )

    if permutation_totals is None:
        return {"total": window_loss, "mcd": window_mcd, "regression": window_regression}

    best_permutation = permutation_totals.argmin(dim=1)  # [B]
    selected_maps = permutation_maps[best_permutation]   # [B, C]

    for network_output, target_class, mask, velocity_target, foreground_mask in frames:
        probabilities = network_output.probabilities
        index = selected_maps[:, None, :].expand(-1, probabilities.shape[1], -1)
        permuted = torch.gather(probabilities, 2, index)
        mcd_loss = criterion(permuted, target_class, mask)
        regression_loss = velocity_regression_loss(
            network_output.velocities, velocity_target, foreground_mask
        )

        window_loss = window_loss + mcd_loss + config.DA_REGRESSION_WEIGHT * regression_loss
        window_mcd = window_mcd + mcd_loss
        window_regression = window_regression + regression_loss

    num_frames = max(outputs.shape[1] - config.HISTORY_STEPS, 1)
    return {
        "total": window_loss / num_frames,
        "mcd": window_mcd / num_frames,
        "regression": window_regression / num_frames,
    }


def train_one_epoch(model, loader, optimizer, criterion, device, max_windows=None):
    model.train()
    total_loss = 0.0
    total_mcd = 0.0
    total_regression = 0.0
    frame_count = 0

    progress = tqdm(loader, desc="train", unit="window")
    for window_index, batch in enumerate(progress):
        if max_windows is not None and window_index >= max_windows:
            break
        optimizer.zero_grad(set_to_none=True)
        losses = compute_window_loss(model, batch, criterion, device)
        losses["total"].backward()
        optimizer.step()
        total_loss += float(losses["total"].detach())
        total_mcd += float(losses["mcd"].detach())
        total_regression += float(losses["regression"].detach())
        frame_count += 1
        progress.set_postfix(
            loss=f"{float(losses['total'].detach()):.4f}",
            mcd=f"{float(losses['mcd'].detach()):.4f}",
            reg=f"{float(losses['regression'].detach()):.4f}",
        )

    windows = max_windows or len(loader)
    return (
        total_loss / max(windows, 1),
        total_mcd / max(windows, 1),
        total_regression / max(windows, 1),
        frame_count,
    )


@torch.no_grad()
def validate(model, loader, criterion, device, max_windows=None):
    model.eval()
    total_loss = 0.0
    total_mcd = 0.0
    total_regression = 0.0
    processed = 0
    progress = tqdm(loader, desc="val", unit="window")
    for window_index, batch in enumerate(progress):
        if max_windows is not None and window_index >= max_windows:
            break
        losses = compute_window_loss(model, batch, criterion, device)
        total_loss += float(losses["total"])
        total_mcd += float(losses["mcd"])
        total_regression += float(losses["regression"])
        processed += 1
        progress.set_postfix(
            loss=f"{float(losses['total']):.4f}",
            mcd=f"{float(losses['mcd']):.4f}",
            reg=f"{float(losses['regression']):.4f}",
        )
    return (
        total_loss / max(processed, 1),
        total_mcd / max(processed, 1),
        total_regression / max(processed, 1),
        processed,
    )


def train_from_config() -> None:
    if not config.IS_COLD:
        raise RuntimeError("当前训练入口只支持 IS_COLD=True 的 M1 冻结缓存")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = load_track_sequence_cold_dataset(
        data_root_folder=str(config.TRAIN_ROOT),
        cache_root_folder=str(config.CACHE_ROOT),
        window_size=config.WINDOW_SIZE,
        stride=config.STRIDE,
        full_sequence=False,
    )
    val_dataset = load_track_sequence_cold_dataset(
        data_root_folder=str(config.VAL_ROOT),
        cache_root_folder=str(config.VAL_CACHE_ROOT),
        window_size=config.WINDOW_SIZE,
        stride=config.STRIDE,
        full_sequence=False,
    )
    loader = DataLoader(
        train_dataset,
        batch_size=config.DA_BATCH_SIZE,
        shuffle=True,
        num_workers=8,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.DA_BATCH_SIZE,
        shuffle=False,
        num_workers=8,
    )
    model = DANet(
        num_tracks=config.NUM_TRACKS,
        history_steps=config.HISTORY_STEPS,
        measurement_dim=config.MEASUREMENT_DIM,
        num_classes=config.NUM_CLASSES,
        d_model=config.DA_D_MODEL,
        num_heads=config.DA_NUM_HEADS,
        encoder_layers=config.DA_ENCODER_LAYERS,
        decoder_layers=config.DA_DECODER_LAYERS,
        feedforward_dim=config.DA_FEEDFORWARD_DIM,
        dropout=config.DA_DROPOUT,
        max_measurements=config.MAX_MEASUREMENTS,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.DA_LR)
    criterion = MCDLoss(smooth=1.0)
    config.DA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, config.DA_EPOCHS + 1):
        train_loss, train_mcd, train_regression, train_windows = train_one_epoch(
            model, loader, optimizer, criterion, device, config.DA_MAX_WINDOWS
        )
        val_loss, val_mcd, val_regression, val_windows = validate(
            model, val_loader, criterion, device, config.DA_VAL_MAX_WINDOWS
        )
        print(
            f"epoch {epoch}/{config.DA_EPOCHS}: "
            f"train_loss={train_loss:.6f} (mcd={train_mcd:.6f}, reg={train_regression:.6f}), "
            f"val_loss={val_loss:.6f} (mcd={val_mcd:.6f}, reg={val_regression:.6f}), "
            f"train_windows={train_windows}, val_windows={val_windows}, device={device}"
        )
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "train_mcd": train_mcd,
            "train_regression": train_regression,
            "val_loss": val_loss,
            "val_mcd": val_mcd,
            "val_regression": val_regression,
            "config": {
                "history_steps": config.HISTORY_STEPS,
                "num_tracks": config.NUM_TRACKS,
                "num_classes": config.NUM_CLASSES,
                "measurement_dim": config.MEASUREMENT_DIM,
                "d_model": config.DA_D_MODEL,
                "num_heads": config.DA_NUM_HEADS,
                "encoder_layers": config.DA_ENCODER_LAYERS,
                "decoder_layers": config.DA_DECODER_LAYERS,
                "feedforward_dim": config.DA_FEEDFORWARD_DIM,
                "dropout": config.DA_DROPOUT,
            },
        }
        torch.save(checkpoint, config.DA_OUTPUT_DIR / "last.pt")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, config.DA_OUTPUT_DIR / "best.pt")
            print(f"saved best checkpoint: {config.DA_OUTPUT_DIR / 'best.pt'}")


if __name__ == "__main__":
    train_from_config()
