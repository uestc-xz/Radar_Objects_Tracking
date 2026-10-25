"""Transformer-DA 最小训练入口。"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
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
from lossfunc import MCDLoss  # noqa: E402
from original.my_dataset.track_sequence_dataset import (  # noqa: E402
    load_track_sequence_cold_dataset,
)
from original.my_track.association import build_assignment_targets  # noqa: E402


def build_history_states(
    outputs: torch.Tensor,
    confidence_logits: torch.Tensor,
    labels: torch.Tensor,
    frame_index: int,
) -> torch.Tensor:
    """从当前窗口历史帧构造 [B,3,H,6] 的编码器输入。"""
    start = frame_index - config.HISTORY_STEPS
    history_positions = labels[:, start:frame_index]
    previous_positions = torch.cat((labels[:, :1], labels[:, :-1]), dim=1)
    history_velocity = (
        labels[:, start:frame_index] - previous_positions[:, start:frame_index]
    ) / config.DELTA_T

    confidence_by_track = []
    for offset in range(config.HISTORY_STEPS):
        absolute_index = start + offset
        distances = torch.cdist(
            history_positions[:, offset], outputs[:, absolute_index, :, :2]
        )
        nearest_query = distances.argmin(dim=-1)
        nearest_confidence = torch.gather(
            confidence_logits[:, absolute_index],
            1,
            nearest_query.unsqueeze(-1).expand(-1, -1, 2),
        )
        valid_target = history_positions[:, offset].abs().sum(dim=-1) > 0
        confidence_by_track.append(nearest_confidence * valid_target.unsqueeze(-1))
    history_confidence = torch.stack(confidence_by_track, dim=1)

    states = torch.cat(
        (
            history_positions[..., 0:1], history_velocity[..., 0:1],
            history_positions[..., 1:2], history_velocity[..., 1:2],
            history_confidence,
        ),
        dim=-1,
    )
    return states.permute(0, 2, 1, 3)


def compute_window_loss(model, batch, criterion, device):
    outputs, confidence_logits, _features, labels = [item.to(device) for item in batch]
    window_loss = outputs.new_zeros(())

    for frame_index in range(config.HISTORY_STEPS, outputs.shape[1]):
        history_states = build_history_states(
            outputs, confidence_logits, labels, frame_index
        )
        measurement_input = torch.cat(
            (outputs[:, frame_index, :, :2], confidence_logits[:, frame_index]),
            dim=-1,
        )
        network_output = model(history_states, measurement_input)
        target_class = build_assignment_targets(
            outputs[:, frame_index, :, :2],
            labels[:, frame_index],
            radius_m=config.ASSIGNMENT_RADIUS_M,
        )
        mask = torch.ones_like(target_class, dtype=network_output.probabilities.dtype)
        window_loss = window_loss + criterion(
            network_output.probabilities, target_class, mask
        )

    return window_loss / max(outputs.shape[1] - config.HISTORY_STEPS, 1)


def train_one_epoch(model, loader, optimizer, criterion, device, max_windows=None):
    model.train()
    total_loss = 0.0
    frame_count = 0

    progress = tqdm(loader, desc="train", unit="window")
    for window_index, batch in enumerate(progress):
        if max_windows is not None and window_index >= max_windows:
            break
        optimizer.zero_grad(set_to_none=True)
        window_loss = compute_window_loss(model, batch, criterion, device)
        window_loss.backward()
        optimizer.step()
        total_loss += float(window_loss.detach())
        frame_count += 1
        progress.set_postfix(loss=f"{float(window_loss.detach()):.4f}")

    windows = max_windows or len(loader)
    return total_loss / max(windows, 1), frame_count


@torch.no_grad()
def validate(model, loader, criterion, device, max_windows=None):
    model.eval()
    total_loss = 0.0
    processed = 0
    progress = tqdm(loader, desc="val", unit="window")
    for window_index, batch in enumerate(progress):
        if max_windows is not None and window_index >= max_windows:
            break
        window_loss = compute_window_loss(model, batch, criterion, device)
        total_loss += float(window_loss)
        processed += 1
        progress.set_postfix(loss=f"{float(window_loss):.4f}")
    return total_loss / max(processed, 1), processed


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
        train_loss, train_windows = train_one_epoch(
            model, loader, optimizer, criterion, device, config.DA_MAX_WINDOWS
        )
        val_loss, val_windows = validate(
            model, val_loader, criterion, device, config.DA_VAL_MAX_WINDOWS
        )
        print(
            f"epoch {epoch}/{config.DA_EPOCHS}: train_loss={train_loss:.6f}, "
            f"val_loss={val_loss:.6f}, train_windows={train_windows}, "
            f"val_windows={val_windows}, device={device}"
        )
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
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
