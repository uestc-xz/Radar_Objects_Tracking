"""Train and evaluate the isolated frozen-M1 single-frame set corrector."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from original.my_dataset.track_sequence_dataset import TrackSequenceColdDataset
from original.my_track import config
from original.my_track.cluster_residual_corrector import (
    ClusterResidualCorrector,
    ClusterResidualOutput,
    cluster_residual_loss,
)
from original.my_track.m1_frame_baseline import match_frame
from original.my_track.single_frame_corrector import (
    SingleFrameSetPredictor,
    single_frame_set_loss,
)


class CachedM1Frames(Dataset):
    """Flatten independent recordings into shuffled, frame-local cache items."""

    def __init__(self, data_root: str, cache_root: str):
        index = TrackSequenceColdDataset(
            data_root, cache_root, window_size=1, stride=1, full_sequence=True
        )
        self.frames = [
            frame
            for record_index in range(len(index))
            for frame in index[record_index]["frames"]
        ]
        if not self.frames:
            raise ValueError(f"No labeled frames found under {data_root}")

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        cache_path, label_path, _sample_id = self.frames[index]
        with np.load(cache_path) as cache:
            positions = np.asarray(cache["out"][:, :2], dtype=np.float32).copy()
            logits = np.asarray(cache["conf"], dtype=np.float32)
        logits = logits - logits.max(axis=-1, keepdims=True)
        exp_logits = np.exp(logits)
        scores = (exp_logits[:, 1] / exp_logits.sum(axis=-1)).astype(np.float32)
        target_positions = np.load(label_path).mean(axis=0).reshape(3, 2).astype(np.float32)
        if positions.shape != (50, 2) or scores.shape != (50,):
            raise ValueError(f"Unexpected M1 cache shape in {cache_path}")
        if not np.isfinite(positions).all() or not np.isfinite(scores).all():
            raise ValueError(f"Non-finite M1 output in {cache_path}")
        return (
            torch.from_numpy(positions),
            torch.from_numpy(scores),
            torch.from_numpy(target_positions),
        )


def _move_batch(batch, device):
    return tuple(value.to(device, non_blocking=True) for value in batch)


def build_single_frame_model(model_type="set"):
    """Keep old checkpoints/defaults valid while isolating the residual trial."""
    if model_type == "set":
        return SingleFrameSetPredictor()
    if model_type == "cluster-residual":
        return ClusterResidualCorrector()
    if model_type == "cluster-evidence":
        return ClusterResidualCorrector(use_cluster_evidence=True)
    raise ValueError(f"Unknown single-frame model: {model_type}")


def single_frame_training_loss(output, targets):
    if isinstance(output, ClusterResidualOutput):
        return cluster_residual_loss(output, targets)
    return single_frame_set_loss(*output, targets)


def save_single_frame_checkpoints(output_dir, checkpoint, best_loss, best_f1):
    """Do not discard a better detector just because another epoch has lower loss."""
    validation = checkpoint["validation"]
    if validation["total"] < best_loss:
        best_loss = validation["total"]
        torch.save(checkpoint, output_dir / "best_single_frame_corrector.pth")
    if validation["f1"] > best_f1:
        best_f1 = validation["f1"]
        torch.save(checkpoint, output_dir / "best_f1_single_frame_corrector.pth")
    torch.save(checkpoint, output_dir / "last_single_frame_corrector.pth")
    return best_loss, best_f1


def _evaluate(model, loader, device, presence_threshold, match_radius_m):
    model.eval()
    totals = {"total": 0.0, "position": 0.0, "presence": 0.0}
    tp = fp = fn = 0
    error_sum_m = 0.0
    frame_count = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="val", leave=False):
            positions, scores, targets = _move_batch(batch, device)
            output = model(positions, scores)
            predicted, presence_logits = output[:2]
            losses = single_frame_training_loss(output, targets)
            batch_size = len(positions)
            frame_count += batch_size
            for name in totals:
                totals[name] += float(losses[name]) * batch_size
            active = presence_logits.sigmoid() >= presence_threshold
            if isinstance(output, ClusterResidualOutput):
                active &= output.valid_mask
            predicted_np = predicted.cpu().numpy()
            active_np = active.cpu().numpy()
            targets_np = targets.cpu().numpy()
            for prediction, is_active, target in zip(predicted_np, active_np, targets_np):
                match = match_frame(
                    prediction[is_active], target, match_radius_m=match_radius_m
                )
                tp += match.true_positives
                fp += match.false_positives
                fn += match.false_negatives
                error_sum_m += match.error_sum_m

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        **{name: value / max(frame_count, 1) for name, value in totals.items()},
        "frames": frame_count,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matched_mae_m": error_sum_m / tp if tp else 0.0,
        "presence_threshold": presence_threshold,
        "match_radius_m": match_radius_m,
    }


def _train_epoch(model, loader, optimizer, device, epoch, epochs):
    model.train()
    totals = {"total": 0.0, "position": 0.0, "presence": 0.0}
    frame_count = 0
    progress = tqdm(loader, desc=f"Epoch {epoch}/{epochs} [train]")
    for positions, scores, targets in progress:
        positions, scores, targets = _move_batch((positions, scores, targets), device)
        output = model(positions, scores)
        losses = single_frame_training_loss(output, targets)
        optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        batch_size = len(positions)
        for name in totals:
            totals[name] += float(losses[name].detach()) * batch_size
        frame_count += batch_size
        progress.set_postfix(loss=f"{totals['total'] / frame_count:.4f}")
    return {name: value / max(frame_count, 1) for name, value in totals.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", default=config.TRAIN_ROOT)
    parser.add_argument("--val-root", default=config.VAL_ROOT)
    parser.add_argument("--cache-root", default=config.CACHE_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New, dedicated directory for this experiment's logs/checkpoint.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--model", choices=("set", "cluster-residual", "cluster-evidence"), default="set")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--presence-threshold", type=float, default=None,
                        help="Defaults to 0.5 for learned presence probabilities; independent of M1 proposal thresholds.")
    parser.add_argument("--match-radius", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    if args.presence_threshold is None:
        args.presence_threshold = 0.5
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0 or args.lr <= 0:
        parser.error("epochs/batch-size/lr must be positive and workers non-negative")
    if not 0 <= args.presence_threshold <= 1 or args.match_radius < 0:
        parser.error("presence threshold must be in [0,1] and match radius non-negative")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("output directory is not empty; choose a new directory to preserve prior runs")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    train_data_root = Path(args.train_root)
    val_data_root = Path(args.val_root)
    train_cache_root = Path(args.cache_root) / "fast_dataset_multi_train"
    val_cache_root = Path(args.cache_root) / "fast_dataset_multi_test"
    train_dataset = CachedM1Frames(str(train_data_root), str(train_cache_root))
    val_dataset = CachedM1Frames(str(val_data_root), str(val_cache_root))
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=device.type == "cuda",
    )
    model = build_single_frame_model(args.model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_val = float("inf")
    best_f1 = -1.0

    print(f"model={args.model}; device={device}; train_frames={len(train_dataset)}; val_frames={len(val_dataset)}")
    if args.model == "cluster-residual":
        initial = _evaluate(model, val_loader, device, args.presence_threshold, args.match_radius)
        (args.output_dir / "initial_validation.json").write_text(
            json.dumps(initial, indent=2), encoding="utf-8"
        )
        best_val, best_f1 = save_single_frame_checkpoints(
            args.output_dir,
            {"epoch": 0, "model_type": args.model, "run_arguments": vars(args) | {"output_dir": str(args.output_dir)},
             "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
             "validation": initial},
            best_val, best_f1,
        )
        print(f"initial: F1={initial['f1']:.4f} MAE={initial['matched_mae_m']:.3f}m")
    for epoch in range(1, args.epochs + 1):
        train_losses = _train_epoch(model, train_loader, optimizer, device, epoch, args.epochs)
        validation = _evaluate(
            model, val_loader, device, args.presence_threshold, args.match_radius
        )
        row = {"epoch": epoch, **{
            f"train_{key}": value for key, value in train_losses.items()
        }, **{
            f"val_{key}": value for key, value in validation.items()
        }}
        history.append(row)
        print(
            f"epoch {epoch}/{args.epochs}: train={train_losses['total']:.5f} "
            f"train_pos={train_losses['position']:.5f} "
            f"train_presence={train_losses['presence']:.5f} "
            f"val={validation['total']:.5f} pos={validation['position']:.5f} "
            f"P={validation['precision']:.4f} R={validation['recall']:.4f} "
            f"F1={validation['f1']:.4f} MAE={validation['matched_mae_m']:.3f}m"
        )
        best_val, best_f1 = save_single_frame_checkpoints(
            args.output_dir,
            {"epoch": epoch, "model_type": args.model,
             "run_arguments": vars(args) | {"output_dir": str(args.output_dir)},
             "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
             "validation": validation},
            best_val, best_f1,
        )
        (args.output_dir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
    print(f"best checkpoint: {args.output_dir / 'best_single_frame_corrector.pth'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
