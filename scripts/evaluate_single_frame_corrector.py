"""Sweep presence thresholds for a saved single-frame corrector checkpoint."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from original.my_track import config
from original.my_track.cluster_residual_corrector import ClusterResidualOutput
from original.my_track.single_frame_evaluation import (
    compare_anchor_localization, evaluate_presence_thresholds,
)
from scripts.train_single_frame_corrector import CachedM1Frames, build_single_frame_model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--val-root", default=config.VAL_ROOT)
    parser.add_argument("--cache-root", default=config.CACHE_ROOT)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--thresholds", type=float, nargs="+",
                        default=[round(value, 2) for value in np.arange(0.1, 1.0, 0.05)])
    parser.add_argument("--match-radius", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")
    if args.batch_size < 1 or args.workers < 0:
        parser.error("batch-size must be positive and workers non-negative")
    if args.output is not None and args.output.exists():
        parser.error(f"output already exists; choose a new path: {args.output}")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_type = checkpoint.get("model_type", "set")
    model = build_single_frame_model(model_type).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    dataset = CachedM1Frames(
        args.val_root, str(Path(args.cache_root) / "fast_dataset_multi_test")
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    all_positions, all_logits, all_targets, all_masks, all_anchors = [], [], [], [], []
    with torch.no_grad():
        for positions, scores, targets in tqdm(loader, desc="threshold-sweep"):
            positions = positions.to(device, non_blocking=True)
            scores = scores.to(device, non_blocking=True)
            output = model(positions, scores)
            predicted_positions, presence_logits = output[:2]
            all_positions.append(predicted_positions.cpu().numpy())
            all_logits.append(presence_logits.cpu().numpy())
            all_targets.append(targets.numpy())
            if isinstance(output, ClusterResidualOutput):
                all_masks.append(output.valid_mask.cpu().numpy())
                all_anchors.append(output.anchors.cpu().numpy())

    threshold_metrics = evaluate_presence_thresholds(
        np.concatenate(all_positions),
        np.concatenate(all_logits),
        np.concatenate(all_targets),
        args.thresholds,
        match_radius_m=args.match_radius,
        prediction_mask=np.concatenate(all_masks) if all_masks else None,
    )
    payload = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "model_type": model_type,
        "validation_frames": len(dataset),
        "mode": "read-only threshold sweep; coordinates and checkpoint unchanged",
        "thresholds": list(threshold_metrics.values()),
    }
    if all_anchors:
        anchors = np.concatenate(all_anchors)
        masks = np.concatenate(all_masks)
        targets = np.concatenate(all_targets)
        # Identical scores/masks isolate coordinate correction from rejection.
        payload["anchor_thresholds"] = list(evaluate_presence_thresholds(
            anchors, np.concatenate(all_logits), targets, args.thresholds,
            match_radius_m=args.match_radius, prediction_mask=masks,
        ).values())
        payload["fixed_anchor_localization"] = compare_anchor_localization(
            anchors, np.concatenate(all_positions), targets, masks, args.match_radius,
        )
    rendered = json.dumps(payload, indent=2)
    if args.output is None:
        print(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"Wrote threshold metrics: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
