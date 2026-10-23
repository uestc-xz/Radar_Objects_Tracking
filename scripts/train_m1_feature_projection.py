"""Train a small identity projection from frozen M1 cache and sequence-local GT."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from original.my_dataset.track_sequence_dataset import TrackSequenceColdDataset
from original.my_track import config
from original.my_track.m1_feature_projection import (
    FeatureProjection,
    build_identity_triplets,
    match_candidate_features,
    triplet_cosine_loss,
)


def collect_triplets(records, score_threshold=.8, gate_m=.5, max_positive_gap=8,
                     max_negative_gap=1, progress=None):
    """Read frozen cache and construct independent triplets per sequence."""
    all_features, all_triplets = [], []
    for record in records:
        features, identities, frame_indices = [], [], []
        for frame_index, (cache_path, label_path, _sample_id) in enumerate(record['frames']):
            with np.load(cache_path) as cache:
                positions = np.asarray(cache['out'][:, :2])
                logits = np.asarray(cache['conf'], dtype=np.float64)
                cached_features = np.asarray(cache['feat'], dtype=np.float32)
            logits -= logits.max(axis=-1, keepdims=True)
            exp_logits = np.exp(logits)
            scores = exp_logits[:, 1] / exp_logits.sum(axis=-1)
            targets = np.load(label_path).mean(axis=0).reshape(3, 2)
            matched, matched_ids = match_candidate_features(
                positions, scores, cached_features, targets,
                gate_m=gate_m, score_threshold=score_threshold,
            )
            if len(matched):
                features.extend(matched)
                identities.extend(matched_ids)
                frame_indices.extend([frame_index] * len(matched))
            if progress is not None and (frame_index + 1) % 1000 == 0:
                progress(record['folder'], frame_index + 1, len(record['frames']))
        if not features:
            continue
        sequence_features = np.asarray(features, dtype=np.float32)
        triplets = build_identity_triplets(
            np.asarray(identities, dtype=np.int64),
            np.asarray(frame_indices, dtype=np.int64),
            max_positive_gap=max_positive_gap,
            max_negative_gap=max_negative_gap,
        )
        if len(triplets):
            offset = sum(len(value) for value in all_features)
            all_triplets.append(triplets + offset)
            all_features.append(sequence_features)
        if progress is not None:
            progress(record['folder'], len(record['frames']), len(record['frames']))
    if not all_features or not all_triplets:
        raise ValueError('no valid same-ID/different-ID triplets found in the selected records')
    return np.concatenate(all_features), np.concatenate(all_triplets)


def evaluate_triplet_retrieval(model, features, triplets, device, batch_size=4096):
    """Measure positive-vs-negative cosine ranking on held-out sequence triplets."""
    model.eval()
    raw_correct = projected_correct = total = 0
    with torch.no_grad():
        for start in range(0, len(triplets), batch_size):
            rows = triplets[start:start + batch_size]
            raw = torch.as_tensor(features[rows], dtype=torch.float32, device=device)
            raw = torch.nn.functional.normalize(raw, dim=-1, eps=1e-8)
            projected = model(raw)
            raw_positive = (raw[:, 0] * raw[:, 1]).sum(dim=-1)
            raw_negative = (raw[:, 0] * raw[:, 2]).sum(dim=-1)
            projected_positive = (projected[:, 0] * projected[:, 1]).sum(dim=-1)
            projected_negative = (projected[:, 0] * projected[:, 2]).sum(dim=-1)
            raw_correct += int((raw_positive > raw_negative).sum().item())
            projected_correct += int((projected_positive > projected_negative).sum().item())
            total += len(rows)
    return {
        'triplets': total,
        'raw_pairwise_accuracy': raw_correct / total if total else 0.,
        'projected_pairwise_accuracy': projected_correct / total if total else 0.,
    }


def _records(root, cache_root, max_sequences=0, max_frames=0):
    dataset = TrackSequenceColdDataset(
        root, cache_root, config.WINDOW_SIZE, config.STRIDE, full_sequence=True,
    )
    records = [dataset[index] for index in range(len(dataset))]
    if max_sequences:
        records = records[:max_sequences]
    if max_frames:
        records = [{**record, 'frames': record['frames'][:max_frames]} for record in records]
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--margin', type=float, default=.2)
    parser.add_argument('--score-threshold', type=float, default=.8)
    parser.add_argument('--gate-m', type=float, default=.5)
    parser.add_argument('--max-positive-gap', type=int, default=8)
    parser.add_argument('--max-negative-gap', type=int, default=1)
    parser.add_argument('--max-train-sequences', type=int, default=0)
    parser.add_argument('--max-val-sequences', type=int, default=0)
    parser.add_argument('--max-frames', type=int, default=0,
                        help='per-sequence prefix; 0 means full sequences')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error('output-dir already exists; choose a fresh experiment directory')
    if (args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or not 0 <= args.margin <= 2 or
            args.max_frames < 0 or args.max_train_sequences < 0 or args.max_val_sequences < 0):
        parser.error('invalid training limits or hyperparameters')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cache_base = Path(config.CACHE_ROOT)
    cache_train = cache_base / 'fast_dataset_multi_train'
    cache_val = cache_base / 'fast_dataset_multi_test'
    train_records = _records(config.TRAIN_ROOT, str(cache_train), args.max_train_sequences,
                             args.max_frames)
    val_records = _records(config.VAL_ROOT, str(cache_val), args.max_val_sequences,
                           args.max_frames)
    train_features, train_triplets = collect_triplets(
        train_records, args.score_threshold, args.gate_m,
        args.max_positive_gap, args.max_negative_gap,
        progress=lambda folder, frame, length: print(
            f'train cache {folder}: {frame}/{length}', flush=True),
    )
    val_features, val_triplets = collect_triplets(
        val_records, args.score_threshold, args.gate_m,
        args.max_positive_gap, args.max_negative_gap,
        progress=lambda folder, frame, length: print(
            f'val cache {folder}: {frame}/{length}', flush=True),
    )
    print(f"records train={len(train_records)} val={len(val_records)}; "
          f"matched train={len(train_features)} triplets={len(train_triplets)}; "
          f"val={len(val_features)} triplets={len(val_triplets)}; device={args.device}",
          flush=True)

    train_tensor = torch.as_tensor(train_features[train_triplets], dtype=torch.float32)
    loader = DataLoader(TensorDataset(train_tensor), batch_size=args.batch_size,
                        shuffle=True, drop_last=False, num_workers=0)
    device = torch.device(args.device)
    model = FeatureProjection(input_dim=train_features.shape[1], output_dim=32).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.
        seen = 0
        for (batch,) in loader:
            batch = batch.to(device)
            projected = model(batch.reshape(-1, batch.shape[-1])).reshape(-1, 3, 32)
            loss = triplet_cosine_loss(projected[:, 0], projected[:, 1], projected[:, 2],
                                       margin=args.margin)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(batch)
            seen += len(batch)
        metrics = evaluate_triplet_retrieval(model, val_features, val_triplets, device)
        row = {'epoch': epoch, 'train_triplet_loss': loss_sum / max(1, seen), **metrics}
        history.append(row)
        print(f"epoch {epoch}/{args.epochs}: loss={row['train_triplet_loss']:.5f} "
              f"raw_pair_acc={row['raw_pairwise_accuracy']:.4f} "
              f"projected_pair_acc={row['projected_pairwise_accuracy']:.4f}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint = args.output_dir / 'feature_projection.pt'
    torch.save({
        'model_state': model.cpu().state_dict(),
        'input_dim': int(train_features.shape[1]), 'output_dim': 32,
        'settings': vars(args) | {'output_dir': str(args.output_dir)},
        'history': history,
    }, checkpoint)
    manifest = {
        'mode': 'independent frozen-M1 feature projection; GT used only to construct offline labels',
        'train_sequences': [record['folder'] for record in train_records],
        'validation_sequences': [record['folder'] for record in val_records],
        'settings': vars(args) | {'output_dir': str(args.output_dir)},
        'train_matched_features': len(train_features), 'train_triplets': len(train_triplets),
        'validation_matched_features': len(val_features),
        'validation_triplets': len(val_triplets), 'history': history,
        'checkpoint': str(checkpoint),
    }
    with (args.output_dir / 'manifest.json').open('x', encoding='utf-8') as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, default=str, allow_nan=False)
        stream.write('\n')
    print(f'checkpoint: {checkpoint}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
