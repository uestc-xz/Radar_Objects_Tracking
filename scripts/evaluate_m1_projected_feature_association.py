"""Compare spatial, raw-M1 and learned-projection association on validation cache."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from original.my_dataset.track_sequence_dataset import TrackSequenceColdDataset
from original.my_track import config
from original.my_track.feature_lifecycle_tracker import FeatureLifecycleTracker
from original.my_track.lifecycle_tracker import LifecycleTracker
from original.my_track.m1_feature_projection import FeatureProjection
from original.my_track.tracking_diagnostics import tracking_diagnostics


def evaluate(records, model, device, feature_weight=.35, max_feature_age=8, progress=None):
    """Replay each sequence independently and aggregate detection/identity metrics."""
    variants = ('spatial', 'raw_feature', 'projected_feature')
    result = {name: [] for name in variants}
    settings = dict(
        hist_len=config.HIST_LEN, max_tracks=3, score_threshold=.8, birth_threshold=.96,
        cluster_radius_m=.75, association_radius_m=1., confirm_fraction=.6,
        tentative_max_missed=2, max_missed=config.HIST_LEN, output_max_missed=2,
        confidence_decay=.95, output_confidence=.5, velocity_alpha=.2,
        max_step_m=.2, prediction_decay=.8, no_death=False,
        weak_updates=True, association_policy='cascade',
    )
    for record in records:
        trackers = {
            'spatial': LifecycleTracker(**settings),
            'raw_feature': FeatureLifecycleTracker(
                **settings, feature_dim=128, feature_weight=feature_weight,
                max_feature_age=max_feature_age,
            ),
            'projected_feature': FeatureLifecycleTracker(
                **settings, feature_dim=model.output_dim, feature_weight=feature_weight,
                max_feature_age=max_feature_age,
            ),
        }
        outputs = {name: [] for name in variants}
        targets = []
        for frame_index, (cache_path, label_path, _sample_id) in enumerate(record['frames']):
            with np.load(cache_path) as cache:
                out = np.asarray(cache['out'])
                conf = np.asarray(cache['conf'])
                features = np.asarray(cache['feat'], dtype=np.float32)
            label = np.load(label_path)
            if (out.shape != (50, 4) or conf.shape != (50, 2) or
                    features.shape != (50, 128) or label.shape != (10, 6) or
                    not all(np.isfinite(value).all() for value in (out, conf, features, label))):
                raise ValueError(f'invalid cache or label shape/value: {cache_path}')
            logits = conf.astype(np.float64)
            logits -= logits.max(axis=-1, keepdims=True)
            exp_logits = np.exp(logits)
            scores = exp_logits[:, 1] / exp_logits.sum(axis=-1)
            positions = out[:, :2].astype(np.float64)
            with torch.no_grad():
                projected = model(torch.as_tensor(features, device=device)).cpu().numpy()
            outputs['spatial'].append(trackers['spatial'].update(positions, scores))
            outputs['raw_feature'].append(trackers['raw_feature'].update(
                positions, scores, candidate_features=features,
            ))
            outputs['projected_feature'].append(trackers['projected_feature'].update(
                positions, scores, candidate_features=projected,
            ))
            targets.append(label.mean(axis=0).reshape(3, 2).astype(np.float64))
            if progress is not None and (frame_index % 1000 == 0 or frame_index + 1 == len(record['frames'])):
                progress(record['folder'], frame_index + 1, len(record['frames']))
        targets = np.asarray(targets)
        for name in variants:
            metrics = tracking_diagnostics(outputs[name], targets)
            metrics_05 = tracking_diagnostics(outputs[name], targets, gate_m=.5)
            result[name].append({
                'folder': record['folder'], 'frames': len(targets),
                'metrics': metrics, 'metrics_0_5m': metrics_05,
                'management': dict(trackers[name].statistics),
                'feature_diagnostics': (dict(trackers[name].feature_statistics)
                                        if name != 'spatial' else None),
            })
    return result


def summarize(rows):
    keys = ('tp', 'fp', 'fn', 'error_sum_m', 'gt_id_switches', 'track_owner_switches',
            'predicted_points', 'gt_points', 'idtp', 'idfp', 'idfn')
    totals = {key: sum(row['metrics'][key] for row in rows) for key in keys}
    tp, fp, fn = totals['tp'], totals['fp'], totals['fn']
    totals.update(
        precision=tp / (tp + fp) if tp + fp else 0.,
        recall=tp / (tp + fn) if tp + fn else 0.,
        f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
        matched_mae_m=totals['error_sum_m'] / tp if tp else None,
        identity_f1=(2 * totals['idtp'] / (totals['predicted_points'] + totals['gt_points'])
                     if totals['predicted_points'] + totals['gt_points'] else 0.),
    )
    return totals


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-sequences', type=int, default=0)
    parser.add_argument('--max-frames', type=int, default=0,
                        help='per-sequence prefix; 0 means full sequences')
    parser.add_argument('--feature-weight', type=float, default=.35)
    parser.add_argument('--max-feature-age', type=int, default=8)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('output already exists; choose a fresh filename')
    if args.max_frames < 0 or args.max_sequences < 0 or args.max_feature_age < 0:
        parser.error('limits must be non-negative')
    if not np.isfinite(args.feature_weight) or args.feature_weight < 0:
        parser.error('feature-weight must be finite and non-negative')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model = FeatureProjection(input_dim=checkpoint['input_dim'],
                              output_dim=checkpoint['output_dim'])
    model.load_state_dict(checkpoint['model_state'])
    device = torch.device(args.device)
    model.to(device).eval()
    dataset = TrackSequenceColdDataset(
        config.VAL_ROOT, str(Path(config.CACHE_ROOT) / 'fast_dataset_multi_test'),
        config.WINDOW_SIZE, config.STRIDE, full_sequence=True,
    )
    records = [dataset[index] for index in range(len(dataset))]
    if args.max_sequences:
        records = records[:args.max_sequences]
    if args.max_frames:
        records = [{**record, 'frames': record['frames'][:args.max_frames]}
                   for record in records]
    if not records:
        parser.error('validation dataset is empty')
    result = evaluate(
        records, model, device, args.feature_weight, args.max_feature_age,
        progress=lambda folder, frame, length: print(f'{folder}: {frame}/{length}', flush=True),
    )
    summaries = {name: summarize(rows) for name, rows in result.items()}
    payload = {
        'mode': 'independent frozen-M1 embedding projection; GT only used for offline metrics',
        'checkpoint': str(args.checkpoint),
        'sequences': [record['folder'] for record in records],
        'settings': {'feature_weight': args.feature_weight,
                     'max_feature_age': args.max_feature_age,
                     'max_frames': args.max_frames},
        'summaries': summaries, 'variants': result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    for name, metrics in summaries.items():
        print(f"{name}: F1={metrics['f1']:.4f} identity_F1={metrics['identity_f1']:.4f} "
              f"IDSW={metrics['gt_id_switches']} MAE={metrics['matched_mae_m']:.4f}m")
    print(f'Wrote: {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
