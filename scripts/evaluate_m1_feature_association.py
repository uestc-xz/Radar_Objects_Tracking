"""Evaluate an opt-in recent M1-feature association experiment against spatial lifecycle."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from original.my_dataset.track_sequence_dataset import TrackSequenceColdDataset
from original.my_track import config
from original.my_track.feature_lifecycle_tracker import FeatureLifecycleTracker
from original.my_track.lifecycle_tracker import LifecycleTracker
from original.my_track.tracking_diagnostics import tracking_diagnostics


def _new_trackers(hist_len, feature_weight, max_feature_age):
    settings = dict(
        hist_len=hist_len, max_tracks=3, score_threshold=.8, birth_threshold=.96,
        cluster_radius_m=.75, association_radius_m=1., confirm_fraction=.6,
        tentative_max_missed=2, max_missed=hist_len, output_max_missed=2,
        confidence_decay=.95, output_confidence=.5, velocity_alpha=.2,
        max_step_m=.2, prediction_decay=.8, no_death=False,
        weak_updates=True, association_policy='cascade',
    )
    return (
        LifecycleTracker(**settings),
        FeatureLifecycleTracker(**settings, feature_dim=128,
                                 feature_weight=feature_weight,
                                 max_feature_age=max_feature_age),
    )


def evaluate_records(records, hist_len, feature_weight, max_feature_age, progress=None):
    """Run spatial and appearance variants on the exact same cached M1 frames."""
    results = {'spatial': [], 'feature': []}
    for record in records:
        if not record['frames']:
            raise ValueError(f"empty sequence: {record['folder']}")
        spatial, feature = _new_trackers(hist_len, feature_weight, max_feature_age)
        predictions = {'spatial': [], 'feature': []}
        changed_assignment_frames = 0
        targets = []
        for index, (cache_path, label_path, sample_id) in enumerate(record['frames']):
            with np.load(cache_path) as cache:
                out, conf = np.asarray(cache['out']), np.asarray(cache['conf'])
                feats = np.asarray(cache['feat'])
            label = np.load(label_path)
            if (out.ndim != 2 or out.shape[1] != 4 or conf.shape != (len(out), 2) or
                    feats.shape != (len(out), 128) or label.shape != (10, 6) or
                    not all(np.isfinite(value).all() for value in (out, conf, feats, label))):
                raise ValueError(f'invalid M1 cache or label: {cache_path}')
            logits = conf.astype(np.float64)
            logits -= logits.max(axis=-1, keepdims=True)
            exp_logits = np.exp(logits)
            scores = exp_logits[:, 1] / exp_logits.sum(axis=-1)
            positions = out[:, :2].astype(np.float64)
            spatial_frame = spatial.update(positions, scores)
            feature_frame = feature.update(positions, scores, candidate_features=feats)
            predictions['spatial'].append(spatial_frame)
            predictions['feature'].append(feature_frame)
            spatial_by_id = {int(track_id): point for track_id, point in
                             zip(spatial_frame.track_ids, spatial_frame.positions)}
            feature_by_id = {int(track_id): point for track_id, point in
                             zip(feature_frame.track_ids, feature_frame.positions)}
            changed_assignment_frames += int(
                spatial_by_id.keys() != feature_by_id.keys() or any(
                    not np.allclose(spatial_by_id[track_id], feature_by_id[track_id], atol=1e-9)
                    for track_id in spatial_by_id.keys() & feature_by_id.keys()
                )
            )
            targets.append(label.mean(axis=0).reshape(3, 2).astype(np.float64))
            if progress is not None and (index % 1000 == 0 or index + 1 == len(record['frames'])):
                progress(record['folder'], index + 1, len(record['frames']))

        targets = np.asarray(targets)
        for name, tracker in (('spatial', spatial), ('feature', feature)):
            metrics = tracking_diagnostics(predictions[name], targets)
            strict_metrics = tracking_diagnostics(predictions[name], targets, gate_m=.5)
            results[name].append({
                'folder': record['folder'], 'frames': len(targets),
                'metrics': metrics, 'metrics_0_5m': strict_metrics,
                'management': dict(tracker.statistics),
                'feature_diagnostics': (dict(tracker.feature_statistics)
                                        if name == 'feature' else None),
                'changed_assignment_frames_vs_spatial': (changed_assignment_frames
                                                         if name == 'feature' else 0),
            })
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sequence', help='optional exact sequence folder, e.g. 20240913_3_8')
    parser.add_argument('--max-frames', type=int, default=0, help='per-sequence prefix; 0 means full')
    parser.add_argument('--hist-len', type=int, default=config.HIST_LEN)
    parser.add_argument('--feature-weight', type=float, default=.35)
    parser.add_argument('--max-feature-age', type=int, default=8)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('output already exists; choose a new filename')
    if args.max_frames < 0 or args.hist_len < 1 or args.max_feature_age < 0:
        parser.error('max-frames/max-feature-age must be non-negative and hist-len positive')
    if not np.isfinite(args.feature_weight) or args.feature_weight < 0:
        parser.error('feature-weight must be finite and non-negative')

    dataset = TrackSequenceColdDataset(
        config.VAL_ROOT, str(Path(config.CACHE_ROOT) / 'fast_dataset_multi_test'),
        config.WINDOW_SIZE, config.STRIDE, full_sequence=True,
    )
    records = []
    for index in range(len(dataset)):
        record = dataset[index]
        if args.sequence and record['folder'] != args.sequence:
            continue
        frames = record['frames'][:args.max_frames] if args.max_frames else record['frames']
        records.append({**record, 'frames': frames})
    if not records:
        parser.error('no validation sequence matched the request')

    result = evaluate_records(
        records, args.hist_len, args.feature_weight, args.max_feature_age,
        progress=lambda folder, frame, length: print(f'{folder}: {frame}/{length}', flush=True),
    )
    payload = {
        'schema_version': 1,
        'mode': 'opt-in recent M1 appearance association; GT used only for offline evaluation',
        'feature_contract': '128-D cached M1 features; spatial gate first; appearance only on ambiguous valid edges',
        'settings': {'hist_len': args.hist_len, 'feature_weight': args.feature_weight,
                     'max_feature_age': args.max_feature_age},
        'sequences': result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    for variant, sequences in result.items():
        for row in sequences:
            metrics = row['metrics']
            print(f"{variant} {row['folder']}: F1={metrics['f1']:.4f} "
                  f"identity_F1={metrics['identity_f1']:.4f} "
                  f"IDSW={metrics['gt_id_switches']} "
                  f"MAE={metrics['matched_mae_m']}")
    print(f'Wrote: {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
