"""同一遍冻结 M1 缓存比较时序基线与生命周期管理，GT 仅供离线评估。"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from original.my_dataset.track_sequence_dataset import TrackSequenceColdDataset
from original.my_track import config
from original.my_track.lifecycle_tracker import LifecycleTracker
from original.my_track.m1_temporal_baseline import TemporalCandidateTracker
from original.my_track.tracking_diagnostics import tracking_diagnostics
from scripts.evaluate_m1_temporal_baseline import _identity_switches, known_target_count


VARIANTS = (
    'simple_miss0', 'simple_miss1', 'lifecycle', 'lifecycle_known',
    'lifecycle_known_no_death', 'lifecycle_no_weak', 'lifecycle_no_coast',
    'lifecycle_short_confirm',
    'lifecycle_cascade',
    'lifecycle_joint',
)


def _tracker(variant, folder, hist_len, association_policy):
    if variant.startswith('simple_'):
        settings = dict(score_threshold=.8, cluster_radius_m=.75,
                        association_radius_m=1., max_missed=int(variant[-1]), max_tracks=3)
        return TemporalCandidateTracker(**settings), settings
    settings = dict(hist_len=hist_len, max_tracks=3, score_threshold=.8,
                    birth_threshold=.96, cluster_radius_m=.75, association_radius_m=1.,
                    confirm_fraction=.6, tentative_max_missed=2, max_missed=hist_len,
                    output_max_missed=2, confidence_decay=.95, output_confidence=.5,
                    velocity_alpha=.2, max_step_m=.2, prediction_decay=.8,
                    no_death=False, weak_updates=True, association_policy=association_policy)
    if variant in ('lifecycle_known', 'lifecycle_known_no_death'):
        settings['max_tracks'] = known_target_count(folder)
    if variant == 'lifecycle_known_no_death':
        settings['no_death'] = True
    if variant == 'lifecycle_no_weak':
        settings['weak_updates'] = False
    if variant == 'lifecycle_no_coast':
        settings['output_max_missed'] = 0
    if variant == 'lifecycle_short_confirm':
        settings['hist_len'] = 3
    if variant == 'lifecycle_cascade':
        settings['association_policy'] = 'cascade'
    if variant == 'lifecycle_joint':
        settings['association_policy'] = 'joint'
    return LifecycleTracker(**settings), settings


def _summarize(sequences):
    additive = (
        'frames', 'tp', 'fp', 'fn', 'error_sum_m', 'predicted_points', 'gt_points',
        'gt_id_switches', 'track_owner_switches', 'fragments', 'emitted_tracks',
        'false_emitted_tracks', 'idtp', 'idfp', 'idfn', 'coasted_tp', 'coasted_fp',
        'legacy_owner_switches',
    )
    totals = {key: sum(row['metrics'][key] for row in sequences) for key in additive}
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


def evaluate_records(records, variants=VARIANTS, hist_len=config.HIST_LEN, progress=None,
                     association_policy='cascade'):
    """每条记录重置所有方法；完整保留标签原槽、确认前缀和输出指纹。"""
    variants = tuple(variants)
    if not variants or len(set(variants)) != len(variants) or any(v not in VARIANTS for v in variants):
        raise ValueError('variants must be a non-empty unique selection of supported names')
    if association_policy not in ('joint', 'cascade'):
        raise ValueError('association_policy must be joint or cascade')
    result = {'variants': {name: {'sequences': []} for name in variants}, 'data_checks': []}
    for record in records:
        folder = record['folder']
        if not record['frames']:
            raise ValueError(f'empty sequence: {folder}')
        trackers, settings, outputs = {}, {}, {name: [] for name in variants}
        hashes = {name: hashlib.sha256() for name in variants}
        for name in variants:
            trackers[name], settings[name] = _tracker(name, folder, hist_len, association_policy)
        targets, sample_ids = [], []
        lower, upper = np.full(2, np.inf), np.full(2, -np.inf)
        first_shapes = None
        for frame_index, (cache_path, label_path, sample_id) in enumerate(record['frames']):
            with np.load(cache_path) as cache:
                out = np.asarray(cache['out'])
                conf = np.asarray(cache['conf'])
            label = np.load(label_path)
            if (out.ndim != 2 or out.shape[1] != 4 or conf.shape != (len(out), 2) or
                    label.shape != (10, 6) or not np.isfinite(out).all() or
                    not np.isfinite(conf).all() or not np.isfinite(label).all()):
                raise ValueError(f'invalid cache or label shape/value: {cache_path}')
            if first_shapes is None:
                first_shapes = dict(out_shape=list(out.shape), conf_shape=list(conf.shape),
                                    label_shape=list(label.shape), out_dtype=str(out.dtype),
                                    conf_dtype=str(conf.dtype), label_dtype=str(label.dtype))
            positions = out[:, :2].astype(np.float64)
            logits = conf.astype(np.float64)
            logits -= logits.max(axis=-1, keepdims=True)
            exp_logits = np.exp(logits)
            scores = exp_logits[:, 1] / exp_logits.sum(axis=-1)
            targets.append(label.mean(axis=0).reshape(3, 2).astype(np.float64))
            sample_ids.append(int(sample_id))
            if len(positions):
                lower = np.minimum(lower, positions.min(axis=0))
                upper = np.maximum(upper, positions.max(axis=0))
            for name, tracker in trackers.items():
                frame = tracker.update(positions, scores)
                outputs[name].append(frame)
                for array in frame:
                    hashes[name].update(str(array.shape).encode('ascii'))
                    hashes[name].update(array.tobytes())
            if progress is not None and (frame_index % 1000 == 0 or frame_index + 1 == len(record['frames'])):
                progress(folder, frame_index + 1, len(record['frames']))
        if any(b <= a for a, b in zip(sample_ids, sample_ids[1:])):
            raise ValueError(f'sample IDs must be strictly increasing within {folder}')
        result['data_checks'].append({
            'folder': folder, **first_shapes, 'frames': len(targets),
            'sample_id_min': min(sample_ids), 'sample_id_max': max(sample_ids),
            'max_sample_id_gap': max(np.diff(sample_ids), default=0).item()
            if len(sample_ids) > 1 else 0,
            'raw_candidate_min_xy_m': lower.tolist() if np.isfinite(lower).all() else None,
            'raw_candidate_max_xy_m': upper.tolist() if np.isfinite(upper).all() else None,
        })
        targets = np.asarray(targets)
        for name, tracker in trackers.items():
            metrics = tracking_diagnostics(outputs[name], targets)
            metrics['legacy_owner_switches'] = _identity_switches(
                [o.positions for o in outputs[name]], targets, [o.track_ids for o in outputs[name]],
            )
            result['variants'][name]['sequences'].append({
                'folder': folder, 'settings': settings[name], 'metrics': metrics,
                'management': dict(tracker.statistics) if hasattr(tracker, 'statistics') else None,
                'prediction_digest': hashes[name].hexdigest(),
            })
    for data in result['variants'].values():
        data['summary'] = _summarize(data['sequences'])
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-frames', type=int, default=0, help='每序列前缀；0=完整记录')
    parser.add_argument('--hist-len', type=int, default=config.HIST_LEN)
    parser.add_argument('--variants', choices=VARIANTS, nargs='+', default=list(VARIANTS))
    parser.add_argument('--association-policy', choices=('cascade', 'joint'), default='cascade')
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('output already exists; choose a new filename')
    if args.max_frames < 0 or args.hist_len < 1:
        parser.error('max-frames must be non-negative; hist-len must be positive')
    dataset = TrackSequenceColdDataset(
        config.VAL_ROOT, str(Path(config.CACHE_ROOT) / 'fast_dataset_multi_test'),
        config.WINDOW_SIZE, config.STRIDE, full_sequence=True,
    )
    records = []
    for index in range(len(dataset)):
        record = dataset[index]
        frames = record['frames'][:args.max_frames] if args.max_frames else record['frames']
        records.append({**record, 'frames': frames})
    if not records:
        parser.error('validation dataset is empty')
    result = evaluate_records(
        records, args.variants, args.hist_len,
        progress=lambda folder, frame, length: print(f'{folder}: {frame}/{length}', flush=True),
        association_policy=args.association_policy,
    )
    payload = {
        'schema_version': 2,
        'mode': 'frozen-M1 independent lifecycle comparison; validation exploration',
        'metric_contract': 'gated 1-to-1 distance diagnostics; not official MOTChallenge/HOTA',
        'match_radius_m': 1., 'hist_len': args.hist_len,
        'association_policy': args.association_policy,
        'data_root': config.VAL_ROOT,
        'cache_root': str(Path(config.CACHE_ROOT) / 'fast_dataset_multi_test'),
        'max_frames_per_sequence': args.max_frames, **result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    for name, data in result['variants'].items():
        summary = data['summary']
        print(f"{name}: P={summary['precision']:.4f} R={summary['recall']:.4f} "
              f"F1={summary['f1']:.4f} identity_F1={summary['identity_f1']:.4f} "
              f"GT_IDSW={summary['gt_id_switches']} fragments={summary['fragments']}")
    print(f'Wrote: {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
