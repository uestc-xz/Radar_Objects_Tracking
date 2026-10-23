"""Compare continuous-frame hysteresis with independent long-gap recovery.

All variants consume the same frozen M1 cache.  Ground truth is used only for
offline diagnostics; it never enters tracker updates.  The default lifecycle
configuration is unchanged and this script is an opt-in experiment.
"""

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
from original.my_track.event_hysteresis import HysteresisLifecycleTracker
from original.my_track.gated_matching import gated_pairs
from original.my_track.identity_recovery import IdentityRecoveryTracker
from original.my_track.lifecycle_tracker import LifecycleTracker
from original.my_track.tracking_diagnostics import tracking_diagnostics


VARIANTS = ('spatial', 'hysteresis', 'spatial_recovery', 'hysteresis_recovery')


def split_id_switch_events(outputs, targets, gate_m=1.):
    """Split same-GT ID switches by the number of unmatched frames in between."""
    targets = np.asarray(targets, dtype=np.float64)
    if targets.ndim != 3 or targets.shape[-1] != 2 or len(outputs) != len(targets):
        raise ValueError('outputs and targets[T,G,2] must have equal frame counts')
    if not np.isfinite(targets).all() or not np.isfinite(gate_m) or gate_m <= 0:
        raise ValueError('targets and positive gate_m must be finite')
    events = {'total': 0, 'consecutive': 0, 'gap_1_2': 0, 'gap_3_5': 0, 'gap_gt5': 0}
    last = {}
    for frame_index, (output, gt) in enumerate(zip(outputs, targets)):
        points = np.asarray(output.positions, dtype=np.float64)
        ids = np.asarray(output.track_ids)
        valid_slots = np.flatnonzero(np.abs(gt).sum(axis=1) > 0)
        pairs = gated_pairs(points, gt[valid_slots], gate_m)
        for point_index, gt_index in pairs:
            slot = int(valid_slots[gt_index])
            track_id = int(ids[point_index])
            previous = last.get(slot)
            if previous is not None and previous[0] != track_id:
                gap = frame_index - previous[1] - 1
                events['total'] += 1
                if gap == 0:
                    events['consecutive'] += 1
                elif gap <= 2:
                    events['gap_1_2'] += 1
                elif gap <= 5:
                    events['gap_3_5'] += 1
                else:
                    events['gap_gt5'] += 1
            last[slot] = (track_id, frame_index)
    return events


def _settings(hist_len, association_policy):
    return dict(
        hist_len=hist_len, max_tracks=3, score_threshold=.8, birth_threshold=.96,
        cluster_radius_m=.75, association_radius_m=1., confirm_fraction=.6,
        tentative_max_missed=2, max_missed=hist_len, output_max_missed=2,
        confidence_decay=.95, output_confidence=.5, velocity_alpha=.2,
        max_step_m=.2, prediction_decay=.8, no_death=False, weak_updates=True,
        association_policy=association_policy,
    )


def _tracker(variant, hist_len, association_policy, hold_gate_m, hold_penalty,
             recovery_memory_steps, recovery_gate_m, recovery_gate_growth_m,
             recovery_max_gate_m, recovery_ambiguity_margin_m):
    settings = _settings(hist_len, association_policy)
    if variant.startswith('hysteresis'):
        tracker = HysteresisLifecycleTracker(
            **settings, hold_gate_m=hold_gate_m, hold_penalty=hold_penalty,
        )
    else:
        tracker = LifecycleTracker(**settings)
    if variant.endswith('_recovery'):
        tracker = IdentityRecoveryTracker(
            tracker,
            memory_steps=recovery_memory_steps,
            gate_m=recovery_gate_m,
            gate_growth_m=recovery_gate_growth_m,
            max_gate_m=recovery_max_gate_m,
            ambiguity_margin_m=recovery_ambiguity_margin_m,
        )
    return tracker, settings


def _summarize(rows):
    keys = ('frames', 'tp', 'fp', 'fn', 'error_sum_m', 'predicted_points', 'gt_points',
            'gt_id_switches', 'track_owner_switches', 'fragments', 'emitted_tracks',
            'false_emitted_tracks', 'idtp', 'idfp', 'idfn', 'coasted_tp', 'coasted_fp')
    totals = {key: sum(row['metrics'][key] for row in rows) for key in keys}
    totals['event_split'] = {
        key: sum(row['event_split'][key] for row in rows)
        for key in ('total', 'consecutive', 'gap_1_2', 'gap_3_5', 'gap_gt5')
    }
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


def evaluate_records(records, hist_len=config.HIST_LEN, association_policy='cascade',
                     hold_gate_m=.5, hold_penalty=.1, recovery_memory_steps=None,
                     recovery_gate_m=1., recovery_gate_growth_m=.02,
                     recovery_max_gate_m=7.64, recovery_ambiguity_margin_m=.5,
                     progress=None):
    if association_policy not in ('joint', 'cascade'):
        raise ValueError('association_policy must be joint or cascade')
    recovery_memory_steps = hist_len * 50 if recovery_memory_steps is None else recovery_memory_steps
    if recovery_memory_steps < 1:
        raise ValueError('recovery_memory_steps must be positive')
    result = {name: [] for name in VARIANTS}
    for record in records:
        trackers = {}
        outputs = {name: [] for name in VARIANTS}
        for name in VARIANTS:
            trackers[name], settings = _tracker(
                name, hist_len, association_policy, hold_gate_m, hold_penalty,
                recovery_memory_steps, recovery_gate_m, recovery_gate_growth_m,
                recovery_max_gate_m, recovery_ambiguity_margin_m,
            )
        targets = []
        for frame_index, (cache_path, label_path, _sample_id) in enumerate(record['frames']):
            with np.load(cache_path) as cache:
                out = np.asarray(cache['out'])
                conf = np.asarray(cache['conf'])
            label = np.load(label_path)
            if (out.ndim != 2 or out.shape[1] != 4 or conf.shape != (len(out), 2) or
                    label.shape != (10, 6) or not np.isfinite(out).all() or
                    not np.isfinite(conf).all() or not np.isfinite(label).all()):
                raise ValueError(f'invalid cache or label shape/value: {cache_path}')
            positions = out[:, :2].astype(np.float64)
            logits = conf.astype(np.float64)
            logits -= logits.max(axis=-1, keepdims=True)
            exp_logits = np.exp(logits)
            scores = exp_logits[:, 1] / exp_logits.sum(axis=-1)
            targets.append(label.mean(axis=0).reshape(3, 2).astype(np.float64))
            for name, tracker in trackers.items():
                outputs[name].append(tracker.update(positions, scores))
            if progress is not None and (frame_index % 1000 == 0 or
                                         frame_index + 1 == len(record['frames'])):
                progress(record['folder'], frame_index + 1, len(record['frames']))
        targets = np.asarray(targets)
        for name, tracker in trackers.items():
            metrics = tracking_diagnostics(outputs[name], targets)
            metrics_05 = tracking_diagnostics(outputs[name], targets, gate_m=.5)
            result[name].append({
                'folder': record['folder'], 'frames': len(targets),
                'metrics': metrics, 'metrics_0_5m': metrics_05,
                'event_split': split_id_switch_events(outputs[name], targets),
                'event_split_0_5m': split_id_switch_events(outputs[name], targets, gate_m=.5),
                'management': dict(tracker.statistics),
            })
    return {'variants': {name: {'sequences': rows, 'summary': _summarize(rows),
                                'summary_0_5m': _summarize([
                                    {**row, 'metrics': row['metrics_0_5m'],
                                     'event_split': row['event_split_0_5m']}
                                    for row in rows])}
                         for name, rows in result.items()}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sequence', default='', help='exact validation folder; empty means all')
    parser.add_argument('--max-sequences', type=int, default=0)
    parser.add_argument('--max-frames', type=int, default=0)
    parser.add_argument('--hist-len', type=int, default=config.HIST_LEN)
    parser.add_argument('--association-policy', choices=('cascade', 'joint'), default='cascade')
    parser.add_argument('--hold-gate-m', type=float, default=.5)
    parser.add_argument('--hold-penalty', type=float, default=.1)
    parser.add_argument('--recovery-memory-steps', type=int, default=0,
                        help='0 means HIST_LEN*50')
    parser.add_argument('--recovery-gate-m', type=float, default=1.)
    parser.add_argument('--recovery-gate-growth-m', type=float, default=.02)
    parser.add_argument('--recovery-max-gate-m', type=float, default=7.64)
    parser.add_argument('--recovery-ambiguity-margin-m', type=float, default=.5)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('output already exists; choose a fresh filename')
    if args.max_frames < 0 or args.max_sequences < 0 or args.hist_len < 1:
        parser.error('limits and hist-len must be non-negative/positive')
    for name in ('hold_gate_m', 'hold_penalty', 'recovery_gate_m',
                 'recovery_gate_growth_m', 'recovery_max_gate_m',
                 'recovery_ambiguity_margin_m'):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            parser.error(f'{name} must be finite and non-negative')
    if args.hold_gate_m == 0 or args.recovery_gate_m == 0 or args.recovery_max_gate_m < args.recovery_gate_m:
        parser.error('hold-gate and recovery-gate must be positive; max recovery gate must cover gate')
    dataset = TrackSequenceColdDataset(
        config.VAL_ROOT, str(Path(config.CACHE_ROOT) / 'fast_dataset_multi_test'),
        config.WINDOW_SIZE, config.STRIDE, full_sequence=True,
    )
    records = [dataset[index] for index in range(len(dataset))]
    if args.sequence:
        records = [record for record in records if record['folder'] == args.sequence]
        if not records:
            parser.error(f'validation sequence not found: {args.sequence}')
    if args.max_sequences:
        records = records[:args.max_sequences]
    if args.max_frames:
        records = [{**record, 'frames': record['frames'][:args.max_frames]}
                   for record in records]
    if not records:
        parser.error('validation dataset is empty')
    result = evaluate_records(
        records, hist_len=args.hist_len, association_policy=args.association_policy,
        hold_gate_m=args.hold_gate_m, hold_penalty=args.hold_penalty,
        recovery_memory_steps=(args.recovery_memory_steps or None),
        recovery_gate_m=args.recovery_gate_m,
        recovery_gate_growth_m=args.recovery_gate_growth_m,
        recovery_max_gate_m=args.recovery_max_gate_m,
        recovery_ambiguity_margin_m=args.recovery_ambiguity_margin_m,
        progress=lambda folder, frame, length: print(f'{folder}: {frame}/{length}', flush=True),
    )
    payload = {
        'schema_version': 1,
        'mode': 'independent frozen-M1 event-level hysteresis and long-gap recovery comparison',
        'metric_contract': 'gated 1-to-1 distance diagnostics; event split uses same GT-slot matches',
        'sequences': [record['folder'] for record in records],
        'settings': vars(args),
        **result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False, default=str)
        stream.write('\n')
    for name, data in result['variants'].items():
        summary = data['summary']
        events = summary['event_split']
        print(f"{name}: F1={summary['f1']:.4f} identity_F1={summary['identity_f1']:.4f} "
              f"IDSW={summary['gt_id_switches']} MAE={summary['matched_mae_m']:.4f}m "
              f"events(consecutive={events['consecutive']}, gap_1_2={events['gap_1_2']}, "
              f"gap_3_5={events['gap_3_5']}, gap_gt5={events['gap_gt5']})")
    print(f'Wrote: {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
