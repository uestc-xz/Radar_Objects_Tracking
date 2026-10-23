"""Dump local evidence around GT-slot ID switches in a frozen-M1 replay.

This is a read-only diagnostic.  It does not change any tracker defaults or
feed GT into inference.  The output is intentionally event-local: each
reported switch contains a small frame window with M1 clusters, pre-update
track state, post-update output and GT distances.
"""

import argparse
from collections import Counter
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
from original.my_track.event_hysteresis import HysteresisLifecycleTracker
from original.my_track.gated_matching import gated_pairs
from original.my_track.identity_recovery import IdentityRecoveryTracker
from original.my_track.lifecycle_tracker import LifecycleTracker
from original.my_track.tentative_weak import TentativeWeakLifecycleTracker


VARIANTS = ('spatial', 'hysteresis', 'spatial_recovery', 'hysteresis_recovery',
            'tentative_weak', 'tentative_weak_recovery',
            'tentative_weak_primary_confirm', 'tentative_weak_primary_confirm_recovery')


def find_id_switch_events(outputs, targets, gate_m=1.):
    """Return GT-slot ownership changes with the number of unmatched frames."""
    targets = np.asarray(targets, dtype=np.float64)
    if targets.ndim != 3 or targets.shape[-1] != 2 or len(outputs) != len(targets):
        raise ValueError('outputs and targets[T,G,2] must have equal frame counts')
    if not np.isfinite(targets).all() or not np.isfinite(gate_m) or gate_m <= 0:
        raise ValueError('targets and positive gate_m must be finite')
    last = {}
    events = []
    for frame_index, (output, gt) in enumerate(zip(outputs, targets)):
        points = np.asarray(output.positions, dtype=np.float64)
        ids = np.asarray(output.track_ids)
        if (points.ndim != 2 or points.shape[1] != 2 or ids.shape != (len(points),) or
                not np.isfinite(points).all() or not np.isfinite(ids).all() or
                not np.equal(ids, np.floor(ids)).all() or len(set(ids.tolist())) != len(ids)):
            raise ValueError('outputs require finite positions and unique aligned IDs')
        valid_slots = np.flatnonzero(np.abs(gt).sum(axis=1) > 0)
        pairs = gated_pairs(points, gt[valid_slots], gate_m)
        for point_index, gt_index in pairs:
            slot = int(valid_slots[gt_index])
            track_id = int(ids[point_index])
            previous = last.get(slot)
            if previous is not None and previous['track_id'] != track_id:
                gap = frame_index - previous['frame'] - 1
                events.append({
                    'frame': frame_index,
                    'gt_slot': slot,
                    'previous_track_id': previous['track_id'],
                    'track_id': track_id,
                    'previous_frame': previous['frame'],
                    'gap': gap,
                    'observed': bool(output.observed[point_index]),
                    'previous_observed': previous['observed'],
                    'match_distance_m': float(np.linalg.norm(points[point_index] - gt[slot])),
                    'nearest_gt_slot': int(valid_slots[np.argmin(
                        np.linalg.norm(points[point_index] - gt[valid_slots], axis=1)
                    )]),
                })
            last[slot] = {'track_id': track_id, 'frame': frame_index,
                          'observed': bool(output.observed[point_index])}
    return events


def classify_event(event, gate_m=1.):
    """Mark GT-neighbour ambiguity, not the causal reason for an ID switch.

    Candidate ambiguity includes both primary and weak proposals; it describes
    geometry relative to GT, not the actual inference assignment graph.
    """
    trace = next(item for item in event['context'] if item['frame'] == event['frame'])
    points = np.asarray(trace['output']['positions'], dtype=np.float64)
    gt = np.asarray(trace['gt'], dtype=np.float64)
    valid_gt = gt[np.abs(gt).sum(axis=1) > 0]
    distances = (np.linalg.norm(points[:, None] - valid_gt[None], axis=-1)
                 if len(points) and len(valid_gt) else np.empty((len(points), len(valid_gt))))
    output_valid = distances <= gate_m
    # Weak proposals can update confirmed tracks too. Omitting them would call
    # a primary/weak competition unambiguous merely because of score threshold.
    proposal_rows = (trace['primary_gt_distances_m'] + trace.get('weak_gt_distances_m', []))
    proposal_distances = np.asarray(proposal_rows, dtype=np.float64)
    candidate_valid = proposal_distances <= gate_m
    output_ambiguous = bool(
        output_valid.size and
        (output_valid.sum(axis=0).max(initial=0) > 1 or output_valid.sum(axis=1).max(initial=0) > 1)
    )
    candidate_ambiguous = bool(
        candidate_valid.size and
        (candidate_valid.sum(axis=0).max(initial=0) > 1 or candidate_valid.sum(axis=1).max(initial=0) > 1)
    )
    if event['gap'] > 0:
        interval = 'gap'
    else:
        interval = 'consecutive'
    return {
        'interval': interval,
        'output_ambiguous': output_ambiguous,
        'candidate_ambiguous': candidate_ambiguous,
    }


def _settings(hist_len, association_policy):
    return dict(
        hist_len=hist_len, max_tracks=3, score_threshold=.8, birth_threshold=.96,
        cluster_radius_m=.75, association_radius_m=1., confirm_fraction=.6,
        tentative_max_missed=2, max_missed=hist_len, output_max_missed=2,
        confidence_decay=.95, output_confidence=.5, velocity_alpha=.2,
        max_step_m=.2, prediction_decay=.8, no_death=False, weak_updates=True,
        association_policy=association_policy,
    )


def build_tracker(variant, hist_len=config.HIST_LEN, association_policy='cascade',
                  hold_gate_m=.5, hold_penalty=.1):
    if variant not in VARIANTS:
        raise ValueError(f'unsupported variant: {variant}')
    settings = _settings(hist_len, association_policy)
    if variant.startswith('hysteresis'):
        tracker = HysteresisLifecycleTracker(
            **settings, hold_gate_m=hold_gate_m, hold_penalty=hold_penalty,
        )
    elif variant.startswith('tentative_weak'):
        tracker = TentativeWeakLifecycleTracker(**settings, primary_confirmation='primary_confirm' in variant)
    else:
        tracker = LifecycleTracker(**settings)
    if variant.endswith('_recovery'):
        tracker = IdentityRecoveryTracker(
            tracker, memory_steps=hist_len * 50, gate_m=1., gate_growth_m=.02,
            max_gate_m=7.64, ambiguity_margin_m=.5,
        )
    return tracker


def _base_tracker(tracker):
    return tracker.tracker if isinstance(tracker, IdentityRecoveryTracker) else tracker


def _tracks_before_update(tracker):
    base = _base_tracker(tracker)
    result = []
    for track in base._tracks:
        predicted = base._predict(track, track.missed + 1)
        result.append({
            'track_id': int(track.track_id),
            'identity_id': (tracker.recovery._mapping.get(int(track.track_id))
                            if isinstance(tracker, IdentityRecoveryTracker) else int(track.track_id)),
            'position': track.position.tolist(),
            'velocity': track.velocity.tolist(),
            'predicted_position': predicted.tolist(),
            'confidence': float(track.confidence),
            'confirmed': bool(track.confirmed),
            'missed': int(track.missed),
        })
    return result


def _cluster_payload(cluster):
    return {
        'positions': cluster.positions.tolist(),
        'scores': cluster.scores.tolist(),
        'sizes': cluster.sizes.tolist(),
    }


def _coverage(points, gt, gate_m):
    """Return independent-neighbour and one-to-one coverage in original GT slots.

    Independent support may reuse a measurement across two nearby GT slots;
    unique support uses the same cardinality-first assignment as evaluation.
    Neither is evidence that the tracker's inference knew the GT identity.
    """
    slots = np.flatnonzero(np.abs(gt).sum(axis=1) > 0)
    distances = np.linalg.norm(points[:, None] - gt[slots][None], axis=-1)
    near = np.zeros(len(gt), dtype=bool)
    unique = np.zeros(len(gt), dtype=bool)
    near[slots] = (distances <= gate_m).any(axis=0)
    for _, column in gated_pairs(points, gt[slots], gate_m):
        unique[slots[column]] = True
    return near.tolist(), unique.tolist()


def _frame_trace(tracker, frame_index, sample_id, positions, scores, gt, output, before, gate_m,
                 association_updates):
    base = _base_tracker(tracker)
    primary, weak = base._proposals(positions, scores)
    valid_slots = np.flatnonzero(np.abs(gt).sum(axis=1) > 0)
    valid_gt = gt[valid_slots]
    primary_distances = (np.linalg.norm(
        primary.positions[:, None] - valid_gt[None], axis=-1
    ) if len(primary.positions) and len(valid_gt) else np.empty((len(primary.positions), len(valid_gt))))
    weak_distances = (np.linalg.norm(
        weak.positions[:, None] - valid_gt[None], axis=-1
    ) if len(weak.positions) and len(valid_gt) else np.empty((len(weak.positions), len(valid_gt))))
    raw_distances = (np.linalg.norm(
        positions[:, None] - valid_gt[None], axis=-1
    ) if len(positions) and len(valid_gt) else np.empty((len(positions), len(valid_gt))))
    support = {}
    for name, points in (
        ('raw08', positions[scores >= base.score_threshold]),
        ('raw096', positions[scores >= base.birth_threshold]),
        ('proposals', np.concatenate((primary.positions, weak.positions))),
    ):
        support[f'{name}_near'], support[f'{name}_unique'] = _coverage(points, gt, gate_m)
    if isinstance(tracker, IdentityRecoveryTracker):
        inverse_mapping = {identity: raw_id for raw_id, identity in tracker.recovery._mapping.items()}
        raw_output_ids = [int(inverse_mapping[int(identity)]) for identity in output.track_ids]
    else:
        raw_output_ids = np.asarray(output.track_ids).tolist()
    traced_updates = []
    for update in association_updates:
        source = 'unknown'
        for name, cluster in (('primary', primary), ('weak', weak)):
            if any(np.array_equal(update['position'], point) and score == update['score']
                   for point, score in zip(cluster.positions, cluster.scores)):
                source = name
                break
        traced_updates.append({**update, 'source': source})
    return {
        'frame': int(frame_index),
        'sample_id': int(sample_id),
        'gt': gt.tolist(),
        'valid_gt_slots': valid_slots.tolist(),
        'raw_candidate_positions': positions.tolist(),
        'raw_candidate_scores': scores.tolist(),
        'raw_gt_distances_m': raw_distances.tolist(),
        'before_tracks': before,
        'association_updates': traced_updates,
        'support': support,
        'primary_clusters': _cluster_payload(primary),
        'weak_clusters': _cluster_payload(weak),
        'primary_gt_distances_m': primary_distances.tolist(),
        'weak_gt_distances_m': weak_distances.tolist(),
        'output': {
            'positions': np.asarray(output.positions).tolist(),
            'scores': np.asarray(output.scores).tolist(),
            'track_ids': np.asarray(output.track_ids).tolist(),
            'raw_track_ids': raw_output_ids,
            'observed': np.asarray(output.observed).tolist(),
        },
    }


def _gap_evidence(event, traces):
    """Describe ALL unmatched input steps, excluding both matched endpoints."""
    interval = traces[event['previous_frame'] + 1:event['frame']]
    slot = event['gt_slot']
    result = {'frames': len(interval)}
    for key in ('raw08_near', 'raw096_near', 'raw08_unique', 'raw096_unique',
                'proposals_near', 'proposals_unique'):
        result[f'{key}_frames'] = sum(bool(trace['support'][key][slot]) for trace in interval)
    missing_run = longest = 0
    for trace in interval:
        missing_run = missing_run + 1 if not trace['support']['raw08_near'][slot] else 0
        longest = max(longest, missing_run)
    result['no_raw08_frames'] = len(interval) - result['raw08_near_frames']
    result['longest_no_raw08_run'] = longest
    return result


def diagnose_record(record, variant='spatial', hist_len=config.HIST_LEN,
                    association_policy='cascade', hold_gate_m=.5, hold_penalty=.1,
                    context=2, max_events=0, gate_m=1., progress=None):
    tracker = build_tracker(variant, hist_len, association_policy, hold_gate_m, hold_penalty)
    if not record['frames']:
        raise ValueError('cannot diagnose an empty sequence')
    if context < 0 or max_events < 0:
        raise ValueError('context/max_events must be non-negative')
    outputs, targets, traces = [], [], []
    sample_ids = []
    fingerprint = hashlib.sha256()
    base = _base_tracker(tracker)
    original_observe = base._observe
    association_updates = []

    def trace_observe(track, position, score):
        # Record only consumed measurements, before the original state update.
        # Births do not call _observe; this is not a list of all proposals.
        association_updates.append({
            'track_id': int(track.track_id),
            'identity_id': (tracker.recovery._mapping.get(int(track.track_id))
                            if isinstance(tracker, IdentityRecoveryTracker)
                            else int(track.track_id)),
            'confirmed_before': bool(track.confirmed),
            'missed_before': int(track.missed),
            'before_position': track.position.tolist(),
            'predicted_position': base._predict(track, track.missed + 1).tolist(),
            'position': np.asarray(position).tolist(),
            'score': float(score),
        })
        original_observe(track, position, score)

    base._observe = trace_observe
    try:
        for frame_index, (cache_path, label_path, sample_id) in enumerate(record['frames']):
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
            gt = label.mean(axis=0).reshape(3, 2).astype(np.float64)
            before = _tracks_before_update(tracker)
            association_updates.clear()
            output = tracker.update(positions, scores)
            outputs.append(output)
            targets.append(gt)
            for field in output:
                fingerprint.update(str(field.shape).encode('ascii'))
                fingerprint.update(field.tobytes())
            trace = _frame_trace(tracker, frame_index, sample_id, positions, scores, gt, output,
                                 before, gate_m, association_updates)
            traces.append(trace)
            sample_ids.append(int(sample_id))
            if progress is not None and (frame_index % 1000 == 0 or
                                         frame_index + 1 == len(record['frames'])):
                progress(record['folder'], frame_index + 1, len(record['frames']))
    finally:
        base._observe = original_observe
    targets = np.asarray(targets)
    if any(right <= left for left, right in zip(sample_ids, sample_ids[1:])):
        raise ValueError('sample IDs must be strictly increasing within a sequence')
    events = find_id_switch_events(outputs, targets, gate_m=gate_m)
    selected = events if not max_events else events[:max_events]
    for event in selected:
        start = max(0, event['frame'] - context)
        end = min(len(traces), event['frame'] + context + 1)
        event['context'] = traces[start:end]
        event['classification'] = classify_event(event, gate_m)
        event['gap_evidence'] = _gap_evidence(event, traces)
    return {
        'folder': record['folder'], 'variant': variant, 'frames': len(traces),
        'sample_id_min': min(sample_ids), 'sample_id_max': max(sample_ids),
        'events_total': len(events), 'events_returned': len(selected),
        'events': selected, 'management': dict(tracker.statistics),
        'prediction_digest': fingerprint.hexdigest(),
        'max_sample_id_gap': int(max(np.diff(sample_ids), default=0)),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequence', default='20240913_3_8')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--variant', choices=VARIANTS, default='spatial')
    parser.add_argument('--max-events', type=int, default=20,
                        help='0 returns every event')
    parser.add_argument('--context', type=int, default=2)
    parser.add_argument('--hist-len', type=int, default=config.HIST_LEN)
    parser.add_argument('--association-policy', choices=('cascade', 'joint'), default='cascade')
    parser.add_argument('--hold-gate-m', type=float, default=.5)
    parser.add_argument('--hold-penalty', type=float, default=.1)
    parser.add_argument('--gate-m', type=float, default=1.)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('output already exists; choose a fresh filename')
    if args.max_events < 0 or args.context < 0 or args.hist_len < 1:
        parser.error('max-events/context must be non-negative; hist-len must be positive')
    for name in ('hold_gate_m', 'hold_penalty', 'gate_m'):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            parser.error(f'{name} must be finite and non-negative')
    if args.gate_m == 0:
        parser.error('gate-m must be positive')
    dataset = TrackSequenceColdDataset(
        config.VAL_ROOT, str(Path(config.CACHE_ROOT) / 'fast_dataset_multi_test'),
        config.WINDOW_SIZE, config.STRIDE, full_sequence=True,
    )
    records = [dataset[index] for index in range(len(dataset))
               if dataset[index]['folder'] == args.sequence]
    if not records:
        parser.error(f'validation sequence not found: {args.sequence}')
    result = diagnose_record(
        records[0], variant=args.variant, hist_len=args.hist_len,
        association_policy=args.association_policy, hold_gate_m=args.hold_gate_m,
        hold_penalty=args.hold_penalty, context=args.context,
        max_events=args.max_events, gate_m=args.gate_m,
        progress=lambda folder, frame, length: print(f'{folder}: {frame}/{length}', flush=True),
    )
    settings = vars(args).copy()
    settings['output'] = str(settings['output'])
    payload = {
        'schema_version': 3,
        'mode': 'read-only frozen-M1 local ID-switch evidence diagnosis',
        'metric_contract': 'events use the same gated 1-to-1 GT-slot matching as tracking diagnostics',
        'settings': settings,
        'coverage_contract': (
            'Gap counts exclude both matched endpoints; steps are input frames. '
            'near allows candidate reuse between GT slots; unique uses gated one-to-one matching. '
            'raw08/raw096 use tracker score/birth thresholds 0.8/0.96. '
            'Ambiguity is geometric evidence only, not causal attribution. '
            'before_tracks.track_id is the raw fragment ID; identity_id is its optional recovery alias. '
            'association_updates records consumed _observe calls only, excluding births; '
            'its identity_id is the alias before observation and may be null. '
            'source identifies a primary/weak proposal, not a GT identity.'
        ),
        **result,
    }
    # Validate serialization before opening the exclusive output, so a type
    # error cannot leave a truncated JSON that looks like a completed run.
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        stream.write(serialized)
        stream.write('\n')
    print(f"{result['folder']}: events={result['events_total']} returned={result['events_returned']} "
          f"management={result['management']}")
    classes = Counter(
        (event['classification']['interval'],
         event['classification']['output_ambiguous'],
         event['classification']['candidate_ambiguous'])
        for event in result['events']
    )
    print(f'event_classes={dict(classes)}')
    print(f'Wrote: {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
