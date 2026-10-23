import unittest
import json
from pathlib import Path
import tempfile

import numpy as np

from original.my_track.m1_temporal_baseline import TemporalFrameOutput
from scripts.diagnose_m1_id_switches import (
    _tracks_before_update, build_tracker, classify_event, diagnose_record, find_id_switch_events,
)


def make_frame(position, track_id):
    return TemporalFrameOutput(
        np.asarray([position], dtype=np.float64), np.asarray([.99]),
        np.asarray([track_id], dtype=np.int64), np.asarray([True], dtype=bool),
    )


class IdSwitchDiagnosticsTests(unittest.TestCase):
    def test_context_records_actual_updates_for_confirmed_tracks(self):
        with tempfile.TemporaryDirectory() as directory:
            frames = []
            for index in range(13):
                cache = Path(directory) / f'{index}.npz'
                label = Path(directory) / f'{index}.npy'
                has_first = index < 4 or index >= 10
                positions = [[1., 2.], [3., 2.]] if has_first else [[3., 2.]]
                out = np.zeros((len(positions), 4), dtype=np.float32)
                out[:, :2] = positions
                conf = np.tile(np.asarray([[0., 10.]], dtype=np.float32), (len(positions), 1))
                gt = np.zeros((10, 6), dtype=np.float32)
                gt[:, :2] = [1., 2.] if has_first else [0., 0.]
                gt[:, 2:4] = [3., 2.]
                np.savez(cache, out=out, conf=conf)
                np.save(label, gt)
                frames.append((cache, label, index + 1))

            record = {'folder': 'synthetic', 'frames': frames}
            result = diagnose_record(record, hist_len=3, context=2)
            self.assertGreater(result['events_total'], 0)
            event = next(event for event in result['events'] if event['frame'] == 12)
            trace = next(item for item in event['context'] if item['frame'] == 12)
            updates = trace['association_updates']
            self.assertTrue(any(
                update['confirmed_before'] and update['source'] == 'primary' and
                update['position'] == [3., 2.]
                for update in updates
            ))
            self.assertTrue(all('predicted_position' in update and 'score' in update
                                for update in updates))

    def test_gap_evidence_counts_support_inside_gap_not_just_at_reappearance(self):
        # Confirm ID 0, expire it, then confirm a new fragment. Two high-score
        # frames inside the output gap must count as available measurements.
        with tempfile.TemporaryDirectory() as directory:
            frames = []
            for index in range(11):
                point_present = index < 3 or index >= 8
                cache = Path(directory) / f'{index}.npz'
                label = Path(directory) / f'{index}.npy'
                out = np.zeros((1, 4), dtype=np.float32)
                out[0, :2] = [1., 2.]
                conf = np.asarray([[0., 10. if point_present else -10.]], dtype=np.float32)
                np.savez(cache, out=out, conf=conf)
                np.save(label, np.tile([1., 2., 0., 0., 0., 0.], (10, 1)))
                frames.append((cache, label, index + 1))
            result = diagnose_record({'folder': 'synthetic', 'frames': frames}, hist_len=3)
            self.assertEqual(result['events_total'], 1)
            event = result['events'][0]
            self.assertEqual((event['previous_frame'], event['frame'], event['gap']), (4, 10, 5))
            evidence = event['gap_evidence']
            self.assertEqual(evidence['frames'], 5)
            self.assertEqual(evidence['raw08_near_frames'], 2)
            self.assertEqual(evidence['raw096_near_frames'], 2)
            self.assertEqual(evidence['proposals_unique_frames'], 2)
            self.assertEqual(evidence['no_raw08_frames'], 3)
            self.assertEqual(evidence['longest_no_raw08_run'], 3)
            self.assertEqual(evidence['raw08_unique_frames'], 2)
            # Context must expose actual pre-update state and raw/output IDs.
            current = next(item for item in event['context'] if item['frame'] == 10)
            self.assertFalse(current['before_tracks'][0]['confirmed'])
            self.assertEqual(current['output']['raw_track_ids'], [1])
            json.dumps(result, allow_nan=False)
            # Labels are diagnostic evidence only and cannot change inference.
            for _, label, _ in frames:
                np.save(label, np.tile([4., 5., 0., 0., 0., 0.], (10, 1)))
            changed_gt = diagnose_record({'folder': 'synthetic', 'frames': frames}, hist_len=3)
            self.assertEqual(result['prediction_digest'], changed_gt['prediction_digest'])

    def test_recovery_trace_distinguishes_fragment_id_from_identity(self):
        tracker = build_tracker('spatial_recovery', hist_len=1)
        tracker.update(np.asarray([[1., 2.]]), np.asarray([.99]))
        for _ in range(2):
            tracker.update(np.empty((0, 2)), np.empty(0))
        output = tracker.update(np.asarray([[1., 2.]]), np.asarray([.99]))
        self.assertEqual(output.track_ids.tolist(), [0])
        state = _tracks_before_update(tracker)
        self.assertEqual(state[0]['track_id'], 1)
        self.assertEqual(state[0]['identity_id'], 0)

    def test_classifier_includes_weak_proposal_competition(self):
        event = {
            'frame': 0, 'gap': 0,
            'context': [{
                'frame': 0, 'gt': [[1., 2.]],
                'output': {'positions': [[1., 2.]]},
                'primary_gt_distances_m': [[.1]],
                'weak_gt_distances_m': [[.2]],
            }],
        }
        self.assertTrue(classify_event(event)['candidate_ambiguous'])

    def test_finder_rejects_fractional_and_nonfinite_ids(self):
        for invalid_id in (1.2, np.nan, np.inf):
            output = make_frame([1., 1.], 0)._replace(track_ids=np.asarray([invalid_id]))
            with self.subTest(invalid_id=invalid_id), self.assertRaises(ValueError):
                find_id_switch_events([output], np.asarray([[[1., 1.]]]))

    def test_finder_reports_previous_owner_and_unmatched_gap(self):
        targets = np.asarray([
            [[1., 1.]], [[1., 1.]], [[1., 1.]], [[1., 1.]],
        ])
        outputs = [
            make_frame([1., 1.], 2),
            make_frame([1., 1.], 3),
            TemporalFrameOutput(
                np.empty((0, 2)), np.empty((0,)), np.empty((0,), dtype=np.int64),
                np.empty((0,), dtype=bool),
            ),
            make_frame([1., 1.], 4),
        ]
        events = find_id_switch_events(outputs, targets)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]['frame'], 1)
        self.assertEqual(events[0]['previous_track_id'], 2)
        self.assertEqual(events[0]['track_id'], 3)
        self.assertEqual(events[0]['gap'], 0)
        self.assertEqual(events[1]['gap'], 1)

    def test_classifier_marks_current_frame_ambiguity(self):
        targets = np.asarray([[[.01, 0.], [10., 10.]]])
        outputs = [TemporalFrameOutput(
            np.asarray([[.2, 0.], [.3, 0.]], dtype=np.float64),
            np.asarray([.99, .98]), np.asarray([2, 3], dtype=np.int64),
            np.asarray([True, True], dtype=bool),
        )]
        events = find_id_switch_events(outputs, targets, gate_m=1.)
        # No prior owner means no switch; construct the event-shaped context
        # directly to verify the classifier's output-side ambiguity contract.
        event = {
            'frame': 0, 'gap': 0,
            'context': [{
                'frame': 0,
                'gt': targets[0].tolist(),
                'output': {'positions': [[.2, 0.], [.3, 0.]]},
                'primary_gt_distances_m': [[.2, 10.], [10., .3]],
            }],
        }
        self.assertEqual(events, [])
        self.assertEqual(classify_event(event), {
            'interval': 'consecutive', 'output_ambiguous': True,
            'candidate_ambiguous': False,
        })


if __name__ == '__main__':
    unittest.main()
