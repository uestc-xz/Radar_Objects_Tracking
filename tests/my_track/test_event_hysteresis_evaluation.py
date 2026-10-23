import unittest

import numpy as np

from original.my_track.m1_temporal_baseline import TemporalFrameOutput
from scripts.evaluate_m1_event_hysteresis import split_id_switch_events


def frame(position, track_id, observed=True):
    if position is None:
        return TemporalFrameOutput(
            np.empty((0, 2)), np.empty((0,)), np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=bool),
        )
    return TemporalFrameOutput(
        np.asarray([position], dtype=np.float64), np.asarray([.99]),
        np.asarray([track_id], dtype=np.int64), np.asarray([observed], dtype=bool),
    )


class EventHysteresisEvaluationTests(unittest.TestCase):
    def test_switches_are_split_by_unmatched_gap_length(self):
        targets = np.asarray([
            [[1., 1.]], [[1., 1.]], [[1., 1.]], [[1., 1.]],
        ])
        outputs = [
            frame([1., 1.], 0),
            frame([1., 1.], 1),
            frame(None, None),
            frame([1., 1.], 2),
        ]
        events = split_id_switch_events(outputs, targets)
        self.assertEqual(events, {
            'total': 2, 'consecutive': 1, 'gap_1_2': 1,
            'gap_3_5': 0, 'gap_gt5': 0,
        })


if __name__ == '__main__':
    unittest.main()
