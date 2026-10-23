import unittest

import numpy as np

from original.my_track.event_hysteresis import HysteresisLifecycleTracker
from original.my_track.lifecycle_tracker import LifecycleTracker


def update(tracker, points=(), scores=None):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    scores = np.ones(len(points), dtype=np.float64) if scores is None else np.asarray(scores)
    return tracker.update(points, scores)


class EventHysteresisTests(unittest.TestCase):
    def test_zero_penalty_is_identical_to_spatial_lifecycle(self):
        kwargs = dict(hist_len=1, cluster_radius_m=.02, association_radius_m=.5,
                      velocity_alpha=1., max_step_m=1., prediction_decay=1.)
        plain = LifecycleTracker(**kwargs)
        hysteresis = HysteresisLifecycleTracker(**kwargs, hold_gate_m=.15,
                                                hold_penalty=.0)
        frames = [
            ([[-.4, 2.], [.4, 2.]], None),
            ([[-.1, 2.], [.1, 2.]], None),
            ([[-.1, 2.], [.1, 2.]], None),
        ]
        for points, scores in frames:
            left, right = update(plain, points, scores), update(hysteresis, points, scores)
            for left_field, right_field in zip(left, right):
                np.testing.assert_array_equal(left_field, right_field)

    def test_hold_penalty_preserves_recent_observation_in_ambiguous_crossing(self):
        kwargs = dict(hist_len=1, cluster_radius_m=.02, association_radius_m=.5,
                      velocity_alpha=1., max_step_m=1., prediction_decay=1.)
        plain = LifecycleTracker(**kwargs)
        hysteresis = HysteresisLifecycleTracker(
            **kwargs, hold_gate_m=.15, hold_penalty=.5,
        )
        for tracker in (plain, hysteresis):
            update(tracker, [[-.4, 2.], [.4, 2.]])
            update(tracker, [[-.1, 2.], [.1, 2.]])
        plain_result = update(plain, [[-.1, 2.], [.1, 2.]])
        hysteresis_result = update(hysteresis, [[-.1, 2.], [.1, 2.]])
        np.testing.assert_allclose(
            plain_result.positions[plain_result.track_ids == 0], [[.1, 2.]],
        )
        np.testing.assert_allclose(
            hysteresis_result.positions[hysteresis_result.track_ids == 0], [[-.1, 2.]],
        )

    def test_hold_penalty_applies_only_to_ambiguous_valid_edges(self):
        tracker = HysteresisLifecycleTracker(
            hist_len=1, cluster_radius_m=.02, association_radius_m=.5,
            hold_gate_m=.15, hold_penalty=.4,
        )
        update(tracker, [[0., 2.]])
        tracker._current_proposal_positions = np.asarray([[.1, 2.], [.4, 2.]])
        penalty = tracker._association_pairwise_penalty(
            [0], None, .5,
        )
        self.assertEqual(penalty.shape, (1, 2))
        # Only one track and two candidates is ambiguous on the candidate side.
        self.assertAlmostEqual(float(penalty[0, 0]), 0.)
        self.assertAlmostEqual(float(penalty[0, 1]), .4)

    def test_invalid_parameters_are_rejected(self):
        for kwargs in ({'hold_gate_m': 0}, {'hold_gate_m': 1.1},
                       {'hold_penalty': -1}, {'hold_penalty': np.nan}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                HysteresisLifecycleTracker(**kwargs)


if __name__ == '__main__':
    unittest.main()
