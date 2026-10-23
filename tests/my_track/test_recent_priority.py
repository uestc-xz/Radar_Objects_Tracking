import unittest

import numpy as np

from original.my_track.lifecycle_tracker import LifecycleTracker
from original.my_track.recent_priority import RecentPriorityTracker


def step(tracker, points=(), score=1.):
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    return tracker.update(points, np.full(len(points), score))


class RecentPriorityTests(unittest.TestCase):
    def test_stale_only_policy_preserves_global_matching_when_all_tracks_are_recent(self):
        tracker = RecentPriorityTracker(hist_len=1, cluster_radius_m=.01, require_stale=True)
        step(tracker, [[0, 2], [.9, 2]])
        result = step(tracker, [[-.7, 2], [.4, 2]])
        self.assertEqual(result.track_ids.tolist(), [0, 1])
        self.assertEqual(result.observed.tolist(), [True, True])
        np.testing.assert_allclose(result.positions, [[-.7, 2], [.4, 2]])

    def test_stale_only_policy_still_protects_recent_track_when_competitor_is_stale(self):
        tracker = RecentPriorityTracker(hist_len=1, max_missed=4, cluster_radius_m=.01,
                                        require_stale=True)
        step(tracker, [[0, 2], [.6, 2]])
        step(tracker, [[0, 2]])
        result = step(tracker, [[.4, 2]])
        self.assertEqual(result.observed.tolist(), [True, False])

    def test_recent_track_keeps_near_measurement_instead_of_losing_it_to_stale_track(self):
        baseline = LifecycleTracker(hist_len=1, max_missed=4, cluster_radius_m=.01)
        tracker = RecentPriorityTracker(hist_len=1, max_missed=4, cluster_radius_m=.01)
        for model in (baseline, tracker):
            step(model, [[0, 2], [.6, 2]])
            step(model, [[0, 2]])
        self.assertEqual(step(baseline, [[.4, 2]]).observed.tolist(), [False, True])
        result = step(tracker, [[.4, 2]])
        self.assertEqual(result.observed.tolist(), [True, False])
        np.testing.assert_allclose(result.positions[result.track_ids == 0], [[.4, 2]])

    def test_distant_measurement_can_return_to_stale_track_outside_priority_gate(self):
        tracker = RecentPriorityTracker(hist_len=1, max_missed=4, cluster_radius_m=.01)
        step(tracker, [[0, 2], [.9, 2]])
        step(tracker, [[0, 2]])
        result = step(tracker, [[.8, 2]])
        self.assertEqual(result.observed.tolist(), [False, True])

    def test_weak_measurement_is_not_used_again_by_second_stage(self):
        tracker = RecentPriorityTracker(hist_len=1, max_missed=4, cluster_radius_m=.01)
        step(tracker, [[0, 2], [.6, 2]])
        step(tracker, [[0, 2]])
        result = step(tracker, [[.4, 2]], score=.9)
        self.assertEqual(result.observed.tolist(), [True, False])
        self.assertEqual(tracker.statistics['created'], 2)

    def test_high_measurement_is_used_once_across_stages_and_birth(self):
        tracker = RecentPriorityTracker(hist_len=1, max_missed=4, cluster_radius_m=.01)
        step(tracker, [[0, 2], [.6, 2]])
        result = step(tracker, [[.3, 2]])
        self.assertEqual(int(result.observed.sum()), 1)
        self.assertEqual(tracker.statistics['created'], 2)

    def test_invalid_gate_and_unsupported_joint_policy_are_rejected(self):
        for kwargs in ({'priority_gate_m': 0}, {'priority_gate_m': 1.1},
                       {'priority_gate_m': np.nan}, {'association_policy': 'joint'}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                RecentPriorityTracker(**kwargs)


if __name__ == '__main__':
    unittest.main()
