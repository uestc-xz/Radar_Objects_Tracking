import unittest

import numpy as np

from original.my_track.m1_temporal_baseline import TemporalCandidateTracker
from scripts.evaluate_m1_temporal_baseline import known_target_count


class TemporalCandidateTrackerTests(unittest.TestCase):
    def test_short_miss_is_propagated_without_creating_a_new_track(self):
        tracker = TemporalCandidateTracker(
            score_threshold=0.8,
            cluster_radius_m=0.2,
            association_radius_m=0.6,
            max_missed=1,
            max_tracks=1,
        )
        first = tracker.update(
            np.asarray([[1.0, 2.0]], dtype=np.float64),
            np.asarray([0.99], dtype=np.float64),
        )
        second = tracker.update(
            np.empty((0, 2), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
        )
        self.assertEqual(first.positions.shape, (1, 2))
        self.assertEqual(second.positions.shape, (1, 2))
        np.testing.assert_allclose(second.positions[0], first.positions[0])
        self.assertEqual(second.track_ids.tolist(), first.track_ids.tolist())

    def test_track_expires_after_missed_limit(self):
        tracker = TemporalCandidateTracker(max_missed=1, max_tracks=1)
        tracker.update(np.asarray([[1.0, 2.0]]), np.asarray([0.99]))
        tracker.update(np.empty((0, 2)), np.empty((0,)))
        expired = tracker.update(np.empty((0, 2)), np.empty((0,)))
        self.assertEqual(expired.positions.shape, (0, 2))

    def test_short_gap_uses_constant_velocity_prediction(self):
        tracker = TemporalCandidateTracker(
            score_threshold=0.8,
            cluster_radius_m=0.1,
            association_radius_m=1.0,
            max_missed=1,
            max_tracks=1,
        )
        tracker.update(np.asarray([[0.0, 1.0]]), np.asarray([0.99]))
        tracker.update(np.asarray([[0.4, 1.0]]), np.asarray([0.99]))
        predicted = tracker.update(np.empty((0, 2)), np.empty((0,)))
        np.testing.assert_allclose(predicted.positions, [[0.8, 1.0]])
        self.assertEqual(predicted.observed.tolist(), [False])

    def test_new_sequence_requires_a_new_tracker(self):
        tracker = TemporalCandidateTracker(max_missed=2, max_tracks=1)
        tracker.update(np.asarray([[1.0, 2.0]]), np.asarray([0.99]))
        fresh = TemporalCandidateTracker(max_missed=2, max_tracks=1)
        result = fresh.update(np.asarray([[-1.0, 3.0]]), np.asarray([0.99]))
        np.testing.assert_allclose(result.positions, [[-1.0, 3.0]])
        self.assertEqual(result.track_ids.tolist(), [0])

    def test_sequence_metadata_target_count_is_read_without_using_labels(self):
        self.assertEqual(known_target_count("20240913_3_8"), 3)
        self.assertEqual(known_target_count("20240916_1_4"), 1)
        with self.assertRaisesRegex(ValueError, "target count"):
            known_target_count("unknown_sequence")


if __name__ == "__main__":
    unittest.main()
