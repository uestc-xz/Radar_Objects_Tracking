import unittest

import numpy as np

from original.my_track.lifecycle_tracker import LifecycleTracker


def step(tracker, points=(), scores=None):
    positions = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    values = np.ones(len(positions)) if scores is None else np.asarray(scores)
    return tracker.update(positions, values)


class LifecycleTrackerTests(unittest.TestCase):
    def test_isolated_high_score_false_alarm_never_becomes_an_output(self):
        tracker = LifecycleTracker(hist_len=3, tentative_max_missed=1)
        self.assertEqual(len(step(tracker, [[1, 2]]).positions), 0)
        step(tracker)
        output = step(tracker)
        self.assertEqual(len(output.positions), 0)
        self.assertEqual(tracker.statistics['confirmed'], 0)
        self.assertEqual(tracker.statistics['deleted'], 1)

    def test_confirmation_uses_history_and_does_not_backfill_startup(self):
        tracker = LifecycleTracker(hist_len=3)
        self.assertEqual(len(step(tracker, [[1, 2]]).positions), 0)
        self.assertEqual(len(step(tracker, [[1, 2]]).positions), 0)
        result = step(tracker, [[1, 2]])
        np.testing.assert_allclose(result.positions, [[1, 2]])
        self.assertEqual(result.track_ids.tolist(), [0])

    def test_low_score_can_update_confirmed_track_but_cannot_create_one(self):
        tracker = LifecycleTracker(hist_len=1)
        self.assertEqual(len(step(tracker, [[1, 2]], [.9]).positions), 0)
        step(tracker, [[1, 2]])
        result = step(tracker, [[1.1, 2]], [.9])
        np.testing.assert_allclose(result.positions, [[1.1, 2]])
        self.assertEqual(result.track_ids.tolist(), [0])
        self.assertEqual(result.observed.tolist(), [True])

    def test_high_score_anchor_is_not_shifted_by_low_score_neighbor(self):
        tracker = LifecycleTracker(hist_len=1)
        result = step(tracker, [[1, 2], [1.5, 2]], [1, .85])
        np.testing.assert_allclose(result.positions, [[1, 2]])

    def test_near_weak_measurement_beats_a_distant_high_score_false_alarm(self):
        tracker = LifecycleTracker(hist_len=1, association_policy='joint')
        step(tracker, [[1, 2]])
        result = step(tracker, [[1.9, 2], [1.1, 2]], [.99, .9])
        # 已知历史在 x=1：0.1m 的低分量测应优于 0.9m 的高分量测。
        np.testing.assert_allclose(result.positions[result.track_ids == 0], [[1.1, 2]])

    def test_short_miss_decays_confidence_and_recovers_the_same_id(self):
        tracker = LifecycleTracker(hist_len=1, max_missed=4)
        first = step(tracker, [[1, 2]])
        missed = step(tracker)
        self.assertAlmostEqual(missed.scores[0], .95)
        self.assertEqual(missed.observed.tolist(), [False])
        second_miss = step(tracker)
        self.assertAlmostEqual(second_miss.scores[0], .9025)
        returned = step(tracker, [[1.1, 2]])
        self.assertEqual(returned.track_ids.tolist(), first.track_ids.tolist())

    def test_output_budget_does_not_delete_internal_identity(self):
        tracker = LifecycleTracker(hist_len=1, max_missed=4, output_max_missed=1)
        first = step(tracker, [[1, 2]])
        step(tracker)
        self.assertEqual(len(step(tracker).positions), 0)
        returned = step(tracker, [[1, 2]])
        self.assertEqual(returned.track_ids.tolist(), first.track_ids.tolist())

    def test_velocity_clipping_and_prediction_decay_bound_gap_displacement(self):
        tracker = LifecycleTracker(hist_len=1, max_missed=20, output_max_missed=20,
                                   output_confidence=0, velocity_alpha=1, max_step_m=.2)
        step(tracker, [[0, 2]])
        step(tracker, [[.5, 2]])
        np.testing.assert_allclose(step(tracker).positions, [[.7, 2]])
        np.testing.assert_allclose(step(tracker).positions, [[.86, 2]])
        for _ in range(8):
            last = step(tracker)
        self.assertLess(last.positions[0, 0], 1.5)

    def test_expired_track_never_reuses_an_old_id(self):
        tracker = LifecycleTracker(hist_len=1, max_missed=1)
        self.assertEqual(step(tracker, [[1, 2]]).track_ids.tolist(), [0])
        step(tracker)
        step(tracker)
        self.assertEqual(step(tracker, [[1, 2]]).track_ids.tolist(), [1])

    def test_no_death_mode_retains_dormant_id_without_infinite_prediction(self):
        tracker = LifecycleTracker(hist_len=1, max_missed=1, no_death=True)
        step(tracker, [[1, 2]])
        for _ in range(6):
            missed = step(tracker)
        self.assertEqual(len(missed.positions), 0)
        self.assertEqual(step(tracker, [[1, 2]]).track_ids.tolist(), [0])
        self.assertEqual(tracker.statistics['deleted'], 0)

    def test_crossing_trajectories_follow_motion_not_x_order(self):
        tracker = LifecycleTracker(
            hist_len=1, cluster_radius_m=.05, association_radius_m=.7,
            velocity_alpha=1, max_step_m=1,
        )
        step(tracker, [[-.6, 2], [.6, 2]])
        step(tracker, [[-.2, 2], [.2, 2]])
        crossed = step(tracker, [[-.2, 2], [.2, 2]])
        self.assertEqual(crossed.track_ids.tolist(), [0, 1])
        np.testing.assert_allclose(crossed.positions, [[.2, 2], [-.2, 2]])

    def test_candidate_permutation_preserves_birth_order_and_capacity(self):
        points = [[1, 2], [-1, 2], [0, 4]]
        scores = [.97, .99, .98]
        first = step(LifecycleTracker(hist_len=1, max_tracks=2), points, scores)
        other = step(LifecycleTracker(hist_len=1, max_tracks=2), points[::-1], scores[::-1])
        np.testing.assert_allclose(first.positions, [[-1, 2], [0, 4]])
        np.testing.assert_allclose(first.positions, other.positions)
        np.testing.assert_array_equal(first.track_ids, other.track_ids)

    def test_reset_isolates_sequences_and_restarts_local_ids(self):
        tracker = LifecycleTracker(hist_len=1)
        step(tracker, [[1, 2]])
        tracker.reset()
        self.assertEqual(step(tracker, [[-1, 3]]).track_ids.tolist(), [0])
        self.assertEqual(tracker.statistics['created'], 1)

    def test_invalid_data_does_not_change_track_state(self):
        tracker = LifecycleTracker(hist_len=1)
        step(tracker, [[1, 2]])
        with self.assertRaises(ValueError):
            step(tracker, [[np.nan, 2]])
        self.assertEqual(step(tracker, [[1, 2]]).track_ids.tolist(), [0])
        with self.assertRaises(ValueError):
            step(tracker, [[1, 2]], [np.inf])

    def test_invalid_parameters_are_rejected(self):
        for kwargs in (
            {'hist_len': 0}, {'hist_len': 1.5}, {'max_tracks': 0},
            {'association_radius_m': np.nan}, {'birth_threshold': .7},
            {'confidence_decay': 1.1}, {'output_max_missed': -1},
            {'velocity_alpha': -1}, {'max_step_m': np.inf},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                LifecycleTracker(**kwargs)


if __name__ == '__main__':
    unittest.main()
