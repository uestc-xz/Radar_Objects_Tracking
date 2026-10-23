import unittest

import numpy as np

from original.my_track.history_motion import HistoryMotionTracker, fit_history_prediction
from original.my_track.lifecycle_tracker import LifecycleTracker


def step(tracker, points=()):
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    return tracker.update(points, np.ones(len(points)))


class HistoryPredictionTests(unittest.TestCase):
    def test_linear_motion_and_gap_decay_match_hand_calculation(self):
        points = np.array([[0., 2], [.1, 2], [.2, 2], [.3, 2]])
        np.testing.assert_allclose(fit_history_prediction(points, [0, 1, 2, 3], 1), [.4, 2])
        np.testing.assert_allclose(fit_history_prediction(points, [0, 1, 2, 3], 2), [.48, 2])
        np.testing.assert_allclose(fit_history_prediction(points, [0, 1, 2, 3], 2,
                                                         prediction_decay=1), [.5, 2])

    def test_irregular_observation_times_do_not_change_velocity_units(self):
        prediction = fit_history_prediction([[0, 2], [.2, 2], [.5, 2]], [3, 5, 8], 1)
        np.testing.assert_allclose(prediction, [.6, 2])

    def test_constant_position_does_not_drift(self):
        np.testing.assert_allclose(fit_history_prediction([[1, 2]] * 4, [0, 1, 2, 3], 9), [1, 2])

    def test_clipped_velocity_uses_a_consistent_fitted_intercept(self):
        # 原 slope=1，被截为 .2；t=[-1,0] 的拟合截距为 .5+.1=.6。
        np.testing.assert_allclose(fit_history_prediction([[0, 2], [1, 2]], [0, 1], 1), [.8, 2])

    def test_invalid_fit_data_and_parameters_are_rejected(self):
        cases = (
            ([[0, 2]], [0], {}),
            ([[0, 2], [1, 2]], [0, 0], {}),
            ([[0, 2], [np.nan, 2]], [0, 1], {}),
            ([[0, 2], [1, 2]], [1, 0], {}),
            ([[0, 2], [1, 2]], [0, 1], {'prediction_decay': 2}),
            ([[0, 2], [1, 2]], [0, 1], {'max_step_m': 0}),
        )
        for points, times, kwargs in cases:
            with self.subTest(points=points, times=times, kwargs=kwargs), self.assertRaises(ValueError):
                fit_history_prediction(points, times, 1, **kwargs)


class HistoryMotionTrackerTests(unittest.TestCase):
    def test_history_prevents_one_noisy_measurement_from_losing_the_original_target(self):
        tracker = HistoryMotionTracker(hist_len=4, cluster_radius_m=.05)
        baseline = LifecycleTracker(hist_len=4, cluster_radius_m=.05)
        for _ in range(3):
            step(tracker, [[0, 2], [1.5, 2]])
            step(baseline, [[0, 2], [1.5, 2]])
        step(tracker, [[.99, 2], [1.5, 2]])
        step(baseline, [[.99, 2], [1.5, 2]])
        result = step(tracker, [[0, 2], [1.1, 2]])
        self.assertEqual(step(baseline, [[0, 2], [1.1, 2]]).observed.tolist(), [True, False])
        self.assertEqual(result.track_ids.tolist(), [0, 1])
        self.assertEqual(result.observed.tolist(), [True, True])
        np.testing.assert_allclose(result.positions, [[0, 2], [1.1, 2]])

    def test_short_history_exactly_reproduces_original_output(self):
        tracker = HistoryMotionTracker(hist_len=3)
        baseline = LifecycleTracker(hist_len=3)
        for points in ([[0, 2]], [[.1, 2]], [[.2, 2]], [], [[.3, 2]]):
            actual, expected = step(tracker, points), step(baseline, points)
            for a, b in zip(actual, expected):
                np.testing.assert_array_equal(a, b)

    def test_observed_coordinates_and_coasting_rule_are_not_smoothed(self):
        tracker = HistoryMotionTracker(hist_len=4)
        baseline = LifecycleTracker(hist_len=4)
        for points in ([[0, 2]], [[0, 2]], [[0, 2]], [[.4, 2]], [[.5, 2]], []):
            actual, expected = step(tracker, points), step(baseline, points)
            for a, b in zip(actual, expected):
                np.testing.assert_array_equal(a, b)

    def test_reset_discards_history_before_a_new_sequence(self):
        tracker = HistoryMotionTracker(hist_len=4)
        for x in [0, .1, .2, .3]:
            step(tracker, [[x, 2]])
        tracker.reset()
        fresh = HistoryMotionTracker(hist_len=4)
        for x in [2., 1.9, 1.8, 1.7, 1.6]:
            actual, expected = step(tracker, [[x, 2]]), step(fresh, [[x, 2]])
            for a, b in zip(actual, expected):
                np.testing.assert_array_equal(a, b)

    def test_elapsed_steps_expire_old_observations_even_when_track_survives(self):
        tracker = HistoryMotionTracker(hist_len=4, min_motion_points=2,
                                       max_missed=20, cluster_radius_m=.01)
        for x in [0, .1, .2, .3]:
            step(tracker, [[x, 2]])
        for _ in range(3):
            step(tracker)
        # 过去四步仅剩一个观测，必须回退到原预测约 .444，而非用旧线性趋势预测 .595。
        result = step(tracker, [[.4, 2], [.65, 2]])
        np.testing.assert_allclose(result.positions[result.track_ids == 0], [[.4, 2]])

    def test_invalid_input_does_not_advance_motion_time(self):
        tracker = HistoryMotionTracker(hist_len=6, cluster_radius_m=.01)
        for x in [0, .1, .2, .3, .4, .5]:
            step(tracker, [[x, 2]])
        with self.assertRaises(ValueError):
            step(tracker, [[np.nan, 2]])
        # 有效下一步预测 .6；若非法帧推进时间，预测变成 .68，会选错为 .71。
        result = step(tracker, [[.59, 2], [.71, 2]])
        np.testing.assert_allclose(result.positions[result.track_ids == 0], [[.59, 2]])


if __name__ == '__main__':
    unittest.main()
