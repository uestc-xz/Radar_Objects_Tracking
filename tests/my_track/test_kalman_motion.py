import unittest

import numpy as np

from original.my_track.kalman_motion import (
    KalmanMotionTracker, correct_motion, predict_motion,
)
from original.my_track.lifecycle_tracker import LifecycleTracker


class KalmanMotionTests(unittest.TestCase):
    def test_prediction_accumulates_independent_acceleration_each_step(self):
        # One axis: P0=0, q=4 -> Q=[[1,2],[2,4]]. Two steps -> [[10,8],[8,8]].
        mean, covariance = predict_motion(
            np.asarray([1., 2., .1, -.2]), np.zeros((4, 4)), [4., 4.], 2,
        )
        np.testing.assert_allclose(mean, [1.2, 1.6, .1, -.2])
        np.testing.assert_allclose(covariance[np.ix_([0, 2], [0, 2])], [[10., 8.], [8., 8.]])

    def test_correction_hand_calculation_and_positive_covariance(self):
        mean, covariance = correct_motion(np.zeros(4), np.eye(4), [2., -4.], [1., 3.])
        np.testing.assert_allclose(mean, [1., -1., 0., 0.])
        np.testing.assert_allclose(np.diag(covariance), [.5, .75, 1., 1.])
        self.assertGreater(np.linalg.eigvalsh(covariance).min(), 0.)

    def test_prediction_rejects_invalid_gap_and_noise(self):
        for gap in (-1, True, 1.5):
            with self.subTest(gap=gap), self.assertRaises(ValueError):
                predict_motion(np.zeros(4), np.eye(4), [1., 1.], gap)
        for noise in ([0., 1.], [np.nan, 1.], [1.], [-1., 1.]):
            with self.subTest(noise=noise), self.assertRaises(ValueError):
                correct_motion(np.zeros(4), np.eye(4), [1., 2.], noise)

    @staticmethod
    def make_tracker(**kwargs):
        return KalmanMotionTracker(
            measurement_var=[1., 1.], acceleration_var=[.0001, .0001],
            initial_velocity_var=[.0001, .0001], hist_len=1, **kwargs,
        )

    def test_filtered_association_retains_target_after_single_measurement_jitter(self):
        filtered = self.make_tracker()
        baseline = LifecycleTracker(hist_len=1)
        for x in [1.] * 10 + [1.99, .7]:
            points, scores = np.asarray([[x, 2.]]), np.asarray([.99])
            a = filtered.update(points, scores)
            b = baseline.update(points, scores)
        # The raw-state baseline cannot return from +.99 jitter to -.3 within 1m.
        self.assertEqual(a.track_ids.tolist(), [0])
        self.assertEqual(a.observed.tolist(), [True])
        np.testing.assert_allclose(a.positions, [[.7, 2.]])
        self.assertIn(False, b.observed.tolist())

    def test_multistage_prediction_is_pure_and_output_remains_measurement(self):
        tracker = self.make_tracker()
        tracker.update(np.asarray([[1., 2.]]), np.asarray([.99]))
        tracker.update(np.asarray([[1.4, 2.]]), np.asarray([.99]))
        a = tracker._association_predictions([0])
        b = tracker._association_predictions([0])
        np.testing.assert_array_equal(a, b)
        self.assertLess(a[0, 0], 1.3)
        output = tracker.update(np.asarray([[1.2, 2.]]), np.asarray([.9]))
        self.assertEqual(output.observed.tolist(), [True])
        np.testing.assert_allclose(output.positions, [[1.2, 2.]])

    def test_invalid_frame_does_not_change_filter_then_reset_clears_history(self):
        tracker = self.make_tracker()
        tracker.update(np.asarray([[1., 2.]]), np.asarray([.99]))
        before = tracker._association_predictions([0]).copy()
        with self.assertRaises(ValueError):
            tracker.update(np.asarray([[np.nan, 2.]]), np.asarray([.99]))
        np.testing.assert_array_equal(tracker._association_predictions([0]), before)
        tracker.reset()
        output = tracker.update(np.asarray([[4., 3.]]), np.asarray([.99]))
        self.assertEqual(output.track_ids.tolist(), [0])
        np.testing.assert_allclose(tracker._association_predictions([0]), [[4., 3.]], atol=1e-12)

    def test_expired_track_filter_is_removed_and_new_birth_is_independent(self):
        tracker = self.make_tracker(max_missed=0)
        tracker.update(np.asarray([[1., 2.]]), np.asarray([.99]))
        tracker.update(np.empty((0, 2)), np.empty(0))
        self.assertEqual(tracker._motion_states, {})
        output = tracker.update(np.asarray([[4., 3.]]), np.asarray([.99]))
        self.assertEqual(output.track_ids.tolist(), [1])
        self.assertEqual(set(tracker._motion_states), {1})


if __name__ == '__main__':
    unittest.main()
