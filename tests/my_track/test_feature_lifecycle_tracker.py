import unittest

import numpy as np

from original.my_track.feature_lifecycle_tracker import FeatureLifecycleTracker
from original.my_track.lifecycle_tracker import LifecycleTracker


def update(tracker, points, scores=None, features=None):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    scores = np.ones(len(points), dtype=np.float64) if scores is None else np.asarray(scores)
    return tracker.update(points, scores, candidate_features=features)


class FeatureLifecycleTrackerTests(unittest.TestCase):
    def test_default_lifecycle_path_remains_identical_without_features(self):
        kwargs = dict(hist_len=1, cluster_radius_m=.05, association_radius_m=1.)
        plain = LifecycleTracker(**kwargs)
        feature = FeatureLifecycleTracker(**kwargs)
        frames = [
            (np.asarray([[-.4, 2.], [.4, 2.]]), np.asarray([.99, .98])),
            (np.asarray([[-.1, 2.], [.1, 2.]]), np.asarray([.98, .99])),
            (np.asarray([[.2, 2.], [-.2, 2.]]), np.asarray([.97, .98])),
        ]
        for points, scores in frames:
            left = plain.update(points, scores)
            right = feature.update(points, scores)
            for left_field, right_field in zip(left, right):
                np.testing.assert_array_equal(left_field, right_field)

    def test_feature_resolves_ambiguous_pairing(self):
        tracker = FeatureLifecycleTracker(
            hist_len=1, cluster_radius_m=.02, association_radius_m=.5,
            velocity_alpha=0., feature_weight=1., max_feature_age=3, feature_dim=2,
        )
        first_features = np.asarray([[1., 0.], [0., 1.]])
        update(tracker, [[-.2, 2.], [.2, 2.]], features=first_features)
        # Both candidates are inside both gates; geometry alone is symmetric.
        second_features = np.asarray([[0., 1.], [1., 0.]])
        result = update(tracker, [[-.1, 2.], [.1, 2.]], features=second_features)
        np.testing.assert_array_equal(result.track_ids, [0, 1])
        np.testing.assert_allclose(result.positions, [[.1, 2.], [-.1, 2.]])

    def test_feature_timeout_falls_back_to_spatial_matching(self):
        tracker = FeatureLifecycleTracker(
            hist_len=1, cluster_radius_m=.02, association_radius_m=.5,
            velocity_alpha=0., feature_weight=1., max_feature_age=1, feature_dim=2,
        )
        update(tracker, [[-.2, 2.], [.2, 2.]], features=[[1., 0.], [0., 1.]])
        update(tracker, [], features=np.empty((0, 2)))
        result = update(tracker, [[-.1, 2.], [.1, 2.]], features=[[0., 1.], [1., 0.]])
        # The old appearance is stale; deterministic geometry/order is used.
        np.testing.assert_array_equal(result.track_ids, [0, 1])
        np.testing.assert_allclose(result.positions, [[-.1, 2.], [.1, 2.]])

    def test_features_are_validated_without_changing_state(self):
        tracker = FeatureLifecycleTracker(hist_len=1, feature_dim=2)
        update(tracker, [[1., 2.]], features=[[1., 0.]])
        with self.assertRaises(ValueError):
            update(tracker, [[1., 2.]], features=[[1., np.nan]])
        with self.assertRaises(ValueError):
            update(tracker, [[1., 2.]], features=np.ones((1, 3)))
        result = update(tracker, [[1., 2.]], features=[[1., 0.]])
        self.assertEqual(result.track_ids.tolist(), [0])

    def test_feature_input_does_not_suppress_unmatched_births(self):
        tracker = FeatureLifecycleTracker(hist_len=1, max_tracks=2, cluster_radius_m=.05,
                                          feature_dim=2)
        result = update(
            tracker, [[0., 2.], [2., 2.]], scores=[.99, .98],
            features=np.asarray([[1., 0.], [0., 1.]]),
        )
        self.assertEqual(len(result.positions), 2)
        self.assertEqual(tracker.statistics['created'], 2)

    def test_clustered_feature_prototype_uses_the_same_score_weighted_members(self):
        tracker = FeatureLifecycleTracker(hist_len=1, feature_dim=2, cluster_radius_m=.1)
        positions = np.asarray([[0., 2.], [.01, 2.], [2., 2.]])
        scores = np.asarray([.99, .98, .97])
        features = np.asarray([[1., 0.], [0., 1.], [0., 1.]])
        primary, weak = tracker._proposals(positions, scores)
        primary_features, weak_features = tracker._proposal_features(
            positions, scores, features, primary, weak,
        )
        self.assertEqual(len(weak.positions), 0)
        expected = np.asarray([.99, .98]) / np.linalg.norm([.99, .98])
        np.testing.assert_allclose(primary_features, [expected, [0., 1.]])
        self.assertEqual(weak_features.shape, (0, 2))


if __name__ == '__main__':
    unittest.main()
