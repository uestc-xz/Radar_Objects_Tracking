"""剩余弱量测续接的行为与真实格式缓存集成测试。"""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from original.my_track.lifecycle_tracker import LifecycleTracker
from original.my_track.tentative_weak import TentativeWeakLifecycleTracker
from scripts.evaluate_m1_lifecycle import evaluate_records
from scripts.diagnose_m1_id_switches import diagnose_record


def step(tracker, points=(), scores=()):
    return tracker.update(np.asarray(points, dtype=np.float64).reshape(-1, 2),
                          np.asarray(scores, dtype=np.float64))


class TentativeWeakTests(unittest.TestCase):
    def test_high_birth_then_weak_updates_confirm_same_fragment_after_full_history(self):
        base = LifecycleTracker(hist_len=4)
        trial = TentativeWeakLifecycleTracker(hist_len=4)
        for index, score in enumerate((.99, .9, .9, .9)):
            original = step(base, [[1, 2]], [score])
            output = step(trial, [[1, 2]], [score])
            self.assertEqual(len(original.positions), 0)
            if index < 3:
                self.assertEqual(len(output.positions), 0)
        np.testing.assert_array_equal(output.track_ids, [0])
        np.testing.assert_allclose(output.positions, [[1, 2]])
        np.testing.assert_array_equal(output.observed, [True])
        self.assertEqual(trial.statistics, dict(created=1, confirmed=1, deleted=0))
        self.assertEqual(base.statistics, dict(created=1, confirmed=0, deleted=1))

    def test_weak_candidates_cannot_birth_even_if_persistent(self):
        tracker = TentativeWeakLifecycleTracker(hist_len=3)
        for _ in range(10):
            self.assertEqual(len(step(tracker, [[1, 2]], [.95]).positions), 0)
        self.assertEqual(tracker.statistics, dict(created=0, confirmed=0, deleted=0))

    def test_confirmed_has_priority_over_closer_tentative_and_weak_is_used_once(self):
        tracker = TentativeWeakLifecycleTracker(hist_len=3, cluster_radius_m=.05)
        for _ in range(3):
            step(tracker, [[0, 2]], [.99])
        step(tracker, [[0, 2], [.8, 2]], [.99, .99])
        output = step(tracker, [[.7, 2]], [.9])
        np.testing.assert_array_equal(output.track_ids, [0])
        np.testing.assert_allclose(output.positions, [[.7, 2]])
        np.testing.assert_array_equal(output.observed, [True])
        tentative = next(t for t in tracker._tracks if t.track_id == 1)
        self.assertEqual(tentative.missed, 1)
        self.assertEqual(list(tentative.hits), [True, False])
        np.testing.assert_allclose(tentative.position, [.8, 2])

    def test_disabled_weak_updates_are_exactly_equivalent_to_original(self):
        base = LifecycleTracker(hist_len=3, weak_updates=False)
        trial = TentativeWeakLifecycleTracker(hist_len=3, weak_updates=False)
        for index in range(30):
            points = [[1 + .01 * index, 2]] if index % 7 != 0 else []
            scores = ([.9 if index % 3 else .99] if points else [])
            for original, output in zip(step(base, points, scores), step(trial, points, scores)):
                np.testing.assert_array_equal(original, output)
        self.assertEqual(base.statistics, trial.statistics)

    def test_joint_policy_is_explicitly_unsupported(self):
        with self.assertRaisesRegex(ValueError, 'cascade'):
            TentativeWeakLifecycleTracker(association_policy='joint')

    def test_primary_confirmation_keeps_weak_fragment_alive_without_confirming(self):
        tracker = TentativeWeakLifecycleTracker(hist_len=4, primary_confirmation=True)
        step(tracker, [[1, 2]], [.99])
        for _ in range(10):
            self.assertEqual(len(step(tracker, [[1, 2]], [.9]).positions), 0)
        self.assertEqual(tracker.statistics, dict(created=1, confirmed=0, deleted=0))
        self.assertEqual(tracker._tracks[0].missed, 0)

    def test_primary_confirmation_accepts_enough_interrupted_strong_evidence(self):
        tracker = TentativeWeakLifecycleTracker(hist_len=5, primary_confirmation=True)
        for index, score in enumerate((.99, .9, .99, .9, .99)):
            output = step(tracker, [[1, 2]], [score])
            self.assertEqual(len(output.positions), int(index == 4))
        self.assertEqual(output.track_ids.tolist(), [0])

    def test_primary_evidence_expires_outside_history_and_reset_clears_it(self):
        tracker = TentativeWeakLifecycleTracker(hist_len=5, primary_confirmation=True)
        # 两个旧高分不足以确认，且会在第三个高分到来前过期。
        for score in (.99, .99, .9, .9, .9, .9, .9, .99):
            self.assertEqual(len(step(tracker, [[1, 2]], [score]).positions), 0)
        tracker.reset()
        for index, score in enumerate((.99, .9, .9, .99, .99)):
            output = step(tracker, [[1, 2]], [score])
            self.assertEqual(len(output.positions), int(index == 4))
        self.assertEqual(output.track_ids.tolist(), [0])
        self.assertEqual(tracker.statistics, dict(created=1, confirmed=1, deleted=0))

    def test_primary_confirmation_without_weak_updates_matches_original(self):
        base = LifecycleTracker(hist_len=4, weak_updates=False)
        trial = TentativeWeakLifecycleTracker(hist_len=4, weak_updates=False,
                                              primary_confirmation=True)
        for index in range(25):
            points = [[1, 2]] if index % 7 else []
            scores = [.99] if points else []
            for original, output in zip(step(base, points, scores), step(trial, points, scores)):
                np.testing.assert_array_equal(original, output)
        self.assertEqual(base.statistics, trial.statistics)

    def test_cache_adapter_composition_and_gt_independence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames = []
            label, changed = root / 'label.npy', root / 'changed.npy'
            np.save(label, np.tile([1, 2, 0, 0, 0, 0], (10, 1)).astype(np.float32))
            np.save(changed, np.tile([-2, 5, 0, 0, 0, 0], (10, 1)).astype(np.float32))
            for index, score in enumerate((.99, .9, .9, .9)):
                path = root / f'{index}.npz'
                np.savez(path, out=np.asarray([[1, 2, .4, .4]], dtype=np.float32),
                         conf=np.asarray([[0, np.log(score / (1 - score))]], dtype=np.float32))
                frames.append((path, label, index))
            names = ['lifecycle', 'recovery_reachable', 'tentative_weak', 'tentative_weak_recovery']
            names += ['tentative_weak_primary_confirm', 'tentative_weak_primary_confirm_recovery']
            record = dict(folder='20240913_1_1', frames=frames)
            result = evaluate_records([record], names, hist_len=4)['variants']
            for name in names[2:]:
                diagnostic = diagnose_record(record, variant=name, hist_len=4)
                self.assertEqual(diagnostic['prediction_digest'],
                                 result[name]['sequences'][0]['prediction_digest'])
            record['frames'] = [(path, changed, index) for path, _, index in frames]
            alternative = evaluate_records([record], names, hist_len=4)['variants']
            self.assertEqual(result['lifecycle']['summary']['tp'], 0)
            self.assertEqual(result['tentative_weak']['summary']['tp'], 1)
            self.assertEqual(result['tentative_weak']['summary']['fn'], 3)
            self.assertEqual(result['tentative_weak_primary_confirm']['summary']['tp'], 0)
            for name in names:
                self.assertEqual(result[name]['sequences'][0]['prediction_digest'],
                                 alternative[name]['sequences'][0]['prediction_digest'])
            self.assertEqual(result['tentative_weak']['sequences'][0]['evidence_digest'],
                             result['tentative_weak_recovery']['sequences'][0]['evidence_digest'])


if __name__ == '__main__':
    unittest.main()
