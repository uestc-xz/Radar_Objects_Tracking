from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.evaluate_m1_lifecycle import DEFAULT_VARIANTS, evaluate_records


NOISE = dict(measurement_var=[.04, .04], acceleration_var=[.001, .001],
             initial_velocity_var=[.01, .01])


class KalmanEvaluationTests(unittest.TestCase):
    def test_opt_in_requires_frozen_noise_without_changing_default_variants(self):
        self.assertNotIn('kalman_motion', DEFAULT_VARIANTS)
        with self.assertRaises(ValueError):
            evaluate_records([], ['kalman_motion'])

    def test_adapter_uses_noise_resets_sequences_and_keeps_gt_out_of_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames = []
            for index, points in enumerate(([[1., 2.]], [], [], [[1.2, 2.]])):
                cache, label = root / f'{index}.npz', root / f'{index}.npy'
                points = np.asarray(points).reshape(-1, 2)
                np.savez(cache, out=np.column_stack((points, np.ones((len(points), 2)))),
                         conf=np.tile([0., 10.], (len(points), 1)))
                np.save(label, np.tile([1., 2., 0., 0., 0., 0.], (10, 1)))
                frames.append((cache, label, index))
            records = [{'folder': name, 'frames': frames} for name in ('a', 'b')]
            variants = ['lifecycle', 'recovery_reachable', 'kalman_motion', 'kalman_motion_recovery']
            results = evaluate_records(records, variants, hist_len=1, motion_noise=NOISE)['variants']
            baseline = evaluate_records(records, ['lifecycle'], hist_len=1)['variants']['lifecycle']
            self.assertEqual(results['lifecycle'], baseline)
            for row in results['kalman_motion']['sequences']:
                self.assertEqual(row['settings']['kalman_motion'], NOISE)
                self.assertEqual(row['metrics']['gt_id_switches'], 1)
            for raw, recovered in zip(results['kalman_motion']['sequences'],
                                      results['kalman_motion_recovery']['sequences']):
                self.assertEqual(raw['evidence_digest'], recovered['evidence_digest'])
                self.assertEqual(recovered['metrics']['gt_id_switches'], 0)
            for _, label, _ in frames:
                np.save(label, np.tile([-4., 8., 0., 0., 0., 0.], (10, 1)))
            changed = evaluate_records(records, ['kalman_motion'], hist_len=1,
                                       motion_noise=NOISE)['variants']['kalman_motion']
            self.assertEqual([r['prediction_digest'] for r in results['kalman_motion']['sequences']],
                             [r['prediction_digest'] for r in changed['sequences']])
            self.assertEqual(changed['summary']['tp'], 0)


if __name__ == '__main__':
    unittest.main()
