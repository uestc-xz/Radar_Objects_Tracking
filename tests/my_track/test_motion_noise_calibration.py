import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.calibrate_m1_motion_noise import estimate_motion_noise, load_motion_noise, main


class MotionNoiseCalibrationTests(unittest.TestCase):
    @staticmethod
    def make_record(root, name, xs, sample_ids=None):
        frames = []
        for index, x in enumerate(xs):
            cache = root / f'{name}_{index}.npz'
            label = root / f'{name}_{index}.npy'
            np.savez(cache, out=np.asarray([[x + .1, 2.2, .4, .4]]), conf=np.asarray([[0., 10.]]))
            np.save(label, np.tile([x, 2., 0., 0., 0., 0.], (10, 1)))
            frames.append((cache, label, index if sample_ids is None else sample_ids[index]))
        return {'folder': name, 'frames': frames}

    def test_hand_calibrated_residual_acceleration_and_velocity_without_cross_sequence_differences(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = [self.make_record(root, 'a', [1., 1.1, 1.3]),
                       self.make_record(root, 'b', [4., 4.])]
            result = estimate_motion_noise(records)
            # Residuals (.1,.2); one acceleration .1; velocities .1,.2,0.
            np.testing.assert_allclose(result['noise']['measurement_var'], [.01, .04])
            np.testing.assert_allclose(result['noise']['acceleration_var'], [.01, 1e-6])
            np.testing.assert_allclose(result['noise']['initial_velocity_var'], [1 / 60, 1e-6])
            self.assertEqual(result['counts'], {'matched_measurements': 5, 'velocity_pairs': 3,
                                               'acceleration_triples': 1})
            self.assertEqual(result['source_split'], 'train')
            path = root / 'noise.json'
            path.write_text(json.dumps(result), encoding='utf-8')
            self.assertEqual(load_motion_noise(path), result['noise'])

    def test_sample_gap_excludes_cross_gap_velocity_and_acceleration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a = self.make_record(root, 'a', [1., 1.1, 1.3])
            b = self.make_record(root, 'gap', [1., 4.], [0, 2])
            result = estimate_motion_noise([a, b])
            self.assertEqual(result['counts']['velocity_pairs'], 2)
            self.assertEqual(result['counts']['acceleration_triples'], 1)
            np.testing.assert_allclose(result['noise']['initial_velocity_var'], [.025, 1e-6])

    def test_ordered_parallel_reads_reproduce_serial_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            record = self.make_record(Path(directory), 'a', [1., 1.1, 1.3])
            self.assertEqual(estimate_motion_noise([record], workers=2), estimate_motion_noise([record]))

    def test_insufficient_motion_evidence_is_rejected_instead_of_using_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            record = self.make_record(Path(directory), 'short', [1., 1.1])
            with self.assertRaises(ValueError):
                estimate_motion_noise([record])

    def test_noise_loader_rejects_wrong_split_or_invalid_variance(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'noise.json'
            for split, variance in [('val', [1., 1.]), ('train', [0., 1.]),
                                    ('train', [1.]), ('train', [float('nan'), 1.])]:
                payload = {'schema_version': 1, 'source_split': split, 'noise': {
                    'measurement_var': variance, 'acceleration_var': [1., 1.],
                    'initial_velocity_var': [1., 1.],
                }}
                path.write_text(json.dumps(payload), encoding='utf-8')
                with self.subTest(split=split, variance=variance), self.assertRaises(ValueError):
                    load_motion_noise(path)

    def test_cli_refuses_existing_output_before_reading_training_data(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'keep.json'
            path.write_text('keep', encoding='utf-8')
            with self.assertRaises(SystemExit):
                main(['--output', str(path)])
            self.assertEqual(path.read_text(encoding='utf-8'), 'keep')


if __name__ == '__main__':
    unittest.main()
