import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.evaluate_m1_lifecycle import DEFAULT_VARIANTS, VARIANTS, evaluate_records, main


class LifecycleEvaluationTests(unittest.TestCase):
    def test_known_no_death_recovery_is_explicit_opt_in_and_records_its_prior(self):
        name = 'lifecycle_known_no_death_recovery'
        self.assertIn(name, VARIANTS)
        self.assertNotIn(name, DEFAULT_VARIANTS)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache, label = root / 'cache.npz', root / 'label.npy'
            np.savez(cache, out=np.asarray([[1., 2., .4, .4]], dtype=np.float32),
                     conf=np.asarray([[0., 10.]], dtype=np.float32))
            np.save(label, np.tile([1., 2., 0., 0., 0., 0.], (10, 1)).astype(np.float32))
            record = {'folder': '20240913_1_1', 'frames': [(cache, label, i) for i in range(4)]}
            variants = evaluate_records([record], [name], hist_len=3)['variants']
            data = variants[name]
            self.assertGreater(data['summary']['tp'], 0)
            settings = data['sequences'][0]['settings']
            self.assertTrue(settings['no_death'])
            self.assertEqual(settings['max_tracks'], 1)
            self.assertEqual(settings['identity_recovery']['known_count'], 1)

    def test_known_no_death_coast_uses_hist_len_only_when_explicitly_selected(self):
        name = 'lifecycle_known_no_death_coast'
        self.assertIn(name, VARIANTS)
        self.assertNotIn(name, DEFAULT_VARIANTS)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache, label = root / 'cache.npz', root / 'label.npy'
            np.savez(cache, out=np.asarray([[1., 2., .4, .4]], dtype=np.float32),
                     conf=np.asarray([[0., 10.]], dtype=np.float32))
            np.save(label, np.tile([1., 2., 0., 0., 0., 0.], (10, 1)).astype(np.float32))
            record = {'folder': '20240913_1_1', 'frames': [(cache, label, i) for i in range(4)]}
            data = evaluate_records([record], [name], hist_len=3)['variants'][name]
            settings = data['sequences'][0]['settings']
            self.assertTrue(settings['no_death'])
            self.assertEqual(settings['max_tracks'], 1)
            self.assertEqual(settings['output_max_missed'], 3)
            self.assertEqual(settings['output_confidence'], .5)

    def test_known_no_death_persistent_coast_disables_confidence_floor_as_opt_in(self):
        name = 'lifecycle_known_no_death_persistent_coast'
        self.assertIn(name, VARIANTS)
        self.assertNotIn(name, DEFAULT_VARIANTS)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache, label = root / 'cache.npz', root / 'label.npy'
            np.savez(cache, out=np.asarray([[1., 2., .4, .4]], dtype=np.float32),
                     conf=np.asarray([[0., 10.]], dtype=np.float32))
            np.save(label, np.tile([1., 2., 0., 0., 0., 0.], (10, 1)).astype(np.float32))
            record = {'folder': '20240913_1_1', 'frames': [(cache, label, i) for i in range(4)]}
            data = evaluate_records([record], [name], hist_len=3)['variants'][name]
            settings = data['sequences'][0]['settings']
            self.assertTrue(settings['no_death'])
            self.assertEqual(settings['max_tracks'], 1)
            self.assertEqual(settings['output_max_missed'], 3)
            self.assertEqual(settings['output_confidence'], 0.)

    def test_real_cache_adapter_counts_startup_and_aggregates_micro_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache, label = root / 'cache.npz', root / 'label.npy'
            np.savez(cache, out=np.asarray([[1, 2, .4, .4]], dtype=np.float32),
                     conf=np.asarray([[0, 10]], dtype=np.float32))
            np.save(label, np.tile([1, 2, 0, 0, 0, 0], (10, 1)).astype(np.float32))
            record = {'folder': '20240913_1_1', 'frames': [(cache, label, i) for i in range(3)]}
            result = evaluate_records([record], ['simple_miss0', 'lifecycle'], hist_len=3)
            self.assertEqual(result['variants']['simple_miss0']['summary']['tp'], 3)
            lifecycle = result['variants']['lifecycle']['summary']
            self.assertEqual((lifecycle['tp'], lifecycle['fn']), (1, 2))
            self.assertAlmostEqual(lifecycle['identity_f1'], .5)
            sample = result['data_checks'][0]
            self.assertEqual(sample['out_shape'], [1, 4])
            self.assertEqual(sample['label_shape'], [10, 6])
            self.assertEqual(sample['sample_id_min'], 0)
            self.assertEqual(sample['sample_id_max'], 2)

    def test_changing_labels_never_changes_inference_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache, first, second = root / 'cache.npz', root / 'first.npy', root / 'second.npy'
            np.savez(cache, out=np.asarray([[1, 2, .4, .4]], dtype=np.float32),
                     conf=np.asarray([[0, 1000]], dtype=np.float32))
            np.save(first, np.tile([1, 2, 0, 0, 0, 0], (10, 1)))
            np.save(second, np.tile([-1, 5, 0, 0, 0, 0], (10, 1)))
            a = {'folder': '20240913_1_1', 'frames': [(cache, first, 0)]}
            b = {'folder': '20240913_1_1', 'frames': [(cache, second, 0)]}
            ra = evaluate_records([a], ['lifecycle'], hist_len=1)['variants']['lifecycle']
            rb = evaluate_records([b], ['lifecycle'], hist_len=1)['variants']['lifecycle']
            self.assertEqual(ra['sequences'][0]['prediction_digest'], rb['sequences'][0]['prediction_digest'])
            self.assertEqual(ra['summary']['tp'], 1)
            self.assertEqual(rb['summary']['tp'], 0)

    def test_each_sequence_resets_state_and_known_count_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache, label = root / 'cache.npz', root / 'label.npy'
            np.savez(cache, out=np.asarray([[1, 2, .4, .4]]), conf=np.asarray([[0, 10.]]))
            np.save(label, np.tile([1, 2, 0, 0, 0, 0], (10, 1)))
            records = [{'folder': name, 'frames': [(cache, label, 0)]}
                       for name in ['20240913_1_1', '20240913_1_2']]
            result = evaluate_records(records, ['lifecycle_known_no_death'], hist_len=3)
            data = result['variants']['lifecycle_known_no_death']
            self.assertEqual(data['summary']['tp'], 0)
            self.assertEqual(data['summary']['fn'], 2)
            for sequence in data['sequences']:
                self.assertTrue(sequence['settings']['no_death'])
                self.assertEqual(sequence['settings']['max_tracks'], 1)

    def test_association_policy_switch_reproduces_the_weak_candidate_tradeoff(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames = []
            for index, points, scores, target in (
                (0, [[1., 2.]], [1.], [1., 2., 0, 0, 0, 0]),
                (1, [[1.9, 2.], [1.1, 2.]], [.99, .9], [1.9, 2., 0, 0, 0, 0]),
            ):
                cache, label = root / f'{index}.npz', root / f'{index}.npy'
                probability = np.clip(np.asarray(scores), 1e-6, 1 - 1e-6)
                logits = np.stack([np.log(1 - probability), np.log(probability)], axis=1)
                np.savez(cache, out=np.column_stack([points, np.ones((len(points), 2))]), conf=logits)
                np.save(label, np.tile(target, (10, 1)))
                frames.append((cache, label, index))
            record = {'folder': '20240913_1_1', 'frames': frames}
            cascade = evaluate_records([record], ['lifecycle'], hist_len=1)
            joint = evaluate_records([record], ['lifecycle'], hist_len=1, association_policy='joint')
            self.assertEqual(cascade['variants']['lifecycle']['summary']['fp'], 0)
            self.assertEqual(joint['variants']['lifecycle']['summary']['fp'], 1)

    def test_cli_rejects_overwrite_before_loading_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'existing.json'
            target.write_text(json.dumps({'keep': 1}), encoding='utf-8')
            with self.assertRaises(SystemExit):
                main(['--output', str(target)])
            self.assertEqual(json.loads(target.read_text(encoding='utf-8')), {'keep': 1})

    def test_identity_recovery_preserves_detection_and_changes_fragment_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames = []
            # hist_len=1，连续两帧漏检后旧状态删除；第 3 步附近出现新确认片段。
            for index, points in enumerate(([[1., 2.]], [], [], [[1.2, 2.]])):
                cache, label = root / f'{index}.npz', root / f'{index}.npy'
                points = np.asarray(points).reshape(-1, 2)
                np.savez(cache, out=np.column_stack([points, np.ones((len(points), 2))]),
                         conf=np.tile([0., 10.], (len(points), 1)))
                np.save(label, np.tile([1., 2., 0, 0, 0, 0], (10, 1)))
                frames.append((cache, label, index))
            result = evaluate_records([{'folder': '20240913_1_1', 'frames': frames}],
                                      ['lifecycle', 'recovery_local'], hist_len=1)['variants']
            baseline, recovered = result['lifecycle'], result['recovery_local']
            self.assertEqual(baseline['summary']['gt_id_switches'], 1)
            self.assertEqual(recovered['summary']['gt_id_switches'], 0)
            for key in ('tp', 'fp', 'fn', 'error_sum_m', 'coasted_tp', 'coasted_fp'):
                self.assertEqual(baseline['summary'][key], recovered['summary'][key])
            self.assertEqual(baseline['sequences'][0]['evidence_digest'],
                             recovered['sequences'][0]['evidence_digest'])
            self.assertEqual(recovered['sequences'][0]['management']['recovered'], 1)
            # 同样候选和片段边界，换掉所有 GT 坐标也不得改变恢复后的完整输出。
            for _, label, _ in frames:
                np.save(label, np.tile([-2., 5., 0, 0, 0, 0], (10, 1)))
            changed = evaluate_records([{'folder': '20240913_1_1', 'frames': frames}],
                                       ['recovery_local'], hist_len=1)['variants']['recovery_local']
            self.assertEqual(recovered['sequences'][0]['prediction_digest'],
                             changed['sequences'][0]['prediction_digest'])
            self.assertNotEqual(recovered['summary']['tp'], changed['summary']['tp'])

    def test_stricter_diagnostic_is_separate_from_original_matching_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache, label = root / 'cache.npz', root / 'label.npy'
            np.savez(cache, out=np.asarray([[1., 2., .4, .4]]), conf=np.asarray([[0., 10.]]))
            np.save(label, np.tile([1.8, 2., 0, 0, 0, 0], (10, 1)))
            records = [{'folder': name, 'frames': [(cache, label, 0)]}
                       for name in ('20240913_1_1', '20240913_1_2')]
            data = evaluate_records(records, ['lifecycle'], hist_len=1)['variants']['lifecycle']
            self.assertEqual(data['summary']['tp'], 2)
            self.assertEqual(data['summary_0_5m']['tp'], 0)
            self.assertEqual(data['summary_0_5m']['fp'], 2)
            self.assertEqual(data['summary_0_5m']['fn'], 2)
            self.assertIsNone(data['summary_0_5m']['matched_mae_m'])

    def test_existing_default_comparison_still_accepts_joint_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache, label = root / 'cache.npz', root / 'label.npy'
            np.savez(cache, out=np.asarray([[1., 2., .4, .4]]), conf=np.asarray([[0., 10.]]))
            np.save(label, np.tile([1., 2., 0, 0, 0, 0], (10, 1)))
            record = {'folder': '20240913_1_1', 'frames': [(cache, label, 0)]}
            data = evaluate_records([record], hist_len=1, association_policy='joint')['variants']['lifecycle']
            self.assertEqual(data['summary']['tp'], 1)
            self.assertEqual(data['sequences'][0]['settings']['association_policy'], 'joint')

    def test_history_variant_runs_history_association_and_ignores_changed_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames = []
            sequence = ([[0, 2], [1.9, 2]],) * 3 + ([[.99, 2], [1.9, 2]], [[0, 2], [1.1, 2]])
            for index, points in enumerate(sequence):
                cache, label = root / f'{index}.npz', root / f'{index}.npy'
                np.savez(cache, out=np.column_stack([points, np.ones((2, 2))]),
                         conf=np.tile([0., 10.], (2, 1)))
                np.save(label, np.tile([0., 2., 1.9, 2., 0, 0], (10, 1)))
                frames.append((cache, label, index))
            records = [{'folder': '20240913_2_1', 'frames': frames}]
            names = ['lifecycle', 'history_motion', 'history_motion_recovery']
            result = evaluate_records(records, names, hist_len=4)['variants']
            self.assertNotEqual(result['lifecycle']['sequences'][0]['evidence_digest'],
                                result['history_motion']['sequences'][0]['evidence_digest'])
            for _, label, _ in frames:
                np.save(label, np.tile([-2., 5., -2., 3., 0, 0], (10, 1)))
            changed = evaluate_records(records, names, hist_len=4)['variants']
            for name in names:
                self.assertEqual(result[name]['sequences'][0]['prediction_digest'],
                                 changed[name]['sequences'][0]['prediction_digest'])

    def test_recent_priority_variants_execute_their_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames = []
            for index, points in enumerate(([[0, 2], [.8, 2]], [[0, 2]], [[.45, 2]])):
                cache, label = root / f'{index}.npz', root / f'{index}.npy'
                np.savez(cache, out=np.column_stack([points, np.ones((len(points), 2))]),
                         conf=np.tile([0., 10.], (len(points), 1)))
                np.save(label, np.tile([0., 2., .8, 2., 0, 0], (10, 1)))
                frames.append((cache, label, index))
            records = [{'folder': '20240913_2_1', 'frames': frames}]
            names = ['lifecycle', 'recent_priority', 'recent_priority_recovery',
                     'recent_priority_stale', 'recent_priority_stale_recovery']
            result = evaluate_records(records, names, hist_len=1)['variants']
            self.assertNotEqual(result['lifecycle']['sequences'][0]['evidence_digest'],
                                result['recent_priority']['sequences'][0]['evidence_digest'])
            self.assertEqual(result['recent_priority']['sequences'][0]['evidence_digest'],
                             result['recent_priority_recovery']['sequences'][0]['evidence_digest'])
            self.assertEqual(result['recent_priority']['sequences'][0]['evidence_digest'],
                             result['recent_priority_stale']['sequences'][0]['evidence_digest'])


if __name__ == '__main__':
    unittest.main()
