import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.evaluate_m1_lifecycle import evaluate_records, main


class LifecycleEvaluationTests(unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main()
