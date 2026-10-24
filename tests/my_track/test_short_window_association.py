"""短窗关联的数值、因果和对称性契约；GT 只用于离线样本。"""

import math
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from original.my_track import short_window_association as association
from original.my_track import short_window_data as data
from scripts import train_m1_short_window_association as training


def frame(step, points, targets=None):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    gt = np.zeros((3, 2), dtype=np.float64)
    if targets is not None:
        gt[:len(targets)] = targets
    return data.AssociationFrame(step, points, np.full(len(points), .99), gt)


class ShortWindowNumericsTests(unittest.TestCase):
    def test_velocity_baseline_uses_actual_gap_and_clips_step_displacement(self):
        history, candidates = association.encode_query(
            [(1, np.array([0., 2.]), .99), (3, np.array([1., 2.]), .99)],
            5, np.array([[1., 2.], [1.4, 2.]]), np.array([.99, .99]),
            np.ones(2, dtype=bool), hist_len=4)
        # 1m/2步限速为 .2m/步；再过两步，预测 1.4m 而非 1m 或 2m。
        result = association.baseline_logits(torch.tensor(history)[None],
                                             torch.tensor(candidates)[None],
                                             torch.ones(1, 2, dtype=torch.bool), True)
        torch.testing.assert_close(result, torch.tensor([[-4., -1.6, 0.]]), atol=1e-6, rtol=1e-6)

    def test_uniform_loss_ignores_padding_and_has_finite_gradients(self):
        # 错误地把 padding 放入分母会从 log(2) 变成约 100。
        logits = torch.tensor([[0., 0., 100.], [0., 100., 100.]], requires_grad=True)
        mask = torch.tensor([[True, False], [False, False]])
        loss = association.ranking_loss(logits, torch.tensor([1, 0]), mask)
        self.assertAlmostEqual(loss.item(), math.log(2) / 2, places=6)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertEqual(logits.grad[0, 2].item(), 0.)

    def test_invalid_target_and_nonfinite_logits_are_rejected(self):
        with self.assertRaises(ValueError):
            association.ranking_loss(torch.zeros(1, 2), torch.tensor([1]),
                                     torch.tensor([[False]]))
        with self.assertRaises(ValueError):
            association.ranking_loss(torch.tensor([[float('nan')]]),
                                     torch.tensor([0]), torch.empty(1, 0, dtype=torch.bool))

    def test_zero_initialization_matches_nearest_last_and_null_gate(self):
        model = association.ShortWindowAssociation(hist_len=3)
        history = torch.zeros(2, 3, 5)
        history[:, -1, 4] = 1
        candidates = torch.tensor([[[.2, 0., .99, 1.], [1.2, 0., .9, 0.]],
                                   [[1.1, 0., .99, 1.], [3., 0., .9, 0.]]])
        mask = torch.ones(2, 2, dtype=torch.bool)
        logits = model(history, candidates, mask)
        torch.testing.assert_close(logits, torch.tensor([[-4., -.8, -4.8],
                                                        [-4., -4.4, -12.]]))
        self.assertEqual(logits.argmax(-1).tolist(), [1, 0])

    def test_empty_candidates_have_zero_loss_and_finite_backward(self):
        model = association.ShortWindowAssociation(hist_len=3)
        history = torch.zeros(2, 3, 5)
        mask = torch.empty(2, 0, dtype=torch.bool)
        logits = model(history, torch.empty(2, 0, 4), mask)
        loss = association.ranking_loss(logits, torch.zeros(2, dtype=torch.long), mask)
        self.assertEqual(loss.item(), 0.)
        loss.backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all()
                            for p in model.parameters()))

    def test_candidate_permutation_and_padding_do_not_change_real_logits(self):
        torch.manual_seed(7)
        model = association.ShortWindowAssociation(hist_len=3)
        # 非零学习头也须保持置换对称性，不能只检查零初始化。
        with torch.no_grad():
            model.edge[-1].weight.normal_()
            model.reject[-1].weight.normal_()
        history = torch.randn(1, 3, 5)
        candidates = torch.tensor([[[.2, .1, .99, 1.], [.8, .2, .85, 0.],
                                    [100., 100., 1., 1.]]])
        mask = torch.tensor([[True, True, False]])
        first = model(history, candidates, mask)
        order = [1, 0, 2]
        second = model(history, candidates[:, order], mask[:, order])
        torch.testing.assert_close(first[:, [0, 2, 1]], second[:, :3])
        without_padding = model(history, candidates[:, :2], mask[:, :2])
        torch.testing.assert_close(first[:, :3], without_padding)

    def test_query_encoding_is_translation_invariant_and_uses_real_age(self):
        observations = [(1, np.array([0., 2.]), .98), (2, np.array([.1, 2.]), .99)]
        args = (5, np.array([[.2, 2.]]), np.array([.99]), np.array([True]))
        history, candidates = association.encode_query(observations, *args, hist_len=4)
        np.testing.assert_allclose(history[:, 4], [1, 1, 0, 0])
        np.testing.assert_allclose(history[:2, 3], [1., .75])
        np.testing.assert_allclose(history[:2, 0], [-.1, 0.], atol=1e-7)
        shift = np.array([4., -3.])
        moved = [(step, xy + shift, score) for step, xy, score in observations]
        h2, c2 = association.encode_query(moved, 5, args[1] + shift, args[2], args[3], hist_len=4)
        np.testing.assert_allclose(history, h2, atol=1e-7)
        np.testing.assert_allclose(candidates, c2, atol=1e-7)
        with self.assertRaises(ValueError):
            association.encode_query(observations, 2, *args[1:], hist_len=4)


class ShortWindowDataTests(unittest.TestCase):
    def test_real_cache_loading_parallel_order_and_bad_shape_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frames = []
            for step in range(1, 5):
                out = np.zeros((50, 4), dtype=np.float32)
                out[0, :2] = [step * .1, 2.]
                conf = np.tile(np.array([10., -10.], dtype=np.float32), (50, 1))
                conf[0] = [-10., 10.]
                label = np.zeros((10, 6), dtype=np.float32)
                label[:, :2] = out[0, :2]
                cache_path, label_path = root / f'{step}.npz', root / f'{step}.npy'
                np.savez(cache_path, out=out, conf=conf)
                np.save(label_path, label)
                frames.append((cache_path, label_path, step))
            records = [{'folder': 'first', 'frames': frames}, {'folder': 'second', 'frames': frames}]
            serial, manifest = data.collect_queries(records, hist_len=4, workers=0)
            parallel, second = data.collect_queries(records, hist_len=4, workers=2)
            self.assertEqual(manifest['query_sha256'], second['query_sha256'])
            for key in serial:
                np.testing.assert_array_equal(serial[key], parallel[key])
            self.assertEqual(serial['sample_ids'].tolist(), [3, 4, 3, 4])
            self.assertEqual(serial['sequence_indices'].tolist(), [0, 0, 1, 1])
            np.save(frames[0][1], np.zeros((6,), dtype=np.float32))
            with self.assertRaises(ValueError):
                data.load_association_frame(frames[0])

    def test_current_measurement_is_not_in_history_and_future_cannot_change_query(self):
        frames = [frame(1, [[0., 2.]], [[0., 2.]]),
                  frame(2, [[.1, 2.]], [[.1, 2.]]),
                  frame(3, [[.2, 2.], [1.2, 2.]], [[.2, 2.], [1.2, 2.]])]
        queries = data.build_sequence_queries(frames, hist_len=4)
        self.assertEqual(queries['labels'].tolist(), [1])
        self.assertEqual(queries['sample_ids'].tolist(), [3])
        np.testing.assert_allclose(queries['history'][0, :, 4], [0, 0, 1, 1])
        np.testing.assert_allclose(queries['history'][0, -2:, 0], [-.1, 0.], atol=1e-7)
        extended = data.build_sequence_queries(frames + [frame(4, [[9., 9.]])], hist_len=4)
        np.testing.assert_array_equal(queries['history'][0], extended['history'][0])

    def test_missing_current_positive_is_null_not_dropped(self):
        frames = [frame(1, [[0., 2.]], [[0., 2.]]), frame(2, [[.1, 2.]], [[.1, 2.]]),
                  frame(3, [[3., 2.]], [[.2, 2.]])]
        queries = data.build_sequence_queries(frames, hist_len=4)
        self.assertEqual(queries['labels'].tolist(), [0])
        self.assertEqual(queries['mask'].sum(), 1)
        frames[-1] = frame(3, [], [[.2, 2.]])
        queries = data.build_sequence_queries(frames, hist_len=4)
        self.assertEqual(queries['labels'].tolist(), [0])
        self.assertEqual(queries['mask'].sum(), 0)

    def test_sample_gaps_expire_history_and_sequence_calls_are_independent(self):
        frames = [frame(1, [[0., 2.]], [[0., 2.]]), frame(2, [[.1, 2.]], [[.1, 2.]]),
                  frame(9, [[.2, 2.]], [[.2, 2.]])]
        self.assertEqual(len(data.build_sequence_queries(frames, hist_len=4)['labels']), 0)
        self.assertEqual(len(data.build_sequence_queries([frames[-1]], hist_len=4)['labels']), 0)
        with self.assertRaises(ValueError):
            data.build_sequence_queries([frames[1], frames[0]], hist_len=4)

    def test_gt_slot_permutation_does_not_change_network_inputs_or_labels(self):
        frames = [frame(t, [[t * .1, 2.]], [[t * .1, 2.]]) for t in [1, 2, 3]]
        original = data.build_sequence_queries(frames, hist_len=4)
        swapped = [data.AssociationFrame(f.sample_id, f.positions, f.scores,
                                         f.targets[[1, 0, 2]]) for f in frames]
        other = data.build_sequence_queries(swapped, hist_len=4)
        for key in ['history', 'candidates', 'mask', 'labels']:
            np.testing.assert_array_equal(original[key], other[key])
        self.assertEqual(other['slots'].tolist(), [1])

    def test_competition_and_ambiguous_labels_are_retained(self):
        # 两个候选相距 .8m，不会被 .75m 聚类合并；到 GT 均为 .4m。
        frames = [frame(1, [[0., 2.]], [[0., 2.]]), frame(2, [[0., 2.]], [[0., 2.]]),
                  frame(3, [[-.4, 2.], [.4, 2.]], [[0., 2.]])]
        queries = data.build_sequence_queries(frames, hist_len=4)
        self.assertEqual(len(queries['labels']), 1)
        self.assertTrue(queries['competition'][0])
        self.assertTrue(queries['ambiguous'][0])


class ShortWindowTrainingTests(unittest.TestCase):
    def test_group_metrics_expose_null_and_competition_denominators(self):
        arrays = {'labels': np.array([1, 1, 0]), 'competition': np.array([True, False, False]),
                  'ambiguous': np.array([False, True, False]), 'sequence_indices': np.array([0, 0, 0])}
        report = training.summarize_predictions(arrays, np.array([1, 0, 1]), np.array([.1, .5, 1.]),
                                                candidate_predictions=np.array([1, 1, 1]))
        self.assertEqual(report['all']['count'], 3)
        self.assertEqual(report['all']['correct'], 1)
        self.assertEqual(report['positive']['accuracy'], .5)
        self.assertEqual(report['null']['accuracy'], 0.)
        self.assertEqual(report['competition']['count'], 1)
        self.assertEqual(report['competition']['accuracy'], 1.)
        self.assertEqual(report['ambiguous']['count'], 1)
        self.assertEqual(report['clean_positive']['count'], 1)
        self.assertEqual(report['positive']['ranking_positive_count'], 2)
        self.assertEqual(report['positive']['ranking_correct'], 2)
        self.assertEqual(report['positive']['ranking_accuracy'], 1.)
        self.assertIsNone(report['null']['ranking_accuracy'])

    def test_training_saves_readable_epoch_and_refuses_existing_output(self):
        frames = [frame(t, [[t * .1, 2.]], [[t * .1, 2.]]) for t in range(1, 7)]
        arrays = data.build_sequence_queries(frames, hist_len=4)
        arrays['sequence_indices'] = np.zeros(len(arrays['labels']), dtype=np.int64)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'run'
            report = training.run_training(arrays, arrays, {'purpose': 'synthetic smoke'}, output,
                                           hist_len=4, epochs=2, batch_size=2, device='cpu', progress=False)
            self.assertEqual(len(report['history']), 2)
            saved = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(saved['history'][0]['epoch'], 1)
            checkpoint = torch.load(output / 'epoch_2.pt', map_location='cpu', weights_only=True)
            restored = association.ShortWindowAssociation(hist_len=4)
            restored.load_state_dict(checkpoint['model_state_dict'])
            with self.assertRaises(FileExistsError):
                training.run_training(arrays, arrays, {}, output, hist_len=4,
                                      epochs=1, device='cpu', progress=False)


if __name__ == '__main__':
    unittest.main()
