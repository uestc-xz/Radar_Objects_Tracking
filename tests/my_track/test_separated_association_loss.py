"""分项关联损失的手算、梯度隔离和训练入口行为。"""

import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from original.my_track.short_window_association import ShortWindowAssociation
from original.my_track import separated_association_loss as separated
from scripts import train_m1_short_window_association as training


class SeparatedLossTests(unittest.TestCase):
    def test_two_uniform_candidates_have_two_log_two_total(self):
        logits = torch.zeros(1, 4, requires_grad=True)
        mask = torch.tensor([[True, True, False]])
        result = separated.separated_losses(logits, torch.tensor([1]), mask, torch.tensor([True]))
        self.assertAlmostEqual(result['ranking'].item(), math.log(2), places=6)
        self.assertAlmostEqual(result['matchability'].item(), math.log(2), places=6)
        self.assertAlmostEqual(result['total'].item(), 2 * math.log(2), places=6)
        result['total'].backward()
        self.assertEqual(logits.grad[0, 3].item(), 0.)

    def test_reject_gradient_does_not_change_candidate_scores(self):
        logits = torch.tensor([[0., 1., .5]], requires_grad=True)
        parts = separated.separated_losses(logits, torch.tensor([0]),
                                            torch.ones(1, 2, dtype=torch.bool), torch.tensor([False]))
        parts['matchability'].backward()
        torch.testing.assert_close(logits.grad[:, 1:], torch.zeros(1, 2))
        self.assertLess(logits.grad[0, 0].item(), 0.)

    def test_rank_gradient_has_zero_sum_and_does_not_change_null(self):
        logits = torch.tensor([[0., 1., .5]], requires_grad=True)
        parts = separated.separated_losses(logits, torch.tensor([1]),
                                            torch.ones(1, 2, dtype=torch.bool), torch.tensor([False]))
        parts['ranking'].backward()
        self.assertEqual(logits.grad[0, 0].item(), 0.)
        self.assertAlmostEqual(logits.grad[0, 1:].sum().item(), 0., places=6)

    def test_group_balance_is_invariant_to_repeating_easy_group(self):
        # competition: log(4)，other: log(2)；两组等权为 1.5log(2)。
        logits = torch.zeros(5, 5, requires_grad=True)
        mask = torch.tensor([[True] * 4] + [[True, True, False, False]] * 4)
        labels = torch.ones(5, dtype=torch.long)
        competition = torch.tensor([True, False, False, False, False])
        parts = separated.separated_losses(logits, labels, mask, competition, balanced=True)
        self.assertAlmostEqual(parts['ranking'].item(), 1.5 * math.log(2), places=6)
        self.assertEqual(parts['competition_count'], 1)
        self.assertEqual(parts['other_count'], 4)
        unbalanced = separated.separated_losses(logits, labels, mask, competition)
        self.assertAlmostEqual(unbalanced['ranking'].item(), 1.2 * math.log(2), places=6)

    def test_single_present_group_keeps_its_full_loss(self):
        for competition in [False, True]:
            parts = separated.separated_losses(torch.zeros(1, 3), torch.tensor([1]),
                                                torch.ones(1, 2, dtype=torch.bool),
                                                torch.tensor([competition]), balanced=True)
            self.assertAlmostEqual(parts['ranking'].item(), math.log(2), places=6)

    def test_no_candidates_is_zero_loss_including_finite_backward(self):
        for width in [0, 3]:
            logits = torch.zeros(2, width + 1, requires_grad=True)
            # 已由模型 mask 的极小有限 padding 不能 sum 后溢出为 NaN。
            with torch.no_grad():
                logits[:, 1:] = torch.finfo(torch.float32).min
            parts = separated.separated_losses(logits, torch.zeros(2, dtype=torch.long),
                                                torch.zeros(2, width, dtype=torch.bool),
                                                torch.zeros(2, dtype=torch.bool), balanced=True)
            self.assertEqual(parts['total'].item(), 0.)
            self.assertEqual(parts['matchability_count'], 0)
            parts['total'].backward()
            self.assertTrue(torch.isfinite(logits.grad).all())

    def test_null_only_and_singleton_positive_have_no_ranking_gradient(self):
        for label in [0, 1]:
            logits = torch.zeros(1, 2, requires_grad=True)
            parts = separated.separated_losses(logits, torch.tensor([label]),
                                                torch.ones(1, 1, dtype=torch.bool), torch.tensor([False]))
            self.assertEqual(parts['ranking'].item(), 0.)
            parts['total'].backward()
            self.assertEqual(logits.grad[0, 1].item(), 0.)

    def test_nonfinite_invalid_target_and_bad_competition_are_rejected(self):
        for logits, labels, mask, competition in [
            (torch.tensor([[0., float('nan')]]), torch.tensor([1]), torch.tensor([[True]]), torch.tensor([False])),
            (torch.zeros(1, 2), torch.tensor([1]), torch.tensor([[False]]), torch.tensor([False])),
            (torch.zeros(1, 2), torch.tensor([0]), torch.tensor([[True]]), torch.tensor([True])),
        ]:
            with self.assertRaises(ValueError):
                separated.separated_losses(logits, labels, mask, competition)


class ParameterRoutingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = ShortWindowAssociation(hist_len=4)
        with torch.no_grad():
            self.model.edge[-1].weight.normal_(std=.1)
            self.model.reject[-1].weight.normal_(std=.1)
        self.history = torch.randn(2, 4, 5)
        self.candidates = torch.tensor([[[.2, 0., .99, 1.], [-.4, .1, .9, 0.]],
                                        [[.5, .1, .99, 1.], [1.4, .2, .8, 0.]]])
        self.mask = torch.ones(2, 2, dtype=torch.bool)

    def test_detach_switch_preserves_forward_values_and_checkpoint_keys(self):
        original = self.model(self.history, self.candidates, self.mask)
        detached = self.model(self.history, self.candidates, self.mask, detach_rejection_history=True)
        torch.testing.assert_close(original, detached, rtol=0, atol=0)
        restored = ShortWindowAssociation(hist_len=4)
        restored.load_state_dict(self.model.state_dict(), strict=True)
        torch.testing.assert_close(original, restored(self.history, self.candidates, self.mask), rtol=0, atol=0)

    def test_matchability_updates_only_reject_parameters(self):
        output = self.model(self.history, self.candidates, self.mask, detach_rejection_history=True)
        parts = separated.separated_losses(output, torch.tensor([1, 0]), self.mask,
                                            torch.tensor([True, False]))
        parts['matchability'].backward()
        for module in [self.model.history_encoder, self.model.edge]:
            self.assertTrue(all(p.grad is None or (p.grad == 0).all() for p in module.parameters()))
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in self.model.reject.parameters()
                               if p.grad is not None), 0.)

    def test_ranking_updates_history_and_edge_but_not_reject(self):
        output = self.model(self.history, self.candidates, self.mask, detach_rejection_history=True)
        parts = separated.separated_losses(output, torch.tensor([2, 0]), self.mask,
                                            torch.tensor([True, False]))
        parts['ranking'].backward()
        for module in [self.model.history_encoder, self.model.edge]:
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in module.parameters()
                                   if p.grad is not None), 0.)
        self.assertTrue(all(p.grad is None or (p.grad == 0).all() for p in self.model.reject.parameters()))


class SeparatedTrainingTests(unittest.TestCase):
    def test_validation_group_means_do_not_depend_on_batch_size(self):
        arrays = {'history': np.zeros((5, 4, 5), dtype=np.float32),
                  'candidates': np.zeros((5, 4, 4), dtype=np.float32),
                  'mask': np.array([[True] * 4] + [[True, True, False, False]] * 4),
                  'labels': np.ones(5, dtype=np.int64),
                  'competition': np.array([True, False, False, False, False]),
                  'ambiguous': np.zeros(5, dtype=bool), 'sequence_indices': np.zeros(5, dtype=np.int64)}
        model = ShortWindowAssociation(hist_len=4)
        for batch_size in [1, 5]:
            report = training.evaluate(model, arrays, 'cpu', batch_size=batch_size,
                                       objective='separated_balanced')
            components = report['model']['objective_components']
            self.assertEqual(components['competition_count'], 1)
            self.assertEqual(components['other_count'], 4)
            self.assertAlmostEqual(components['ranking'], 1.5 * math.log(2), places=6)

    def test_explicit_objectives_save_components_and_preserve_common_joint_metric(self):
        arrays = {'history': np.zeros((3, 4, 5), dtype=np.float32),
                  'candidates': np.array([[[.2, 0., .99, 1.], [-.4, 0., .9, 0.]]] * 3, dtype=np.float32),
                  'mask': np.ones((3, 2), dtype=bool), 'labels': np.array([1, 2, 0], dtype=np.int64),
                  'competition': np.array([True, True, False]), 'ambiguous': np.zeros(3, dtype=bool),
                  'sequence_indices': np.zeros(3, dtype=np.int64)}
        with tempfile.TemporaryDirectory() as temporary:
            for objective in ['separated', 'separated_balanced']:
                output = Path(temporary) / objective
                report = training.run_training(arrays, arrays, {}, output, hist_len=4, epochs=1,
                                               batch_size=3, objective=objective, progress=False)
                self.assertEqual(report['settings']['objective'], objective)
                components = report['history'][0]['validation']['objective_components']
                self.assertEqual(components['positive_count'], 2)
                self.assertEqual(components['competition_count'], 2)
                self.assertEqual(components['matchability_count'], 3)
                checkpoint = torch.load(output / 'epoch_1.pt', weights_only=True)
                self.assertEqual(checkpoint['objective'], objective)
                model = ShortWindowAssociation(hist_len=4)
                model.load_state_dict(checkpoint['model_state_dict'])
                logits = model(*(torch.from_numpy(arrays[k]) for k in ['history', 'candidates', 'mask']))
                joint = F.cross_entropy(logits, torch.from_numpy(arrays['labels']))
                self.assertAlmostEqual(report['history'][0]['validation']['all']['nll'], joint.item(), places=6)
                saved = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
                self.assertEqual(len(saved['history']), 1)


if __name__ == '__main__':
    unittest.main()
