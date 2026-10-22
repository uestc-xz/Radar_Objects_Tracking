import unittest

import torch

from original.my_track.track_state import TrackBank, batch_track_states


def update_bank(bank, observed=True, existence=0.9, confidence=0.8, x=1.0):
    final_pos = torch.tensor([[x, 2.0], [0.0, 0.0], [0.0, 0.0]])
    fused_feat = torch.ones(3, 128)
    observed_mask = torch.tensor([observed, False, False])
    existence_prob = torch.tensor([existence, 0.0, 0.0])
    track_confidence = torch.tensor([confidence, 0.0, 0.0])
    bank.update(final_pos, fused_feat, observed_mask, existence_prob, track_confidence, dt=0.24)


class TrackBankTests(unittest.TestCase):
    def test_confirmation_waits_for_hist_len_and_existence_mean(self):
        bank = TrackBank(hist_len=4, feat_dim=128, confirm_threshold=0.5)
        for frame in range(3):
            update_bank(bank, existence=0.9, x=float(frame + 1))
            self.assertFalse(bank.confirmed[0])
        update_bank(bank, existence=0.9, x=4.0)
        self.assertTrue(bank.initialized[0])
        self.assertTrue(bank.confirmed[0])
        self.assertEqual(bank.age[0].item(), 4)

    def test_low_presence_probability_does_not_confirm_and_later_history_can(self):
        bank = TrackBank(hist_len=3, feat_dim=128, confirm_threshold=0.6)
        for _ in range(3):
            update_bank(bank, existence=0.2)
        self.assertFalse(bank.confirmed[0])
        update_bank(bank, existence=0.9)
        self.assertFalse(bank.confirmed[0])
        update_bank(bank, existence=0.9)
        self.assertTrue(bank.confirmed[0])

    def test_misses_keep_confirmed_id_write_prediction_and_decay_confidence(self):
        bank = TrackBank(hist_len=2, feat_dim=128, confirm_threshold=0.5, miss_decay=0.95)
        update_bank(bank, existence=0.9, confidence=0.8, x=1.0)
        update_bank(bank, existence=0.9, confidence=0.8, x=2.0)
        before_hist = bank.hist[0].clone()
        before_confidence = bank.confidence[0].item()

        update_bank(bank, observed=False, existence=0.9, confidence=0.99, x=2.24)

        self.assertTrue(bank.confirmed[0])
        self.assertEqual(bank.miss[0].item(), 1)
        self.assertLess(bank.confidence[0].item(), before_confidence)
        self.assertFalse(torch.equal(bank.hist[0], before_hist))
        torch.testing.assert_close(bank.hist[0, -1, :2], torch.tensor([2.24, 2.0]))

    def test_reset_clears_sequence_state_and_batch_helper_stacks_banks(self):
        bank_a = TrackBank(hist_len=3, feat_dim=128)
        bank_b = TrackBank(hist_len=3, feat_dim=128)
        update_bank(bank_a)
        states = batch_track_states([bank_a, bank_b], device="cpu")

        self.assertEqual(states["hist"].shape, (2, 3, 3, 4))
        self.assertEqual(states["confirmed"].shape, (2, 3))
        bank_a.reset()
        self.assertFalse(bank_a.initialized.any())
        self.assertFalse(bank_a.confirmed.any())
        self.assertEqual(bank_a.frames_seen, 0)


if __name__ == "__main__":
    unittest.main()
