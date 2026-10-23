import unittest

import numpy as np

from original.my_track.identity_recovery import TrackletIdentityRecovery
from original.my_track.m1_temporal_baseline import TemporalFrameOutput


def frame(points=(), ids=(), observed=None):
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    return TemporalFrameOutput(points, np.ones(len(points)), np.asarray(ids, dtype=np.int64),
                               np.ones(len(points), dtype=bool) if observed is None
                               else np.asarray(observed, dtype=bool))


class IdentityRecoveryTests(unittest.TestCase):
    def test_confirmed_new_fragment_recovers_absent_identity_without_changing_evidence(self):
        recovery = TrackletIdentityRecovery()
        self.assertEqual(recovery.remap(frame([[0, 2]], [10])).track_ids.tolist(), [0])
        recovery.remap(frame())
        source = frame([[.2, 2]], [20])
        result = recovery.remap(source)
        self.assertEqual(result.track_ids.tolist(), [0])
        for index in (0, 1, 3):
            np.testing.assert_array_equal(result[index], source[index])
        self.assertEqual(recovery.statistics['recovered'], 1)

    def test_original_fragment_returns_with_same_identity_without_recovery(self):
        recovery = TrackletIdentityRecovery()
        recovery.remap(frame([[0, 2]], [10]))
        recovery.remap(frame())
        result = recovery.remap(frame([[3, 2]], [10]))
        self.assertEqual(result.track_ids.tolist(), [0])
        self.assertEqual(recovery.statistics['recovered'], 0)

    def test_reachable_gate_grows_with_gap_but_local_gate_does_not(self):
        for growth, expected in ((0, 1), (.02, 0)):
            recovery = TrackletIdentityRecovery(gate_growth_m=growth)
            recovery.remap(frame([[0, 2]], [10]))
            for _ in range(100):
                recovery.remap(frame())
            self.assertEqual(recovery.remap(frame([[2, 2]], [20])).track_ids.tolist(), [expected])

    def test_ambiguous_old_identities_are_not_forced_onto_new_fragment(self):
        recovery = TrackletIdentityRecovery()
        recovery.remap(frame([[-.4, 2], [.4, 2]], [10, 11]))
        recovery.remap(frame())
        self.assertEqual(recovery.remap(frame([[0, 2]], [20])).track_ids.tolist(), [2])

    def test_two_new_fragments_competing_for_one_identity_are_rejected(self):
        recovery = TrackletIdentityRecovery()
        recovery.remap(frame([[0, 2]], [10]))
        recovery.remap(frame())
        result = recovery.remap(frame([[-.1, 2], [.1, 2]], [20, 21]))
        self.assertEqual(result.track_ids.tolist(), [1, 2])

    def test_active_identity_cannot_be_given_to_another_fragment(self):
        recovery = TrackletIdentityRecovery()
        recovery.remap(frame([[0, 2]], [10]))
        result = recovery.remap(frame([[0, 2], [.1, 2]], [10, 20]))
        self.assertEqual(result.track_ids.tolist(), [0, 1])

    def test_return_after_ownership_transfer_cannot_duplicate_an_identity(self):
        recovery = TrackletIdentityRecovery()
        recovery.remap(frame([[0, 2]], [10]))
        recovery.remap(frame([[.1, 2]], [20]))
        result = recovery.remap(frame([[.2, 2], [0, 2]], [20, 10]))
        self.assertEqual(result.track_ids.tolist(), [0, 1])

    def test_coasting_does_not_move_the_remembered_observation_endpoint(self):
        recovery = TrackletIdentityRecovery()
        recovery.remap(frame([[0, 2]], [10]))
        recovery.remap(frame([[2, 2]], [10], [False]))
        self.assertEqual(recovery.remap(frame([[.1, 2]], [20])).track_ids.tolist(), [0])

    def test_expired_memory_cannot_be_recovered_by_a_new_fragment(self):
        recovery = TrackletIdentityRecovery(memory_steps=2)
        recovery.remap(frame([[0, 2]], [10]))
        recovery.remap(frame())
        recovery.remap(frame())
        self.assertEqual(recovery.remap(frame([[0, 2]], [20])).track_ids.tolist(), [1])

    def test_known_single_missing_identity_can_return_at_a_distant_location(self):
        recovery = TrackletIdentityRecovery(known_count=2)
        recovery.remap(frame([[0, 2], [2, 2]], [10, 11]))
        recovery.remap(frame([[0, 2]], [10]))
        self.assertEqual(recovery.remap(frame([[0, 2], [-2, 5]], [10, 20])).track_ids.tolist(), [0, 1])

    def test_incomplete_known_identity_set_does_not_force_a_distant_recovery(self):
        recovery = TrackletIdentityRecovery(known_count=2)
        recovery.remap(frame([[0, 2]], [10]))
        recovery.remap(frame())
        self.assertEqual(recovery.remap(frame([[3, 2]], [20])).track_ids.tolist(), [1])

    def test_reset_removes_identity_memory(self):
        recovery = TrackletIdentityRecovery()
        recovery.remap(frame([[0, 2]], [10]))
        recovery.reset()
        self.assertEqual(recovery.remap(frame([[2, 3]], [30])).track_ids.tolist(), [0])
        self.assertEqual(recovery.statistics['recovered'], 0)

    def test_invalid_input_does_not_expire_memory(self):
        recovery = TrackletIdentityRecovery(memory_steps=1)
        recovery.remap(frame([[0, 2]], [10]))
        with self.assertRaises(ValueError):
            recovery.remap(frame([[np.nan, 2]], [20]))
        self.assertEqual(recovery.remap(frame([[0, 2]], [20])).track_ids.tolist(), [0])

    def test_invalid_parameters_are_rejected(self):
        for kwargs in ({'memory_steps': 0}, {'gate_growth_m': -1}, {'gate_m': np.nan},
                       {'ambiguity_margin_m': -1}, {'known_count': 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                TrackletIdentityRecovery(**kwargs)


if __name__ == '__main__':
    unittest.main()
