"""Opt-in event-level association hysteresis for the frozen-M1 tracker.

The base :class:`LifecycleTracker` remains unchanged.  This module adds a
small, explicitly configured cost only on ambiguous spatially valid edges:
when a confirmed track was observed on the previous input frame, candidates
that are outside its short hold gate receive a non-negative penalty.  The
global gated assignment is still responsible for the final one-to-one match.

Long-gap identity recovery is intentionally kept in ``identity_recovery`` so
that continuous-frame hysteresis and re-capture can be evaluated separately.
"""

import numpy as np

from .lifecycle_tracker import LifecycleTracker


class HysteresisLifecycleTracker(LifecycleTracker):
    """Lifecycle tracker with an opt-in short-term ambiguity penalty.

    ``hold_gate_m`` describes how far a candidate may move from the previous
    observed position before the hold penalty applies.  The penalty is used
    only for confirmed tracks that were observed in the immediately preceding
    input frame and only on edges belonging to an ambiguous valid-edge
    component.  It is therefore inactive for ordinary one-to-one matches and
    cannot make an invalid edge valid.
    """

    def __init__(self, *args, hold_gate_m=.5, hold_penalty=.1, **kwargs):
        if not np.isfinite(hold_gate_m) or hold_gate_m <= 0:
            raise ValueError('hold_gate_m must be finite and positive')
        if not np.isfinite(hold_penalty) or hold_penalty < 0:
            raise ValueError('hold_penalty must be finite and non-negative')
        super().__init__(*args, **kwargs)
        if hold_gate_m > self.association_radius_m:
            raise ValueError('hold_gate_m must not exceed association_radius_m')
        self.hold_gate_m = float(hold_gate_m)
        self.hold_penalty = float(hold_penalty)
        self.hysteresis_statistics = {'frames': 0, 'penalized_edges': 0}

    def reset(self):
        super().reset()
        self.hysteresis_statistics = {'frames': 0, 'penalized_edges': 0}

    def _association_pairwise_penalty(self, indices, proposal_features, gate_m):
        del proposal_features
        if self.hold_penalty == 0 or not indices:
            return None
        points = getattr(self, '_current_proposal_positions', None)
        if points is None or len(points) == 0:
            return None

        predicted = self._association_predictions(indices)
        distances = np.linalg.norm(predicted[:, None] - points[None], axis=-1)
        valid = distances <= float(gate_m)
        ambiguous = valid & (
            (valid.sum(axis=1, keepdims=True) > 1) |
            (valid.sum(axis=0, keepdims=True) > 1)
        )
        previous = np.stack([self._tracks[index].position for index in indices])
        previous_distances = np.linalg.norm(previous[:, None] - points[None], axis=-1)
        recent_confirmed = np.asarray([
            self._tracks[index].confirmed and self._tracks[index].missed == 0
            for index in indices
        ], dtype=bool)
        penalized = ambiguous & recent_confirmed[:, None] & (
            previous_distances > self.hold_gate_m
        )
        self.hysteresis_statistics['frames'] += int(np.any(penalized))
        self.hysteresis_statistics['penalized_edges'] += int(penalized.sum())
        return np.where(penalized, self.hold_penalty, 0.).astype(np.float64)
