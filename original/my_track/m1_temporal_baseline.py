"""Small, training-free temporal tracker for frozen M1 candidate clusters."""
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from .m1_frame_baseline import cluster_candidates


class TemporalFrameOutput(NamedTuple):
    positions: np.ndarray
    scores: np.ndarray
    track_ids: np.ndarray
    observed: np.ndarray


@dataclass
class _Track:
    track_id: int
    position: np.ndarray
    velocity: np.ndarray
    score: float
    missed: int = 0


class TemporalCandidateTracker:
    """Associate clustered M1 candidates and briefly propagate missed tracks.

    Each instance represents one sequence. Call ``reset`` or construct a new
    instance at sequence boundaries; IDs are local to that instance.
    """

    def __init__(
        self,
        score_threshold: float = 0.8,
        cluster_radius_m: float = 0.75,
        association_radius_m: float = 1.0,
        max_missed: int = 1,
        max_tracks: int = 3,
        miss_score_decay: float = 0.8,
    ):
        if max_missed < 0 or max_tracks < 1:
            raise ValueError("max_missed must be non-negative and max_tracks positive")
        if not np.isfinite(association_radius_m) or association_radius_m <= 0:
            raise ValueError("association_radius_m must be finite and positive")
        if not np.isfinite(miss_score_decay) or not 0 <= miss_score_decay <= 1:
            raise ValueError("miss_score_decay must be in [0,1]")
        self.score_threshold = float(score_threshold)
        self.cluster_radius_m = float(cluster_radius_m)
        self.association_radius_m = float(association_radius_m)
        self.max_missed = int(max_missed)
        self.max_tracks = int(max_tracks)
        self.miss_score_decay = float(miss_score_decay)
        self.reset()

    def reset(self):
        self._tracks = []
        self._next_track_id = 0

    def update(self, candidate_positions, candidate_scores) -> TemporalFrameOutput:
        """Consume one frame and return current tracks, including short gaps."""
        candidates = cluster_candidates(
            candidate_positions,
            candidate_scores,
            self.score_threshold,
            self.cluster_radius_m,
        )
        old_tracks = self._tracks
        matched_tracks = set()
        matched_candidates = set()

        if old_tracks and len(candidates.positions):
            predictions = np.stack([
                track.position + track.velocity * (track.missed + 1)
                for track in old_tracks
            ])
            distances = np.linalg.norm(
                predictions[:, None, :] - candidates.positions[None, :, :], axis=-1
            )
            track_count, candidate_count = distances.shape
            unmatched_cost = self.association_radius_m * 0.5
            invalid_cost = self.association_radius_m * (track_count + candidate_count + 2)
            costs = np.full(
                (track_count + candidate_count, track_count + candidate_count),
                invalid_cost,
                dtype=np.float64,
            )
            costs[:track_count, :candidate_count] = np.where(
                distances <= self.association_radius_m, distances, invalid_cost
            )
            costs[:track_count, candidate_count:] = unmatched_cost
            costs[track_count:, :candidate_count] = unmatched_cost
            costs[track_count:, candidate_count:] = 0.0
            rows, columns = linear_sum_assignment(costs)
            for row, column in zip(rows, columns):
                if row >= track_count or column >= candidate_count:
                    continue
                if distances[row, column] > self.association_radius_m:
                    continue
                track = old_tracks[row]
                gap = track.missed + 1
                previous_position = track.position.copy()
                track.position = candidates.positions[column].copy()
                track.velocity = (track.position - previous_position) / gap
                track.score = float(candidates.scores[column])
                track.missed = 0
                matched_tracks.add(row)
                matched_candidates.add(column)

        retained = []
        for index, track in enumerate(old_tracks):
            if index not in matched_tracks:
                track.missed += 1
            if track.missed <= self.max_missed:
                retained.append(track)

        for index, (position, score) in enumerate(zip(candidates.positions, candidates.scores)):
            if index in matched_candidates or len(retained) >= self.max_tracks:
                continue
            retained.append(_Track(
                track_id=self._next_track_id,
                position=position.copy(),
                velocity=np.zeros(2, dtype=np.float64),
                score=float(score),
            ))
            self._next_track_id += 1

        self._tracks = retained
        ordered = sorted(retained, key=lambda track: track.track_id)
        output_positions = np.asarray([
            track.position + track.velocity * track.missed for track in ordered
        ], dtype=np.float64).reshape(-1, 2)
        return TemporalFrameOutput(
            positions=output_positions,
            scores=np.asarray([
                track.score * self.miss_score_decay ** track.missed for track in ordered
            ], dtype=np.float64),
            track_ids=np.asarray([track.track_id for track in ordered], dtype=np.int64),
            observed=np.asarray([track.missed == 0 for track in ordered], dtype=bool),
        )
