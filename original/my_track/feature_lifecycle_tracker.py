"""Opt-in lifecycle tracker that uses recent frozen-M1 features on ambiguous pairs."""

from numbers import Integral

import numpy as np

from .lifecycle_tracker import LifecycleTracker
from .m1_frame_baseline import cluster_candidates


class FeatureLifecycleTracker(LifecycleTracker):
    """生命周期规则不变，仅在空间门内歧义关联时使用短时外观原型。"""

    def __init__(self, *args, feature_dim=128, feature_weight=.35,
                 feature_momentum=.7, max_feature_age=8, **kwargs):
        if isinstance(feature_dim, bool) or not isinstance(feature_dim, Integral) or feature_dim < 1:
            raise ValueError('feature_dim must be a positive integer')
        if isinstance(max_feature_age, bool) or not isinstance(max_feature_age, Integral) or max_feature_age < 0:
            raise ValueError('max_feature_age must be a non-negative integer')
        if not np.isfinite(feature_weight) or feature_weight < 0:
            raise ValueError('feature_weight must be finite and non-negative')
        if not np.isfinite(feature_momentum) or not 0 <= feature_momentum < 1:
            raise ValueError('feature_momentum must be finite and in [0,1)')
        self.feature_dim = int(feature_dim)
        self.feature_weight = float(feature_weight)
        self.feature_momentum = float(feature_momentum)
        self.max_feature_age = int(max_feature_age)
        super().__init__(*args, **kwargs)

    def reset(self):
        super().reset()
        self.feature_statistics = {'ambiguous_frames': 0, 'comparable_pairs': 0}

    @staticmethod
    def _normalize(vector):
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 0 else np.zeros_like(vector)

    def _proposal_features(self, positions, scores, features, primary, weak):
        if features is None:
            return None, None
        positions = np.asarray(positions, dtype=np.float64)
        scores = np.asarray(scores, dtype=np.float64)
        features = np.asarray(features, dtype=np.float64)
        if (features.shape != (len(positions), self.feature_dim) or
                not np.isfinite(features).all()):
            raise ValueError(f'candidate_features must be finite [N,{self.feature_dim}]')

        order = np.lexsort((positions[:, 1], positions[:, 0], -scores))
        sorted_positions, sorted_scores, sorted_features = positions[order], scores[order], features[order]
        primary_clusters, primary_members = cluster_candidates(
            sorted_positions, sorted_scores, self.birth_threshold, self.cluster_radius_m,
            return_membership=True,
        )
        weak_mask = (sorted_scores >= self.score_threshold) & (sorted_scores < self.birth_threshold)
        if len(primary_clusters.positions):
            distances = np.linalg.norm(sorted_positions[:, None] - primary_clusters.positions[None], axis=-1)
            weak_mask &= (distances > self.cluster_radius_m).all(axis=1)
        weak_positions, weak_scores = sorted_positions[weak_mask], sorted_scores[weak_mask]
        weak_clusters, weak_local_members = cluster_candidates(
            weak_positions, weak_scores, self.score_threshold, self.cluster_radius_m,
            return_membership=True,
        )
        weak_global_indices = np.flatnonzero(weak_mask)
        weak_members = [weak_global_indices[members] for members in weak_local_members]

        def prototypes(memberships):
            result = []
            for members in memberships:
                member_scores = sorted_scores[members]
                if member_scores.sum() > 0:
                    vector = np.average(sorted_features[members], axis=0, weights=member_scores)
                else:
                    vector = sorted_features[members].mean(axis=0)
                result.append(self._normalize(vector))
            return np.asarray(result, dtype=np.float64).reshape(-1, self.feature_dim)

        # _proposals and this method use the same deterministic sort and exact cluster membership.
        if (len(primary_clusters.positions) != len(primary.positions) or
                len(weak_clusters.positions) != len(weak.positions)):
            raise RuntimeError('feature proposals diverged from lifecycle proposal clustering')
        return prototypes(primary_members), prototypes(weak_members)

    def _update_appearance(self, track, feature):
        if feature is None:
            return
        feature = self._normalize(np.asarray(feature, dtype=np.float64))
        if np.linalg.norm(feature) == 0:
            return
        if track.appearance is None:
            track.appearance = feature.copy()
        else:
            blended = (self.feature_momentum * track.appearance +
                       (1 - self.feature_momentum) * feature)
            track.appearance = self._normalize(blended)
        track.last_feature_step = self._frame_index

    def _association_pairwise_penalty(self, indices, proposal_features, gate_m):
        if proposal_features is None or not len(indices) or not len(proposal_features):
            return None
        predicted = self._association_predictions(indices)
        points = self._current_proposal_positions
        distances = np.linalg.norm(predicted[:, None] - points[None], axis=-1)
        valid = distances <= gate_m
        ambiguous = valid & ((valid.sum(axis=1, keepdims=True) > 1) |
                             (valid.sum(axis=0, keepdims=True) > 1))
        track_objects = [self._tracks[index] for index in indices]
        track_features = np.stack([
            np.zeros(self.feature_dim) if track.appearance is None else track.appearance
            for track in track_objects
        ])
        recent = np.asarray([
            track.appearance is not None and self._frame_index - track.last_feature_step <= self.max_feature_age
            for track in track_objects
        ], dtype=bool)
        dots = np.clip(track_features @ proposal_features.T, -1., 1.)
        track_valid = np.linalg.norm(track_features, axis=1) > 0
        candidate_valid = np.linalg.norm(proposal_features, axis=1) > 0
        usable = ambiguous & recent[:, None] & track_valid[:, None] & candidate_valid[None, :]
        if usable.any():
            self.feature_statistics['ambiguous_frames'] += 1
            self.feature_statistics['comparable_pairs'] += int(usable.sum())
        return np.where(usable, self.feature_weight * (1 - dots), 0.)
