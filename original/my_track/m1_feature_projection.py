"""Independent projection and sample construction for frozen-M1 identity features."""

from numbers import Integral

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .gated_matching import gated_pairs


class FeatureProjection(nn.Module):
    """Map frozen M1 embeddings to a compact, L2-normalized identity space."""

    def __init__(self, input_dim=128, hidden_dim=64, output_dim=32):
        super().__init__()
        for name, value in (('input_dim', input_dim), ('hidden_dim', hidden_dim),
                            ('output_dim', output_dim)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.layers = nn.Sequential(
            nn.Linear(self.input_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), self.output_dim),
        )

    def forward(self, features):
        if (not torch.is_tensor(features) or features.ndim < 2 or
                features.shape[-1] != self.input_dim):
            raise ValueError(f'features must be a tensor [...,{self.input_dim}]')
        if not torch.isfinite(features).all():
            raise ValueError('features must be finite')
        return F.normalize(self.layers(features), p=2, dim=-1, eps=1e-8)


def match_candidate_features(positions, scores, features, targets, gate_m=.5,
                            score_threshold=.8):
    """Assign confident M1 candidates to GT slots for offline training labels only."""
    positions = np.asarray(positions, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    features = np.asarray(features, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float64)
    if (positions.ndim != 2 or positions.shape[1] != 2 or
            scores.shape != (len(positions),) or features.ndim != 2 or
            features.shape[0] != len(positions) or targets.shape != (3, 2)):
        raise ValueError('positions/scores/features/targets have incompatible shapes')
    if (not all(np.isfinite(value).all() for value in (positions, scores, features, targets)) or
            ((scores < 0) | (scores > 1)).any() or not np.isfinite(gate_m) or gate_m <= 0 or
            not np.isfinite(score_threshold) or not 0 <= score_threshold <= 1):
        raise ValueError('inputs must be finite and thresholds must be in range')
    valid_slots = np.flatnonzero(np.abs(targets).sum(axis=1) > 0)
    selected = np.flatnonzero(scores >= score_threshold)
    if not len(selected) or not len(valid_slots):
        return np.empty((0, features.shape[1]), dtype=np.float32), np.empty(0, dtype=np.int64)
    pairs = gated_pairs(positions[selected], targets[valid_slots], gate_m)
    if not pairs:
        return np.empty((0, features.shape[1]), dtype=np.float32), np.empty(0, dtype=np.int64)
    candidate_indices = np.asarray([selected[candidate] for candidate, _ in pairs], dtype=np.int64)
    identities = np.asarray([valid_slots[target] for _, target in pairs], dtype=np.int64)
    return features[candidate_indices].copy(), identities


def build_identity_triplets(identities, frame_indices, max_positive_gap=8,
                            max_negative_gap=1):
    """Build deterministic anchor/positive/negative rows within one recording."""
    identities = np.asarray(identities)
    frame_indices = np.asarray(frame_indices)
    if (identities.ndim != 1 or frame_indices.shape != identities.shape or
            not np.issubdtype(identities.dtype, np.integer) or
            not np.issubdtype(frame_indices.dtype, np.integer)):
        raise ValueError('identities and frame_indices must be aligned integer vectors')
    for name, value in (('max_positive_gap', max_positive_gap),
                        ('max_negative_gap', max_negative_gap)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
            raise ValueError(f'{name} must be a non-negative integer')

    triplets = []
    for anchor in range(len(identities)):
        gaps = np.abs(frame_indices - frame_indices[anchor])
        positive_candidates = np.flatnonzero(
            (identities == identities[anchor]) & (gaps > 0) & (gaps <= max_positive_gap)
        )
        negative_candidates = np.flatnonzero(
            (identities != identities[anchor]) & (gaps <= max_negative_gap)
        )
        if not len(positive_candidates) or not len(negative_candidates):
            continue
        positive = int(positive_candidates[np.argmin(gaps[positive_candidates])])
        negative = int(negative_candidates[np.argmin(gaps[negative_candidates])])
        triplets.append((anchor, positive, negative))
    return np.asarray(triplets, dtype=np.int64).reshape(-1, 3)


def triplet_cosine_loss(anchor, positive, negative, margin=.2):
    """Hinge loss requiring same-ID cosine similarity to exceed negative similarity."""
    if (anchor.shape != positive.shape or anchor.shape != negative.shape or
            anchor.ndim != 2 or not 0 <= margin <= 2):
        raise ValueError('triplet embeddings must be aligned [N,D] and margin in [0,2]')
    if not all(torch.isfinite(value).all() for value in (anchor, positive, negative)):
        raise ValueError('triplet embeddings must be finite')
    positive_similarity = F.cosine_similarity(anchor, positive, dim=-1)
    negative_similarity = F.cosine_similarity(anchor, negative, dim=-1)
    return F.relu(negative_similarity - positive_similarity + margin).mean()
