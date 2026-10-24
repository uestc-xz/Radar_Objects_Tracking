"""只在离线监督阶段读取 GT，生成因果、序列独立的短窗候选查询。"""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib

import numpy as np

from .config import HIST_LEN
from .gated_matching import gated_pairs
from .lifecycle_tracker import LifecycleTracker
from .short_window_association import encode_query


@dataclass(frozen=True)
class AssociationFrame:
    sample_id: int
    positions: np.ndarray
    scores: np.ndarray
    targets: np.ndarray


def _is_ambiguous(candidate, slot, positions, targets, valid_slots, margin=.15):
    """标记几何代理标签不唯一/一对一强制非最近；不删除这些正样本。"""
    distance = np.linalg.norm(positions[candidate] - targets[slot])
    other_slots = valid_slots[valid_slots != slot]
    other_points = np.delete(positions, candidate, axis=0)
    return bool((len(other_slots) and
                 (np.linalg.norm(targets[other_slots] - positions[candidate], axis=1)
                  <= distance + margin).any()) or
                (len(other_points) and
                 (np.linalg.norm(other_points - targets[slot], axis=1) <= distance + margin).any()))


def build_sequence_queries(frames, hist_len=HIST_LEN, label_radius_m=.5):
    """一条独立序列 -> 固定最大 50 提议的数组；先查询再插入本帧历史。

    所有三个历史槽均可查询，不依赖当前 GT 是否有效。历史观测由离线 GT
    配对挑选，因此属于条件历史诊断；有歧义的过去配对也可能混入历史。
    """
    if isinstance(hist_len, bool) or not isinstance(hist_len, int) or hist_len < 2:
        raise ValueError('hist_len must be an integer >= 2')
    if not np.isfinite(label_radius_m) or label_radius_m <= 0:
        raise ValueError('label_radius_m must be finite and positive')
    histories = [deque() for _ in range(3)]
    rows, previous_step = [], None
    proposal_builder = LifecycleTracker(hist_len=hist_len)
    for frame in frames:
        step = frame.sample_id
        if (not isinstance(step, (int, np.integer)) or
                (previous_step is not None and step <= previous_step)):
            raise ValueError('sample IDs must be strictly increasing integers')
        previous_step = step
        targets = np.asarray(frame.targets, dtype=np.float64)
        if targets.shape != (3, 2) or not np.isfinite(targets).all():
            raise ValueError('targets must be finite [3,2]')
        if len(frame.positions) > 50:
            raise ValueError('M1 cache can have at most 50 candidates')
        primary, weak = proposal_builder._proposals(frame.positions, frame.scores)
        positions = np.concatenate((primary.positions, weak.positions))
        scores = np.concatenate((primary.scores, weak.scores))
        is_primary = np.arange(len(positions)) < len(primary.positions)
        valid_slots = np.flatnonzero(np.abs(targets).sum(axis=1) > 0)
        assignments = {int(valid_slots[slot]): int(candidate)
                       for candidate, slot in gated_pairs(positions, targets[valid_slots], label_radius_m)}
        for slot, observations in enumerate(histories):
            while observations and step - observations[0][0] > hist_len:
                observations.popleft()
            if len(observations) < 2:
                continue
            history, candidates = encode_query(observations, step, positions, scores,
                                                is_primary, hist_len)
            padded = np.zeros((50, 4), dtype=np.float32)
            padded[:len(candidates)] = candidates
            mask = np.arange(50) < len(candidates)
            candidate = assignments.get(slot)
            positive = candidate is not None
            competition = positive and (np.linalg.norm(candidates[:, :2], axis=1) <= 1.).sum() >= 2
            ambiguous = positive and _is_ambiguous(candidate, slot, positions, targets, valid_slots)
            rows.append((history, padded, mask, 0 if candidate is None else candidate + 1,
                         bool(competition), bool(ambiguous), step, slot))
        # 当前 GT 与量测只在构造完全部历史查询后使用，禁止教师量测提前进入输入。
        for slot, candidate in assignments.items():
            histories[slot].append((step, positions[candidate].copy(), float(scores[candidate])))
    names = ['history', 'candidates', 'mask', 'labels', 'competition', 'ambiguous', 'sample_ids', 'slots']
    shapes = [(hist_len, 5), (50, 4), (50,), (), (), (), (), ()]
    dtypes = [np.float32, np.float32, bool, np.int64, bool, bool, np.int64, np.int64]
    return {name: np.asarray([row[i] for row in rows], dtype=dtype).reshape((-1,) + shape)
            for i, (name, shape, dtype) in enumerate(zip(names, shapes, dtypes))}


def load_association_frame(paths):
    """按已冻结冷缓存契约读一帧；不加载 feat、不运行 M1。"""
    cache_path, label_path, sample_id = paths
    with np.load(cache_path) as cache:
        out = np.asarray(cache['out'])
        conf = np.asarray(cache['conf'])
    label = np.load(label_path)
    if (out.shape != (50, 4) or conf.shape != (50, 2) or label.shape != (10, 6) or
            out.dtype != np.float32 or conf.dtype != np.float32 or label.dtype != np.float32):
        raise ValueError(f'expected float32 out[50,4]/conf[50,2]/label[10,6]: {cache_path}')
    if not all(np.isfinite(value).all() for value in (out, conf, label)):
        raise ValueError(f'nonfinite frozen input: {cache_path}')
    logits = conf.astype(np.float64)
    logits -= logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return AssociationFrame(sample_id, out[:, :2], exp_logits[:, 1] / exp_logits.sum(axis=1),
                            label.mean(axis=0).reshape(3, 2))


def collect_queries(records, hist_len=HIST_LEN, workers=4, progress=None):
    """仅 I/O 并行，有序配对和历史构造在主线程；每序列重新建立历史。"""
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 0:
        raise ValueError('workers must be an integer >= 0')
    parts, sequence_stats = [], []
    digest = hashlib.sha256()
    for sequence_index, record in enumerate(records):
        frame_count = len(record['frames'])
        def report_frames(iterator):
            for index, frame in enumerate(iterator, 1):
                if progress is not None and (index % 500 == 0 or index == frame_count):
                    progress(record['folder'], index, frame_count)
                yield frame
        if workers:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                queries = build_sequence_queries(report_frames(pool.map(load_association_frame, record['frames'])),
                                                 hist_len=hist_len)
        else:
            queries = build_sequence_queries(report_frames(map(load_association_frame, record['frames'])),
                                             hist_len=hist_len)
        queries['sequence_indices'] = np.full(len(queries['labels']), sequence_index, dtype=np.int64)
        parts.append(queries)
        sequence_stats.append({'folder': record['folder'], 'frames': frame_count,
                               'queries': len(queries['labels']),
                               'positive': int((queries['labels'] > 0).sum()),
                               'null': int((queries['labels'] == 0).sum()),
                               'competition': int(queries['competition'].sum()),
                               'ambiguous': int(queries['ambiguous'].sum())})
        digest.update(record['folder'].encode('utf-8'))
        for name in sorted(queries):
            digest.update(name.encode('ascii'))
            digest.update(queries[name].tobytes())
    if not parts or not sum(len(part['labels']) for part in parts):
        raise ValueError('no queries with at least two past observations in selected records')
    arrays = {name: np.concatenate([part[name] for part in parts]) for name in parts[0]}
    return arrays, {'sequences': sequence_stats, 'query_sha256': digest.hexdigest(),
                    'queries': len(arrays['labels']), 'hist_len': hist_len,
                    'history_range': [arrays['history'].min(axis=(0, 1)).tolist(),
                                      arrays['history'].max(axis=(0, 1)).tolist()],
                    'candidate_range': [arrays['candidates'].min(axis=(0, 1)).tolist(),
                                        arrays['candidates'].max(axis=(0, 1)).tolist()]}
