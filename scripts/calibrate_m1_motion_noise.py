"""仅用训练集标定独立卡尔曼关联实验参数，不训练/调用 M1 或读取验证标签。"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from numbers import Integral
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from original.my_dataset.track_sequence_dataset import TrackSequenceColdDataset
from original.my_track import config
from original.my_track.gated_matching import gated_pairs
from original.my_track.kalman_motion import validate_motion_noise
from original.my_track.m1_frame_baseline import cluster_candidates


def load_motion_noise(path):
    """只接受带训练来源声明的完整参数文件；绝不从验证结果补参数。"""
    with Path(path).open(encoding='utf-8') as stream:
        payload = json.load(stream)
    if payload.get('schema_version') != 1 or payload.get('source_split') != 'train':
        raise ValueError('schema 1 training-split calibration required')
    return validate_motion_noise(payload.get('noise'))


def _load_frame(frame):
    with np.load(frame[0]) as cache:
        out, conf = np.asarray(cache['out']), np.asarray(cache['conf'])
    return out, conf, np.load(frame[1])


def _read_frames(frames, workers):
    if not workers:
        yield from map(_load_frame, frames)
        return
    # map 保持输入顺序。只有文件读取并行，GT 差分和浮点累计仍严格有序。
    with ThreadPoolExecutor(max_workers=workers) as executor:
        yield from executor.map(_load_frame, frames)


def estimate_motion_noise(records, progress=None, workers=0):
    """训练期 GT 只用于离线残差/运动统计；返回参数及可审计样本计数。"""
    if isinstance(workers, bool) or not isinstance(workers, Integral) or workers < 0:
        raise ValueError('workers must be a non-negative integer')
    residual_sum, velocity_sum, acceleration_sum = (np.zeros(2) for _ in range(3))
    matched_count = velocity_count = acceleration_count = 0
    checks = []
    folders = set()
    for record in records:
        if not record['frames'] or record['folder'] in folders:
            raise ValueError('nonempty records with unique sequence folders required')
        folders.add(record['folder'])
        previous = older = None
        first_shapes = None
        lower, upper = np.full(2, np.inf), np.full(2, -np.inf)
        loaded = _read_frames(record['frames'], workers)
        for frame_index, (frame, arrays) in enumerate(zip(record['frames'], loaded)):
            cache_path, _, sample_id = frame
            out, conf, label = arrays
            if (out.ndim != 2 or out.shape[1] != 4 or conf.shape != (len(out), 2) or
                    label.shape != (10, 6) or not np.isfinite(out).all() or
                    not np.isfinite(conf).all() or not np.isfinite(label).all()):
                raise ValueError(f'invalid cache/label shape or value: {cache_path}')
            if previous is not None and sample_id <= previous[0]:
                raise ValueError('sample IDs must be strictly increasing in each sequence')
            if first_shapes is None:
                first_shapes = dict(out_shape=list(out.shape), conf_shape=list(conf.shape),
                                    label_shape=list(label.shape), out_dtype=str(out.dtype),
                                    conf_dtype=str(conf.dtype), label_dtype=str(label.dtype))
            logits = conf.astype(np.float64)
            logits -= logits.max(axis=-1, keepdims=True)
            probabilities = np.exp(logits)
            scores = probabilities[:, 1] / probabilities.sum(axis=-1)
            points = out[:, :2].astype(np.float64)
            primary = cluster_candidates(points, scores, .96, .75)
            gt = label.mean(axis=0).reshape(3, 2).astype(np.float64)
            valid = np.abs(gt).sum(axis=1) > 0
            pairs = gated_pairs(primary.positions, gt[valid], 1.)
            for i, j in pairs:
                residual_sum += (primary.positions[i] - gt[valid][j]) ** 2
            matched_count += len(pairs)
            if valid.any():
                lower = np.minimum(lower, gt[valid].min(axis=0))
                upper = np.maximum(upper, gt[valid].max(axis=0))
            if previous is not None and sample_id - previous[0] == 1:
                mask = valid & previous[2]
                velocity_sum += ((gt[mask] - previous[1][mask]) ** 2).sum(axis=0)
                velocity_count += int(mask.sum())
                if older is not None and previous[0] - older[0] == 1:
                    mask &= older[2]
                    acceleration = gt[mask] - 2 * previous[1][mask] + older[1][mask]
                    acceleration_sum += (acceleration ** 2).sum(axis=0)
                    acceleration_count += int(mask.sum())
            older, previous = previous, (sample_id, gt, valid)
            if progress is not None and (frame_index % 1000 == 0 or
                                         frame_index + 1 == len(record['frames'])):
                progress(record['folder'], frame_index + 1, len(record['frames']))
        checks.append({'folder': record['folder'], 'frames': len(record['frames']),
                       **first_shapes, 'gt_min_xy_m': lower.tolist() if np.isfinite(lower).all() else None,
                       'gt_max_xy_m': upper.tolist() if np.isfinite(upper).all() else None})
    if not min(matched_count, velocity_count, acceleration_count):
        raise ValueError('matched measurements, consecutive GT pairs and triples are all required')
    floor = 1e-6
    noise = validate_motion_noise({
        'measurement_var': np.maximum(residual_sum / matched_count, floor).tolist(),
        'acceleration_var': np.maximum(acceleration_sum / acceleration_count, floor).tolist(),
        'initial_velocity_var': np.maximum(velocity_sum / velocity_count, floor).tolist(),
    })
    return {
        'schema_version': 1, 'source_split': 'train', 'noise': noise,
        'counts': {'matched_measurements': matched_count, 'velocity_pairs': velocity_count,
                   'acceleration_triples': acceleration_count},
        'estimator': 'axis second moments; primary score .96 / cluster .75m / match 1m',
        'variance_floor': floor, 'time_unit': 'input step', 'data_checks': checks,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4, help='有序文件预读线程数；0=串行')
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('output already exists; choose a new filename')
    if args.workers < 0:
        parser.error('workers must be non-negative')
    cache_root = str(Path(config.CACHE_ROOT) / 'fast_dataset_multi_train')
    dataset = TrackSequenceColdDataset(config.TRAIN_ROOT, cache_root,
                                       config.WINDOW_SIZE, config.STRIDE, full_sequence=True)
    payload = estimate_motion_noise(
        [dataset[i] for i in range(len(dataset))],
        progress=lambda folder, step, length: print(f'{folder}: {step}/{length}', flush=True),
        workers=args.workers,
    )
    payload.update(data_root=config.TRAIN_ROOT, cache_root=cache_root, workers=args.workers)
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        stream.write(serialized + '\n')
    print(f"Frozen train-only noise: {payload['noise']}")
    print(f"Counts: {payload['counts']}; wrote {args.output}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
