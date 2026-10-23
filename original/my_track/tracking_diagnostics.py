"""变长航迹的离线诊断；原始 GT 槽是身份，GT 不参与任何推理。

门控一对一距离匹配和身份计数是本项目诊断口径，不冒充标准 MOTChallenge/HOTA。
"""

from collections import Counter

import numpy as np
from scipy.optimize import linear_sum_assignment

from .gated_matching import gated_pairs


def tracking_diagnostics(outputs, targets, gate_m=1.):
    """检测微平均、GT 换 ID、预测换 GT、碎片与全序列身份配对分数。"""
    targets = np.asarray(targets, dtype=np.float64)
    if targets.ndim != 3 or targets.shape[-1] != 2 or len(outputs) != len(targets):
        raise ValueError('outputs and targets[T,G,2] must have equal frame counts')
    if not np.isfinite(targets).all() or not np.isfinite(gate_m) or gate_m <= 0:
        raise ValueError('targets and positive gate_m must be finite')
    tp = fp = fn = predicted_total = gt_total = 0
    error_sum = 0.
    gt_switches = owner_switches = fragments = coast_tp = coast_fp = 0
    last_gt_id, last_owner, previous_matched = {}, {}, set()
    contingency, emitted, matched_ids = Counter(), set(), set()
    for frame, gt in zip(outputs, targets):
        ids = np.asarray(frame.track_ids)
        seen = np.asarray(frame.observed, dtype=bool)
        points = np.asarray(frame.positions, dtype=np.float64)
        if (ids.shape != (len(points),) or seen.shape != ids.shape or
                len(set(ids.tolist())) != len(ids) or
                not np.isfinite(ids).all() or not np.equal(ids, np.floor(ids)).all()):
            raise ValueError('each frame requires unique integral IDs and aligned fields')
        valid_slots = np.flatnonzero(np.abs(gt).sum(axis=1) > 0)
        valid_gt = gt[valid_slots]
        pairs = gated_pairs(points, valid_gt, gate_m)
        tp += len(pairs)
        fp += len(points) - len(pairs)
        fn += len(valid_gt) - len(pairs)
        predicted_total += len(points)
        gt_total += len(valid_gt)
        emitted.update(int(i) for i in ids)
        matched_now = set()
        observed_coast_tp = 0
        for pred_index, gt_index in pairs:
            track_id, slot = int(ids[pred_index]), int(valid_slots[gt_index])
            error_sum += float(np.linalg.norm(points[pred_index] - valid_gt[gt_index]))
            if slot in last_gt_id:
                gt_switches += int(last_gt_id[slot] != track_id)
                fragments += int(slot not in previous_matched)
            if track_id in last_owner:
                owner_switches += int(last_owner[track_id] != slot)
            last_gt_id[slot], last_owner[track_id] = track_id, slot
            matched_now.add(slot)
            matched_ids.add(track_id)
            contingency[(track_id, slot)] += 1
            observed_coast_tp += int(not seen[pred_index])
        coast_tp += observed_coast_tp
        coast_fp += int((~seen).sum()) - observed_coast_tp
        previous_matched = matched_now
    # 全序列最优 ID 映射：碎片分成多个 ID 时只有一个能匹配同一 GT。
    predicted_ids = sorted(emitted)
    slots = sorted({slot for _, slot in contingency})
    idtp = 0
    if predicted_ids and slots:
        matrix = np.asarray([[contingency[(pid, slot)] for slot in slots] for pid in predicted_ids])
        rows, columns = linear_sum_assignment(-matrix)
        idtp = int(matrix[rows, columns].sum())
    precision = tp / (tp + fp) if tp + fp else 0.
    recall = tp / (tp + fn) if tp + fn else 0.
    return {
        'frames': len(outputs), 'tp': tp, 'fp': fp, 'fn': fn,
        'error_sum_m': error_sum, 'predicted_points': predicted_total, 'gt_points': gt_total,
        'precision': precision, 'recall': recall,
        'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
        'matched_mae_m': error_sum / tp if tp else None,
        'gt_id_switches': gt_switches, 'track_owner_switches': owner_switches,
        'fragments': fragments, 'emitted_tracks': len(emitted),
        'false_emitted_tracks': len(emitted - matched_ids),
        'idtp': idtp, 'idfp': predicted_total - idtp, 'idfn': gt_total - idtp,
        'identity_f1': 2 * idtp / (predicted_total + gt_total) if predicted_total + gt_total else 0.,
        'coasted_tp': coast_tp, 'coasted_fp': coast_fp,
    }
