"""独立短窗候选排序筛查：条件量测历史，不是闭环跟踪训练。"""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from original.my_track import config
from original.my_track.short_window_association import (
    ShortWindowAssociation, baseline_logits, ranking_loss,
)
from original.my_track.short_window_data import collect_queries
from original.my_track.separated_association_loss import separated_losses
from scripts.train_m1_feature_projection import _records

OBJECTIVES = ('joint', 'separated', 'separated_balanced')
COMPONENT_COUNTS = {'ranking': 'positive_count', 'competition_ranking': 'competition_count',
                    'other_ranking': 'other_count', 'matchability': 'matchability_count'}


def _accumulate_components(accumulator, parts):
    for name, count_name in COMPONENT_COUNTS.items():
        count = parts[count_name]
        accumulator[name] = accumulator.get(name, 0.) + float(parts[name].detach()) * count
        accumulator[count_name] = accumulator.get(count_name, 0) + count


def _component_summary(accumulator, balanced):
    result = {}
    for name, count_name in COMPONENT_COUNTS.items():
        count = accumulator.get(count_name, 0)
        result[count_name] = count
        result[name] = accumulator.get(name, 0.) / count if count else 0.
    if balanced:
        groups = int(result['competition_count'] > 0) + int(result['other_count'] > 0)
        result['ranking'] = (result['competition_ranking'] + result['other_ranking']) / groups if groups else 0.
    result['total'] = result['ranking'] + result['matchability']
    return result


def _gradient_audit(model, parts):
    """只读分项梯度，不更新参数或写入 .grad；检查实际训练图的参数路由。"""
    named = list(model.named_parameters())
    result = {}
    for component in ('ranking', 'matchability'):
        gradients = torch.autograd.grad(parts[component], [p for _, p in named],
                                        retain_graph=True, allow_unused=True)
        squares = {group: 0. for group in ('history_encoder', 'edge', 'reject')}
        for (name, _), gradient in zip(named, gradients):
            if gradient is not None:
                squares[name.split('.')[0]] += float(gradient.detach().square().sum())
        result[component] = {group: value ** .5 for group, value in squares.items()}
    return result


def summarize_predictions(arrays, predictions, losses, candidate_predictions=None):
    """显示分母；无该类样本的准确率为 None，不能当成零或满分。"""
    labels = arrays['labels']
    positive = labels > 0
    groups = {'all': np.ones(len(labels), dtype=bool), 'positive': positive,
              'null': ~positive, 'competition': arrays['competition'],
              'ambiguous': arrays['ambiguous'],
              'clean_positive': positive & ~arrays['ambiguous']}
    result = {}
    for name, selected in groups.items():
        count = int(selected.sum())
        correct = int((predictions[selected] == labels[selected]).sum())
        result[name] = {'count': count, 'correct': correct,
                        'accuracy': correct / count if count else None,
                        'nll': float(losses[selected].mean()) if count else None,
                        'predicted_null_rate': float((predictions[selected] == 0).mean()) if count else None,
                        'wrong_candidate': int((selected & positive & (predictions > 0) &
                                                (predictions != labels)).sum()),
                        'false_reject': int((selected & positive & (predictions == 0)).sum()),
                        'false_accept': int((selected & ~positive & (predictions > 0)).sum())}
        if candidate_predictions is not None:
            ranked = selected & positive
            ranking_count = int(ranked.sum())
            ranking_correct = int((candidate_predictions[ranked] == labels[ranked]).sum())
            result[name].update({'ranking_positive_count': ranking_count,
                                 'ranking_correct': ranking_correct,
                                 'ranking_accuracy': ranking_correct / ranking_count if ranking_count else None})
    return result


def _loader(arrays, batch_size, shuffle=False, seed=7, include_competition=False):
    names = ('history', 'candidates', 'mask', 'labels')
    if include_competition:
        names += ('competition',)
    dataset = TensorDataset(*(torch.from_numpy(arrays[name])
                              for name in names))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                      generator=torch.Generator().manual_seed(seed))


def evaluate(model, arrays, device, batch_size=2048, include_baselines=False, objective='joint'):
    """相同查询上的条件排序评估；不维护或更新 tracker 状态。"""
    methods = ['model', 'nearest_last', 'last_velocity'] if include_baselines else ['model']
    predictions = {method: [] for method in methods}
    candidate_predictions = {method: [] for method in methods}
    losses = {method: [] for method in methods}
    components = {method: {} for method in methods}
    model.eval()
    with torch.no_grad():
        for tensors in _loader(arrays, batch_size, include_competition=True):
            history, candidates, mask, labels, competition = (tensor.to(device) for tensor in tensors)
            logits = {'model': model(history, candidates, mask)}
            if include_baselines:
                logits['nearest_last'] = baseline_logits(history, candidates, mask)
                logits['last_velocity'] = baseline_logits(history, candidates, mask, True)
            for method in methods:
                predictions[method].append(logits[method].argmax(-1).cpu().numpy())
                # 仅正样本统计“排除 null 后选谁”，区分排序退化与拒绝阈值变化。
                candidate_predictions[method].append((logits[method][:, 1:].argmax(-1) + 1).cpu().numpy())
                losses[method].append(F.cross_entropy(logits[method], labels, reduction='none').cpu().numpy())
                if objective != 'joint':
                    parts = separated_losses(logits[method], labels, mask, competition,
                                             balanced=objective == 'separated_balanced')
                    _accumulate_components(components[method], parts)
    result = {}
    for method in methods:
        predicted, nll = np.concatenate(predictions[method]), np.concatenate(losses[method])
        candidate_ranked = np.concatenate(candidate_predictions[method])
        result[method] = summarize_predictions(arrays, predicted, nll, candidate_ranked)
        if objective != 'joint':
            result[method]['objective_components'] = _component_summary(
                components[method], balanced=objective == 'separated_balanced')
        result[method]['per_sequence'] = {}
        for index in np.unique(arrays['sequence_indices']):
            selected = arrays['sequence_indices'] == index
            subset = {name: arrays[name][selected] for name in
                      ('labels', 'competition', 'ambiguous', 'sequence_indices')}
            result[method]['per_sequence'][str(index)] = summarize_predictions(
                subset, predicted[selected], nll[selected], candidate_ranked[selected])
    return result


def run_training(train, validation, metadata, output_dir, hist_len=config.HIST_LEN,
                 epochs=2, batch_size=512, lr=1e-3, device='cpu', seed=7, progress=True, objective='joint'):
    """每轮落盘独立 checkpoint/指标；不覆盖已有实验，不自动挑验证最优模型。"""
    for name, value in [('epochs', epochs), ('batch_size', batch_size)]:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f'{name} must be an integer >= 1')
    if not np.isfinite(lr) or lr <= 0:
        raise ValueError('lr must be finite and positive')
    if not len(train['labels']) or not len(validation['labels']):
        raise ValueError('train and validation must both contain queries')
    if objective not in OBJECTIVES:
        raise ValueError(f'objective must be one of {OBJECTIVES}')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device)
    model = ShortWindowAssociation(hist_len).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    report = {'schema': 1, 'task': 'conditional_short_window_ranking_not_closed_loop',
              'metadata': metadata, 'settings': {'hist_len': hist_len, 'epochs': epochs,
                'batch_size': batch_size, 'lr': lr, 'seed': seed, 'device': str(device), 'objective': objective,
                'label_radius_m': .5, 'logit_temperature_m': .25, 'null_radius_m': 1.,
                'train_queries': len(train['labels']), 'val_queries': len(validation['labels']),
                'parameters': sum(parameter.numel() for parameter in model.parameters())},
              'initial_validation': evaluate(model, validation, device, include_baselines=True, objective=objective),
              'history': []}
    def save_manifest():
        with (output_dir / 'manifest.json').open('w', encoding='utf-8') as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
    save_manifest()
    for method, values in report['initial_validation'].items():
        print(f"initial {method}: nll={values['all']['nll']:.5f} "
              f"accuracy={values['all']['accuracy']:.4f} "
              f"competition={values['competition']['accuracy']} "
              f"competition_rank={values['competition']['ranking_accuracy']} "
              f"null={values['null']['accuracy']}", flush=True)
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        model.train()
        total_loss = total_grad = 0.
        component_totals = {}
        count = batches = 0
        loader = tqdm(_loader(train, batch_size, shuffle=True, seed=seed + epoch, include_competition=True),
                      desc=f'Epoch {epoch}/{epochs} [rank train]', disable=not progress, unit='batch')
        for tensors in loader:
            history, candidates, mask, labels, competition = (tensor.to(device) for tensor in tensors)
            optimizer.zero_grad(set_to_none=True)
            if objective == 'joint':
                loss = ranking_loss(model(history, candidates, mask), labels, mask)
            else:
                logits = model(history, candidates, mask, detach_rejection_history=True)
                parts = separated_losses(logits, labels, mask, competition,
                                         balanced=objective == 'separated_balanced')
                loss = parts['total']
                _accumulate_components(component_totals, parts)
                if epoch == 1 and batches == min(1, len(loader) - 1):
                    report['gradient_audit'] = {'batch_index': batches, 'routes': _gradient_audit(model, parts)}
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite ranking loss')
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            total_loss += float(loss.detach()) * len(labels)
            total_grad += float(grad)
            count += len(labels)
            batches += 1
            if progress:
                loader.set_postfix(loss=f'{total_loss / count:.5f}', grad=f'{float(grad):.3f}')
        validation_metrics = evaluate(model, validation, device, objective=objective)
        entry = {'epoch': epoch, 'train_objective': total_loss / count,
                 'mean_preclip_grad': total_grad / batches,
                 'validation': validation_metrics['model'], 'seconds': time.perf_counter() - started}
        if objective == 'joint':
            entry['train_nll'] = total_loss / count
        else:
            entry['train_components'] = _component_summary(component_totals, objective == 'separated_balanced')
        report['history'].append(entry)
        torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch, 'hist_len': hist_len, 'seed': seed, 'objective': objective},
                   output_dir / f'epoch_{epoch}.pt')
        save_manifest()
        val = entry['validation']
        train_name = 'train_nll' if objective == 'joint' else 'train_objective'
        component_text = '' if objective == 'joint' else (
            f" train_rank={entry['train_components']['ranking']:.5f}"
            f" train_match={entry['train_components']['matchability']:.5f}")
        print(f"{objective} epoch {epoch}/{epochs}: {train_name}={entry['train_objective']:.5f}"
              f"{component_text} val_joint_nll={val['all']['nll']:.5f} accuracy={val['all']['accuracy']:.4f} "
              f"competition={val['competition']['accuracy']} "
              f"competition_rank={val['competition']['ranking_accuracy']} null={val['null']['accuracy']} "
              f"({entry['seconds']:.1f}s)", flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--workers', type=int, default=4, help='ordered cache I/O threads only')
    parser.add_argument('--max-sequences', type=int, default=0)
    parser.add_argument('--max-frames', type=int, default=0, help='prefix per sequence; 0 means all')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--no-progress', action='store_true', help='hide per-batch bars, keep epoch summaries')
    parser.add_argument('--objectives', nargs='+', choices=OBJECTIVES, default=['joint'],
                        help='one objective writes output-dir; multiple write one subdirectory each')
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error(f'output directory already exists: {args.output_dir}')
    if len(set(args.objectives)) != len(args.objectives):
        parser.error('objectives must be distinct')
    if (args.epochs < 1 or args.batch_size < 1 or args.workers < 0 or
            args.max_sequences < 0 or args.max_frames < 0 or not np.isfinite(args.lr) or args.lr <= 0):
        parser.error('epochs/batch-size/lr must be positive; limits/workers must be nonnegative')
    torch.set_num_threads(min(4, torch.get_num_threads()))
    collected, metadata = {}, {'note': 'GT-linked measurement histories; not autonomous tracking',
                              'splits': {}}
    for split, root in [('train', config.TRAIN_ROOT), ('validation', config.VAL_ROOT)]:
        cache_root = Path(config.CACHE_ROOT) / Path(root).name
        records = _records(root, str(cache_root), args.max_sequences, args.max_frames)
        print(f'collecting {split}: {len(records)} sequences', flush=True)
        started = time.perf_counter()
        def show(folder, done, total):
            print(f'  {split} {folder}: {done}/{total} frames', flush=True)
        collected[split], details = collect_queries(records, workers=args.workers, progress=show)
        details.update({'data_root': str(root), 'cache_root': str(cache_root),
                        'collection_seconds': time.perf_counter() - started})
        metadata['splits'][split] = details
        print(f"{split}: {details['queries']} queries ({details['collection_seconds']:.1f}s)", flush=True)
    if len(args.objectives) > 1:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    for objective in args.objectives:
        output = args.output_dir / objective if len(args.objectives) > 1 else args.output_dir
        print(f'training objective={objective}: {output}', flush=True)
        run_training(collected['train'], collected['validation'], metadata, output,
                     epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                     device=args.device, seed=args.seed, progress=not args.no_progress, objective=objective)


if __name__ == '__main__':
    main()
