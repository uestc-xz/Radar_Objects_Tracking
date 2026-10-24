"""短窗实验：候选条件排序与可匹配性分项监督，不改变预测类别定义。"""

import torch
from torch.nn import functional as F

from .short_window_association import masked_logits


def separated_losses(logits, labels, candidate_mask, competition, balanced=False):
    """返回 total/分项均值和各自计数；两个子任务的参数路由由前向共同保证。

    排序只看 positive 的 valid 候选；接收 margin 使用候选最大值的 detach，
    不能通过整体压低候选分数训练拒绝。调用方还须对拒绝分支的历史编码
    detach，避免该分支通过共享编码器间接更新排序表示。
    """
    logits = masked_logits(logits, candidate_mask)
    if (labels.shape != (len(logits),) or labels.dtype != torch.long or
            (labels < 0).any() or (labels >= logits.shape[1]).any()):
        raise ValueError('labels must be long[B] inside [0,N]')
    valid_classes = torch.cat((torch.ones_like(labels[:, None], dtype=torch.bool), candidate_mask), dim=1)
    if not valid_classes.gather(1, labels[:, None]).all():
        raise ValueError('positive target cannot refer to padding')
    positive = labels > 0
    if (competition.shape != labels.shape or competition.dtype != torch.bool or
            (competition & ~positive).any()):
        raise ValueError('competition must be bool[B] and a subset of positive queries')
    if not isinstance(balanced, bool):
        raise ValueError('balanced must be bool')

    # 先清除 padding 才构造可反传零，防止多个 dtype.min 相加溢出后乘零。
    zero = logits[:, 1:].masked_fill(~candidate_mask, 0.).sum() * 0.
    positive_count = int(positive.sum().item())
    competition_count = int(competition.sum().item())
    other_count = positive_count - competition_count
    competition_rank = other_rank = ranking = zero
    if positive_count:
        candidate_ce = F.cross_entropy(logits[positive, 1:], labels[positive] - 1, reduction='none')
        group = competition[positive]
        competition_rank = candidate_ce[group].mean() if competition_count else zero
        other_rank = candidate_ce[~group].mean() if other_count else zero
        if balanced:
            ranking = (competition_rank + other_rank) / int(bool(competition_count) + bool(other_count))
        else:
            ranking = candidate_ce.mean()

    has_candidates = candidate_mask.any(dim=1)
    matchability_count = int(has_candidates.sum().item())
    matchability = zero
    if matchability_count:
        best_candidate = logits[has_candidates, 1:].amax(dim=1).detach()
        margin = best_candidate - logits[has_candidates, 0]
        matchability = F.binary_cross_entropy_with_logits(margin, positive[has_candidates].to(logits.dtype))
    return {'total': ranking + matchability, 'ranking': ranking, 'matchability': matchability,
            'competition_ranking': competition_rank, 'other_ranking': other_rank,
            'positive_count': positive_count, 'competition_count': competition_count,
            'other_count': other_count, 'matchability_count': matchability_count}
