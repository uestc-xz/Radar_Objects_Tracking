import torch
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F
import torch.nn as nn
#### 返回二者
def hungarian_loss_with_classification(pred_logits, pred_boxes, targets,epoch = 0,is_train = True):
    batch_size = pred_boxes.shape[0]
    num_queries = pred_boxes.shape[1]

    # 初始化累加变量
    total_coord_loss = torch.tensor(0.0, device=pred_boxes.device)
    total_class_loss = torch.tensor(0.0, device=pred_boxes.device)

    # mse_loss = nn.MSELoss()
    mse_loss = nn.SmoothL1Loss(beta = 0.5)
    ce_loss = nn.CrossEntropyLoss()
    # mse_weight = (epoch//2)*0.05
    if (is_train == True):
        # mse_weight = (epoch//2)*0.05
        mse_weight = 1.0
        if(mse_weight>=1):
            mse_weight = 1.0
    else:
        mse_weight = 1.0
    for b in range(batch_size):
        target_current = targets[b]  # [3, 4]
        is_invalid = torch.all(torch.isclose(target_current[:, :2],
                                             torch.tensor([0.0, 0.0], device=pred_boxes.device)), dim=1)
        valid_mask = ~is_invalid
        valid_targets = target_current[valid_mask][:, :2]  # [n, 2]

        if len(valid_targets) == 0:
            # 没有目标时
            coord_loss = 0.1 * mse_loss(
                pred_boxes[b][:, :2],
                torch.zeros_like(pred_boxes[b][:, :2])
            )
            class_loss = ce_loss(
                pred_logits[b],
                torch.zeros(num_queries, dtype=torch.long, device=pred_boxes.device)
            )
            total_coord_loss += coord_loss
            total_class_loss += class_loss
            continue

        # 匈牙利匹配
        pred_coords = pred_boxes[b][:, :2]  # [50, 2]
        dist_matrix = torch.cdist(valid_targets, pred_coords, p=2)  # [n, 50]
        dist_matrix = dist_matrix.T  # [50, n]

        row_idx, col_idx = linear_sum_assignment(dist_matrix.detach().cpu().numpy())

        coord_loss = torch.tensor(0.0, device=pred_boxes.device)
        class_loss_matched = torch.tensor(0.0, device=pred_boxes.device)

        if len(row_idx) > 0:
            matched_pred = pred_coords[row_idx]
            matched_target = valid_targets[col_idx]
            coord_loss = mse_loss(matched_pred, matched_target)

            matched_logits = pred_logits[b][row_idx]
            class_targets_matched = torch.ones(len(row_idx), dtype=torch.long, device=pred_boxes.device)
            class_loss_matched = ce_loss(matched_logits, class_targets_matched)

        # 未匹配的惩罚
        if num_queries > len(row_idx) and len(row_idx) > 0:
            unmatched_mask = torch.ones(num_queries, dtype=torch.bool, device=pred_boxes.device)
            unmatched_mask[row_idx] = False

            unmatched_pred = pred_coords[unmatched_mask]
            penalty_loss = mse_weight * mse_loss(unmatched_pred, torch.zeros_like(unmatched_pred))
            coord_loss += penalty_loss

            unmatched_logits = pred_logits[b][unmatched_mask]
            class_targets_unmatched = torch.zeros(len(unmatched_logits), dtype=torch.long, device=pred_boxes.device)
            class_loss_unmatched = 1*ce_loss(unmatched_logits, class_targets_unmatched)
            class_loss_matched += class_loss_unmatched

        # 累加到总损失
        total_coord_loss += coord_loss
        total_class_loss += class_loss_matched

    return total_coord_loss / batch_size, total_class_loss / batch_size