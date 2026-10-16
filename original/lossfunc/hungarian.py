import torch
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
# def hungarian_loss(pred, target):
#     batch_size, timesteps, _ = pred.shape
#     nums = 3
#     pred = pred.view(batch_size, timesteps, nums, 2)  # (B, T, 5, 2)
#     target = target.view(batch_size, timesteps, nums, 2)  # (B, T, 5, 2)
#
#     loss = torch.tensor(0.0, device=pred.device)
#
#     for b in range(batch_size):
#         for t in range(timesteps):
#             # 计算当前预测点和目标点之间的欧氏距离矩阵 (5, 5)
#             dist = torch.cdist(pred[b, t], target[b, t], p=2)  # (5, 5)
#
#             # 使用匈牙利算法找到最优匹配
#             row_idx, col_idx = linear_sum_assignment(dist.detach().cpu().numpy())
#
#             # 计算匹配后的总距离（可以取平均或总和）
#             matched_dist = dist[row_idx, col_idx].mean()
#             loss += matched_dist
#
#     # 归一化：除以 batch_size 和 timesteps
#     return loss / (batch_size * timesteps) if batch_size > 0 else loss
# def hungarian_loss(pred, target):
#     batch_size, target_num, _ = pred.shape
#     nums = 3
#     target = target.view(batch_size, nums, 2)  # (B, T, 5, 2)
#
#     loss = torch.tensor(0.0, device=pred.device)
#
#     for b in range(batch_size):
#
#         # 计算当前预测点和目标点之间的欧氏距离矩阵 (5, 5)
#         dist = torch.cdist(pred[b], target[b], p=2)  # (5, 5)
#
#         # 使用匈牙利算法找到最优匹配
#         row_idx, col_idx = linear_sum_assignment(dist.detach().cpu().numpy())
#
#         # 计算匹配后的总距离（可以取平均或总和）
#         matched_dist = dist[row_idx, col_idx].mean()
#         loss += matched_dist
#
#     # 归一化：除以 batch_size 和 timesteps
#     return loss / batch_size  if batch_size > 0 else loss
#
# def hungarian_loss(pred,target, num_max_targets=3):
#
#     batch_size = pred.shape[0]
#     total_loss = torch.tensor(0.0, device=pred.device)
#     # 将标签reshape为 [B, 3, 2]，方便处理
#     target_reshaped = target.view(batch_size, num_max_targets, 2)  # [B,3,2]
#     smooth_l1_loss = nn.SmoothL1Loss()
#     # 定义无效目标的坐标 (0, 0)，用于匹配判断
#     invalid_coords = torch.tensor([0.0, 0.0], device=pred.device)
#     for b in range(batch_size):
#         # 1. 筛选当前样本的有效目标（排除等于(0,0)的无效目标）
#         # 逐个目标判断：x==0且y==3.5 视为无效
#         is_invalid = torch.all(torch.isclose(target_reshaped[b], invalid_coords), dim=1)  # [3]，True表示无效
#         valid_mask = ~is_invalid  # [3]，True表示有效目标
#         valid_targets = target_reshaped[b][valid_mask]  # [n,2]，n=1,2,3（数据保证n≥1）
#         # num_people = len(valid_targets)
#
#         # 2. 预测取全部3个查询（与最大目标数一致）
#         pred_current = pred[b]  # [3,2]
#
#         # 3. 构建距离矩阵 [n, 3]（有效目标数 × 预测数）
#         dist_matrix = torch.cdist(valid_targets, pred_current, p=2)
#
#         # 4. 匈牙利算法匹配（n个目标匹配n个预测）
#         row_idx, col_idx = linear_sum_assignment(dist_matrix.detach().cpu().numpy())
#
#         # 5. 计算匹配对的损失（L1损失更稳定）
#         matched_pred = pred_current[col_idx]  # [n,2]
#
#         loss = smooth_l1_loss(matched_pred, valid_targets)
#
#         # 6. 对未匹配的预测（3-n个）加惩罚：让它们接近无效标签(0,3.5)
#         if 3 > len(col_idx):
#             unmatched_mask = torch.ones(3, dtype=torch.bool, device=pred.device)
#             unmatched_mask[col_idx] = False  # 标记未匹配的预测
#             unmatched_pred = pred_current[unmatched_mask]  # [3-n,2]
#             # 惩罚未匹配预测偏离(0,0)，权重可调整
#             loss += 0.1 * smooth_l1_loss(unmatched_pred, invalid_coords.repeat(unmatched_pred.shape[0], 1))
#
#         total_loss += (loss)
#     return (total_loss / batch_size if batch_size > 0 else total_loss)
### zheshi haoyongde
# def hungarian_loss(pred, target, num_max_targets=3):
#     batch_size = pred.shape[0]
#     total_loss = torch.tensor(0.0, device=pred.device)
#
#     # 现在target也是 [B, 3, 4] - (cx, cy, w, h)
#     smooth_l1_loss = nn.SmoothL1Loss()
#
#     # 定义无效目标的坐标 (0, 0, 0, 0)
#     invalid_coords = torch.tensor([0.0, 0.0, 0.0, 0.0], device=pred.device)
#
#     for b in range(batch_size):
#         # 1. 筛选有效目标（根据中心点判断）
#         target_current = target[b]  # [3,4]
#
#         is_invalid = torch.all(torch.isclose(target_current[:, :2],
#                                              torch.tensor([0.0, 0.0], device=pred.device)),dim=1)
#         valid_mask = ~is_invalid
#         valid_targets = target_current[valid_mask]  # [n,4] - 包含真实宽高
#
#         # 2. 当前样本的预测 [3,4] - (cx, cy, w, h)
#         pred_current = pred[b]  # [3,4]
#         pred_points = pred_current[:, :2]  # [3,2] - 只取中心点用于匹配
#
#         # 3. 用中心点距离进行匈牙利匹配
#         dist_matrix = torch.cdist(valid_targets[:, :2], pred_points, p=2)  # 只用中心点匹配
#         row_idx, col_idx = linear_sum_assignment(dist_matrix.detach().cpu().numpy())
#
#         # 4. 计算中心点损失
#         matched_pred_points = pred_points[col_idx]  # 匹配的预测中心点 [n,2]
#         matched_target_points = valid_targets[row_idx, :2]  # 匹配的真实中心点 [n,2]
#         center_loss = smooth_l1_loss(matched_pred_points, matched_target_points)
#
#         # 5. 计算宽高损失（使用真实的宽高标签！）
#         matched_pred_wh = pred_current[col_idx, 2:]  # 匹配的预测宽高 [n,2]
#         matched_target_wh = valid_targets[row_idx, 2:]  # 匹配的真实宽高 [n,2]
#         wh_loss = smooth_l1_loss(matched_pred_wh, matched_target_wh)
#         #
#         # # 6. 总损失 = 中心点损失 + 宽高损失
#         loss = center_loss + wh_loss  # 可以调整权重，比如 center_loss + 0.5 * wh_loss
#         # loss = center_loss   # 可以调整权重，比如 center_loss + 0.5 * wh_loss
#         # 7. 对未匹配的预测加惩罚（中心点到无效位置）
#         if num_max_targets > len(col_idx):
#             unmatched_mask = torch.ones(num_max_targets, dtype=torch.bool, device=pred.device)
#             unmatched_mask[col_idx] = False
#             unmatched_pred_points = pred_points[unmatched_mask]  # [3-n,2]
#
#             penalty_loss = 0.1 * smooth_l1_loss(
#                 unmatched_pred_points,
#                 torch.tensor([0.0, 0.0], device=pred.device).repeat(unmatched_pred_points.shape[0], 1)
#             )
#             loss += penalty_loss
#
#         total_loss += loss
#
#     return total_loss / batch_size if batch_size > 0 else total_loss

def hungarian_loss(pred, target):
    """
    支持可变查询数量的匈牙利损失
    pred: [B, num_queries, 4] - 50个查询
    target: [B, num_targets, 4] - 3个目标
    """
    batch_size = pred.shape[0]
    num_queries = pred.shape[1]
    total_loss = torch.tensor(0.0, device=pred.device)

    smooth_l1_loss = nn.MSELoss()

    for b in range(batch_size):
        target_current = target[b]  # [3,4]
        pred_current = pred[b]  # [50,4]

        # 筛选有效目标
        is_invalid = torch.all(torch.isclose(target_current[:, :2],
                                             torch.tensor([0.0, 0.0], device=pred.device)), dim=1)
        valid_mask = ~is_invalid
        valid_targets = target_current[valid_mask]  # [n,4]

        if len(valid_targets) == 0:
            # 如果没有目标，惩罚所有查询
            penalty_loss = 0.1 * smooth_l1_loss(
                pred_current[:, :2],
                torch.zeros_like(pred_current[:, :2])
            )
            total_loss += penalty_loss
            continue

        # 匈牙利匹配：n个目标 vs 50个查询
        dist_matrix = torch.cdist(valid_targets[:, :2], pred_current[:, :2], p=2)
        row_idx, col_idx = linear_sum_assignment(dist_matrix.detach().cpu().numpy())

        # 匹配的损失
        matched_pred = pred_current[col_idx]
        matched_target = valid_targets[row_idx]

        center_loss = smooth_l1_loss(matched_pred[:, :2], matched_target[:, :2])
        wh_loss = smooth_l1_loss(matched_pred[:, 2:], matched_target[:, 2:])

        loss = center_loss + wh_loss

        # 未匹配查询的惩罚
        if num_queries > len(col_idx):
            unmatched_mask = torch.ones(num_queries, dtype=torch.bool, device=pred.device)
            unmatched_mask[col_idx] = False
            unmatched_pred = pred_current[unmatched_mask]

            penalty_loss = 0.1 * smooth_l1_loss(
                unmatched_pred[:, :2],
                torch.zeros_like(unmatched_pred[:, :2])
            )
            loss += penalty_loss

        total_loss += loss

    return total_loss / batch_size

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


def box_cxcywh_to_xyxy(x):
    """
    将 (x_center, y_center, w, h) 转换为 (x1, y1, x2, y2)
    """
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def generalized_box_iou(boxes1, boxes2):
    """
    计算广义IoU损失
    boxes1, boxes2: [N, 4], [M, 4] 格式为 (x1, y1, x2, y2)
    返回: [N, M] 的GIoU矩阵
    """
    # 计算交集面积
    inter_x1y1 = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    inter_x2y2 = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]
    inter_wh = (inter_x2y2 - inter_x1y1).clamp(min=0)  # [N, M, 2]
    inter_area = inter_wh[:, :, 0] * inter_wh[:, :, 1]  # [N, M]

    # 计算各自面积
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])  # [N]
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])  # [M]

    # 计算并集面积
    union_area = area1[:, None] + area2 - inter_area  # [N, M]

    # 计算IoU
    iou = inter_area / union_area

    # 计算最小封闭框
    enclose_x1y1 = torch.min(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    enclose_x2y2 = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]
    enclose_wh = (enclose_x2y2 - enclose_x1y1).clamp(min=0)  # [N, M, 2]
    enclose_area = enclose_wh[:, :, 0] * enclose_wh[:, :, 1]  # [N, M]

    # 计算GIoU
    giou = iou - (enclose_area - union_area) / enclose_area
    return giou

#
# def complete_detr_loss(pred_logits, pred_boxes, targets):
#     """
#     完整的DETR损失函数，包含中心点和宽高回归
#     - 使用L1损失 + GIoU损失进行框回归
#     - 正确处理全零填充的无效目标
#
#     Args:
#         pred_logits: [batch_size, num_queries, 2]    # 2分类：背景(0) vs 目标(1)
#         pred_boxes: [batch_size, num_queries, 4]     # [x, y, w, h]
#         targets: [batch_size, max_targets, 4]        # 包含全零填充的无效目标
#     """
#     batch_size = pred_logits.shape[0]
#     num_queries = pred_logits.shape[1]
#
#     total_loss = 0
#     total_class_loss = 0
#     total_bbox_l1_loss = 0
#     total_giou_loss = 0
#
#     for b in range(batch_size):
#         target_boxes = targets[b]  # [max_targets, 4]
#
#         # 过滤全零的无效目标
#         valid_mask = ~torch.all(torch.eq(target_boxes, 0.0), dim=1)
#         valid_targets = target_boxes[valid_mask]
#
#         num_valid_targets = len(valid_targets)
#
#         # 情况1：没有有效目标
#         if num_valid_targets == 0:
#             target_classes = torch.zeros(num_queries, dtype=torch.long, device=pred_logits.device)
#             class_loss = F.cross_entropy(pred_logits[b], target_classes)
#             total_class_loss += class_loss
#             continue
#
#         # 情况2：有有效目标
#         # 1. 计算匹配代价（关键改进！）
#         with torch.no_grad():
#             # 分类代价：使用目标类的预测概率（负号是因为我们要最小化代价）
#             cost_class = -pred_logits[b, :, 1].softmax(-1)  # [num_queries]
#
#             # 框回归代价：L1距离 + GIoU距离
#             pred_boxes_flat = pred_boxes[b]  # [num_queries, 4]
#
#             # L1代价
#             cost_bbox = torch.cdist(pred_boxes_flat, valid_targets, p=1)  # [num_queries, num_valid_targets]
#
#             # GIoU代价
#             pred_boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes_flat)
#             valid_targets_xyxy = box_cxcywh_to_xyxy(valid_targets)
#             giou_matrix = generalized_box_iou(pred_boxes_xyxy, valid_targets_xyxy)
#             cost_giou = -giou_matrix  # GIoU越大越好，所以加负号
#
#             # 总匹配代价 = 分类 + L1 + GIoU
#             C = 1.0 * cost_class.unsqueeze(1) + 2.0 * cost_bbox + 1.0 * cost_giou
#             C = C.cpu().numpy()
#
#             # 匈牙利匹配
#             row_idx, col_idx = linear_sum_assignment(C)
#
#         # 2. 计算损失
#         idx = torch.as_tensor(row_idx, device=pred_logits.device)
#         jdx = torch.as_tensor(col_idx, device=pred_logits.device)
#
#         # 分类损失：匹配上的为目标类(1)，未匹配上的为背景类(0)
#         target_classes = torch.zeros(num_queries, dtype=torch.long, device=pred_logits.device)
#         target_classes[idx] = 1
#         class_loss = F.cross_entropy(pred_logits[b], target_classes)
#
#         # 框回归损失：只计算匹配上的框
#         if len(idx) > 0:
#             matched_pred_boxes = pred_boxes[b][idx]  # [num_matched, 4]
#             matched_target_boxes = valid_targets[jdx]  # [num_matched, 4]
#
#             # L1损失
#             bbox_l1_loss = F.l1_loss(matched_pred_boxes, matched_target_boxes, reduction='mean')
#
#             # GIoU损失
#             matched_pred_xyxy = box_cxcywh_to_xyxy(matched_pred_boxes)
#             matched_target_xyxy = box_cxcywh_to_xyxy(matched_target_boxes)
#             giou_scores = generalized_box_iou(matched_pred_xyxy, matched_target_xyxy)
#             giou_loss = (1.0 - giou_scores.diag()).mean()  # 只取对角线上的匹配对
#
#             total_bbox_l1_loss += bbox_l1_loss
#             total_giou_loss += giou_loss
#         else:
#             bbox_l1_loss = torch.tensor(0.0, device=pred_boxes.device)
#             giou_loss = torch.tensor(0.0, device=pred_boxes.device)
#
#         total_class_loss += class_loss
#
#     # 取batch平均
#     total_class_loss = total_class_loss / batch_size
#     total_bbox_l1_loss = total_bbox_l1_loss / batch_size
#     total_giou_loss = total_giou_loss / batch_size
#
#     # 总损失
#     total_loss = total_class_loss + 2.0 * total_bbox_l1_loss + 1.0 * total_giou_loss
#
#     return total_loss, total_class_loss, total_bbox_l1_loss, total_giou_loss
# def complete_detr_loss(pred_logits, pred_boxes, targets):
#     """
#     完整的DETR损失函数，包含中心点和宽高回归
#     - 使用L1损失 + GIoU损失进行框回归
#     - 正确处理全零填充的无效目标
#
#     Args:
#         pred_logits: [batch_size, num_queries, 2]    # 2分类：背景(0) vs 目标(1)
#         pred_boxes: [batch_size, num_queries, 4]     # [x, y, w, h]
#         targets: [batch_size, max_targets, 4]        # 包含全零填充的无效目标
#     """
#     batch_size = pred_logits.shape[0]
#     num_queries = pred_logits.shape[1]
#
#     total_loss = 0
#     total_class_loss = 0
#     total_bbox_l1_loss = 0
#     total_giou_loss = 0
#
#     for b in range(batch_size):
#         target_boxes = targets[b]  # [max_targets, 4]
#
#         # 过滤全零的无效目标
#         valid_mask = ~torch.all(torch.eq(target_boxes, 0.0), dim=1)
#         valid_targets = target_boxes[valid_mask]
#
#         num_valid_targets = len(valid_targets)
#
#         # 情况1：没有有效目标
#         if num_valid_targets == 0:
#             target_classes = torch.zeros(num_queries, dtype=torch.long, device=pred_logits.device)
#             class_loss = F.cross_entropy(pred_logits[b], target_classes)
#             total_class_loss += class_loss
#             # 对于没有目标的情况，框回归损失为0
#             continue
#
#         # 情况2：有有效目标
#         # 1. 计算匹配代价
#         with torch.no_grad():
#             # 分类代价：使用目标类的预测概率
#             # pred_logits[b] 形状: [num_queries, 2]
#             cost_class = -pred_logits[b, :, 1:2]  # 取目标类的logits [num_queries, 1]
#             cost_class = cost_class.repeat(1, num_valid_targets)  # [num_queries, num_valid_targets]
#
#             # 框回归代价：L1距离
#             pred_boxes_flat = pred_boxes[b]  # [num_queries, 4]
#             cost_bbox = torch.cdist(pred_boxes_flat, valid_targets, p=1)  # [num_queries, num_valid_targets]
#
#             # GIoU代价
#             pred_boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes_flat)
#             valid_targets_xyxy = box_cxcywh_to_xyxy(valid_targets)
#             giou_matrix = generalized_box_iou(pred_boxes_xyxy, valid_targets_xyxy)
#             cost_giou = -giou_matrix  # GIoU越大越好，所以加负号
#
#             # 总匹配代价 = 分类 + L1 + GIoU
#             C = 1.0 * cost_class + 2.0 * cost_bbox + 1.0 * cost_giou  # [num_queries, num_valid_targets]
#             C = C.cpu().numpy()
#
#             # 匈牙利匹配：确保是方阵
#             if num_queries > num_valid_targets:
#                 # 查询数多于目标数，需要填充
#                 C_padded = np.full((num_queries, num_queries), 1e9)  # 用大数填充
#                 C_padded[:, :num_valid_targets] = C
#                 row_idx, col_idx = linear_sum_assignment(C_padded)
#                 # 过滤掉匹配到填充位置的索引
#                 mask = col_idx < num_valid_targets
#                 row_idx = row_idx[mask]
#                 col_idx = col_idx[mask]
#             else:
#                 # 目标数多于查询数，需要填充
#                 C_padded = np.full((num_valid_targets, num_valid_targets), 1e9)
#                 C_padded[:num_queries, :num_valid_targets] = C
#                 row_idx, col_idx = linear_sum_assignment(C_padded)
#                 # 过滤掉匹配到填充位置的索引
#                 mask = row_idx < num_queries
#                 row_idx = row_idx[mask]
#                 col_idx = col_idx[mask]
#
#         # 2. 计算损失
#         if len(row_idx) > 0:
#             idx = torch.as_tensor(row_idx, device=pred_logits.device)
#             jdx = torch.as_tensor(col_idx, device=pred_logits.device)
#
#             # 分类损失：匹配上的为目标类(1)，未匹配上的为背景类(0)
#             target_classes = torch.zeros(num_queries, dtype=torch.long, device=pred_logits.device)
#             target_classes[idx] = 1
#             class_loss = F.cross_entropy(pred_logits[b], target_classes)
#
#             # 框回归损失：只计算匹配上的框
#             matched_pred_boxes = pred_boxes[b][idx]  # [num_matched, 4]
#             matched_target_boxes = valid_targets[jdx]  # [num_matched, 4]
#
#             # L1损失
#             bbox_l1_loss = F.l1_loss(matched_pred_boxes, matched_target_boxes, reduction='mean')
#
#             # GIoU损失
#             matched_pred_xyxy = box_cxcywh_to_xyxy(matched_pred_boxes)
#             matched_target_xyxy = box_cxcywh_to_xyxy(matched_target_boxes)
#             giou_scores = generalized_box_iou(matched_pred_xyxy, matched_target_xyxy)
#             giou_loss = (1.0 - giou_scores.diag()).mean()  # 只取对角线上的匹配对
#
#             total_bbox_l1_loss += bbox_l1_loss
#             total_giou_loss += giou_loss
#             total_class_loss += class_loss
#         else:
#             # 没有匹配到任何目标，所有查询都是背景
#             target_classes = torch.zeros(num_queries, dtype=torch.long, device=pred_logits.device)
#             class_loss = F.cross_entropy(pred_logits[b], target_classes)
#             total_class_loss += class_loss
#
#     # 取batch平均
#     total_class_loss = total_class_loss / batch_size
#     total_bbox_l1_loss = total_bbox_l1_loss / batch_size if batch_size > 0 else torch.tensor(0.0)
#     total_giou_loss = total_giou_loss / batch_size if batch_size > 0 else torch.tensor(0.0)
#
#     # 总损失
#     total_loss = total_class_loss + 2.0 * total_bbox_l1_loss + 1.0 * total_giou_loss
#
#     return total_loss, total_class_loss, total_bbox_l1_loss, total_giou_loss
def complete_detr_loss(pred_logits, pred_boxes, targets):
    """
    完整的DETR损失函数 + 未匹配查询惩罚
    """
    batch_size = pred_logits.shape[0]
    num_queries = pred_logits.shape[1]

    total_loss = 0
    total_class_loss = 0
    total_bbox_l1_loss = 0
    total_giou_loss = 0
    total_unmatched_penalty = 0  # 新增：未匹配惩罚

    for b in range(batch_size):
        target_boxes = targets[b]  # [max_targets, 4]

        # 过滤全零的无效目标
        valid_mask = ~torch.all(torch.eq(target_boxes, 0.0), dim=1)
        valid_targets = target_boxes[valid_mask]

        num_valid_targets = len(valid_targets)

        # 情况1：没有有效目标
        if num_valid_targets == 0:
            target_classes = torch.zeros(num_queries, dtype=torch.long, device=pred_logits.device)
            class_loss = F.cross_entropy(pred_logits[b], target_classes)

            # 新增：没有目标时，所有查询都应该预测为背景，坐标接近0
            unmatched_penalty = 0.1 * F.l1_loss(
                pred_boxes[b],
                torch.zeros_like(pred_boxes[b])
            )

            total_class_loss += class_loss
            total_unmatched_penalty += unmatched_penalty
            continue

        # 情况2：有有效目标
        # 1. 计算匹配代价
        with torch.no_grad():
            # 分类代价：使用目标类的预测概率
            cost_class = -pred_logits[b, :, 1:2]  # [num_queries, 1]
            cost_class = cost_class.repeat(1, num_valid_targets)  # [num_queries, num_valid_targets]

            # 框回归代价：L1距离
            pred_boxes_flat = pred_boxes[b]  # [num_queries, 4]
            cost_bbox = torch.cdist(pred_boxes_flat, valid_targets, p=1)  # [num_queries, num_valid_targets]

            # GIoU代价
            pred_boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes_flat)
            valid_targets_xyxy = box_cxcywh_to_xyxy(valid_targets)
            giou_matrix = generalized_box_iou(pred_boxes_xyxy, valid_targets_xyxy)
            cost_giou = -giou_matrix  # GIoU越大越好，所以加负号

            # 总匹配代价
            C = 1.0 * cost_class + 2.0 * cost_bbox + 1.0 * cost_giou
            C = C.cpu().numpy()

            # 匈牙利匹配
            row_idx, col_idx = linear_sum_assignment(C)

        # 2. 计算匹配的损失
        idx = torch.as_tensor(row_idx, device=pred_logits.device)
        jdx = torch.as_tensor(col_idx, device=pred_logits.device)

        # 分类损失：匹配上的为目标类(1)，未匹配上的为背景类(0)
        target_classes = torch.zeros(num_queries, dtype=torch.long, device=pred_logits.device)
        target_classes[idx] = 1
        class_loss = F.cross_entropy(pred_logits[b], target_classes)

        # 框回归损失：只计算匹配上的框
        bbox_l1_loss = torch.tensor(0.0, device=pred_boxes.device)
        giou_loss = torch.tensor(0.0, device=pred_boxes.device)

        if len(idx) > 0:
            matched_pred_boxes = pred_boxes[b][idx]
            matched_target_boxes = valid_targets[jdx]

            # L1损失
            bbox_l1_loss = F.l1_loss(matched_pred_boxes, matched_target_boxes, reduction='mean')

            # GIoU损失
            matched_pred_xyxy = box_cxcywh_to_xyxy(matched_pred_boxes)
            matched_target_xyxy = box_cxcywh_to_xyxy(matched_target_boxes)
            giou_scores = generalized_box_iou(matched_pred_xyxy, matched_target_xyxy)
            giou_loss = (1.0 - giou_scores.diag()).mean()

        # 3. 新增：未匹配查询的惩罚
        unmatched_penalty = torch.tensor(0.0, device=pred_boxes.device)
        if num_queries > len(idx) and len(idx) > 0:
            # 创建未匹配掩码
            unmatched_mask = torch.ones(num_queries, dtype=torch.bool, device=pred_boxes.device)
            unmatched_mask[idx] = False

            # 获取未匹配的预测框
            unmatched_pred_boxes = pred_boxes[b][unmatched_mask]

            # 惩罚1：坐标应该接近0（背景位置）
            coord_penalty = 0.1 * F.l1_loss(
                unmatched_pred_boxes,
                torch.zeros_like(unmatched_pred_boxes)
            )

            # 惩罚2：尺寸应该接近0（背景框）
            size_penalty = 0.1 * F.l1_loss(
                unmatched_pred_boxes[:, 2:],  # 只惩罚宽高
                torch.zeros_like(unmatched_pred_boxes[:, 2:])
            )

            unmatched_penalty = coord_penalty + size_penalty

        total_class_loss += class_loss
        total_bbox_l1_loss += bbox_l1_loss
        total_giou_loss += giou_loss
        total_unmatched_penalty += unmatched_penalty

    # 取batch平均
    total_class_loss = total_class_loss / batch_size
    total_bbox_l1_loss = total_bbox_l1_loss / batch_size
    total_giou_loss = total_giou_loss / batch_size
    total_unmatched_penalty = total_unmatched_penalty / batch_size

    # 总损失 = 原有损失 + 未匹配惩罚
    total_loss = (total_class_loss +
                  2.0 * total_bbox_l1_loss +
                  1.0 * total_giou_loss +
                  0.5 * total_unmatched_penalty)  # 未匹配惩罚权重可以调整

    return total_loss, total_class_loss, total_bbox_l1_loss, total_giou_loss, total_unmatched_penalty
######    返回总损失
# def hungarian_loss_with_classification(pred_logits, pred_boxes, targets):
#
#     batch_size = pred_boxes.shape[0]
#     num_queries = pred_boxes.shape[1]
#     total_loss = torch.tensor(0.0, device=pred_boxes.device)
#
#     mse_loss = nn.MSELoss()
#     ce_loss = nn.CrossEntropyLoss()
#
#     for b in range(batch_size):
#         target_current = targets[b]  # [3, 4]
#
#         # 筛选有效目标
#         is_invalid = torch.all(torch.isclose(target_current[:, :2],
#                                              torch.tensor([0.0, 0.0], device=pred_boxes.device)), dim=1)
#         valid_mask = ~is_invalid
#         valid_targets = target_current[valid_mask][:, :2]  # [n, 2]
#
#         if len(valid_targets) == 0:
#             # 没有目标时
#             coord_loss = 0.1 * mse_loss(
#                 pred_boxes[b][:, :2],
#                 torch.zeros_like(pred_boxes[b][:, :2])
#             )
#             class_loss = ce_loss(
#                 pred_logits[b],
#                 torch.zeros(num_queries, dtype=torch.long, device=pred_boxes.device)
#             )
#             total_loss += coord_loss + class_loss
#             continue
#
#         # 匈牙利匹配
#         pred_coords = pred_boxes[b][:, :2]  # [50, 2]
#         dist_matrix = torch.cdist(valid_targets, pred_coords, p=2)  # [n, 50]
#         dist_matrix = dist_matrix.T  # [50, n]
#
#         row_idx, col_idx = linear_sum_assignment(dist_matrix.detach().cpu().numpy())
#
#         # 修复：正确的索引使用
#         coord_loss = torch.tensor(0.0, device=pred_boxes.device)
#         class_loss_matched = torch.tensor(0.0, device=pred_boxes.device)
#
#         if len(row_idx) > 0:
#             # 正确的索引对应：
#             # row_idx: 查询索引, col_idx: 目标索引
#             matched_pred = pred_coords[row_idx]  # 匹配的查询预测坐标
#             matched_target = valid_targets[col_idx]  # 对应的真实目标坐标
#
#             coord_loss = mse_loss(matched_pred, matched_target)
#
#             # 分类损失：匹配的查询应该预测为"有目标"
#             matched_logits = pred_logits[b][row_idx]  # 使用row_idx（查询索引）
#             class_targets_matched = torch.ones(len(row_idx), dtype=torch.long, device=pred_boxes.device)
#             class_loss_matched = ce_loss(matched_logits, class_targets_matched)
#
#         # 未匹配的惩罚
#         if num_queries > len(row_idx) and len(row_idx) > 0:
#             unmatched_mask = torch.ones(num_queries, dtype=torch.bool, device=pred_boxes.device)
#             unmatched_mask[row_idx] = False  # 使用row_idx标记已匹配的查询
#
#             unmatched_pred = pred_coords[unmatched_mask]
#
#             # 坐标惩罚
#             penalty_loss = 0.1 * mse_loss(
#                 unmatched_pred,
#                 torch.zeros_like(unmatched_pred)
#             )
#             coord_loss += penalty_loss
#
#             # 未匹配分类损失
#             unmatched_logits = pred_logits[b][unmatched_mask]
#             class_targets_unmatched = torch.zeros(len(unmatched_logits), dtype=torch.long, device=pred_boxes.device)
#             class_loss_unmatched = ce_loss(unmatched_logits, class_targets_unmatched)
#             class_loss_matched += class_loss_unmatched
#
#         total_loss += coord_loss + class_loss_matched
#
#     return total_loss / batch_size
def hungarian_loss_with_matches(pred, target):
    """
    返回：损失值, 匹配信息
    """
    batch_size = pred.shape[0]
    total_loss = 0
    all_matches = []  # 保存每个batch的匹配结果
    smooth_l1_loss = nn.MSELoss()
    for b in range(batch_size):
        target_current = target[b]
        pred_current = pred[b]

        # 你的现有匹配逻辑...
        is_invalid = torch.all(torch.isclose(target_current[:, :2],
                                             torch.tensor([0.0, 0.0], device=pred.device)), dim=1)
        valid_mask = ~is_invalid
        valid_targets = target_current[valid_mask]

        if len(valid_targets) == 0:
            # 没有有效目标
            matches = {'row_idx': [], 'col_idx': []}
            loss = 0.1 * smooth_l1_loss(pred_current[:, :2], torch.zeros_like(pred_current[:, :2]))
        else:
            # 匈牙利匹配
            dist_matrix = torch.cdist(valid_targets[:, :2], pred_current[:, :2], p=2)
            row_idx, col_idx = linear_sum_assignment(dist_matrix.detach().cpu().numpy())

            # 计算边界框损失
            matched_pred = pred_current[col_idx]
            matched_target = valid_targets[row_idx]
            center_loss = smooth_l1_loss(matched_pred[:, :2], matched_target[:, :2])
            wh_loss = smooth_l1_loss(matched_pred[:, 2:], matched_target[:, 2:])
            loss = center_loss + wh_loss

            matches = {'row_idx': row_idx, 'col_idx': col_idx}

        total_loss += loss
        all_matches.append(matches)

    return total_loss / batch_size, all_matches


























#
# def hungarian_loss(pred, class_pred, target, target_num, confidence,num_max_targets=3):
#     """
#     处理1-3个目标的匈牙利匹配损失，无效目标标签为(0, 3.5)
#     pred: 模型输出 [B, num_queries=3, 2] （固定3个预测）
#     target: 标签 [B, 6] （格式：[x1,y1,x2,y2,x3,y3]，无效目标为(0,3.5)）
#     """
#     batch_size = pred.shape[0]
#     total_loss = torch.tensor(0.0, device=pred.device)
#     total_confidence_loss = torch.tensor(0.0, device=pred.device)
#     target_reshaped = target.view(batch_size, num_max_targets, 2)  # [B,3,2]
#     smooth_l1_loss = nn.SmoothL1Loss()
#     MSELoss = nn.MSELoss()
#     bce_loss = nn.BCELoss()  # 二分类交叉熵用于置信度
#     # 定义无效目标的坐标 (0, 0)，用于匹配判断
#     invalid_coords = torch.tensor([0.0, 0.0], device=pred.device)
#
#     for b in range(batch_size):
#         try:
#             # 1. 筛选当前样本的有效目标（排除等于(0,3.5)的无效目标）
#             # 逐个目标判断：x==0且y==3.5 视为无效
#             is_invalid = torch.all(torch.isclose(target_reshaped[b], invalid_coords), dim=1)  # [3]，True表示无效
#             valid_mask = ~is_invalid  # [3]，True表示有效目标
#             valid_targets = target_reshaped[b][valid_mask]  # [n,2]，n=1,2,3（数据保证n≥1）
#             num_people = len(valid_targets)
#             conf_current = confidence[b]  # [3, 1]
#             # 2. 预测取全部3个查询（与最大目标数一致）
#             pred_current = pred[b]  # [3,2]
#
#             # 3. 构建距离矩阵 [n, 3]（有效目标数 × 预测数）
#             dist_matrix = torch.cdist(valid_targets, pred_current, p=2)
#
#             # 检查距离矩阵是否有无效值
#             if torch.isnan(dist_matrix).any() or torch.isinf(dist_matrix).any():
#                 print(f"⚠️ 距离矩阵包含无效值 - 样本 {b}")
#                 print(f"   valid_targets: {valid_targets}")
#                 print(f"   pred_current: {pred_current}")
#                 print(f"   dist_matrix: {dist_matrix}")
#                 continue
#
#             # 4. 匈牙利算法匹配（n个目标匹配n个预测）
#             dist_matrix_np = dist_matrix.detach().cpu().numpy()
#
#             # 检查numpy数组是否有无效值
#             if np.isnan(dist_matrix_np).any() or np.isinf(dist_matrix_np).any():
#                 print(f"⚠️ NumPy距离矩阵包含无效值 - 样本 {b}")
#                 print(f"   dist_matrix_np: {dist_matrix_np}")
#                 continue
#
#             row_idx, col_idx = linear_sum_assignment(dist_matrix_np)
#             target_conf = torch.zeros(3, 1, device=pred.device)
#             target_conf[col_idx] = 1.0  # 匹配的query置信度=1
#             # 5. 计算匹配对的损失（L1损失更稳定）
#             matched_pred = pred_current[col_idx]  # [n,2]
#             loss = MSELoss(matched_pred, valid_targets)
#             # loss = smooth_l1_loss(matched_pred, valid_targets)
#             conf_loss = bce_loss(conf_current, target_conf)
#             # 6. 对未匹配的预测（3-n个）加惩罚：让它们接近无效标签(0,3.5)
#             if 3 > len(col_idx):
#                 unmatched_mask = torch.ones(3, dtype=torch.bool, device=pred.device)
#                 unmatched_mask[col_idx] = False  # 标记未匹配的预测
#                 unmatched_pred = pred_current[unmatched_mask]  # [3-n,2]
#                 # 惩罚未匹配预测偏离(0,0)，权重可调整
#                 # loss += 0.1 * smooth_l1_loss(unmatched_pred, invalid_coords.repeat(unmatched_pred.shape[0], 1))
#                 loss += 0.1 * MSELoss(unmatched_pred, invalid_coords.repeat(unmatched_pred.shape[0], 1))
#             total_loss += loss
#             total_confidence_loss += conf_loss
#
#         except ValueError as e:
#             if "matrix contains invalid numeric entries" in str(e):
#                 print(f"❌ 匈牙利匹配错误 - 样本 {b}: {e}")
#                 print(f"   距离矩阵形状: {dist_matrix.shape}")
#                 print(f"   距离矩阵值: {dist_matrix}")
#                 print(f"   有效目标数: {len(valid_targets)}")
#                 print(f"   有效目标: {valid_targets}")
#                 print(f"   预测值: {pred_current}")
#             continue
#
#         except Exception as e:
#             print(f"❌ 未知错误 - 样本 {b}: {e}")
#             print(f"   有效目标数: {len(valid_targets)}")
#             print(f"   预测形状: {pred_current.shape}")
#             continue
#
#     if total_loss == 0 and batch_size > 0:
#         print("⚠️ 所有样本处理失败，返回零损失")
#
#     return (total_loss / batch_size if batch_size > 0 else total_loss) ,(total_confidence_loss / batch_size if batch_size > 0 else total_confidence_loss)