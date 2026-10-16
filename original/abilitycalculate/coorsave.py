import matplotlib.pyplot as plt
import os
import torch
# from original.my_dataset import process_temporal_labels
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from original.my_dataset import add_wh_to_targets
import  numpy as np
import torch.nn.functional as F
def test_and_write_results_simple(model, test_loader, device, save_dir='test_results'):
    """测试并保存结果txt（带调试信息）"""
    model.eval()

    os.makedirs(save_dir, exist_ok=True)
    result_file = open(os.path.join(save_dir, 'results.txt'), 'w')

    total_samples = 0
    total_loss = 0

    with torch.no_grad():
        for batch_idx, (radar_data, coordinates) in enumerate(tqdm(test_loader, desc='Testing')):
            radar_data = radar_data.to(device)
            pred = model(radar_data)  # [B, 50, 4]

            # 处理标签
            B, T, C = coordinates.shape
            target_coords_with_wh = []

            # 调试信息：检查输入数据
            print(f"\n=== Batch {batch_idx} ===")
            print(f"雷达数据形状: {radar_data.shape}")
            print(f"原始坐标形状: {coordinates.shape}")
            print(f"原始坐标范围 - 最小: {coordinates.min():.3f}, 最大: {coordinates.max():.3f}")

            for i in range(B):
                avg_coords = coordinates[i].mean(dim=0)
                avg_coords_reshaped = avg_coords.reshape(-1, 2)
                coords_with_wh = add_wh_to_targets(avg_coords_reshaped, normalize_coords=False)
                target_coords_with_wh.append(coords_with_wh)

                # 只打印第一个样本的调试信息
                if i == 0 and batch_idx == 0:
                    print(f"样本0平均前坐标 (第一个时间步): {coordinates[i][0]}")
                    print(f"样本0平均后坐标: {avg_coords_reshaped}")
                    print(f"样本0添加宽高后: {coords_with_wh}")

            target_coords_with_wh = torch.stack(target_coords_with_wh).to(device)  # [B, 3, 4]

            # 调试信息：检查目标标签
            print(f"目标标签形状: {target_coords_with_wh.shape}")
            print(f"目标标签范围 - X: [{target_coords_with_wh[..., 0].min():.3f}, {target_coords_with_wh[..., 0].max():.3f}]")
            print(f"目标标签范围 - Y: [{target_coords_with_wh[..., 1].min():.3f}, {target_coords_with_wh[..., 1].max():.3f}]")

            # 调试信息：检查预测输出
            print(f"预测输出形状: {pred.shape}")
            print(f"预测输出范围 - X: [{pred[..., 0].min():.3f}, {pred[..., 0].max():.3f}]")
            print(f"预测输出范围 - Y: [{pred[..., 1].min():.3f}, {pred[..., 1].max():.3f}]")

            for sample_idx in range(B):
                # 对每个样本单独计算损失
                pred_sample = pred[sample_idx:sample_idx + 1]  # [1, 50, 4]
                target_sample = target_coords_with_wh[sample_idx:sample_idx + 1]  # [1, 3, 4]

                sample_loss = hungarian_loss(pred_sample, target_sample)

                # 调试信息：检查样本损失
                if sample_idx == 0 and batch_idx == 0:
                    print(f"样本0损失计算:")
                    print(f"  预测样本形状: {pred_sample.shape}")
                    print(f"  目标样本形状: {target_sample.shape}")
                    print(f"  样本损失: {sample_loss.item():.6f}")

                pred_current = pred[sample_idx]  # [50, 4]
                target_current = target_coords_with_wh[sample_idx]  # [3, 4]

                # 匈牙利匹配（用于显示结果）
                is_invalid = torch.all(torch.isclose(target_current[:, :2],
                                                     torch.tensor([0.0, 0.0], device=pred.device)), dim=1)
                valid_mask = ~is_invalid
                valid_targets = target_current[valid_mask]  # [n, 4]

                if len(valid_targets) > 0:
                    pred_points = pred_current[:, :2]  # [50, 2]
                    dist_matrix = torch.cdist(valid_targets[:, :2], pred_points, p=2)  # [n, 50]
                    row_idx, col_idx = linear_sum_assignment(dist_matrix.detach().cpu().numpy())

                    # 创建匹配后的预测数组
                    matched_pred = torch.zeros_like(valid_targets)  # [n, 4]
                    matched_pred[row_idx] = pred_current[col_idx]

                    # 构建匹配信息
                    match_info = ""
                    center_errors = []
                    for t_idx, p_idx in zip(row_idx, col_idx):
                        center_diff = torch.sqrt(((pred_current[p_idx, :2] - valid_targets[t_idx, :2]) ** 2).sum())
                        center_errors.append(center_diff.item())
                        match_info += f"T{t_idx}->P{p_idx}({center_diff.item():.4f}) "

                    # 调试信息：检查匹配误差
                    if sample_idx == 0 and batch_idx == 0:
                        print(f"样本0匹配信息:")
                        print(f"  有效目标数: {len(valid_targets)}")
                        print(f"  匹配对: {list(zip(row_idx, col_idx))}")
                        print(f"  中心点误差: {center_errors}")
                        print(f"  平均中心点误差: {np.mean(center_errors):.4f}")
                else:
                    matched_pred = torch.zeros((0, 4), device=pred.device)
                    match_info = "No targets"

                # 写入结果
                sample_id = batch_idx * B + sample_idx
                result_file.write(f"=== Sample {sample_id} ===\n")
                result_file.write(f"样本损失: {sample_loss.item():.6f}\n")

                # 真实标签
                result_file.write("真实标签:\n")
                for i, target in enumerate(valid_targets):
                    x, y, w, h = target.cpu().numpy()
                    result_file.write(f"  目标{i}: ({x:.4f}, {y:.4f}, {w:.4f}, {h:.4f})\n")

                # 匹配预测
                result_file.write("匹配预测:\n")
                for i, pred_val in enumerate(matched_pred):
                    x, y, w, h = pred_val.cpu().numpy()
                    result_file.write(f"  预测{i}: ({x:.4f}, {y:.4f}, {w:.4f}, {h:.4f})\n")

                # 匹配关系
                result_file.write(f"匹配关系: {match_info}\n\n")

                total_samples += 1
                total_loss += sample_loss.item()

            # 只打印前2个batch的调试信息
            if batch_idx >= 1:
                print("... 后续batch省略调试信息 ...")
                continue

    # 总体统计
    if total_samples > 0:
        avg_loss = total_loss / total_samples
        result_file.write("=" * 50 + "\n")
        result_file.write(f"总体统计:\n")
        result_file.write(f"总样本数: {total_samples}\n")
        result_file.write(f"平均损失: {avg_loss:.6f}\n")

    result_file.close()
    print(f"\n=== 最终测试结果 ===")
    print(f"测试集平均损失: {avg_loss:.6f}")
    print(f"结果已保存到 {save_dir}/results.txt")

def test_and_save_results_detr_style(model, test_loader, device, save_dir='test_results_detr', confidence_threshold=0.5):
    """DETR风格的测试函数"""
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    total_samples = 0
    total_loss = 0
    ce_loss_sum = 0
    bbox_loss_sum = 0

    # 评估指标
    total_objects = 0
    detected_objects = 0
    false_positives = 0

    with torch.no_grad():
        for batch_idx, (radar_data, coordinates) in enumerate(tqdm(test_loader, desc='Testing')):
            radar_data = radar_data.to(device)

            # 准备目标数据
            B, T, C = coordinates.shape
            target_coords_with_wh = []
            for i in range(B):
                avg_coords = coordinates[i].mean(dim=0)
                avg_coords_reshaped = avg_coords.reshape(-1, 2)
                coords_with_wh = add_wh_to_targets(avg_coords_reshaped, normalize_coords=False)
                target_coords_with_wh.append(coords_with_wh)
            target_coords_with_wh = torch.stack(target_coords_with_wh).to(device)

            # 前向传播 - 现在输出边界框和分类logits
            pred_boxes, pred_logits = model(radar_data)

            # 计算DETR风格损失
            total_loss_val, ce_loss_val, bbox_loss_val = detr_style_loss(
                pred_boxes, pred_logits, target_coords_with_wh
            )

            # 使用分类概率过滤（关键变化！）
            pred_probs = F.softmax(pred_logits, dim=-1)  # [B, num_queries, 2]
            human_probs = pred_probs[..., 0]  # 人体类别的概率 [B, num_queries]

            # 处理每个样本
            for b in range(B):
                # 过滤低置信度预测 - 使用人体类别的概率
                keep_mask = human_probs[b] > confidence_threshold
                filtered_boxes = pred_boxes[b][keep_mask]  # [N, 4]
                filtered_scores = human_probs[b][keep_mask]  # [N]
                filtered_logits = pred_logits[b][keep_mask]  # [N, 2] - 用于调试

                # 统计有效目标数量
                target_current = target_coords_with_wh[b]
                is_invalid = torch.all(torch.isclose(target_current[:, :2],
                                                   torch.tensor([0.0, 0.0], device=device)), dim=1)
                valid_mask = ~is_invalid
                valid_targets = target_current[valid_mask]
                num_gt = len(valid_targets)

                # 更新评估指标
                total_objects += num_gt
                num_detected = len(filtered_boxes)
                detected_objects += min(num_detected, num_gt)
                false_positives += max(0, num_detected - num_gt)

                # 保存样本结果到txt
                sample_id = batch_idx * B + b
                save_sample_result_detr_style(
                    sample_id,
                    filtered_boxes.cpu().numpy(),
                    filtered_scores.cpu().numpy(),
                    filtered_logits.cpu().numpy(),  # 保存logits用于调试
                    valid_targets.cpu().numpy(),
                    total_loss_val.item(),
                    ce_loss_val.item(),
                    bbox_loss_val.item(),
                    save_dir
                )

                total_samples += 1
                total_loss += total_loss_val.item()
                ce_loss_sum += ce_loss_val.item()
                bbox_loss_sum += bbox_loss_val.item()

    # 计算总体指标
    precision = detected_objects / (detected_objects + false_positives) if (detected_objects + false_positives) > 0 else 0
    recall = detected_objects / total_objects if total_objects > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    avg_total_loss = total_loss / total_samples if total_samples > 0 else 0
    avg_ce_loss = ce_loss_sum / total_samples if total_samples > 0 else 0
    avg_bbox_loss = bbox_loss_sum / total_samples if total_samples > 0 else 0

    # 保存总体统计
    save_overall_stats_detr_style(
        avg_total_loss, avg_ce_loss, avg_bbox_loss,
        precision, recall, f1_score,
        detected_objects, total_objects, false_positives,
        total_samples, save_dir
    )

    print(f"\n=== DETR风格测试完成 ===")
    print(f"平均损失 - 总: {avg_total_loss:.6f}, 分类: {avg_ce_loss:.6f}, 边界框: {avg_bbox_loss:.6f}")
    print(f"检测指标 - 精确率: {precision:.4f}, 召回率: {recall:.4f}, F1: {f1_score:.4f}")
    print(f"检测统计 - 检测到: {detected_objects}/{total_objects}, 误检: {false_positives}")
    print(f"结果保存至: {save_dir}")

    return {
        'total_loss': avg_total_loss,
        'ce_loss': avg_ce_loss,
        'bbox_loss': avg_bbox_loss,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'detected_objects': detected_objects,
        'total_objects': total_objects,
        'false_positives': false_positives
    }


def save_sample_result_detr_style(sample_id, pred_boxes, pred_scores, pred_logits, gt_boxes,
                                  total_loss, ce_loss, bbox_loss, save_dir):
    """保存单个样本的详细结果（DETR风格）"""
    result_file = os.path.join(save_dir, f'sample_{sample_id:06d}.txt')

    with open(result_file, 'w') as f:
        f.write(f"=== 样本 {sample_id} ===\n")
        f.write(f"总损失: {total_loss:.6f}\n")
        f.write(f"分类损失: {ce_loss:.6f}\n")
        f.write(f"边界框损失: {bbox_loss:.6f}\n\n")

        f.write("真实目标:\n")
        if len(gt_boxes) == 0:
            f.write("  无真实目标\n")
        else:
            for i, box in enumerate(gt_boxes):
                x, y, w, h = box
                f.write(f"  目标{i}: ({x:.4f}, {y:.4f}, {w:.4f}, {h:.4f})\n")

        f.write("\n检测结果:\n")
        if len(pred_boxes) == 0:
            f.write("  未检测到目标\n")
        else:
            for i, (box, score, logit) in enumerate(zip(pred_boxes, pred_scores, pred_logits)):
                x, y, w, h = box
                background_prob = 1.0 - score  # 背景类别的概率
                f.write(f"  检测{i}: ({x:.4f}, {y:.4f}, {w:.4f}, {h:.4f}) ")
                f.write(f"人体概率: {score:.4f}, 背景概率: {background_prob:.4f}\n")
                f.write(f"        Logits: 人体={logit[0]:.4f}, 背景={logit[1]:.4f}\n")


def save_overall_stats_detr_style(avg_total_loss, avg_ce_loss, avg_bbox_loss,
                                  precision, recall, f1_score,
                                  detected_objects, total_objects, false_positives,
                                  total_samples, save_dir):
    """保存总体统计信息（DETR风格）"""
    stats_file = os.path.join(save_dir, 'overall_stats.txt')

    with open(stats_file, 'w') as f:
        f.write("=== DETR风格总体测试统计 ===\n\n")
        f.write(f"总样本数: {total_samples}\n\n")

        f.write("平均损失:\n")
        f.write(f"  总损失: {avg_total_loss:.6f}\n")
        f.write(f"  分类损失: {avg_ce_loss:.6f}\n")
        f.write(f"  边界框损失: {avg_bbox_loss:.6f}\n\n")

        f.write("检测性能:\n")
        f.write(f"  精确率: {precision:.4f}\n")
        f.write(f"  召回率: {recall:.4f}\n")
        f.write(f"  F1分数: {f1_score:.4f}\n\n")

        f.write("检测统计:\n")
        f.write(f"  总真实目标: {total_objects}\n")
        f.write(f"  正确检测: {detected_objects}\n")
        f.write(f"  误检数量: {false_positives}\n")
        f.write(f"  检测率: {detected_objects / total_objects:.4f}\n")
def save_sample_result(sample_id, pred_boxes, pred_scores, gt_boxes,
                       total_loss, bbox_loss, conf_loss, save_dir):
    """保存单个样本的详细结果"""
    result_file = os.path.join(save_dir, f'sample_{sample_id:06d}.txt')

    with open(result_file, 'w') as f:
        f.write(f"=== 样本 {sample_id} ===\n")
        f.write(f"总损失: {total_loss:.6f}\n")
        f.write(f"边界框损失: {bbox_loss:.6f}\n")
        f.write(f"置信度损失: {conf_loss:.6f}\n\n")

        f.write("真实目标:\n")
        if len(gt_boxes) == 0:
            f.write("  无真实目标\n")
        else:
            for i, box in enumerate(gt_boxes):
                x, y, w, h = box
                f.write(f"  目标{i}: ({x:.4f}, {y:.4f}, {w:.4f}, {h:.4f})\n")

        f.write("\n检测结果:\n")
        if len(pred_boxes) == 0:
            f.write("  未检测到目标\n")
        else:
            for i, (box, score) in enumerate(zip(pred_boxes, pred_scores)):
                x, y, w, h = box
                f.write(f"  检测{i}: ({x:.4f}, {y:.4f}, {w:.4f}, {h:.4f}) 置信度: {score:.4f}\n")


def save_overall_stats(avg_total_loss, avg_bbox_loss, avg_conf_loss,
                       precision, recall, f1_score,
                       detected_objects, total_objects, false_positives,
                       total_samples, save_dir):
    """保存总体统计信息"""
    stats_file = os.path.join(save_dir, 'overall_stats.txt')

    with open(stats_file, 'w') as f:
        f.write("=== 总体测试统计 ===\n\n")
        f.write(f"总样本数: {total_samples}\n\n")

        f.write("平均损失:\n")
        f.write(f"  总损失: {avg_total_loss:.6f}\n")
        f.write(f"  边界框损失: {avg_bbox_loss:.6f}\n")
        f.write(f"  置信度损失: {avg_conf_loss:.6f}\n\n")

        f.write("检测性能:\n")
        f.write(f"  精确率: {precision:.4f}\n")
        f.write(f"  召回率: {recall:.4f}\n")
        f.write(f"  F1分数: {f1_score:.4f}\n\n")

        f.write("检测统计:\n")
        f.write(f"  总真实目标: {total_objects}\n")
        f.write(f"  正确检测: {detected_objects}\n")
        f.write(f"  误检数量: {false_positives}\n")
        f.write(f"  检测率: {detected_objects / total_objects:.4f}\n")



def compute_confidence_loss_single(pred_confidence, matches):
    """
    pred_confidence: [num_queries, 1] 当前batch的置信度
    matches: 从匈牙利匹配得到的匹配信息
    """
    num_queries = pred_confidence.shape[0]

    # 创建目标标签
    target_labels = torch.zeros(num_queries, device=pred_confidence.device)

    # 匹配到的查询设为1，未匹配的设为0
    if 'col_idx' in matches and len(matches['col_idx']) > 0:
        target_labels[matches['col_idx']] = 1.0

    # 计算二分类交叉熵损失
    conf_loss = F.binary_cross_entropy_with_logits(
        pred_confidence.squeeze(-1),  # [num_queries]
        target_labels  # [num_queries]
    )

    return conf_loss

def detr_style_loss(pred_boxes, pred_logits, target_boxes):
    """
    DETR风格的损失函数
    pred_boxes: [B, num_queries, 4]
    pred_logits: [B, num_queries, 2]  # 2个类别：人体(0) vs 背景(1)
    target_boxes: [B, num_targets, 4]
    """
    batch_size = pred_boxes.shape[0]
    total_loss = 0
    total_ce_loss = 0
    total_bbox_loss = 0

    for b in range(batch_size):
        # 当前batch的数据
        target_current = target_boxes[b]  # [num_targets, 4]
        pred_boxes_current = pred_boxes[b]  # [num_queries, 4]
        pred_logits_current = pred_logits[b]  # [num_queries, 2]

        # 1. 检测无效目标
        is_invalid = torch.all(torch.isclose(target_current[:, :2],
                                             torch.tensor([0.0, 0.0], device=pred_boxes.device)), dim=1)
        valid_mask = ~is_invalid
        valid_targets = target_current[valid_mask]  # [n, 4]

        if len(valid_targets) == 0:
            # 没有有效目标，跳过这个batch
            continue

        # 2. 计算代价矩阵（同时考虑分类和位置）
        # 分类代价：取"人体"类别的概率（我们希望匹配到的查询有高概率）
        class_probs = F.softmax(pred_logits_current, dim=-1)  # [num_queries, 2]
        human_probs = class_probs[:, 0]  # [num_queries] - 人体类别的概率
        cost_class = -human_probs.unsqueeze(0).repeat(len(valid_targets), 1)  # [n, num_queries]

        # 边界框代价
        cost_bbox = torch.cdist(valid_targets[:, :2], pred_boxes_current[:, :2], p=2)  # [n, num_queries]

        # 总代价矩阵
        cost_matrix = cost_class + cost_bbox  # [n, num_queries]

        # 3. 匈牙利匹配
        row_idx, col_idx = linear_sum_assignment(cost_matrix.detach().cpu().numpy())

        # 4. 计算损失（只对匹配到的查询）
        # 分类损失
        matched_logits = pred_logits_current[col_idx]  # [n, 2]
        # 所有匹配到的目标都是"人体"类别（类别0）
        target_labels = torch.zeros(len(col_idx), dtype=torch.long, device=pred_logits.device)  # [n] 全为0
        loss_ce = F.cross_entropy(matched_logits, target_labels)

        # 边界框损失
        matched_pred_boxes = pred_boxes_current[col_idx]  # [n, 4]
        matched_target_boxes = valid_targets[row_idx]  # [n, 4]
        loss_bbox = F.mse_loss(matched_pred_boxes, matched_target_boxes)

        total_ce_loss += loss_ce
        total_bbox_loss += loss_bbox
        total_loss += loss_ce + loss_bbox

    if batch_size > 0:
        avg_loss = total_loss / batch_size
        avg_ce_loss = total_ce_loss / batch_size
        avg_bbox_loss = total_bbox_loss / batch_size
    else:
        avg_loss = torch.tensor(0.0)
        avg_ce_loss = torch.tensor(0.0)
        avg_bbox_loss = torch.tensor(0.0)

    return avg_loss, avg_ce_loss, avg_bbox_loss