import torch


def boxes_iou(label, pred, r=1.0, resolution=100, smooth=1e-6):
    # 保存原始batch_size用于后续形状恢复
    original_batch_size = label.shape[0]

    # 处理输入形状，统一转换为(batch_size, time, 3, 2)
    # 支持:
    # - (batch_size, 3, 2): 原始格式，视为time=1的特殊情况
    # - (batch_size, time, 6): 新格式，将6拆分为3×2

    # 处理label
    if label.dim() == 3 and label.shape[-1] == 6:
        # 对于(batch, time, 6)格式，拆分6为3×2
        label = label.view(label.shape[0], label.shape[1], 3, 2)  # 变为(batch, time, 3, 2)
    elif label.dim() == 3 and label.shape[1] == 3 and label.shape[2] == 2:
        # 对于原始格式，增加time维度(设为1)
        label = label.unsqueeze(1)  # 变为(batch, 1, 3, 2)
    else:
        raise ValueError(f"不支持的label形状: {label.shape}，请使用(batch, 3, 2)或(batch, time, 6)")

    # 处理pred，与label保持一致
    if pred.dim() == 3 and pred.shape[-1] == 6:
        pred = pred.view(pred.shape[0], pred.shape[1], 3, 2)
    elif pred.dim() == 3 and pred.shape[1] == 3 and pred.shape[2] == 2:
        pred = pred.unsqueeze(1)
    else:
        raise ValueError(f"不支持的pred形状: {pred.shape}，请使用(batch, 3, 2)或(batch, time, 6)")

    # 确保label和pred处理后的形状一致
    assert label.shape == pred.shape, f"处理后label和pred形状不一致: {label.shape} vs {pred.shape}"

    # 获取处理后的形状参数
    batch_size, time_steps = label.shape[0], label.shape[1]

    # 场地尺寸（全程浮点运算）
    court_size = 5.0 * resolution
    r_pixels = r * resolution / 2.0  # 保持浮点

    # 创建网格坐标（可微）
    x = torch.linspace(0.0, court_size - 1.0, int(court_size), device=label.device)
    y = torch.linspace(0.0, court_size - 1.0, int(court_size), device=label.device)
    xx, yy = torch.meshgrid(x, y, indexing='xy')  # 形状：(res*5, res*5)
    # 扩展维度以适应(batch, time, 3, ...)的形状
    xx = xx.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1, 1, 1, res*5, res*5)
    yy = yy.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1, 1, 1, res*5, res*5)

    # 转换GT坐标到网格（可微）
    label_x = (label[..., 0] + 2.5) * resolution  # (batch, time, 3)
    label_y = (label[..., 1] + 2.5) * resolution  # (batch, time, 3)
    label_x = label_x.unsqueeze(-1).unsqueeze(-1)  # (batch, time, 3, 1, 1)
    label_y = label_y.unsqueeze(-1).unsqueeze(-1)  # (batch, time, 3, 1, 1)

    # 关键修复：用sigmoid实现软掩码（替代不可微的布尔运算）
    # 软判断x在[left, right]范围内
    def soft_in_range(x, left, right, k=10.0):
        # 左边：x >= left → 用sigmoid近似
        left_prob = torch.sigmoid(k * (x - left))
        # 右边：x <= right → 用sigmoid近似
        right_prob = torch.sigmoid(k * (right - x))
        # 同时满足则概率接近1，否则接近0
        return left_prob * right_prob

    # 计算GT软掩码（可微）
    gt_masks = soft_in_range(xx, label_x - r_pixels, label_x + r_pixels) * \
               soft_in_range(yy, label_y - r_pixels, label_y + r_pixels)
    # 合并3个点的掩码（用max替代any，可微）
    court = gt_masks.max(dim=2).values  # (batch, time, res*5, res*5)

    # 计算Pred软掩码（可微）
    pred_x = (pred[..., 0] + 2.5) * resolution  # (batch, time, 3)
    pred_y = (pred[..., 1] + 2.5) * resolution  # (batch, time, 3)
    pred_x = pred_x.unsqueeze(-1).unsqueeze(-1)  # (batch, time, 3, 1, 1)
    pred_y = pred_y.unsqueeze(-1).unsqueeze(-1)  # (batch, time, 3, 1, 1)

    pred_masks = soft_in_range(xx, pred_x - r_pixels, pred_x + r_pixels) * \
                 soft_in_range(yy, pred_y - r_pixels, pred_y + r_pixels)
    # 合并3个点的掩码
    grid_pred = pred_masks.max(dim=2).values  # (batch, time, res*5, res*5)

    # 计算IoU（完全可微）
    intersection = (court * grid_pred).sum(dim=(2, 3))  # (batch, time)
    union = (court + grid_pred - court * grid_pred).sum(dim=(2, 3))  # (batch, time)
    iou = intersection / (union + smooth)  # (batch, time)

    # 如果原始输入是(batch, 3, 2)格式（已转换为(batch, 1, 3, 2)），去掉time维度
    if original_batch_size == batch_size and time_steps == 1:
        return iou.squeeze(1)  # 变回(batch,)

    return iou  # 保持(batch, time)形状


# if __name__ == "__main__":
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#     # 测试原始格式 (batch_size, 3, 2)
#     gt1 = torch.tensor([[[0.2, 0.2], [0.5, 0.5], [0.8, 0.8]]],
#                        device=device, requires_grad=True, dtype=torch.float32)
#     pred1 = torch.tensor([[[1.1, 1.1], [1.4, 1.4], [0.9, 0.80]]],
#                          device=device, requires_grad=True, dtype=torch.float32)
#
#     iou1 = grid_iou(gt1, pred1, r=0.3)
#     print("原始格式IoU:", iou1.shape, iou1)
#     iou1.mean().backward()
#     print("原始格式GT梯度形状:", gt1.grad.shape)
#     print("原始格式Pred梯度梯度形状:", pred1.grad.shape)
#
#     # 测试新格式 (batch_size, time, 6)
#     # 6个值对应3个点的(x1,y1,x2,y2,x3,y3)
#     base_gt = [
#         [0.2, 0.2, 0.5, 0.5, 0.8, 0.8],  # time step 1 基础模式
#         [0.3, 0.3, 0.6, 0.6, 0.9, 0.9]  # time step 2 基础模式
#     ]
#
#     base_pred = [
#         [1.1, 1.1, 1.4, 1.4, 0.9, 0.80],  # time step 1 基础模式
#         [1.2, 1.2, 1.5, 1.5, 1.0, 0.90]  # time step 2 基础模式
#     ]
#     batch_size = 3
#     timesteps = 15
#
#     # 初始化存储列表
#     gt_list = []
#     pred_list = []
#
#     for b in range(batch_size):
#         # 为每个样本生成15个时间步
#         sample_gt = []
#         sample_pred = []
#
#         for t in range(timesteps):
#             # 循环使用基础模式并添加微小变化，使数据更真实
#             base_idx = t % 2  # 0或1，循环使用两个基础模式
#             # 添加随批次和时间步变化的微小偏移，增加数据多样性
#             offset = 0.01 * (b - 1) + 0.005 * t
#
#             # 生成当前时间步的gt数据
#             current_gt = [
#                 base_gt[base_idx][0] + offset,
#                 base_gt[base_idx][1] + offset,
#                 base_gt[base_idx][2] + offset,
#                 base_gt[base_idx][3] + offset,
#                 base_gt[base_idx][4] + offset,
#                 base_gt[base_idx][5] + offset
#             ]
#
#             # 生成当前时间步的pred数据（与gt有一定差异）
#             current_pred = [
#                 base_pred[base_idx][0] + offset * 0.8,
#                 base_pred[base_idx][1] + offset * 0.8,
#                 base_pred[base_idx][2] + offset * 0.8,
#                 base_pred[base_idx][3] + offset * 0.8,
#                 base_pred[base_idx][4] + offset * 0.8,
#                 base_pred[base_idx][5] + offset * 0.8
#             ]
#
#             sample_gt.append(current_gt)
#             sample_pred.append(current_pred)
#
#         gt_list.append(sample_gt)
#         pred_list.append(sample_pred)
#
#     # 转换为PyTorch张量
#     gt2 = torch.tensor(gt_list, device=device, requires_grad=True, dtype=torch.float32)
#     pred2 = torch.tensor(pred_list, device=device, requires_grad=True, dtype=torch.float32)
#
#     # 打印形状验证
#     print(f"gt2形状: {gt2.shape}  (batch_size={gt2.shape[0]}, timesteps={gt2.shape[1]})")
#     print(f"pred2形状: {pred2.shape}  (batch_size={pred2.shape[0]}, timesteps={pred2.shape[1]})")
#     iou2 = grid_iou(gt2, pred2, r=0.3)
#     print("\n新格式IoU:", iou2.shape, iou2)
#     # 计算每个batch的loss并取平均
#     loss = (1 - iou2).mean()  # 先计算1-IoU作为损失，再求平均
#     loss.backward()
#     print("新格式GT梯度形状:", gt2.grad.shape)
#     print("新格式Pred梯度形状:", pred2.grad.shape)
#     print("损失值:", loss.item())
