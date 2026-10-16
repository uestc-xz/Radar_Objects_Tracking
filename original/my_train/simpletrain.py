from tqdm import tqdm
from torch import torch
from original.my_dataset import add_wh_to_targets


def validate_with_temporal_avg(model, dataloader, criterion, device):
    """验证函数（带时间维平均）"""
    model.eval()
    total_loss = 0
    total_class_loss = 0
    total_coord_loss = 0  # 修正：改为total_coord_loss

    with torch.no_grad():
        for batch_idx, (radar_data, coordinates) in enumerate(dataloader):
            radar_data = radar_data.to(device)

            # 对坐标取时间平均，然后添加宽高
            B, T, C = coordinates.shape
            target_coords_with_wh = []

            for i in range(B):
                avg_coords = coordinates[i].mean(dim=0)
                avg_coords_reshaped = avg_coords.reshape(-1, 2)
                coords_with_wh = add_wh_to_targets(avg_coords_reshaped, normalize_coords=False)
                target_coords_with_wh.append(coords_with_wh)

            target_coords_with_wh = torch.stack(target_coords_with_wh).to(device)

            coorout, conf = model(radar_data)

            # 计算损失 - 修正：接收两个返回值
            batch_coord_loss, batch_class_loss = criterion(conf, coorout, target_coords_with_wh,is_train=False)
            total_batch_loss = batch_coord_loss + batch_class_loss

            # 统计 - 修正：使用batch级别的损失
            total_loss += total_batch_loss.item()
            total_class_loss += batch_class_loss.item()  # 使用batch_class_loss
            total_coord_loss += batch_coord_loss.item()  # 使用batch_coord_loss

    num_batches = len(dataloader)
    avg_loss = total_loss / num_batches
    avg_class_loss = total_class_loss / num_batches
    avg_coord_loss = total_coord_loss / num_batches

    return avg_loss, avg_class_loss, avg_coord_loss  # 修正：返回三个值

def train_epoch_with_temporal_avg(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0
    total_class_loss = 0
    total_coord_loss = 0

    # 解码器层的权重 - 设置为[0,0,1.0]时应该与之前完全一致
    layer_weights = [0.2, 0.2, 0.6]  # 只使用最后一层

    pbar = tqdm(dataloader, desc=f'Epoch {epoch} Training')

    for batch_idx, (radar_data, coordinates) in enumerate(pbar):
        radar_data = radar_data.to(device)

        # 准备目标数据（保持不变）
        B, T, C = coordinates.shape
        target_coords_with_wh = []
        for i in range(B):
            avg_coords = coordinates[i].mean(dim=0)
            avg_coords_reshaped = avg_coords.reshape(-1, 2)
            coords_with_wh = add_wh_to_targets(avg_coords_reshaped, normalize_coords=False)
            target_coords_with_wh.append(coords_with_wh)
        target_coords_with_wh = torch.stack(target_coords_with_wh).to(device)

        optimizer.zero_grad()

        # 前向传播 - 现在返回所有层
        all_outputs = model(radar_data)  # 返回列表 [(out, conf), (out, conf), (out, conf)]

        # 计算加权损失
        total_batch_loss = 0
        total_batch_coord_loss = 0
        total_batch_class_loss = 0

        for layer_idx, (layer_out, layer_conf) in enumerate(all_outputs):
            # 计算当前层损失
            layer_coord_loss, layer_class_loss = criterion(
                layer_conf,
                layer_out,
                target_coords_with_wh,
                epoch,
                is_train=True
            )

            # 应用层权重
            weight = layer_weights[layer_idx]
            weighted_layer_loss = weight * (layer_coord_loss + layer_class_loss)
            weighted_coord_loss = weight * layer_coord_loss
            weighted_class_loss = weight * layer_class_loss

            total_batch_loss += weighted_layer_loss
            total_batch_coord_loss += weighted_coord_loss
            total_batch_class_loss += weighted_class_loss

        # 反向传播
        total_batch_loss.backward()
        optimizer.step()

        # 统计
        total_loss += total_batch_loss.item()
        total_class_loss += total_batch_class_loss.item()
        total_coord_loss += total_batch_coord_loss.item()

        pbar.set_postfix({
            'TotalLoss': f'{total_batch_loss.item():.6f}',
            'ClassLoss': f'{total_batch_class_loss.item():.6f}',
            'CoordLoss': f'{total_batch_coord_loss.item():.6f}',
        })

    avg_loss = total_loss / len(dataloader)
    avg_class_loss = total_class_loss / len(dataloader)
    avg_coord_loss = total_coord_loss / len(dataloader)

    return avg_loss, avg_class_loss, avg_coord_loss