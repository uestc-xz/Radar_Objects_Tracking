import torch.nn as nn
import torch.nn.functional as F
from original.my_dataset import add_wh_to_targets,load_fast_dataset1
from original.my_train import train_epoch_with_temporal_avg,validate_with_temporal_avg
from datetime import datetime
from original.lossfunc import hungarian_loss_with_classification
from original.abilitycalculate import count_parameters
from tqdm import tqdm
from torch.utils.data import DataLoader
from original.my_models import EncoderLayer,Decoder,SimpleFusionBackbone
import math
from thop import profile, clever_format
import torch
import os
os.environ['PYTHONUNBUFFERED'] = '1'
heatmap_size1 = 96


def test_and_write_result_with_tracking(model, test_loader, device, save_dir='test_results',
                                              confidence_threshold=0.5):
    """测试并保存结果txt（修正后的跟踪版本）"""
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    result_file = open(os.path.join(save_dir, 'results_with_tracking_fixed.txt'), 'w', encoding='utf-8')

    total_samples = 0

    # 跟踪变量
    prev_real_gt_count = None  # 真实有效的目标数
    jump_count = 0
    sequence_info = []

    with torch.no_grad():
        for batch_idx, (radar_data, coordinates) in enumerate(tqdm(test_loader, desc='Testing')):
            radar_data = radar_data.to(device)

            # 模型推理
            pred_boxes, pred_logits = model(radar_data)

            # 处理标签
            B, T, C = coordinates.shape
            target_coords_with_wh = []

            for i in range(B):
                avg_coords = coordinates[i].mean(dim=0)
                avg_coords_reshaped = avg_coords.reshape(-1, 2)
                coords_with_wh = add_wh_to_targets(avg_coords_reshaped, normalize_coords=False)
                target_coords_with_wh.append(coords_with_wh)

            target_coords_with_wh = torch.stack(target_coords_with_wh).to(device)

            for sample_idx in range(B):
                # 获取真实有效的目标数（不是全零的）
                target_current = target_coords_with_wh[sample_idx]
                is_invalid = torch.all(torch.isclose(target_current[:, :2],
                                                     torch.tensor([0.0, 0.0], device=device)), dim=1)
                valid_mask = ~is_invalid
                valid_targets = target_current[valid_mask]
                current_real_gt_count = len(valid_targets)  # 这才是真实有效的目标数

                # 原始的"槽位"数
                avg_coords = coordinates[sample_idx].mean(dim=0)
                avg_coords_reshaped = avg_coords.reshape(-1, 2)
                slot_count = len(avg_coords_reshaped)  # 这应该是3

                # 跟踪目标数变化
                if prev_real_gt_count is not None:
                    if current_real_gt_count != prev_real_gt_count:
                        jump_count += 1
                        print(f"时序跳变: 样本 {total_samples} ({prev_real_gt_count}人 -> {current_real_gt_count}人)")

                prev_real_gt_count = current_real_gt_count

                # 记录序列信息
                sequence_info.append({
                    'sample_id': total_samples,
                    'batch_idx': batch_idx,
                    'sample_idx_in_batch': sample_idx,
                    'slot_count': slot_count,  # 固定为3
                    'real_gt_count': current_real_gt_count,  # 真实有效的1/2/3
                    'coordinates_shape': coordinates[sample_idx].shape  # 原始形状
                })

                # 推理处理
                scores = F.softmax(pred_logits[sample_idx], dim=-1)[:, 1]
                keep = scores > confidence_threshold

                filtered_boxes = pred_boxes[sample_idx][keep]
                filtered_scores = scores[keep]

                # 写入结果
                sample_id = total_samples
                result_file.write("=" * 80 + "\n")
                result_file.write(f"样本 ID: {sample_id:04d}\n")
                result_file.write(f"批次索引: {batch_idx}, 批次内索引: {sample_idx}\n")
                result_file.write(f"坐标形状: {coordinates[sample_idx].shape}\n")
                result_file.write(f"槽位数: {slot_count}\n")
                result_file.write(f"真实有效目标数: {current_real_gt_count}\n")
                result_file.write(f"时序跳变次数: {jump_count}\n")
                result_file.write("-" * 80 + "\n")
                result_file.write(f"置信度阈值: {confidence_threshold}\n")
                result_file.write(f"检测到目标数: {len(filtered_boxes)}\n")
                result_file.write("\n")

                # 真实标签表格 - 只显示有效目标
                result_file.write("【真实标签（有效目标）】\n")
                result_file.write("-" * 40 + "\n")
                if len(valid_targets) > 0:
                    result_file.write("序号      X         Y         W         H      是否有效\n")
                    result_file.write("-" * 40 + "\n")
                    for i, target in enumerate(valid_targets):
                        x, y, w, h = target.cpu().numpy()
                        # 检查是否真的有效（非零）
                        is_valid = not (abs(x) < 0.01 and abs(y) < 0.01)
                        result_file.write(
                            f"{i:2d}    {x:8.4f}  {y:8.4f}  {w:8.4f}  {h:8.4f}  {'有效' if is_valid else '无效'}\n")
                else:
                    result_file.write("无有效真实目标\n")
                result_file.write("\n")

                # 显示所有槽位（包括无效的）
                result_file.write("【所有目标槽位】\n")
                result_file.write("-" * 50 + "\n")
                result_file.write("序号      X         Y         W         H      状态\n")
                result_file.write("-" * 50 + "\n")
                for i in range(len(target_current)):
                    x, y, w, h = target_current[i].cpu().numpy()
                    is_zero = abs(x) < 0.01 and abs(y) < 0.01
                    result_file.write(
                        f"{i:2d}    {x:8.4f}  {y:8.4f}  {w:8.4f}  {h:8.4f}  {'有效' if not is_zero else '无效(全零)'}\n")
                result_file.write("\n")

                # 预测结果表格
                result_file.write("【预测结果】\n")
                result_file.write("-" * 50 + "\n")
                if len(filtered_boxes) > 0:
                    result_file.write("序号      X         Y         W         H        置信度\n")
                    result_file.write("-" * 50 + "\n")
                    for i, (box, score) in enumerate(zip(filtered_boxes, filtered_scores)):
                        x, y, w, h = box.cpu().numpy()
                        result_file.write(f"{i:2d}    {x:8.4f}  {y:8.4f}  {w:8.4f}  {h:8.4f}  {score:.4f}\n")
                else:
                    result_file.write("无检测目标 (置信度均低于阈值)\n")
                result_file.write("\n")

                # 对比分析
                result_file.write("【对比分析】\n")
                result_file.write("-" * 40 + "\n")
                result_file.write(f"真实有效目标数: {current_real_gt_count}\n")
                result_file.write(f"检测到目标数: {len(filtered_boxes)}\n")
                result_file.write(f"目标槽位数: {slot_count}\n")

                if current_real_gt_count == len(filtered_boxes):
                    result_file.write("目标数量: ✓ 匹配\n")
                elif len(filtered_boxes) > current_real_gt_count:
                    result_file.write("目标数量: ⚠ 可能过检\n")
                else:
                    result_file.write("目标数量: ⚠ 可能漏检\n")
                result_file.write("\n")

                total_samples += 1

                # 只处理前50个样本用于分析
            #     if total_samples >= 50:
            #         break
            # if total_samples >= 50:
            #     break

    # 分析报告
    result_file.write("=" * 80 + "\n")
    result_file.write("【详细时序分析报告】\n")
    result_file.write("=" * 80 + "\n")

    # 统计真实目标数分布
    real_gt_counts = [info['real_gt_count'] for info in sequence_info]
    slot_counts = [info['slot_count'] for info in sequence_info]

    from collections import Counter
    real_count_dist = Counter(real_gt_counts)
    slot_count_dist = Counter(slot_counts)

    result_file.write(f"总样本数: {total_samples}\n")
    result_file.write(f"目标数跳变次数: {jump_count}\n")
    result_file.write(f"跳变频率: {jump_count / total_samples * 100:.2f}%\n\n")

    result_file.write("【槽位数分布】\n")
    result_file.write(f"固定槽位数: {list(slot_count_dist.keys())[0]}\n\n")

    result_file.write("【真实有效目标数分布】\n")
    for count, freq in sorted(real_count_dist.items()):
        result_file.write(f"{count}人目标: {freq}个样本 ({freq / total_samples * 100:.2f}%)\n")

    # 分析连续序列
    result_file.write("\n【连续序列分析】\n")
    current_count = None
    current_length = 0
    sequences = []

    for info in sequence_info:
        if info['real_gt_count'] == current_count:
            current_length += 1
        else:
            if current_count is not None:
                sequences.append((current_count, current_length))
            current_count = info['real_gt_count']
            current_length = 1

    if current_count is not None:
        sequences.append((current_count, current_length))

    result_file.write(f"连续序列总数: {len(sequences)}\n")
    for i, (count, length) in enumerate(sequences):
        result_file.write(f"序列{i + 1}: {count}人目标连续{length}个样本\n")

    # 前50个样本的详细列表
    result_file.write("\n【前50个样本详情】\n")
    result_file.write("-" * 60 + "\n")
    result_file.write("样本ID   槽位数   真实目标数\n")
    result_file.write("-" * 60 + "\n")
    for info in sequence_info[:50]:
        result_file.write(f"{info['sample_id']:04d}       {info['slot_count']}         {info['real_gt_count']}\n")

    result_file.close()

    print(f"\n=== 测试完成（修正跟踪版）===")
    print(f"分析样本数: {total_samples}")
    print(f"目标数跳变次数: {jump_count}")
    print(f"槽位数: {list(slot_count_dist.keys())[0]} (应该是固定的)")
    print(f"真实目标数分布: {dict(real_count_dist)}")
    print(f"连续序列数: {len(sequences)}")

    # 打印序列详情
    print("\n【序列详情】")
    for i, (count, length) in enumerate(sequences):
        print(f"序列{i + 1}: {count}人目标 × {length}个样本")

    return sequence_info


# 使用这个函数替换原来的测试函数
# test_and_write_result_with_tracking(model, test_loader, device, save_dir='test_results', confidence_threshold=0.96)
def test_and_write_result(model, test_loader, device, save_dir='test_results', confidence_threshold=0.5):
    """测试并保存结果txt（纯推理，无匹配）"""
    model.eval()

    os.makedirs(save_dir, exist_ok=True)
    result_file = open(os.path.join(save_dir, 'results.txt'), 'w', encoding='utf-8')

    total_samples = 0

    with torch.no_grad():
        for batch_idx, (radar_data, coordinates) in enumerate(tqdm(test_loader, desc='Testing')):
            radar_data = radar_data.to(device)

            # 模型推理
            pred_boxes, pred_logits = model(radar_data)  # [B, 50, 4], [B, 50, 2]

            # 处理标签（只是为了显示，不用于匹配）
            B, T, C = coordinates.shape
            target_coords_with_wh = []

            for i in range(B):
                avg_coords = coordinates[i].mean(dim=0)
                avg_coords_reshaped = avg_coords.reshape(-1, 2)
                coords_with_wh = add_wh_to_targets(avg_coords_reshaped, normalize_coords=False)
                target_coords_with_wh.append(coords_with_wh)

            target_coords_with_wh = torch.stack(target_coords_with_wh).to(device)  # [B, 3, 4]

            for sample_idx in range(B):
                # 推理处理：应用置信度阈值
                scores = F.softmax(pred_logits[sample_idx], dim=-1)[:, 1]  # [50] - 有目标概率
                keep = scores > confidence_threshold

                # 过滤后的预测
                filtered_boxes = pred_boxes[sample_idx][keep]  # [num_detections, 4]
                filtered_scores = scores[keep]  # [num_detections]

                # 真实目标（只用于显示对比）
                target_current = target_coords_with_wh[sample_idx]  # [3, 4]
                is_invalid = torch.all(torch.isclose(target_current[:, :2],
                                                     torch.tensor([0.0, 0.0], device=device)), dim=1)
                valid_mask = ~is_invalid
                valid_targets = target_current[valid_mask]  # [n, 4]

                # 写入结果
                sample_id = batch_idx * B + sample_idx
                result_file.write("=" * 60 + "\n")
                result_file.write(f"样本 ID: {sample_id:04d}\n")
                result_file.write("-" * 60 + "\n")
                result_file.write(f"置信度阈值: {confidence_threshold}\n")
                result_file.write(f"检测到目标数: {len(filtered_boxes)}\n")
                result_file.write("\n")

                # 真实标签表格
                result_file.write("【真实标签】\n")
                result_file.write("-" * 40 + "\n")
                if len(valid_targets) > 0:
                    result_file.write("序号      X         Y         W         H\n")
                    result_file.write("-" * 40 + "\n")
                    for i, target in enumerate(valid_targets):
                        x, y, w, h = target.cpu().numpy()
                        result_file.write(f"{i:2d}    {x:8.4f}  {y:8.4f}  {w:8.4f}  {h:8.4f}\n")
                else:
                    result_file.write("无真实目标\n")
                result_file.write("\n")

                # 预测结果表格
                result_file.write("【预测结果】\n")
                result_file.write("-" * 50 + "\n")
                if len(filtered_boxes) > 0:
                    result_file.write("序号      X         Y         W         H        置信度\n")
                    result_file.write("-" * 50 + "\n")
                    for i, (box, score) in enumerate(zip(filtered_boxes, filtered_scores)):
                        x, y, w, h = box.cpu().numpy()
                        result_file.write(f"{i:2d}    {x:8.4f}  {y:8.4f}  {w:8.4f}  {h:8.4f}  {score:.4f}\n")
                else:
                    result_file.write("无检测目标 (置信度均低于阈值)\n")
                result_file.write("\n")

                # 简单对比（不进行匹配，只是并列显示）
                result_file.write("【对比分析】\n")
                result_file.write("-" * 30 + "\n")
                result_file.write(f"真实目标数: {len(valid_targets)}\n")
                result_file.write(f"检测目标数: {len(filtered_boxes)}\n")

                # 简单的数量统计
                if len(valid_targets) == len(filtered_boxes):
                    result_file.write("目标数量: ✓ 匹配\n")
                elif len(filtered_boxes) > len(valid_targets):
                    result_file.write("目标数量: ⚠ 可能过检\n")
                else:
                    result_file.write("目标数量: ⚠ 可能漏检\n")
                result_file.write("\n")

                # 统计信息
                result_file.write("【统计信息】\n")
                result_file.write("-" * 30 + "\n")
                result_file.write(f"总查询数: 50\n")
                result_file.write(f"高置信度查询: {len(filtered_boxes)}\n")
                result_file.write(f"低置信度查询: {50 - len(filtered_boxes)}\n")
                if len(filtered_boxes) > 0:
                    result_file.write(f"最高置信度: {filtered_scores.max().item():.4f}\n")
                    result_file.write(f"平均置信度: {filtered_scores.mean().item():.4f}\n")
                result_file.write("\n\n")

                total_samples += 1

            # 只处理前几个batch用于调试
            if batch_idx >= 2:
                continue

    # 总体统计
    result_file.write("=" * 60 + "\n")
    result_file.write("【总体测试统计】\n")
    result_file.write("=" * 60 + "\n")
    result_file.write(f"总样本数:     {total_samples}\n")
    result_file.write(f"置信度阈值:   {confidence_threshold}\n")
    result_file.write(f"测试时间:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    result_file.close()
    print(f"\n=== 测试完成 ===")
    print(f"测试样本数: {total_samples}")
    print(f"置信度阈值: {confidence_threshold}")
    print(f"结果已保存到: {save_dir}/results.txt")

class Mymodule(nn.Module):
    def __init__(self, input_channels=8, freq_bins=89, time_steps=10):
        super(Mymodule, self).__init__()
        # self.backbone = CNNfeat(input_channels=8, freq_bins=89, time_steps=10)
        self.backbone = SimpleFusionBackbone(input_channels=8, freq_bins=89, time_steps=10)
        self.DETR_Encoder = nn.ModuleList([  # 用ModuleList装多个EncoderLayer
            EncoderLayer(attention_dim=128, attention_heads=8)
            for _ in range(4)  # 堆叠enc_num_layers层
        ])
        self.position_encoding = self._create_sin_cos_encoding(48, 128)
        self.DETR_Decoder = Decoder(
            d_model=128,
            num_queries=50,  # 最多预测5个目标
            num_layers=3 # 3层解码器
        )
    def _create_sin_cos_encoding(self, seq_len, d_model):
        # 确保 d_model 与 backbone 输出维度一致
        # 如果 backbone 输出是 128 维，这里也应该是 128
        position = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) *
                             (-math.log(10000.0) / d_model))

        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        return pe  # [seq_len, d_model]

    def forward(self, x):
        batch_size, channels, dim, timesteps = x.shape

        # Backbone特征提取
        backbonefeat = self.backbone(x)

        # 重塑为序列形式
        backbonefeat_reshaped = backbonefeat.view(batch_size, 128, -1)
        encoder_input = backbonefeat_reshaped.transpose(1, 2)

        # 添加位置编码
        pos_enc = self.position_encoding.to(encoder_input.device)
        encoder_input_with_pos = encoder_input + pos_enc.unsqueeze(0)

        # 编码器前向传播
        encoder_output = encoder_input_with_pos
        for enc_layer in self.DETR_Encoder:
            encoder_output = enc_layer(encoder_input_with_pos)

        # 解码器 - 获取所有层输出
        coor_pred, wh_pred, conf, _ = self.DETR_Decoder(encoder_output)

        # 训练时返回所有层，测试时只返回最后一层
        if self.training:
            # 返回所有层的信息
            all_outputs = []
            for layer_idx in range(len(coor_pred)):
                coor_layer = coor_pred[layer_idx].reshape(batch_size, -1, 2)
                wh_layer = wh_pred[layer_idx].reshape(batch_size, -1, 2)
                conf_layer = conf[layer_idx].reshape(batch_size, -1, 2)
                layer_out = torch.cat([coor_layer, wh_layer], dim=2)
                all_outputs.append((layer_out, conf_layer))
            return all_outputs
        else:
            # 测试/推理时只返回最后一层（与之前完全一致）
            coor_out = coor_pred[-1].reshape(batch_size, -1, 2)
            wh_out = wh_pred[-1].reshape(batch_size, -1, 2)
            conf_out = conf[-1].reshape(batch_size, -1, 2)
            out = torch.cat([coor_out, wh_out], dim=2)
            return out, conf_out


if __name__ == '__main__':
    # 测试参数
    num_channels = 8
    batch_size =32
    # test_label_processing()
    seed = 42
    g = torch.Generator()
    g.manual_seed(seed)
    generator = torch.Generator().manual_seed(2023)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Mymodule(input_channels = num_channels).to(device)


    # # 使用
    total, trainable = count_parameters(model)
    dummy_input = torch.randn(batch_size, 8, 89, 10).to(complex).to(device)
    flops, params = profile(model, inputs=(dummy_input,))
    print(f"FLOPs: {flops}")
    print(f"Parameters: {params}")

    # 使用clever_format转换为更易读的格式(G, M, K)
    flops, params = clever_format([flops, params], "%.3f")
    print(f"FLOPs: {flops}, Params: {params}")
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,  # 初始学习率（BiGRU通常需要较小学习率）
        betas=(0.9, 0.999),  # 动量参数
        weight_decay=1e-4,  # 解耦权重衰减（防止过拟合）
        eps=1e-8
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=40,  # 20个epoch
        eta_min=1e-4  # 最小学习率
    )
    # 纯多人
    train_dataset = load_fast_dataset1(
        data_root_folder=r"E:\Jupyter\track\fast_dataset_multi_train",
        seq_length=10,  # 这个参数现在只是兼容性保留，实际从文件读取
        add_noise=True,
        delete_occlusion=False
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=True, generator=g,)
    test_dataset = load_fast_dataset1(
        data_root_folder=r"E:\Jupyter\track\fast_dataset_multi_test",
        seq_length=10,  # 这个参数现在只是兼容性保留，实际从文件读取
        add_noise=False,
        delete_occlusion=True
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    # criterion = TWPLoss(alpha=0.95, gamma=3, regression_weight=0.3) #5.30
    # criterion = hungarian_loss
    criterion = hungarian_loss_with_classification


    #
    # # # criterion = new_complete_loss
    # checkpoint = torch.load('best_model_multilayerloss128.pth')
    # # 提取模型权重
    # model.load_state_dict(checkpoint['model_state_dict'])
    # # #
    # test_and_write_result(
    #     model, test_loader, device,
    #     save_dir='test_results_0.96_numworks = 0',
    #     confidence_threshold=0.96
    # )
    # test_and_write_result_with_tracking(
    #     model, test_loader, device,
    #     save_dir='test_results_0.96_numworks = 0',
    #     confidence_threshold=0.96
    # )
    # ssss

    num_epochs = 40
    best_val_loss = float('inf')

    for epoch in range(1, num_epochs + 1):
        print(f'\nEpoch {epoch}/{num_epochs}')
        print('-' * 50)

        # 训练 - 现在返回3个值
        train_total_loss, train_class_loss, train_coord_loss = train_epoch_with_temporal_avg(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # 验证 - 现在返回3个值
        val_total_loss, val_class_loss, val_coord_loss = validate_with_temporal_avg(
            model, test_loader, criterion, device
        )

        # 学习率调整
        scheduler.step()
        # 打印结果
        print(f'Train Total Loss: {train_total_loss:.6f} | Val Total Loss: {val_total_loss:.6f}')
        print(f'Train Class Loss: {train_class_loss:.6f} | Val Class Loss: {val_class_loss:.6f}')
        print(f'Train Coord Loss: {train_coord_loss:.6f} | Val Coord Loss: {val_coord_loss:.6f}')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.8f}')

        # 保存最佳模型
        if val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
            }, 'best_model_multilayerloss128___121.pth')
            print(f'Saved best model with val_loss: {val_total_loss:.6f}')