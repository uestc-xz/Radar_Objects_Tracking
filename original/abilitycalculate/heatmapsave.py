import matplotlib.pyplot as plt
import os
import torch
from original.my_dataset import process_temporal_labels
from tqdm import tqdm
def test_and_save_results(model, test_loader, device,heatmap_size1, save_dir='test_results' ):
    """测试并保存结果图片"""
    model.eval()

    # 创建保存文件夹
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for batch_idx, (radar_data, coordinates) in enumerate(tqdm(test_loader, desc='Testing')):
            # 移动到设备
            radar_data = radar_data.to(device)

            # 预测 - 修复这里的维度问题
            out1 = model(radar_data)  # 形状: (B, 1, 128, 128)

            # 正确的索引方式 - 取第一个通道的所有空间位置
            pred_heatmap = out1[:, 0, :, :].cpu().detach().numpy()  # 形状: (B, 128, 128)

            # 处理每个样本
            for sample_idx in range(radar_data.shape[0]):
                # 生成真实标签热图
                target_heatmap = process_temporal_labels(
                    coordinates[sample_idx:sample_idx + 1],
                    heatmap_size=(heatmap_size1, heatmap_size1)  # 使用128×128
                )[0, 0].cpu().numpy()

                # 提取第一个时间步的目标坐标
                targets_first_frame = []
                for i in range(0, 6, 2):
                    x = coordinates[sample_idx, 0, i]
                    y = coordinates[sample_idx, 0, i + 1]
                    # 使用2.7作为范围判断
                    if -2.7 <= x <= 2.7 and -2.7+3.5 <= y <= 2.7+3.5 and not (abs(x) < 1e-5 and abs(y) < 1e-5):
                        targets_first_frame.append((x, y, i // 2 + 1))

                # 创建图片
                plt.figure(figsize=(10, 8))

                # 预测热图 - 现在pred_heatmap[sample_idx]是(128, 128)
                plt.imshow(pred_heatmap[sample_idx],
                           cmap='viridis',
                           extent=[-2.7, 2.7, -2.7+3.5, 2.7+3.5],  # 改为2.7范围
                           origin='lower')

                # 标记真实坐标点
                colors = ['red', 'blue', 'green']
                markers = ['o', 's', 'D']
                for idx, (x, y, target_id) in enumerate(targets_first_frame):
                    plt.scatter(x, y, color=colors[idx], s=100, marker=markers[idx],
                                edgecolors='white', linewidth=2, label=f'Target {target_id}')

                plt.colorbar(label='Predicted Confidence')
                plt.xlabel('X (m)')
                plt.ylabel('Y (m)')
                plt.title(f'Test Sample {batch_idx * test_loader.batch_size + sample_idx + 1}')
                plt.grid(True, alpha=0.3)
                plt.xticks([-2.7, -2, -1, 0, 1, 2, 2.7])
                plt.yticks([-2.7+3.5, -2+3.5, -1+3.5, 0+3.5, 1+3.5, 2+3.5, 2.7+3.5])
                if targets_first_frame:
                    plt.legend()

                # 保存图片
                plt.savefig(f'{save_dir}/test_{batch_idx * test_loader.batch_size + sample_idx + 1:04d}.png',
                            dpi=150, bbox_inches='tight')
                plt.close()

                # 打印坐标信息到文件
                with open(f'{save_dir}/test_{batch_idx * test_loader.batch_size + sample_idx + 1:04d}_coords.txt',
                          'w') as f:
                    f.write("标记的目标坐标:\n")
                    for x, y, target_id in targets_first_frame:
                        f.write(f"Target {target_id}: ({x:.2f}, {y:.2f})\n")

    print(f"测试完成！结果保存在 {save_dir} 文件夹中")