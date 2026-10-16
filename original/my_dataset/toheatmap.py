import torch


def process_temporal_labels(coordinates, heatmap_size=(96, 96), sigma=1, eps=0):
    B, T, C = coordinates.shape
    H, W = heatmap_size
    batch_heatmaps = []
    x_range = 2.7
    y_range = 6.2

    def normalize_coord(coord, axis='x', size=96):
        if axis == 'x':
            normalized = (coord + x_range) / (2 * x_range)
        else:
            normalized = (coord - 0.8) / (6.2 - 0.8)
        return normalized * (size - 1)

    for batch_idx in range(B):
        # 先对坐标取时间平均
        avg_coords = coordinates[batch_idx].mean(dim=0)  # (C,)
        avg_coords = avg_coords.reshape(-1, 2)  # (C/2, 2)

        # 创建单帧热图
        heatmap = torch.zeros((H, W))
        targets = []

        # 处理平均后的坐标
        for i in range(avg_coords.shape[0]):
            x = avg_coords[i, 0]
            y = avg_coords[i, 1]

            x_valid = (-x_range <= x <= x_range)
            y_valid = (0.8 <= y <= 6.2)
            is_near_origin = (abs(x) < eps and abs(y) < eps)

            if x_valid and y_valid and not is_near_origin:
                pixel_x = normalize_coord(x, 'x', W)
                pixel_y = normalize_coord(y, 'y', H)
                targets.append((pixel_x.item(), pixel_y.item()))

        # 为平均后的目标创建高斯热图
        for target in targets:
            x, y = target
            x = max(0, min(W - 1, x))
            y = max(0, min(H - 1, y))

            x_range_tensor = torch.arange(W, dtype=torch.float32)
            y_range_tensor = torch.arange(H, dtype=torch.float32)
            xx, yy = torch.meshgrid(x_range_tensor, y_range_tensor, indexing='xy')

            gaussian = torch.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
            heatmap = torch.maximum(heatmap, gaussian)

        batch_heatmaps.append(heatmap.unsqueeze(0))

    return torch.stack(batch_heatmaps)  # (B, 1, H, W)
# def process_temporal_labels(coordinates, heatmap_size=(96, 96), sigma=1, eps=0):
#     B, T, C = coordinates.shape
#     H, W = heatmap_size
#     batch_heatmaps = []
#     x_range = 2.7
#     y_range = 6.2
#
#     def normalize_coord(coord, axis='x', size=96):
#         if axis == 'x':
#             normalized = (coord + x_range) / (2 * x_range)
#         else:
#             normalized = (coord - 0.8) / (6.2 - 0.8)
#         return normalized * (size - 1)
#
#     for batch_idx in range(B):
#         # 为每个时间步创建热图，然后求平均
#         temporal_heatmaps = []
#
#         for t in range(T):
#             # 创建当前时间步的热图
#             heatmap_t = torch.zeros((H, W))
#             targets = []
#
#             for i in range(0, C, 2):
#                 x = coordinates[batch_idx, t, i]
#                 y = coordinates[batch_idx, t, i + 1]
#
#                 x_valid = (-x_range <= x <= x_range)
#                 y_valid = (0.8 <= y <= 6.2)
#                 is_near_origin = (abs(x) < eps and abs(y ) < eps)
#
#                 if x_valid and y_valid and not is_near_origin:
#                     pixel_x = normalize_coord(x, 'x', W)
#                     pixel_y = normalize_coord(y, 'y', H)
#                     targets.append((pixel_x.item(), pixel_y.item()))
#
#             # 为当前时间步创建高斯热图
#             for target in targets:
#                 x, y = target
#                 x = max(0, min(W - 1, x))
#                 y = max(0, min(H - 1, y))
#
#                 x_range_tensor = torch.arange(W, dtype=torch.float32)
#                 y_range_tensor = torch.arange(H, dtype=torch.float32)
#                 xx, yy = torch.meshgrid(x_range_tensor, y_range_tensor, indexing='xy')
#
#                 gaussian = torch.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
#                 heatmap_t = torch.maximum(heatmap_t, gaussian)
#
#             temporal_heatmaps.append(heatmap_t.unsqueeze(0))
#
#         # 对时间步求平均
#         if temporal_heatmaps:
#             avg_heatmap = torch.mean(torch.cat(temporal_heatmaps, dim=0), dim=0)
#         else:
#             avg_heatmap = torch.zeros((H, W))
#
#         batch_heatmaps.append(avg_heatmap.unsqueeze(0))
#
#     return torch.stack(batch_heatmaps)  # (B, 1, H, W)

def create_gaussian_heatmap(heatmap_size, centers, sigma=2):
    """创建时间序列热图"""
    T, H, W = heatmap_size
    heatmap = torch.zeros((T, H, W))
    x_range = 2.7
    y_range = 6.2

    def normalize_coord(coord, axis='x', size=96):
        if axis == 'x':
            normalized = (coord + x_range) / (2 * x_range)
        else:  # y轴
            normalized = (coord - 0.8) / (6.2 - 0.8)
        return normalized * (size - 1)

    for t in range(T):
        if t < len(centers):  # 这个时间步有目标
            for center in centers[t]:
                x, y = center
                # 映射到像素坐标
                pixel_x = normalize_coord(x, 'x', W)
                pixel_y = normalize_coord(y, 'y', H)

                # 确保坐标在范围内
                pixel_x = max(0, min(W - 1, pixel_x))
                pixel_y = max(0, min(H - 1, pixel_y))

                # 创建高斯分布
                x_range_tensor = torch.arange(W)
                y_range_tensor = torch.arange(H)
                xx, yy = torch.meshgrid(x_range_tensor, y_range_tensor, indexing='xy')

                gaussian = torch.exp(-((xx - pixel_x) ** 2 + (yy - pixel_y) ** 2) / (2 * sigma ** 2))
                heatmap[t] = torch.maximum(heatmap[t], gaussian)

    return heatmap.unsqueeze(0)  # (1, T, H, W)