import torch
import torch.nn as nn
import torch.nn.functional as F
class TWPLoss(nn.Module):
    def __init__(self, alpha=0.9, gamma=2, regression_weight=0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.regression_weight = regression_weight

    def focal_loss(self, pred, target):
        """Focal Loss for segmentation"""
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

    def charbonnier_loss(self, pred, target, eps=1e-3):
        """Charbonnier Loss for heatmap quality"""
        diff = pred - target
        loss = torch.sqrt(diff * diff + eps)
        return loss.mean()
    def regression_loss(self, pred_coords, target_coords):
        """回归损失 - 计算坐标预测的L1损失"""
        # pred_coords: (B, 2) 或 (B, 4)  [距离, 角度] 或 [x1,y1,x2,y2]
        # target_coords: (B, 2) 或 (B, 4)
        return F.l1_loss(pred_coords, target_coords)
    def forward(self, pred_heatmap, target_heatmap, pred_coords=None, target_coords=None):
        """
        Args:
            pred_heatmap: (B, 1, T, H, W) 网络预测的热图
            target_heatmap: (B, 1, T, H, W) 真实热图标签
        """
        # 分割损失
        seg_loss = self.focal_loss(pred_heatmap, target_heatmap)

        # Charbonnier损失 (用sigmoid确保值在0-1之间)
        pred_sigmoid = torch.sigmoid(pred_heatmap)
        char_loss = self.charbonnier_loss(pred_sigmoid, target_heatmap)

        # 总损失
        total_loss = seg_loss +  char_loss
        if pred_coords is not None and target_coords is not None:
            reg_loss = self.regression_loss(pred_coords, target_coords)
            total_loss = total_loss + self.regression_weight * reg_loss
        else:
            reg_loss = torch.tensor(0.0)
        return {
            'total_loss': total_loss,
            'seg_loss': seg_loss,
            'char_loss': char_loss,
            'reg_loss': reg_loss
        }