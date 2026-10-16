import torch
import torch.nn as nn

from torchvision.models.swin_transformer import SwinTransformerBlock
heatmap_size1 = 128
class Encoderlayer(nn.Module):
    def __init__(self, in_channels, out_channels,use_swin):
        super().__init__()
        self.use_swin = use_swin
        self.block = nn.Sequential(
            # 2D卷积：仅处理空间维度（H,W）
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),  # 去掉时间维度，kernel_size=3x3
            nn.BatchNorm2d(out_channels),  # 同步改为BatchNorm2d
            nn.ReLU(inplace=True),
            # 空间下采样：stride=2（仅H,W下采样）
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),  # 去掉时间维度的stride
            nn.BatchNorm2d(out_channels),  # 同步改为BatchNorm2d
            nn.ReLU(inplace=True)
        )
        if use_swin:
            self.swin_block = SwinTransformerBlock(
                dim=out_channels,        # 输入特征维度
                num_heads=4,             # 注意力头数
                window_size=(6,6),           # 窗口大小
                shift_size=[0,0],            # 移位大小（0=W-MSA, 3=SW-MSA）
                mlp_ratio=4.0,           # MLP扩展比例
                dropout=0.0,             # Dropout率
                attention_dropout=0.0,   # 注意力Dropout率
                stochastic_depth_prob=0.0 # 随机深度概率
            )
    def forward(self, x):
        x = self.block(x)
        if self.use_swin:
            B, C, H, W = x.shape

            # 动态计算合适的窗口大小
            window_size = min(7, H, W)  # 不超过特征图尺寸

            if self.use_swin:
                # SwinTransformerBlock需要 [B, H, W, C] 格式
                B, C, H, W = x.shape
                x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
                x = self.swin_block(x)
                x = x.permute(0, 3, 1, 2)  # [B, C, H, W] 恢复原格式
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels, channels_list):
        super().__init__()
        self.layers = nn.ModuleList()

        # 第一层
        self.layers.append(Encoderlayer(in_channels, channels_list[0],use_swin = False))

        # 后续层
        for i in range(1, len(channels_list)):
            self.layers.append(Encoderlayer(channels_list[i - 1], channels_list[i],use_swin = False))

    def forward(self, x):
        features = []  # 保存每一层的输出特征
        for layer in self.layers:
            x = layer(x)
            features.append(x)
        return features  # 返回所有层的特征列表


class ChannelFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),  # 1x1 2D卷积
            nn.BatchNorm2d(out_channels),  # 2D批归一化
            nn.LeakyReLU(inplace=True)
        )
    def forward(self, mag_feat, pha_feat):
        # 在通道维度拼接
        fused = torch.cat([mag_feat, pha_feat], dim=1)  # dim=1 是通道维
        # 通道融合和压缩
        return self.fusion_conv(fused)


class DecoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),  # 2D卷积
            nn.BatchNorm2d(out_channels),  # 2D批归一化
            nn.ReLU(inplace=True),
            # 2D转置卷积：仅空间上采样（H,W）
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2, padding=0),  # kernel=2x2，stride=2
            nn.BatchNorm2d(out_channels),  # 2D批归一化
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.block(x)

class SimpleFusionBackbone(nn.Module):
    def __init__(self, input_channels=8, freq_bins=89, time_steps=10):
        super(SimpleFusionBackbone, self).__init__()
        self.freq_bins1 = freq_bins
        self.freq_bins1 = 128
        self.real_feat_ext_cnn = nn.Conv2d(in_channels=input_channels, out_channels=self.freq_bins1,
                                           kernel_size=(3, time_steps), stride=1, padding=(1, 0), bias=True)
        self.imag_feat_ext_cnn = nn.Conv2d(in_channels=input_channels, out_channels=self.freq_bins1,
                                           kernel_size=(3, time_steps), stride=1, padding=(1, 0), bias=True)
        self.mag_conv = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.pha_conv = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)

        # 编码器（已经是2D的，保持不变）
        self.encoderw1 = 16
        self.encoderw2 = 32
        self.encoderw3 = 64
        self.encoderw4 = 128
        self.fusionw1 = self.encoderw1
        self.fusionw2 = self.encoderw2
        self.fusionw3 = self.encoderw3
        self.fusionw4 = self.encoderw4
        self.mag_encoder = Encoder(in_channels=8,
                                   channels_list=[self.encoderw1, self.encoderw2, self.encoderw3, self.encoderw4])
        self.pha_encoder = Encoder(in_channels=8,
                                   channels_list=[self.encoderw1, self.encoderw2, self.encoderw3, self.encoderw4])

        # 融合层和解码器（保持不变）
        self.fusion1 = ChannelFusion(in_channels=2 * self.encoderw1, out_channels=self.fusionw1)
        self.fusion2 = ChannelFusion(in_channels=2 * self.encoderw2, out_channels=self.fusionw2)
        self.fusion3 = ChannelFusion(in_channels=2 * self.encoderw3, out_channels=self.fusionw3)
        self.fusion4 = ChannelFusion(in_channels=2 * self.encoderw4, out_channels=self.fusionw4)
    def forward(self, x):
        batch_size, channels, freq_bins, time_steps = x.shape
        x_2d = x  # (B, 10, 8, 89)
        # print(f"x_2d{x_2d.shape}")
        x_real = x_2d.real.float()
        x_imag = x_2d.imag.float()

        x_real_ext = self.real_feat_ext_cnn(x_real)
        x_imag_ext = self.imag_feat_ext_cnn(x_imag)
        # print(f"x_real_ext  {x_real_ext.shape}")

        amp = torch.sqrt(x_real_ext ** 2 + x_imag_ext ** 2).squeeze(3).unsqueeze(1)
        pha = torch.atan2(x_imag_ext, x_real_ext).squeeze(3).unsqueeze(1)

        # print(f"amp  {amp.shape}")

        ra_mag = self.mag_conv(amp)  # (B, 4, 128, 128)
        ra_pha = self.pha_conv(pha)  # (B, 4, 128, 128)
        # print(f"ra_mag  {ra_mag.shape}")
        # 后续编码器-解码器部分保持不变
        mag_features = self.mag_encoder(ra_mag)
        pha_features = self.pha_encoder(ra_pha)
        # print(f"mag_features[0]  {mag_features[0].shape}")
        # print(f"mag_features[1]  {mag_features[1].shape}")
        # print(f"mag_features[2]  {mag_features[2].shape}")
        # print(f"mag_features[3]  {mag_features[3].shape}")
        fused_feat1 = self.fusion1(mag_features[0], pha_features[0])
        fused_feat2 = self.fusion2(mag_features[1], pha_features[1])
        fused_feat3 = self.fusion3(mag_features[2], pha_features[2])
        fused_feat4 = self.fusion4(mag_features[3], pha_features[3])

        return fused_feat4