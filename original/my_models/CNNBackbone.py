import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
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


class CNNfull(nn.Module):
    def __init__(self, input_channels=8, freq_bins=89, time_steps=10):
        super(CNNfull, self).__init__()

        # 2D特征提取
        self.real_feat_ext_cnn = nn.Conv2d(in_channels=time_steps, out_channels=freq_bins,
                                           kernel_size=(1, freq_bins), stride=1, padding=(0, 0), bias=True)
        self.imag_feat_ext_cnn = nn.Conv2d(in_channels=time_steps, out_channels=freq_bins,
                                           kernel_size=(1, freq_bins), stride=1, padding=(0, 0), bias=True)

        # 2D RA处理
        self.ra_real = nn.Conv2d(in_channels=input_channels, out_channels=freq_bins,
                                 kernel_size=1, stride=1, padding=0)
        self.ra_imag = nn.Conv2d(in_channels=input_channels, out_channels=freq_bins,
                                 kernel_size=1, stride=1, padding=0)

        # 2D幅度和相位卷积
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

        self.deoutc1 = 128
        self.deoutc2 = 64
        self.deoutc3 = 32
        self.deoutc4 = 16
        self.decoder1 = DecoderLayer(in_channels=self.fusionw4 + self.fusionw3, out_channels=self.deoutc1)
        self.decoder2 = DecoderLayer(in_channels=self.deoutc1 + self.fusionw2, out_channels=self.deoutc2)
        self.decoder3 = DecoderLayer(in_channels=self.deoutc2 + self.fusionw1, out_channels=self.deoutc3)
        self.decoder4 = DecoderLayer(in_channels=self.deoutc3, out_channels=self.deoutc4)

        self.final_conv = nn.Conv2d(self.deoutc4, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        batch_size, channels, freq_bins, time_steps = x.shape  # (B, 8, 89, 10)
        # x = x[:,:,:,-1].unsqueeze(3)
        # print(x.shape)
        # 重塑数据以适应2D卷积
        # 方法1: 将时间步移到通道维度
        x_2d = x.permute(0, 3, 1, 2).contiguous()  # (B, 10, 8, 89)

        x_real = x_2d.real.float()
        x_imag = x_2d.imag.float()

        # 提取相位信息（用于后续可能的使用）
        # x_pha = torch.atan2(x_imag, x_real).mean(dim=1).reshape(batch_size, -1)

        # 2D特征提取
        x_real_ext = self.real_feat_ext_cnn(x_real)  # (B, 89, 8, 1)
        x_imag_ext = self.imag_feat_ext_cnn(x_imag)  # (B, 89, 8, 1)

        # 调整维度以适应RA处理
        x_real_ext = x_real_ext.permute(0, 2, 1, 3).contiguous()  # (B, 8, 89, 1)
        x_imag_ext = x_imag_ext.permute(0, 2, 1, 3).contiguous()  # (B, 8, 89, 1)

        # 2D RA处理
        ra_real = self.ra_real(x_real_ext)  # (B, 89, 89, 1)
        ra_imag = self.ra_imag(x_imag_ext)  # (B, 89, 89, 1)

        # 计算幅度和相位
        amp = torch.sqrt(ra_real ** 2 + ra_imag ** 2)
        pha = torch.atan2(ra_imag, ra_real)

        # 移除多余的维度并插值到目标尺寸
        amp = amp.squeeze(-1)  # (B, 89, 89)
        pha = pha.squeeze(-1)  # (B, 89, 89)
        # 插值到热图尺寸
        amp = F.interpolate(amp.unsqueeze(1), size=(heatmap_size1, heatmap_size1), mode='nearest')  # (B, 1, 128, 128)
        pha = F.interpolate(pha.unsqueeze(1), size=(heatmap_size1, heatmap_size1), mode='nearest')  # (B, 1, 128, 128)

        # 2D卷积处理
        ra_mag = self.mag_conv(amp)  # (B, 4, 128, 128)
        ra_pha = self.pha_conv(pha)  # (B, 4, 128, 128)

        # 后续编码器-解码器部分保持不变
        mag_features = self.mag_encoder(ra_mag)
        pha_features = self.pha_encoder(ra_pha)

        fused_feat1 = self.fusion1(mag_features[0], pha_features[0])
        fused_feat2 = self.fusion2(mag_features[1], pha_features[1])
        fused_feat3 = self.fusion3(mag_features[2], pha_features[2])
        fused_feat4 = self.fusion4(mag_features[3], pha_features[3])

        # 解码器部分保持不变
        x = fused_feat4
        print(f"fused_feat4.shape {fused_feat4.shape}")
        fused_feat3_upsampled = F.interpolate(fused_feat3, size=(x.shape[2], x.shape[3]), mode='nearest')

        print(f"fused_feat3_upsampled.shape {fused_feat3_upsampled.shape}")
        x = torch.cat([x, fused_feat3_upsampled], dim=1)

        x = self.decoder1(x)
        print(f"self.decoder1.shape {x.shape}")

        fused_feat2_upsampled = F.interpolate(fused_feat2, size=(x.shape[2], x.shape[3]), mode='nearest')
        print(f"self.decoder1.shape {x.shape}")
        x = torch.cat([x, fused_feat2_upsampled], dim=1)
        x = self.decoder2(x)

        fused_feat1_upsampled = F.interpolate(fused_feat1, size=(x.shape[2], x.shape[3]), mode='nearest')
        x = torch.cat([x, fused_feat1_upsampled], dim=1)
        x = self.decoder3(x)

        x = self.decoder4(x)
        output = self.final_conv(x)

        return output


class FusionEncoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels, fusion_method='add'):
        super().__init__()
        self.fusion_method = fusion_method
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 主路径（幅度特征）
        self.mag_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # 相位特征投影 - 现在预先创建，但根据输入动态调整
        self.pha_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)  # 预先创建

        if fusion_method == 'concat':
            self.fusion_conv = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1)

    def forward(self, mag_feat, pha_feat=None):
        # 处理幅度特征
        mag_out = self.mag_block(mag_feat)

        # 如果有相位特征，进行融合
        if pha_feat is not None:
            # 确保相位特征通道数匹配
            if pha_feat.size(1) != self.in_channels:
                # 如果通道数不匹配，使用1x1卷积调整
                if not hasattr(self, 'pha_adapter') or self.pha_adapter.in_channels != pha_feat.size(1):
                    self.pha_adapter = nn.Conv2d(pha_feat.size(1), self.in_channels, kernel_size=1).to(pha_feat.device)
                pha_feat = self.pha_adapter(pha_feat)

            # 对相位特征进行投影和下采样
            pha_projected = self.pha_proj(pha_feat)
            pha_down = F.interpolate(pha_projected, size=mag_out.shape[2:], mode='bilinear')

            if self.fusion_method == 'add':
                fused_out = mag_out + pha_down
            elif self.fusion_method == 'concat':
                concatenated = torch.cat([mag_out, pha_down], dim=1)
                fused_out = self.fusion_conv(concatenated)
            else:
                fused_out = mag_out
        else:
            fused_out = mag_out

        return fused_out

class CNNfeat(nn.Module):
    def __init__(self, input_channels=8, freq_bins=89, time_steps=10):
        super(CNNfeat, self).__init__()
        self.freq_bins1 = heatmap_size1
        # 2D特征提取
        self.real_feat_ext_cnn = nn.Conv2d(in_channels=time_steps, out_channels=self.freq_bins1,
                                           kernel_size=(1, freq_bins), stride=1, padding=(0, 0), bias=True)
        self.imag_feat_ext_cnn = nn.Conv2d(in_channels=time_steps, out_channels=self.freq_bins1,
                                           kernel_size=(1, freq_bins), stride=1, padding=(0, 0), bias=True)

        # 2D RA处理
        self.ra_real = nn.Conv2d(in_channels=input_channels, out_channels=self.freq_bins1,
                                 kernel_size=1, stride=1, padding=0)
        self.ra_imag = nn.Conv2d(in_channels=input_channels, out_channels=self.freq_bins1,
                                 kernel_size=1, stride=1, padding=0)

        # 2D幅度和相位卷积
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
        #
        # self.deoutc1 = 128
        # self.deoutc2 = 64
        # self.deoutc3 = 32
        # self.deoutc4 = 16
        # self.decoder1 = DecoderLayer(in_channels=self.fusionw4 + self.fusionw3, out_channels=self.deoutc1)
        # self.decoder2 = DecoderLayer(in_channels=self.deoutc1 + self.fusionw2, out_channels=self.deoutc2)
        # self.decoder3 = DecoderLayer(in_channels=self.deoutc2 + self.fusionw1, out_channels=self.deoutc3)
        # self.decoder4 = DecoderLayer(in_channels=self.deoutc3, out_channels=self.deoutc4)
        #
        # self.final_conv = nn.Conv2d(self.deoutc4, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x, return_heatmap=False):
        batch_size, channels, freq_bins, time_steps = x.shape  # (B, 8, 89, 10)
        # x = x[:,:,:,-1].unsqueeze(3)
        # print(x.shape)
        # 重塑数据以适应2D卷积
        # 方法1: 将时间步移到通道维度
        x_2d = x.permute(0, 3, 1, 2).contiguous()  # (B, 10, 8, 89)

        x_real = x_2d.real.float()
        x_imag = x_2d.imag.float()

        # 提取相位信息（用于后续可能的使用）
        # x_pha = torch.atan2(x_imag, x_real).mean(dim=1).reshape(batch_size, -1)
        # 2D特征提取
        x_real_ext = self.real_feat_ext_cnn(x_real)  # (B, 89, 8, 1)
        x_imag_ext = self.imag_feat_ext_cnn(x_imag)  # (B, 89, 8, 1)   10 时间步----1

        # 调整维度以适应RA处理
        x_real_ext = x_real_ext.permute(0, 2, 1, 3).contiguous()  # (B, 8, 89, 1)
        x_imag_ext = x_imag_ext.permute(0, 2, 1, 3).contiguous()  # (B, 8, 89, 1)

        # 2D RA处理
        ra_real = self.ra_real(x_real_ext)  # (B, 89, 89, 1)
        ra_imag = self.ra_imag(x_imag_ext)  # (B, 89, 89, 1)

        # 计算幅度和相位
        amp = torch.sqrt(ra_real ** 2 + ra_imag ** 2)
        pha = torch.atan2(ra_imag, ra_real)

        # 移除多余的维度并插值到目标尺寸
        amp = amp.squeeze(-1).unsqueeze(1)  # (B, 89, 89)
        pha = pha.squeeze(-1).unsqueeze(1)  # (B, 89, 89)
        # print(f"amp{amp.shape}")
        heatmap = amp.clone()
        # 插值到热图尺寸
        # amp = F.interpolate(amp.unsqueeze(1), size=(heatmap_size1, heatmap_size1), mode='nearest')  # (B, 1, 128, 128)
        # pha = F.interpolate(pha.unsqueeze(1), size=(heatmap_size1, heatmap_size1), mode='nearest')  # (B, 1, 128, 128)
        # print(f"amp{amp.shape}")
        # 2D卷积处理
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
        # print(f"fused_feat1  {fused_feat1.shape}")
        # print(f"fused_feat2  {fused_feat2.shape}")
        # print(f"fused_feat3  {fused_feat3.shape}")
        # print(f"fused_feat4  {fused_feat4.shape}")
        if return_heatmap:
            return fused_feat4, heatmap
        else:
            return fused_feat4
            # return fused_feat1, fused_feat2, fused_feat3, fused_feat4
        # return fused_feat4
        # return fused_feat1,fused_feat2,fused_feat3,fused_feat4


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





class SimpleFusionBackbone_nophanomag(nn.Module):
    def __init__(self, input_channels=8, freq_bins=89, time_steps=10):
        super(SimpleFusionBackbone_nophanomag, self).__init__()
        self.freq_bins1 = freq_bins
        self.freq_bins1 = 89
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

        amp = x_real_ext.squeeze(3).unsqueeze(1)
        pha = x_imag_ext.squeeze(3).unsqueeze(1)

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

















# 用于幅度和相位分支的消融
class ChannelFusion_ablation(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 在单分支消融实验中，in_channels 应该是单分支的维度（例如 self.encoderw1）
        # 但为了不改动 Backbone 的初始化参数 (2 * self.encoderw1)，我们在这里兼容一下
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels // 2, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, mag_feat, pha_feat=None):
        # 消融实验逻辑：忽略 pha_feat，只处理 mag_feat
        # 此时输入的 mag_feat 维度已经是单分支维度
        return self.fusion_conv(mag_feat)
class SimpleFusionBackbone_phafusion(nn.Module):
    def __init__(self, input_channels=8, freq_bins=89, time_steps=10):
        super(SimpleFusionBackbone_phafusion, self).__init__()
        self.freq_bins1 = freq_bins
        self.freq_bins1 = 128
        self.real_feat_ext_cnn = nn.Conv2d(in_channels=input_channels, out_channels=self.freq_bins1,
                                           kernel_size=(3, time_steps), stride=1, padding=(1, 0), bias=True)
        self.imag_feat_ext_cnn = nn.Conv2d(in_channels=input_channels, out_channels=self.freq_bins1,
                                           kernel_size=(3, time_steps), stride=1, padding=(1, 0), bias=True)
        self.mag_conv = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)

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
        # 融合层和解码器（保持不变）
        self.fusion1 = ChannelFusion_ablation(in_channels=2 * self.encoderw1, out_channels=self.fusionw1)
        self.fusion2 = ChannelFusion_ablation(in_channels=2 * self.encoderw2, out_channels=self.fusionw2)
        self.fusion3 = ChannelFusion_ablation(in_channels=2 * self.encoderw3, out_channels=self.fusionw3)
        self.fusion4 = ChannelFusion_ablation(in_channels=2 * self.encoderw4, out_channels=self.fusionw4)
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

        # print(f"amp  {amp.shape}")

        ra_mag = self.mag_conv(amp)  # (B, 4, 128, 128)
        # print(f"ra_mag  {ra_mag.shape}")
        # 后续编码器-解码器部分保持不变
        mag_features = self.mag_encoder(ra_mag)
        # print(f"mag_features[0]  {mag_features[0].shape}")
        # print(f"mag_features[1]  {mag_features[1].shape}")
        # print(f"mag_features[2]  {mag_features[2].shape}")
        # print(f"mag_features[3]  {mag_features[3].shape}")
        fused_feat1 = self.fusion1(mag_features[0])
        fused_feat2 = self.fusion2(mag_features[1])
        fused_feat3 = self.fusion3(mag_features[2])
        fused_feat4 = self.fusion4(mag_features[3])

        return fused_feat4
class SimpleFusionBackbone_magfusion(nn.Module):
    def __init__(self, input_channels=8, freq_bins=89, time_steps=10):
        super(SimpleFusionBackbone_magfusion, self).__init__()
        self.freq_bins1 = freq_bins
        self.freq_bins1 = 128
        self.real_feat_ext_cnn = nn.Conv2d(in_channels=input_channels, out_channels=self.freq_bins1,
                                           kernel_size=(3, time_steps), stride=1, padding=(1, 0), bias=True)
        self.imag_feat_ext_cnn = nn.Conv2d(in_channels=input_channels, out_channels=self.freq_bins1,
                                           kernel_size=(3, time_steps), stride=1, padding=(1, 0), bias=True)
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
        self.pha_encoder = Encoder(in_channels=8,
                                   channels_list=[self.encoderw1, self.encoderw2, self.encoderw3, self.encoderw4])
        # 融合层和解码器（保持不变）
        self.fusion1 = ChannelFusion_ablation(in_channels=2 * self.encoderw1, out_channels=self.fusionw1)
        self.fusion2 = ChannelFusion_ablation(in_channels=2 * self.encoderw2, out_channels=self.fusionw2)
        self.fusion3 = ChannelFusion_ablation(in_channels=2 * self.encoderw3, out_channels=self.fusionw3)
        self.fusion4 = ChannelFusion_ablation(in_channels=2 * self.encoderw4, out_channels=self.fusionw4)
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
        ra_pha = self.pha_conv(pha)  # (B, 4, 128, 128)
        # print(f"ra_mag  {ra_mag.shape}")
        # 后续编码器-解码器部分保持不变

        pha_features = self.pha_encoder(ra_pha)
        # print(f"mag_features[0]  {mag_features[0].shape}")
        # print(f"mag_features[1]  {mag_features[1].shape}")
        # print(f"mag_features[2]  {mag_features[2].shape}")
        # print(f"mag_features[3]  {mag_features[3].shape}")
        fused_feat1 = self.fusion1(pha_features[0])
        fused_feat2 = self.fusion2(pha_features[1])
        fused_feat3 = self.fusion3(pha_features[2])
        fused_feat4 = self.fusion4(pha_features[3])

        return fused_feat4

class CrossModalAttention(nn.Module):
    """实部和虚部的交叉注意力机制"""

    def __init__(self, channels=48, num_heads=8):
        super(CrossModalAttention, self).__init__()

        # 自注意力层
        self.real_self_attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True
        )

        self.imag_self_attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True
        )

        # 交叉注意力层
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True
        )

        # 层归一化
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.norm3 = nn.LayerNorm(channels)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.ReLU(),
            nn.Linear(channels * 4, channels)
        )

    def forward(self, real_features, imag_features):
        # 输入: 多个 [B, 48, 128, 1] 的特征
        # 首先平均所有分支
        real_mean = sum(real_features) / len(real_features)  # [B, 48, 128, 1]
        imag_mean = sum(imag_features) / len(imag_features)  # [B, 48, 128, 1]

        # 转换为序列 [B, 128, 48]
        real_seq = real_mean.squeeze(-1).transpose(1, 2)  # [B, 128, 48]
        imag_seq = imag_mean.squeeze(-1).transpose(1, 2)  # [B, 128, 48]

        # 1. 自注意力
        real_attn, _ = self.real_self_attn(real_seq, real_seq, real_seq)
        imag_attn, _ = self.imag_self_attn(imag_seq, imag_seq, imag_seq)

        real_seq = self.norm1(real_seq + real_attn)
        imag_seq = self.norm1(imag_seq + imag_attn)

        # 2. 交叉注意力
        real_cross, _ = self.cross_attention(real_seq, imag_seq, imag_seq)
        imag_cross, _ = self.cross_attention(imag_seq, real_seq, real_seq)

        real_seq = self.norm2(real_seq + real_cross)
        imag_seq = self.norm2(imag_seq + imag_cross)

        # 3. FFN
        real_ffn = self.ffn(real_seq)
        imag_ffn = self.ffn(imag_seq)

        real_seq = self.norm3(real_seq + real_ffn)
        imag_seq = self.norm3(imag_seq + imag_ffn)

        # 转换回原形状 [B, 48, 128, 1]
        real_output = real_seq.transpose(1, 2).unsqueeze(-1)
        imag_output = imag_seq.transpose(1, 2).unsqueeze(-1)

        return real_output, imag_output
class SimpleFusionBackbone1(nn.Module):
    def __init__(self, input_channels=8, freq_bins=89, time_steps=10):
        super(SimpleFusionBackbone1, self).__init__()
        self.freq_bins1 = freq_bins
        self.freq_bins1 = 2
        self.target_dim = 128
        self.adaptive_pool = nn.AdaptiveAvgPool2d((self.target_dim, 1))
        self.attention_fusion = CrossModalAttention(channels=48)
        self.real_feat_ext_cnn = nn.Conv2d(in_channels=input_channels, out_channels=self.freq_bins1,
                                           kernel_size=(3, time_steps), stride=1, padding=(1, 0), bias=True)
        self.imag_feat_ext_cnn = nn.Conv2d(in_channels=input_channels, out_channels=self.freq_bins1,
                                           kernel_size=(3, time_steps), stride=1, padding=(1, 0), bias=True)
        self.real_feat_ext_cnn1 = nn.Conv2d(in_channels=input_channels, out_channels=self.freq_bins1,
                                           kernel_size=(3*2, time_steps), stride=1, padding=(1, 0), bias=True)
        self.imag_feat_ext_cnn1 = nn.Conv2d(in_channels=input_channels, out_channels=self.freq_bins1,
                                           kernel_size=(3*2, time_steps), stride=1, padding=(1, 0), bias=True)
        self.real_feat_ext_cnn2 = nn.Conv2d(in_channels=input_channels, out_channels=self.freq_bins1,
                                           kernel_size=(3*3, time_steps), stride=1, padding=(1, 0), bias=True)
        self.imag_feat_ext_cnn2 = nn.Conv2d(in_channels=input_channels, out_channels=self.freq_bins1,
                                           kernel_size=(3*3, time_steps), stride=1, padding=(1, 0), bias=True)

        self.imag_feat_ext_cnn11 = nn.Conv2d(in_channels=self.freq_bins1, out_channels=48,
                                            kernel_size=(3, 1), stride=1, padding=(1, 0), bias=True)
        self.real_feat_ext_cnn11 = nn.Conv2d(in_channels=self.freq_bins1, out_channels=48,
                                            kernel_size=(3, 1), stride=1, padding=(1, 0), bias=True)
        self.imag_feat_ext_cnn21 = nn.Conv2d(in_channels=self.freq_bins1, out_channels=48,
                                            kernel_size=(3, 1), stride=1, padding=(1, 0), bias=True)
        self.real_feat_ext_cnn21 = nn.Conv2d(in_channels=self.freq_bins1, out_channels=48,
                                            kernel_size=(3, 1), stride=1, padding=(1, 0), bias=True)
        self.imag_feat_ext_cnn31 = nn.Conv2d(in_channels=self.freq_bins1, out_channels=48,
                                            kernel_size=(3, 1), stride=1, padding=(1, 0), bias=True)
        self.real_feat_ext_cnn31 = nn.Conv2d(in_channels=self.freq_bins1, out_channels=48,
                                            kernel_size=(3, 1), stride=1, padding=(1, 0), bias=True)

    def forward(self, x):
        batch_size, channels, freq_bins, time_steps = x.shape
        x_2d = x  # (B, 10, 8, 89)
        # print(f"x_2d{x_2d.shape}") #x_2dtorch.Size([32, 8, 89, 10])
        x_real = x_2d.real.float()
        x_imag = x_2d.imag.float()

        x_real_ext = self.real_feat_ext_cnn(x_real)
        x_imag_ext = self.imag_feat_ext_cnn(x_imag)
        x_real_ext1 = self.real_feat_ext_cnn1(x_real)
        x_imag_ext1 = self.imag_feat_ext_cnn1(x_imag)
        x_real_ext2 = self.real_feat_ext_cnn2(x_real)
        x_imag_ext2 = self.imag_feat_ext_cnn2(x_imag)
        # print(f"x_real_ext  {x_real_ext.shape}")
        # print(f"x_real_ext1  {x_real_ext1.shape}")
        # print(f"x_real_ext2  {x_real_ext2.shape}")

        x_real_ext11 = self.real_feat_ext_cnn11(x_real_ext)
        x_imag_ext11 = self.imag_feat_ext_cnn11(x_imag_ext)

        x_real_ext21 = self.real_feat_ext_cnn21(x_real_ext1)
        x_imag_ext21 = self.imag_feat_ext_cnn21(x_imag_ext1)

        x_real_ext31 = self.real_feat_ext_cnn31(x_real_ext2)
        x_imag_ext31 = self.imag_feat_ext_cnn31(x_imag_ext2)



        x_real_ext11 = self.adaptive_pool(x_real_ext11)  # [32, 48, 128, 1]
        x_imag_ext11 = self.adaptive_pool(x_imag_ext11)
        x_real_ext21 = self.adaptive_pool(x_real_ext21)
        x_imag_ext21 = self.adaptive_pool(x_imag_ext21)
        x_real_ext31 = self.adaptive_pool(x_real_ext31)
        x_imag_ext31 = self.adaptive_pool(x_imag_ext31)

        # print(f"x_real_ext11  {x_real_ext11.shape}")
        # print(f"x_real_ext21  {x_real_ext21.shape}")
        # print(f"x_real_ext31  {x_real_ext31.shape}")

        real_features = [x_real_ext11, x_real_ext21, x_real_ext31]
        imag_features = [x_imag_ext11, x_imag_ext21, x_imag_ext31]
        real_fused, imag_fused = self.attention_fusion(real_features, imag_features)

        # print(f"real_fused  {real_fused.shape}")
        amp = torch.sqrt(real_fused ** 2 + imag_fused ** 2).squeeze(3)
        pha = torch.atan2(imag_fused, real_fused).squeeze(3)
        # print(f"amp  {amp.shape}")
        output = amp + pha
        output = output.contiguous()
        return output



