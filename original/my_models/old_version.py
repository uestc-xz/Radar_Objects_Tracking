import torch
import os
import torch.nn as nn

class old_version_modle_for_new_start(nn.Module):
    def __init__(self, input_channels=8, freq_bins=89, hidden_size=640, num_layers=2, num_targets=1):
        super(old_version_modle_for_new_start, self).__init__()
        self.num_targets = num_targets
        self.freq_bins = freq_bins
        self.x_loss = nn.MSELoss()
        self.y_loss = nn.MSELoss()
        # 双向GRU处理时间序列
        # 幅度走向
        self.GRUA = nn.GRU(
            input_size=freq_bins,
            hidden_size=hidden_size, num_layers=num_layers,
            bidirectional=True, batch_first=True)
        # 相位走向
        self.GRUP = nn.GRU(
            input_size=freq_bins,
            hidden_size=hidden_size, num_layers=num_layers,
            bidirectional=True, batch_first=True)

        # 为每个通道设计自己的实部、虚部和相位的线性层
        self.shared_output_layersA = nn.ModuleList([self._create_shared_output_layer(hidden_size) for _ in range(input_channels)])
        self.shared_output_layersP = nn.ModuleList([self._create_shared_output_layer(hidden_size) for _ in range(input_channels)])
        self.out_putx = nn.ModuleList(self._create_output_layerx() for _ in range(input_channels))
        self.out_puty = nn.ModuleList(self._create_output_layery() for _ in range(input_channels))

    def _create_shared_output_layer(self, hidden_size,num_targets=1):
        return nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size , 384), nn.ReLU(),
            nn.Linear(384, 384), nn.ReLU(),
            nn.Linear(384, 128), nn.ReLU(),
        )

    def _create_output_layerx(self, num_targets=1):
        return nn.Sequential(
            nn.Linear(128 , 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, num_targets*1)
        )
    def _create_output_layery(self, num_targets=1):
        return nn.Sequential(
            nn.Linear(128 , 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, num_targets * 1)
        )

    def forward(self, x):
        # x形状: [batch_size, 8, 300, 25] (complex64)
        batch_size, channels, freq_bins, time_steps = x.shape
        num_targets = 1 #根据数据集人数调整
        # 分离实部和虚部
        x_real = x.real.float()
        x_imag = x.imag.float()
        x_ampl = x.abs().float()
        x_phas = torch.atan2(x_imag, x_real)  # 相位计算
        # 幅度归一化

        x_ampl = x_ampl / (x_ampl.abs().amax(dim=[1, 2, 3], keepdim=True) + 1e-8)

        output_shared = torch.zeros(batch_size, time_steps, 128, device=x.device)

        for i in range(channels):
            input_tempampl = x_ampl[:, i, :, :].permute(0, 2, 1)  # [batch, time, freq]
            input_tempphas = x_phas[:, i, :, :].permute(0, 2, 1)

            gru_outampl, _ = self.GRUA(input_tempampl)  # [batch, time, 2*hidden_size]
            gru_outphas, _ = self.GRUP(input_tempphas)

            out_ampl = self.shared_output_layersA[i](gru_outampl)
            out_phas = self.shared_output_layersP[i](gru_outphas)

            output_shared += (out_ampl + out_phas )  # 各通道结果相加
            # output_shared += out_ampl
            outx = self.out_putx[i](output_shared)
            outy = self.out_puty[i](output_shared)
            # print(outx.shape)
            output = torch.cat([outx,outy],dim = 2)

        return output


def get_old_verssion_feat_for_new_start(x=None, old_model_path="老的网络_xy分开计算loss-2-分阶段训练-2-1-2allc.pth", device="cpu",
                          return_model=False):
    # 加载原始完整网络的权重
    old_full_net = old_version_modle_for_new_start(
        input_channels=8,
        freq_bins=89,
        hidden_size=640,
        num_layers=2,
        num_targets=1
    ).to(device)
    # old_full_net.load_state_dict(torch.load(old_model_path, map_location=device), strict=False)
    for param in old_full_net.parameters():
        param.requires_grad = True

    # 设置为训练模式
    old_full_net.train()
    # old_full_net.eval()

    if return_model:
        return old_full_net

    # 否则返回特征（保持原有功能）
    with torch.no_grad():
        output_shared = old_full_net(x.to(device))
    return output_shared




class old_version_modle(nn.Module):
    def __init__(self, input_channels=8, freq_bins=89, hidden_size=640, num_layers=2, num_targets=1):
        super(old_version_modle, self).__init__()
        self.num_targets = num_targets
        self.freq_bins = freq_bins
        self.x_loss = nn.MSELoss()
        self.y_loss = nn.MSELoss()
        # 双向GRU处理时间序列
        # 幅度走向
        self.GRUA = nn.GRU(
            input_size=freq_bins,
            hidden_size=hidden_size, num_layers=num_layers,
            bidirectional=True, batch_first=True)
        # 相位走向
        self.GRUP = nn.GRU(
            input_size=freq_bins,
            hidden_size=hidden_size, num_layers=num_layers,
            bidirectional=True, batch_first=True)
        self.GRUPD = nn.GRU(
            input_size=freq_bins,
            hidden_size=hidden_size, num_layers=num_layers,
            bidirectional=True, batch_first=True)

        # 为每个通道设计自己的实部、虚部和相位的线性层
        self.shared_output_layersA = nn.ModuleList([self._create_shared_output_layer(hidden_size) for _ in range(input_channels)])
        self.shared_output_layersP = nn.ModuleList([self._create_shared_output_layer(hidden_size) for _ in range(input_channels)])
        self.out_putx = nn.ModuleList(self._create_output_layerx() for _ in range(input_channels))
        self.out_puty = nn.ModuleList(self._create_output_layery() for _ in range(input_channels))

    def _create_shared_output_layer(self, hidden_size,num_targets=1):
        return nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size , 384), nn.ReLU(),
            nn.Linear(384, 384), nn.ReLU(),
            nn.Linear(384, 128), nn.ReLU(),
        )

    def _create_output_layerx(self, num_targets=1):
        return nn.Sequential(
            nn.Linear(128 , 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, num_targets*1)
        )
    def _create_output_layery(self, num_targets=1):
        return nn.Sequential(
            nn.Linear(128 , 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, num_targets * 1)
        )

    def forward(self, x):
        # x形状: [batch_size, 8, 300, 25] (complex64)
        batch_size, channels, freq_bins, time_steps = x.shape
        num_targets = 1 #根据数据集人数调整
        # 分离实部和虚部
        x_real = x.real.float()
        x_imag = x.imag.float()
        x_ampl = x.abs().float()
        x_phas = torch.atan2(x_imag, x_real)  # 相位计算  
        # 幅度归一化
        x_ampl = x_ampl / (x_ampl.abs().amax(dim=[1, 2, 3], keepdim=True) + 1e-8)
        output_shared = torch.zeros(batch_size, time_steps, 128, device=x.device)

        for i in range(channels):
            input_tempampl = x_ampl[:, i, :, :].permute(0, 2, 1)  # [batch, time, freq]
            input_tempphas = x_phas[:, i, :, :].permute(0, 2, 1)

            gru_outampl, _ = self.GRUA(input_tempampl)  # [batch, time, 2*hidden_size]
            gru_outphas, _ = self.GRUP(input_tempphas)

            out_ampl = self.shared_output_layersA[i](gru_outampl)
            out_phas = self.shared_output_layersP[i](gru_outphas)

            output_shared += (out_ampl + out_phas )  # 各通道结果相加
            outx = self.out_putx[i](output_shared)
            outy = self.out_puty[i](output_shared)
            # print(outx.shape)
            output = torch.cat([outx,outy],dim = 2)

        return output

def get_old_verssion_feat(x=None, old_model_path="老的网络_xy分开计算loss-2-分阶段训练-2-1-2allc.pth", device="cpu",
                          return_model=False):
    # 加载原始完整网络的权重
    old_full_net = old_version_modle(
        input_channels=8,
        freq_bins=89,
        hidden_size=640,
        num_layers=2,
        num_targets=1
    ).to(device)
    # old_full_net.load_state_dict(torch.load(old_model_path, map_location=device), strict=False)
    for param in old_full_net.parameters():
        param.requires_grad = True

    # 设置为训练模式
    old_full_net.train()
    # old_full_net.eval()

    if return_model:
        return old_full_net

    # 否则返回特征（保持原有功能）
    with torch.no_grad():
        output_shared = old_full_net(x.to(device))
    return output_shared


class old_version_multi_modle(nn.Module):
    def __init__(self, input_channels=8, freq_bins=89, hidden_size=640, num_layers=2, num_targets=3):
        super(old_version_multi_modle, self).__init__()
        self.num_targets = num_targets
        self.freq_bins = freq_bins
        self.x_loss = nn.MSELoss()
        self.y_loss = nn.MSELoss()
        # 双向GRU处理时间序列
        # 幅度走向
        self.GRUA = nn.GRU(
            input_size=freq_bins,
            hidden_size=hidden_size, num_layers=num_layers,
            bidirectional=True, batch_first=True)
        # 相位走向
        self.GRUP = nn.GRU(
            input_size=freq_bins,
            hidden_size=hidden_size, num_layers=num_layers,
            bidirectional=True, batch_first=True)
        self.GRUPD = nn.GRU(
            input_size=freq_bins,
            hidden_size=hidden_size, num_layers=num_layers,
            bidirectional=True, batch_first=True)

        # 为每个通道设计自己的实部、虚部和相位的线性层
        self.shared_output_layersA = nn.ModuleList([self._create_shared_output_layer(hidden_size) for _ in range(input_channels)])
        self.shared_output_layersP = nn.ModuleList([self._create_shared_output_layer(hidden_size) for _ in range(input_channels)])
        self.shared_output_layersPD = nn.ModuleList([self._create_shared_output_layer(hidden_size) for _ in range(input_channels)])
        self.out_putx = nn.ModuleList(self._create_output_layerx() for _ in range(input_channels))
        self.out_puty = nn.ModuleList(self._create_output_layery() for _ in range(input_channels))

    def _create_shared_output_layer(self, hidden_size,num_targets=1):
        return nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size , 384), nn.ReLU(),
            nn.Linear(384, 384), nn.ReLU(),
            nn.Linear(384, 128), nn.ReLU(),
            # nn.Linear(128, 64), nn.ReLU(),
            # nn.Linear(64, num_targets*1)
        )

    def _create_output_layerx(self, num_targets=3):
        return nn.Sequential(
            nn.Linear(128 , 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, num_targets*1)
        )
    def _create_output_layery(self, num_targets=3):
        return nn.Sequential(
            nn.Linear(128 , 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, num_targets * 1)
        )

    def forward(self, x):
        # x形状: [batch_size, 8, 300, 25] (complex64)
        batch_size, channels, freq_bins, time_steps = x.shape
        num_targets = 1 #根据数据集人数调整
        # 分离实部和虚部
        x_real = x.real.float()
        x_imag = x.imag.float()
        x_ampl = x.abs().float()
        x_phas = torch.atan2(x_imag, x_real)  # 相位计算  # 避免除以零
        # 幅度归一化
        x_ampl = x_ampl / (x_ampl.abs().amax(dim=[1, 2, 3], keepdim=True) + 1e-8)
        output_shared = torch.zeros(batch_size, time_steps, 128, device=x.device)

        for i in range(channels):
            input_tempampl = x_ampl[:, i, :, :].permute(0, 2, 1)  # [batch, time, freq]
            input_tempphas = x_phas[:, i, :, :].permute(0, 2, 1)

            gru_outampl, _ = self.GRUA(input_tempampl)  # [batch, time, 2*hidden_size]
            gru_outphas, _ = self.GRUP(input_tempphas)

            out_ampl = self.shared_output_layersA[i](gru_outampl)
            out_phas = self.shared_output_layersP[i](gru_outphas)

            output_shared += (out_ampl + out_phas )  # 各通道结果相加
            # print(output_shared)
            # ppp
            outx = self.out_putx[i](output_shared)
            outy = self.out_puty[i](output_shared)
            # print(outx.shape)
            output = torch.cat([outx,outy],dim = 2)

        return output