import torch
import torch.nn as nn
import torch.nn.functional as F
from .old_version import old_version_modle
from .old_version import old_version_modle_for_new_start
from .old_version import old_version_multi_modle
class OLD_TimeSeries_Backbone(nn.Module):
    def __init__(self, input_channels=8, freq_bins=89, hidden_size=640, num_layers=2):
        super(OLD_TimeSeries_Backbone, self).__init__()

        self.input_channels = input_channels
        self.freq_bins = freq_bins
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.GRUA = nn.GRU(
            input_size=freq_bins,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True
        )
        self.GRUP = nn.GRU(
            input_size=freq_bins,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True
        )

        self.shared_output_layersA = nn.ModuleList(
            [self._create_shared_output_layer(hidden_size) for _ in range(input_channels)]
        )
        self.shared_output_layersP = nn.ModuleList(
            [self._create_shared_output_layer(hidden_size) for _ in range(input_channels)]
        )

    def _create_shared_output_layer(self, hidden_size):
        return nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            # 最初版本没加
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 384),
            # 最初版本没加
            nn.LayerNorm(384),
            nn.ReLU(),
            nn.Linear(384, 384), nn.ReLU(),
            nn.Linear(384, 128), nn.ReLU(),
        )

    def forward(self, x):
        batch_size, channels, freq_bins, time_steps = x.shape
        x_real = x.real.float()
        x_imag = x.imag.float()
        x_ampl = x.abs().float()
        x_phas = torch.atan2(x_imag, x_real)

        x_ampl = x_ampl / (x_ampl.abs().amax(dim=[1, 2, 3], keepdim=True) + 1e-8)


        x_ampl = x_ampl / (x_ampl.abs().amax(dim=[2], keepdim=True))
        # x_phas = x_phas / (x_phas.abs().amax(dim=[2], keepdim=True) + 1e-8)###相位归一化并不好！！！！！
        output_shared = torch.zeros(batch_size, time_steps, 128, device=x.device)
        for i in range(channels):

            input_tempampl = x_ampl[:, i, :, :].permute(0, 2, 1)
            input_tempphas = x_phas[:, i, :, :].permute(0, 2, 1)

            gru_outampl, _ = self.GRUA(input_tempampl)
            gru_outphas, _ = self.GRUP(input_tempphas)

            out_ampl = self.shared_output_layersA[i](gru_outampl)
            out_phas = self.shared_output_layersP[i](gru_outphas)

            output_shared += (out_ampl + out_phas)
            # print(output_shared.shape)
        return output_shared

class OLD_TimeSeries_Backbone_for_new_start(nn.Module):
    def __init__(self, input_channels=8, freq_bins=89, hidden_size=640, num_layers=2,attention_heads=3):
        super(OLD_TimeSeries_Backbone_for_new_start, self).__init__()

        self.input_channels = input_channels
        self.freq_bins = freq_bins+1
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.attention_dim = freq_bins+1
        self.attention_heads = attention_heads
        self.head_dim = (freq_bins+1) // attention_heads
        self.norm1 = nn.LayerNorm(self.attention_dim)
        self.norm2 = nn.LayerNorm(self.attention_dim)
        # 检查维度是否能被头数整除
        assert self.head_dim * attention_heads == (freq_bins+1), "attention_dim必须能被attention_heads整除"
        self.GRUA = nn.GRU(
            input_size=freq_bins,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(self.attention_dim, self.attention_dim * 4),
            nn.ReLU(),
            nn.Linear(self.attention_dim * 4,self.attention_dim)
        )
        self.out_proj = nn.Linear(self.attention_dim, self.attention_dim)
        self.Pq_proj = nn.Linear(self.attention_dim,self.attention_dim)

        self.Akv_proj = nn.Linear(self.attention_dim,2*self.attention_dim)



    def _create_shared_output_layer(self, hidden_size):
        return nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 384),
            nn.ReLU(),
            nn.Linear(384, 384), nn.ReLU(),
            nn.Linear(384, 128), nn.ReLU(),
        )

    def forward(self, x):
        batch_size, channels, freq_bins, time_steps = x.shape
        # print(x[0,0,0,0])
        xpad = F.pad(x, (0, 0, 0, 1, 0, 0, 0, 0))
        # print(xpad[0, 0, 0, 0])
        # print(xpad.shape)
        xpad_real = xpad.real.float()
        xpad_imag = xpad.imag.float()
        x_pad_phas = torch.atan2(xpad_imag, xpad_real)
        x_pad_ampl = xpad.abs().float()
        x_pad_ampl = x_pad_ampl / (x_pad_ampl.abs().amax(dim=[2], keepdim=True))
        x_pad_ampl = x_pad_ampl.mean(dim = 1)
        x_pad_phas = x_pad_phas.mean(dim=1)
        residual = x_pad_phas.view(batch_size,time_steps,-1)
        # print(f"x_pad_ampl shape : {x_pad_ampl.shape}")


        q = self.Pq_proj(x_pad_phas.view(batch_size,time_steps,-1))
        kv = self.Akv_proj(x_pad_ampl.view(batch_size,time_steps,-1))
        # print(f"q(来自 Phas) shape : {q.shape}")
        # print(f"kv(来自 Ampl) shape : {kv.shape}")
        q = q.reshape(batch_size, time_steps, self.attention_heads, self.head_dim)
        kv = kv.reshape(batch_size, time_steps, 2, self.attention_heads, self.head_dim)
        k , v  = kv.unbind(2)
        # print(f"k shape{k.shape}v shape : {v.shape}")

        q = q.transpose(1, 2)  # [batch_size, heads, time_steps_x, head_dim]
        k = k.transpose(1, 2)  # [batch_size, heads, time_steps_y, head_dim]
        v = v.transpose(1, 2)  # [batch_size, heads, time_steps_y, head_dim]
        attn_weights = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, H, T_x, T_y]
        # print(f"attn_weights shape : {attn_weights.shape}")
        attn_weights = torch.softmax(attn_weights, dim=-1)

        # # 应用注意力权重到V
        attn_output = attn_weights @ v  # [B, H, T_x, D_h]
        # print(f"attn_output shape :{attn_output.shape}")

        attn_output = attn_output.transpose(1, 2).contiguous()  # [B, T_x, H, D_h]
        attn_output = attn_output.reshape(batch_size, time_steps, self.attention_dim)

        # 输出投影和残差连接
        attn_output = self.out_proj(attn_output)
        # print(f"attn_output shape :{attn_output.shape}")
        attn_output = residual + attn_output
        residual = attn_output
        attn_output_norm = self.norm2(attn_output)
        # print(f"attn_output_norm shape :{attn_output_norm.shape}")
        ffn_output = self.ffn(attn_output_norm)
        output = residual + ffn_output
        # print(f"output shape :{output.shape}")

        return output
def get_old_backbone_feat(x=None, old_model_path="老的网络_xy分开计算loss-2-分阶段训练-2-1-2allc.pth", device="cpu",
                          return_model=False):
    # 加载原始完整网络的权重
    old_full_net = old_version_modle(
        input_channels=8,
        freq_bins=89,
        hidden_size=128,
        num_layers=2,
        num_targets=1
    ).to(device)
    # old_full_net.load_state_dict(torch.load(old_model_path, map_location=device), strict=False)
    old_full_net.eval()

    # 初始化新的骨干网络
    backbone = OLD_TimeSeries_Backbone(
        input_channels=8,
        freq_bins=89,
        hidden_size=128,
        num_layers=2
    ).to(device)

    # # 权重移植
    # backbone.GRUA.load_state_dict(old_full_net.GRUA.state_dict())
    # backbone.GRUP.load_state_dict(old_full_net.GRUP.state_dict())
    # for i in range(backbone.input_channels):
    #     backbone.shared_output_layersA[i].load_state_dict(
    #         old_full_net.shared_output_layersA[i].state_dict()
    #     )
    #     backbone.shared_output_layersP[i].load_state_dict(
    #         old_full_net.shared_output_layersP[i].state_dict()
    #     )
    #
    # backbone.eval()
    # random_tensor = torch.randn(1, 8, 89, 15)
    # old_out = old_full_net(random_tensor)
    # backbone_out = backbone(random_tensor)
    # 如果只需要模型，就返回模型
    if return_model:
        return backbone

    # 否则返回特征（保持原有功能）
    with torch.no_grad():
        output_shared = backbone(x.to(device))
    return output_shared

def get_old_backbone_feat_for_new_start(x=None, old_model_path="老的网络_xy分开计算loss-2-分阶段训练-2-1-2allc.pth", device="cpu",
                          return_model=False):
    # 加载原始完整网络的权重
    old_full_net = old_version_modle_for_new_start(
        input_channels=8,
        freq_bins=89,
        hidden_size=75,
        num_layers=2,
        num_targets=1
    ).to(device)
    # old_full_net.load_state_dict(torch.load(old_model_path, map_location=device), strict=False)
    old_full_net.eval()

    # 初始化新的骨干网络
    backbone = OLD_TimeSeries_Backbone_for_new_start(
        input_channels=8,
        freq_bins=89,
        hidden_size=75,
        num_layers=2
    ).to(device)

    if return_model:
        return backbone

    # 否则返回特征（保持原有功能）
    with torch.no_grad():
        output_shared = backbone(x.to(device))
    return output_shared

class OLD_TimeSeries_Multi_Backbone(nn.Module):
    def __init__(self, input_channels=8, freq_bins=89, hidden_size=640, num_layers=2, num_targets=3):
        super(OLD_TimeSeries_Multi_Backbone, self).__init__()
        self.num_targets = num_targets
        self.input_channels = input_channels
        self.freq_bins = freq_bins
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

    def _create_shared_output_layer(self, hidden_size,num_targets=1):
        return nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size , 384), nn.ReLU(),
            nn.Linear(384, 384), nn.ReLU(),
            nn.Linear(384, 128), nn.ReLU(),

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

        return output_shared

def get_old_multi_backbone_feat(x=None, old_model_path="老的网络2-多人-用于收集backbone参数.pth", device="cpu",
                          return_model=False):
    # 加载原始完整网络的权重
    old_full_net = old_version_multi_modle(
        input_channels=8,
        freq_bins=89,
        hidden_size=640,
        num_layers=2,
        num_targets=3
    ).to(device)
    old_full_net.load_state_dict(torch.load(old_model_path, map_location=device), strict=False)
    old_full_net.eval()

    # 初始化新的骨干网络
    backbone = OLD_TimeSeries_Multi_Backbone(
        input_channels=8,
        freq_bins=89,
        hidden_size=640,
        num_layers=2
    ).to(device)

    # 权重移植
    backbone.GRUA.load_state_dict(old_full_net.GRUA.state_dict())
    backbone.GRUP.load_state_dict(old_full_net.GRUP.state_dict())
    for i in range(backbone.input_channels):
        backbone.shared_output_layersA[i].load_state_dict(
            old_full_net.shared_output_layersA[i].state_dict()
        )
        backbone.shared_output_layersP[i].load_state_dict(
            old_full_net.shared_output_layersP[i].state_dict()
        )

    backbone.eval()
    # random_tensor = torch.randn(1, 8, 89, 15)
    # old_out = old_full_net(random_tensor)
    # backbone_out = backbone(random_tensor)
    # 如果只需要模型，就返回模型
    if return_model:
        return backbone

    # 否则返回特征（保持原有功能）
    with torch.no_grad():
        output_shared = backbone(x.to(device))
    return output_shared