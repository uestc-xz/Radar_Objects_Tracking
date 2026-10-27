"""Transformer-DA 的 Transformer 网络部分。

本文件只实现论文中的编码器、解码器和关联分类头，不包含卡尔曼滤波、
轨迹管理或训练循环。

与论文的适配约定：
    - 论文单步目标状态为 [t, x, vx, y, vy]，这里省略时间 t；
    - 每个历史时刻使用 [x, vx, y, vy, conf_bg, conf_fg] 共 6 维；
    - 默认使用 4 个历史时刻，因此单条航迹编码特征为 4 * 6 = 24 维；
    - 当前数据集最多 50 个 M1 量测，类别为 0=杂波、1~3=三个目标。
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
from torch import nn


class DANetOutput(NamedTuple):
    """网络输出：逐量测分类 logits/概率，以及逐量测速度回归位移。"""

    logits: torch.Tensor
    probabilities: torch.Tensor
    velocities: torch.Tensor


class SinusoidalPositionEncoding(nn.Module):
    """标准正弦位置编码，避免为固定序列位置引入可学习参数。"""

    def __init__(self, d_model: int, max_length: int):
        super().__init__()
        positions = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        encoding = torch.zeros(max_length, d_model)
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        encoding[:, 1::2] = torch.cos(positions * frequencies[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        return sequence + self.encoding[:, : sequence.shape[1]]


class DANet(nn.Module):
    """论文 Transformer-DA 的网络主体。

    Args:
        num_tracks: 最大目标数；当前数据集为 3。
        history_steps: 使用的历史时刻数；论文默认 4。
        measurement_dim: 当前量测特征维度；当前适配为 M1 的 4 维输出。
        num_classes: 杂波加目标类别数；当前为 4。
    """

    def __init__(self, num_tracks: int = 3, history_steps: int = 4, measurement_dim: int = 4,
            num_classes: int = 4, d_model: int = 512, num_heads: int = 8, encoder_layers: int = 6,
            decoder_layers: int = 6, feedforward_dim: int = 512, dropout: float = 0.1, max_measurements: int = 50,):
        super().__init__()
        self.num_tracks = num_tracks
        self.history_steps = history_steps
        self.measurement_dim = measurement_dim
        self.num_classes = num_classes
        self.max_measurements = max_measurements

        self.history_state_dim = 6
        history_dim = history_steps * self.history_state_dim
        self.history_projection = nn.Linear(history_dim, d_model)
        self.measurement_projection = nn.Linear(measurement_dim, d_model)
        self.encoder_position = SinusoidalPositionEncoding(d_model, num_tracks)
        self.decoder_position = SinusoidalPositionEncoding(d_model, max_measurements)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)
        self.classifier = nn.Linear(d_model, num_classes)
        # 速度回归头：与分类头共享解码器输出，回归逐量测的 (vx, vy) 位移。
        self.velocity_head = nn.Linear(d_model, 2)

    def forward(self, history_states: torch.Tensor, measurements: torch.Tensor,) -> DANetOutput:
        """前向传播。

        Args:
            history_states: [B, num_tracks, history_steps, 6]，每步为
                [x, vx, y, vy, conf_bg, conf_fg]。
            measurements: [B, num_measurements, 4]，每个量测为
                [x, y, conf_bg, conf_fg]。

        Returns:
            DANetOutput：logits/probabilities [B, num_measurements, num_classes]，
            以及 velocities [B, num_measurements, 2]（相邻帧位置差位移）。
        """
        batch_size, num_tracks, history_steps, state_dim = history_states.shape
        batch_size_m, num_measurements, measurement_dim = measurements.shape
        history = history_states.reshape(batch_size, num_tracks, history_steps * state_dim)
        encoder_tokens = self.encoder_position(self.history_projection(history))
        memory = self.encoder(encoder_tokens)

        decoder_tokens = self.decoder_position(self.measurement_projection(measurements))
        decoded = self.decoder(decoder_tokens, memory)
        logits = self.classifier(decoded)
        probabilities = logits.softmax(dim=-1)
        # 速度回归：单位为相邻帧位置差（未除以 Δt，Δt 为常数会被权重吸收）。
        velocities = self.velocity_head(decoded)
        return DANetOutput(logits=logits, probabilities=probabilities, velocities=velocities)
