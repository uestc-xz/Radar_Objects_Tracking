"""
M2 量测生成 (Measurement)
=========================
作用：
    把定位前端 M1 输出的三路张量（坐标 / 置信度 logits / 特征）打包为结构化的
    「量测集合」，供后续 M5（残差 GRU）、M6（融合打分）、M7（关联决策）使用。
    本模块无学习参数，是纯数据封装。

输入（来自 M1，即 Mymodule 推理分支的三路返回）：
    coor : [B, 50, 2]   每个 query 回归出的中心坐标 (x, y)
    conf : [B, 50, 2]   conf_head 的两分类 logits（[0]背景 / [1]有目标）
    feat : [B, 50, 128] dec_output 每个 query 的 128 维特征向量

输出（Measurements 命名元组）：
    pos   : [B, 50, 2]   中心坐标 (x, y)，单位米
    score : [B, 50]      目标置信度 s_i = softmax(conf)[..., 1] ∈ [0,1]
    feat  : [B, 50, 128] 128 维身份特征 f_i = dec_output[i]

约定：
    跟踪逐帧进行，实际 batch=1；此处保留 B 维以兼容模型批量推理。
"""
from typing import NamedTuple

import torch
import torch.nn.functional as F


class Measurements(NamedTuple):
    """量测集合 M = { m_i = (p_i, s_i, f_i) }，i = 1..50。

    字段：
        pos   : [B, 50, 2]   中心坐标 p_i = (x_i, y_i)
        score : [B, 50]      目标置信度 s_i
        feat  : [B, 50, 128] 身份特征 f_i
    """
    pos: torch.Tensor
    score: torch.Tensor
    feat: torch.Tensor


def generate_measurements(coor, conf, feat):
    """M2 主入口：把 M1 三路输出打包成量测集合。

    Args:
        coor : [B, 50, 2]   中心坐标 (x, y)
        conf : [B, 50, 2]   两分类 logits
        feat : [B, 50, 128] 128 维特征

    Returns:
        Measurements(pos=[B,50,2], score=[B,50], feat=[B,50,128])
    """
    # s_i = P(该 query 是真实目标) = softmax(conf) 的「有目标」通道（索引 1）
    score = F.softmax(conf, dim=-1)[..., 1]  # [B, 50]
    return Measurements(pos=coor, score=score, feat=feat)
