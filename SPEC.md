# 穿墙雷达数据与原始方案说明

> 已裁剪：仅保留原始方案与数据说明部分，在线跟踪方案、模块实施表与实验记录已移除。

## 当前实现模块：my_track 教师关联课程训练与稳定性诊断（真实训练复核中）

- 2026-10-23 验证提速：完整记录验证按 `VAL_SEQUENCE_BATCH_SIZE=8` 合批独立序列，每序列独立维护 TrackBank、按原时间顺序递归，结束的序列不填充。验证位置损失保持原逐帧等权口径，MAE/IDSW 等仍逐序列计算。冷/暖路径合成串并行一致性测试通过；实际 GPU 加速幅度待测。文件读取仍为按需串行；不改变训练入口、学习率或输出目录。
- 2026-10-23 定点回退：按用户要求撤销模块 10 的自主主损失/教师辅助双路径训练和融合分母 `1e-3` 下限，恢复单路径教师课程与 dtype tiny 分母保护。保留模块 9 的训练/验证独立权重、模块 11 的完整序列验证合批、当前学习率与输出目录；39 项合成测试中 38 项通过，唯一失败为旧测试写死学习率，详见 `CODE_LOG.md` 模块 12。真实训练效果尚未复测。
- 2026-10-23 训练调权试验：新增独立 `TRACKER_TRAIN_LOSS_WEIGHTS`，训练总损失为 `2*association + 5*position + 0.25*existence + 0.1*confidence`；验证继续使用原 `TRACKER_LOSS_WEIGHTS`（position 权重为 1）。原始子损失、MAE 和其他指标定义不变，checkpoint 额外记录训练权重。当前 `LR=3e-5`、输出目录、模型和教师课程均不修改；按用户要求仅完成代码，不运行测试、验证或训练，效果待后续实测。
- 目的：在冻结 M1 的前提下，先用固定 GT 槽标签指导候选融合与坐标纠偏，再逐步过渡到模型自主关联，降低早期错误关联对位置回归和历史状态的污染。
- 训练接口：`JointTracker.forward(...)` 仅在训练时接收可选的候选标签 `[B,50]` 与混合比例 `r∈[0,1]`；有效融合分配为 `r * one_hot(GT标签) + (1-r) * ST(模型关联)`。
- 推理/验证：不传标签且 `r=0`，必须严格保持自主关联的原有行为。
- 退火：前 `TEACHER_ASSOC_WARMUP_EPOCHS` 轮取 `r=1`，随后经 `TEACHER_ASSOC_ANNEAL_EPOCHS` 线性降至 `0`；关联 CE 始终训练。
- 验证：合成测试覆盖教师融合、零比例等价及调度边界；2026-10-22 全套 33 项 `my_track` 单元测试与 `py_compile` 已通过。真实数据首轮使用 `LR=1e-3`，在第 5 个完整 epoch 后中止：训练损失持续下降，但自主验证关联 CE 由 `3.55` 升至 `12.12`，位置 MAE 发生大幅波动，因此该运行不作为模型选择依据。
- 当前受控对照：除学习率降为 `3e-4` 和新增只读诊断外，冻结 M1、数据划分、初始化流程、损失、教师退火和验证自主推理均保持不变；输出写入独立目录，避免混入首轮日志。
- 启动状态：2026-10-22 受控对照尚未成功运行。首次被沙箱拒绝创建 DataLoader 管道；获准在沙箱外启动后，首个 batch 前向发生 CUDA OOM，进程退出，没有优化更新、epoch 指标或新 checkpoint。原配置保持 `BATCH_SIZE=32`、`num_workers=8`，未自动缩批或重启。

## 1. 数据与已有基础

- 雷达体制：2 发 4 收，共 8 个通道。
- 单样本信号：`complex64 (8,89,10)`；89 为距离单元，10 为样本内部雷达时间步，复数保存实部和虚部。
- 原始标签：`float32 (10,6)`；对 10 行取平均后重构为 `(3,2)`，全零槽 `(0,0)` 表示无目标。
- 坐标单位为米；当前快速数据集中的有效 `y` 已加 `3.5 m` 偏移，不得重复添加。抽样覆盖全部序列后，有效 `y` 约为 `0.98–6.02 m`。
- 探测区域（物理边界）：`x ∈ [-2.7, 2.7] m`，`y ∈ [0.8, 6.2] m`（实测范围，与 CODE_LOG 一致）。
- 文件夹命名为 `日期_人数_序号`；同一文件夹是一条独立雷达记录，样本按数字文件名连续排列。
- 训练集含 `新建文件夹`、`新建文件夹 (2)` 两个可疑目录（可能为空或测试数据），读取前需确认。
- 相邻 `.npy` 约间隔 24–25 ms，40 个样本约 1 秒；禁止跨文件夹构造时间序列。
- 每条序列人数固定为 1–3，过程中无真实人员进入或离开；模型必须自主判断人数。
- 同一序列内 GT 槽位顺序固定，可作为序列内伪 ID；ID 不要求跨序列一致。
- 遮挡样本：训练集保留、测试集剔除（`delete_occlusion` 训练传 `False`、测试传 `True`，见 §2.1）；模型应在短时遮挡、靠近和交叉时维持原 ID。
- 可复用资产：已发表的时空特征提取网络、DETR 结构与预训练权重、现有数据划分和标签。

## 2. 数据读取方式

### 2.1 主流程（当前实际使用）

入口文件 `original/123ren.py` 的 `__main__` 中：

```python
train_dataset = load_fast_dataset1(
    data_root_folder=r"E:\Jupyter\track\fast_dataset_multi_train",
    seq_length=10, add_noise=True,  delete_occlusion=False)
test_dataset = load_fast_dataset1(
    data_root_folder=r"E:\Jupyter\track\fast_dataset_multi_test",
    seq_length=10, add_noise=False, delete_occlusion=True)
```

- `load_fast_dataset1(...)` 定义在 `original/my_dataset/fast_sample_dataset.py`，返回 `FastNpyDataset1` 实例。
- `FastNpyDataset1`：
  - `__init__` → `_build_sample_list()`：遍历根目录下所有子文件夹（**每个文件夹 = 一条完整序列**），收集 `signals/` 与 `labels/` 子目录中**同名** `.npy` 文件对，并记录所在文件夹名；`delete_occlusion=True` 时再调用 `_filter_occluded_samples()`，按每个文件夹里的 `遮挡样本id统计.txt`（跳过标题行）剔除遮挡样本 id。
  - `__getitem__(idx)`：`np.load` 读信号与标签 → `torch.from_numpy` → 可选 `_add_gaussian_noise`（std=0.03，50% 概率，实部虚部分别加）→ 返回 `(signal, label)`。
  - 返回 shape：signal `complex64 (8,89,10)`；label `float32 (10,6)`（6 列 = 3 目标 × (x,y)）。

### 2.2 标签后处理（训练循环内）

`original/my_train/simpletrain.py` 的 `train_epoch_with_temporal_avg` / `validate_with_temporal_avg`：

1. `coordinates[i].mean(dim=0)`：把 `(10,6)` 的 10 帧平均 → `(6,)`；
2. `.reshape(-1, 2)` → `(3,2)`；
3. `add_wh_to_targets(coords, normalize_coords=False)`（定义在 `original/my_dataset/YoLostylelabel.py`）→ `(3,4)` = `[x, y, w, h]`，固定宽高 `w=h=0.4`，全零槽保持为零。

## 3. 网络结构（original/123ren.py 的 `Mymodule`）

整体框架：

```
complex (8,89,10)
  → SimpleFusionBackbone（时空特征提取）→ (B,128,8,6)
  → 展平成 48 个 token (B,48,128) + 正弦位置编码
  → DETR Encoder（4 层，128 维，8 头自注意力）
  → DETR Decoder（3 层，50 个 object query）
  → 50 个候选：(50,4) [x,y,w,h] + (50,2) 置信度
```

`Mymodule` 定义在 `original/123ren.py`（`class Mymodule`），`__init__` 组装四个部分：

1. **`self.backbone = SimpleFusionBackbone(input_channels=8, freq_bins=89, time_steps=10)`**
   定义在 `original/my_models/CNNBackbone.py`。实部/虚部分别过 `Conv2d(8→128, kernel=(3,10))` 提取 → 合成幅度 `amp` 与相位 `pha` → 各过 `Conv2d(1→8)` → 幅度/相位**两条独立的 4 层 CNN Encoder**（通道 16→32→64→128，每层 3×3 卷积 + stride2 下采样）→ 每一层用 `ChannelFusion`（1×1 卷积，幅度+相位 concat 后压缩）逐层融合 → 输出 `fused_feat4`，形状 `(B,128,8,6)`。

2. **`self.DETR_Encoder`**：`ModuleList([EncoderLayer(attention_dim=128, attention_heads=8) for _ in range(4)])`，4 层。`EncoderLayer` 定义在 `original/my_models/DETR_Encoder_Decoder.py`：多头自注意力（128 维、8 头）+ FFN，LayerNorm 残差。
   前向先把 backbone 输出 `view(B,128,-1)` 再 `transpose` 成 `(B,48,128)`（48 个 token），加上 `self.position_encoding`（`_create_sin_cos_encoding(48,128)` 生成的正弦位置编码）。

3. **`self.DETR_Decoder = Decoder(d_model=128, num_queries=50, num_layers=3)`**
   `Decoder` 定义在 `DETR_Encoder_Decoder.py`：50 个可学习 object query + 正弦位置编码；3 层 `DecoderLayer`，每层 = **掩码自注意力 + 对 encoder 输出的交叉注意力 + FFN**。输出头含 `coord_headx`、`coord_heady`、`classifier`（2 类）、`conf_head`。

4. **输出**：训练时返回 3 个解码层的全部输出（`all_outputs`，每层 `(coor+wh, conf)`，按 `[0.2,0.2,0.6]` 加权）；测试时只返回最后一层 `(out, conf)`，`out` shape `(B,50,4)`、`conf` shape `(B,50,2)`。

## 4. 已确认项

| # | 问题 | 结论 |
|---|------|------|
| 1 | 时空特征提取输出维度 | `(128,8,6)`，展平为 48 个 token |
| 2 | 单样本与标签 | complex64 `(8,89,10)`；标签平均后 `(3,2)`，空槽 `(0,0)` |
| 3 | y 坐标 | 有效 y 已加 3.5m，不得重复偏移 |
| 4 | 序列边界 | 每文件夹一条序列，禁止跨文件夹拼接 |
| 5 | 时间尺度 | 相邻 `.npy` 约 24–25ms，40 样本约 1 秒 |
| 6 | 人数与 ID | 每序列固定 1–3 人；模型自主判断；ID 仅需序列内稳定 |
| 7 | 遮挡 | 保留并要求维持 ID |
| 8 | 训练硬件 | GTX 1660 Super 6GB，目标约 10 分钟/epoch 以内 |

## 5. 损失函数

主流程用 `hungarian_loss_with_classification`（`original/lossfunc/hungarian.py`，
`123ren.py` 中 `criterion = hungarian_loss_with_classification`）。

调用签名（`original/my_train/simpletrain.py` 的 `train_epoch_with_temporal_avg` / `validate_with_temporal_avg`）：

    coord_loss, class_loss = criterion(conf, out, target_coords_with_wh, epoch, is_train=...)

- 返回**两个标量**：`coord_loss`（坐标回归）+ `class_loss`（分类），相加为总损失。
- 输入：`conf` (B,50,2) 分类 logits、`out` (B,50,4) 为 `[x,y,w,h]`、`target` (B,3,4)。
- 匹配：每样本用**中心点 `(x,y)`** 的欧氏距离矩阵 (50×n) 做匈牙利匹配
  （`scipy.optimize.linear_sum_assignment`）；**w/h 不参与匹配、也不参与回归**（固定 0.4）。
- `coord_loss`：匹配对用 `SmoothL1Loss(beta=0.5)` 回归中心点；未匹配的 50−n 个 query
  加 SmoothL1(坐标, 0) 惩罚拉向原点。
- `class_loss`：`CrossEntropyLoss`，匹配 query 标签=1（有目标）、未匹配=0（背景）。
- 无有效目标时：所有 query 坐标惩罚到 0、分类标签全 0。
- 训练时 3 个解码层输出按 `[0.2, 0.2, 0.6]` 加权（`123ren.py` 的 `layer_weights`）；
  验证/测试只用最后一层。

## 6. 训练超参与环境

来自 `original/123ren.py` 的 `__main__`（当前实际配置）：

| 项 | 值 |
|---|---|
| 优化器 | `AdamW(lr=1e-3, betas=(0.9,0.999), weight_decay=1e-4, eps=1e-8)` |
| 学习率调度 | `CosineAnnealingLR(T_max=40, eta_min=1e-4)` |
| epoch | 40 |
| batch_size | 32（train，shuffle，`num_workers=0`）；test 不 shuffle |
| 随机种子 | 42（`torch.Generator().manual_seed(42)`，用于 train loader） |
| 解码器层权重 | `[0.2, 0.2, 0.6]` |
| 最佳模型保存名 | `best_model_multilayerloss128___121.pth`（含 model/optimizer state_dict + epoch） |
| 训练硬件 | GTX 1660 Super 6GB，目标 ≤10 分钟/epoch |

工具路径：
- Python：`E:\Miniconda3\python.exe`
- Git：`E:\Git\bin\git.exe`

## 7. 实际使用的文件（`123ren.py` 主流程）

| 文件 | 使用的内容 |
|---|---|
| `original/123ren.py` | 主入口：`Mymodule` + 测试 |
| `original/my_train/simpletrain.py` | `train_epoch_with_temporal_avg`、`validate_with_temporal_avg` |
| `original/my_models/CNNBackbone.py` | `SimpleFusionBackbone` |
| `original/my_models/DETR_Encoder_Decoder.py` | `EncoderLayer`、`Decoder` |
| `original/my_dataset/fast_sample_dataset.py` | `load_fast_dataset1` → `FastNpyDataset1` |
| `original/my_dataset/YoLostylelabel.py` | `add_wh_to_targets` |
| `original/lossfunc/hungarian.py` | `hungarian_loss_with_classification` |
| `original/abilitycalculate/count_parameters.py` | `count_parameters`（仅打印参数量） |

## 8. 已知坑

- **`add_wh_to_targets` 的 normalize 分支有 bug**：`normalize_coords=True` 时 x、y **都按
  `[-2.7, 2.7]` 归一化**，但真实 y∈[0.8, 6.2]；主流程用 `normalize_coords=False` 规避。
  **若日后启用归一化，必须修正 y 的归一化区间。**
- **损失只回归中心点**：`hungarian_loss_with_classification` 用 `[:, :2]` 匹配和回归，w/h 不参与（见 §5）。
- **损失签名含未生效参数**：`hungarian_loss_with_classification` 的 `epoch`/`is_train` 实际不影响计算（`mse_weight` 恒为 1.0）。
- **训练/测试遮挡处理不对称**：训练保留遮挡、测试剔除（见 §1、§2.1）。
