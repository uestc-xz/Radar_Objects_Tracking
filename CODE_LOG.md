# 代码变更日志

## my_track 联合航迹模块 — 2026-10-21 至 2026-10-22

- 目标：用固定身份槽的联合 tracker 替换旧的逐帧一对一关联训练路径；按候选融合、预测/观测状态分离和整记录身份连续性组织实现。
- 配置：新增监督半径、身份确认初值、漏检置信度衰减、质量标签尺度、Transformer 结构、M1 分块大小及四项损失权重。旧 `CONF_THRESHOLD` 仅保留兼容，不用于新 tracker 门控。
- `association.py` 输入/输出：候选 `[B,M,2]` 与 GT `[B,3,2]` 生成固定槽标签 `[B,M]`；`tracker_losses` 返回 association/position/existence/confidence 四项标量。支持空 GT、同槽多个候选和有限梯度。
- `joint_tracker.py` 输入/输出：`Measurements(pos[B,50,2],score[B,50],feat[B,50,128])` 与 TrackBank 状态字典，输出四类候选关联及 `[B,3]` 槽位坐标/观测/置信字段。候选不做一对一匹配；槽内按 M1 score 加权，回归头可纠正至候选范围外。
- `track_state.py` 输入/输出：每帧模型最终坐标、融合特征、观测标记和概率，更新三槽历史并供下一帧导出。HIST_LEN 预热后按滚动存在均值确认；确认 ID 不因漏检清除。
- `train_tracker.py` / `train.py`：冷路径直接接缓存，暖路径冻结并分块运行 M1；滑窗所有帧计算损失，窗口起点 reset。只优化 JointTracker；验证按完整文件夹逐帧回放并记录 IDSW、欧氏位置误差、目标覆盖率、漏检保持、观测/预测计数。
- 训练可观测性：新增 `SHOW_PROGRESS` 开关；每个 epoch 分别显示训练 batch 和验证序列进度，动态更新平均 loss、学习率及累计验证帧数，结束汇总格式保持不变。
- `track_sequence_dataset.py` / `metrics.py`：默认窗口模式保持原行为；新增只返回帧路径索引的 full-sequence 模式，整序列验证按需读取。槽位映射在 warm-up 只初始化一次并固定。
- 验证：补齐目标覆盖率与训练进度行为测试后，全量 27 项标准库测试通过；`py_compile` 与 `git diff --check` 通过。
- 未验证：未读取真实 M1 缓存做正标签比例/确认阈值扫描；未运行长序列训练；未覆盖已有 `original/my_track/runs/` 文件。

## 数据集探索 — 2026-06-29

- 样本路径：`E:\Jupyter\track\fast_dataset_multi_train/` 和 `fast_dataset_multi_test/`
- 信号 shape：(8, 89, 10)，dtype=complex64，实部范围约 [-0.05, 0.15]，虚部范围约 [-0.15, 0.05]
- 标签 shape：(10, 6)，dtype=float32，6 列 = 3 目标 × (x, y)，坐标范围 x∈[-2.7, 2.7], y∈[0.8, 6.2]
- 文件夹命名：`日期_人数_序号`，如 `20240913_3_1` = 3人，`20240913_1_3` = 1人
- 训练集：24 个文件夹（含 新建文件夹），测试集：10 个文件夹
- 每文件夹内 signals/ 和 labels/ 子目录，.npy 格式，一一对应
- 抽样验证：连续 10 个样本的标签身份顺序一致 ✅（如 T1/T2/T3 的坐标范围各自稳定，无身份交换）
- 遮挡文件：部分文件夹有 `遮挡样本id统计.txt`
- 异常发现：训练集中含 `新建文件夹` 和 `新建文件夹 (2)`，可能为空或测试数据
- **精度评估**：当前已使用 complex64 + float32，符合低精度策略。可进一步将 int64 索引改为 int32

