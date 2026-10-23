# 冻结 M1 特征身份投影实验（模块 24）

## 问题与边界

模块 23 的原始 128 维 M1 特征作为当前关联次级代价，没有在八条验证序列上降低 IDSW。本实验只回答一个窄问题：这些 embedding 是否已有短时身份信号，只是原始空间/相似度不适合直接用于 tracker？

实验不修改 M1、M1 缓存、默认 `LifecycleTracker`、`FeatureLifecycleTracker` 的默认参数、联合训练器或现有损失。所有新行为通过两个独立脚本显式执行，投影 checkpoint 写入新实验目录。

## 标签与数据流

对每条记录独立处理，绝不把不同记录中的相同 GT 槽当作同一 ID。每帧从 M1 缓存读取候选坐标、置信度和特征；置信度取前景 softmax。仅保留分数 `>=0.8` 的候选，再以 `0.5 m` 距离门对候选与非空 GT 槽作最大基数一对一匹配。此 GT 配对只用于离线训练标签，不参与模型输入或 tracker 更新。

匹配候选的 3 元组标签为 `(特征, 序列内 GT 槽, 帧序号)`。同身份且帧间隔 1–8 的最近样本作为正例；不同身份且间隔不超过 1 帧的最近样本作为负例。缺正例或负例的 anchor 被跳过。三元组在每条序列内部独立构造，再拼接索引，禁止跨记录正/负配对。

模型为 `128→64→32` 两层 MLP，GELU 后输出 L2 单位向量。训练目标是余弦 triplet hinge：

`L = mean(max(0, cos(a,n) - cos(a,p) + 0.2))`。

这是身份度量头，不回归位置、身份分类 logits 或全局 ID 表。输入仍是冻结 M1 embedding，训练不运行 M1。

## 评估

每个 epoch 仅报告完整验证序列三元组上的 pairwise ranking accuracy（raw 与 projected 并列），不据此挑选 best checkpoint。最后一轮 checkpoint 在完整验证集同一 M1 cache 上依次与三个 tracker 变体比较：纯空间；原始 128D；投影 32D。所有变体使用模块 23 相同生命周期参数和 feature weight `.35`、max age `8`，GT 仅用于整段离线度量。输出检测 P/R/F1、匹配欧氏 MAE、GT-IDSW、identity-F1，并按序列列出结果。

判定需同时满足：

1. projected 验证三元组 pairwise accuracy 高于 raw；
2. 八序列 aggregate IDSW/identity-F1 相对纯空间和 raw feature 有明确净改善；
3. 检测 F1 和匹配 MAE 没有实质退化；
4. 改善不只来自单条序列，须查看每序列分布及最难三目标序列。

若检索变好但 tracker 指标不变，结论是 embedding 可分性不等于当前生命周期算法可利用，不推广。若训练和验证 triplet 指标严重背离，结论为泛化不足，不增加网络复杂度。

## 实现文件

- `original/my_track/m1_feature_projection.py`：投影 MLP、GT 门控配对、序列内三元组生成及损失。
- `scripts/train_m1_feature_projection.py`：独立冷缓存训练器，拒绝覆盖已有目录，输出 checkpoint 和 manifest。
- `scripts/evaluate_m1_projected_feature_association.py`：完整序列 spatial/raw/projected 同帧回放。
- `tests/my_track/test_m1_feature_projection.py`：投影形状/归一化/输入校验、候选门控配对、身份样本时间边界、triplet loss。

## 验证与结果

实现阶段 7 项单测先在模块缺失时按预期 RED；实现后目标测试 7/7 通过，相关 Python 文件可编译。

## 首轮结果（2026-10-23）

使用 25 条训练序列和 8 条完整验证序列，`score_threshold=.8`、候选/GT 距离门 `0.5 m`、正样本最大间隔 8 帧、负样本最大间隔 1 帧，MLP 训练 2 轮、batch 512、AdamW `lr=1e-3`。训练匹配特征 86,506、三元组 83,018；验证匹配特征 17,914、三元组 16,916。

原始 embedding 在验证三元组上的正例相似度胜过负例准确率为 `.9030`；投影后第 1/2 轮为 `.9775/.9791`。这证明投影头能拟合并泛化到短时序列内的构造任务，但不等于 tracker 的身份指标一定改善。

完整八序列生命周期回放（相同空间/生命周期参数，feature weight `.35`、max age `8`）如下：

| 变体 | Precision | Recall | F1 | GT-IDSW | identity-F1 | MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| spatial | 0.9443 | 0.8338 | 0.8856 | 200 | 0.3965 | 0.2805 m |
| raw feature | 0.9437 | 0.8334 | 0.8851 | 203 | 0.3929 | 0.2804 m |
| projected feature | 0.9449 | 0.8349 | 0.8867 | 216 | 0.4242 | 0.2809 m |

逐序列变化不是一致改善：`20240915_1_4` identity-F1 `.3942→.6597`、IDSW `2→1`；三目标 `20240913_3_3` IDSW `42→51`，`20240916_2_3` `52→60`；最难 `20240913_3_8` IDSW `76→77` 且 identity-F1 `.1481→.1474`。因此投影特征在部分单目标记录上有效，但会在多目标竞争中增加身份所有权切换；检测 F1 只小幅提高、MAE 略升，不满足默认推广条件。

结论：保留独立投影和回放脚本作为 opt-in 诊断，不替换默认空间 tracker，也不把 `feature_weight` 提高。若继续研究，应先针对多目标交叉样本做按事件的门控/滞回实验，而不是继续扩大投影网络或训练轮数。

复现命令：

```powershell
E:\Miniconda3\python.exe scripts\train_m1_feature_projection.py --output-dir original\my_track\runs\feature_projection_trial1 --epochs 2 --batch-size 512 --device cuda
E:\Miniconda3\python.exe scripts\evaluate_m1_projected_feature_association.py --checkpoint original\my_track\runs\feature_projection_trial1\feature_projection.pt --output original\my_track\runs\feature_projection_trial1\validation_full.json --device cuda
```
