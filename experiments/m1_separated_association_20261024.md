# 短窗排序与可匹配性分项监督 — 2026-10-24

基线 `65998dc`；模块 30。沿用上一轮条件历史诊断范围，不改变旧 tracker/M1/配置，不声称闭环身份收益。用户已授权自行细化、实现、验证；提交由用户执行。

## 证据与假设

上一轮错误接收 4102→2047，但 positive 误拒 153→980；竞争 pure ranking 821/958→822/958。代码中的联合 softmax 同时训练排序及拒绝，候选头能通过整体压低 logits 来降低 null 样本损失；当前这只是可确认的梯度路径，不是全部退化的因果证明。

手算 logits `[null=0,c1=1,c2=.5]`：null 标签 CE=1.680270，候选梯度 `[.506480,.307196]`，候选整体被压低；c1 标签 CE=.680270，候选梯度 `[-.493520,.307196]`，候选梯度和=-.186324。候选头承担了可匹配性学习；条件排序的候选梯度和应为零。

假设 A：将排序与是否匹配的参数梯度分离，能减少排序分支学拒绝的干扰。假设 B：竞争只占训练查询 2.07%，单独平均这类排序损失能提高困难候选学习贡献。分别做固定对照，不在验证集扫描权重、温度或阈值。

## 冻结条件与接口

- 数据生成完全不动：.5m GT 代理配对、历史先查询后更新、sample 年龄、主/弱提议、歧义标记、查询分组、训练验证划分和指纹均不变。
- `ShortWindowAssociation.forward(..., detach_rejection_history=False)` 增加仅控制 autograd 的显式选项。默认计算不变；开启时 `reject(encoding.detach(), context)`，数值、输出类别/shape、参数和 checkpoint 键完全相同。
- 单候选/多候选保留，当前无候选的查询仍计指标；模型接口仅量测，没有 GT 或目标槽编号。只改独立短窗训练脚本，不改联合 tracker。
- 新 `original/my_track/separated_association_loss.py` 负责数值损失与梯度边界；现有短窗脚本新增 `--objectives joint separated separated_balanced`，共享一次数据加载，各分支相同 seed7/batch512/lr1e-3/AdamW/clip5/两轮。

## 分项公式与路由

标签 `y=0` 表示 null，`y=j+1` 表示正候选 j。valid candidate 集合为 V，mask 不进入分母。

- `L_rank(q)=CE(candidate_logits[V], y-1)`，只在 positive 查询计算。只有一个 valid 候选时 CE=0，不伪造负候选。普通分项版对所有 positive 等权平均。
- `L_match(q)=BCEWithLogits(max(candidate_logits[V]).detach()-null_logit, y>0)`，只在 V 非空的查询计算；margin>0 的阈值与既有 argmax 接收一致，使用 max 而非 logsumexp 是显式的新对照，不声称只是联合 CE 的代数展开。
- 新总损失 `L_rank + L_match`，两项权重固定 1。null BCE 不向候选 logits 回传，拒绝历史 detach 阻断其更新历史编码器；排序不含 null logit，不更新拒绝头。
- balanced 版的 `L_rank=.5*mean(competition positive CE)+.5*mean(other positive CE)`，两组都有时成立；只有一组则该组全权，全部为空返回可反传的零。组平衡按每个训练 batch 计算；验证分项使用全验证集的各组计数汇总，不平均 batch 均值。
- V 为空时，推理始终 null，训练该查询分项为零；仍保留在共同 joint CE、正确率和查询分母。非法 positive 指向 padding 必须拒绝，NaN/Inf 拒绝。
- 手算零 logits、两 valid 候选、positive：rank=log(2)，match=log(2)，total=2log(2)。两个组的 rank CE 分别 log(2)、log(4)，balanced=1.5log(2)，不随复制容易组样本而改变。

## 执行与验收

1. 先写数值和真实模型参数路由测试，再实现分项损失和前向 detach；原测试必须继续通过。检查单独 backward 时两组参数梯度是否隔离，不用 mock 验证。
2. 扩展训练/evaluate：默认 joint 不变；显式记录 objective、分项与计数；旧共同 joint NLL、预测、分组和纯排序保持原定义。多分支输出独立子目录，存在即拒绝覆盖。每轮 checkpoint 写 objective。
3. 真实前缀先检查 shape、dtype、标签/历史及进度/文件；完整数据构造一次，joint/两项分离/再平衡各两轮。复现上一轮 joint，并核对 train/validation query 指纹。
4. 保存每轮、逐序列和错误类型对照；只有多目标竞争排序有可解释收益、误拒和整体无实质退化时，才讨论后续闭环。没有净收益则不推广，保留失败证据。

## 状态

实现、前缀和完整三组实验均已完成；保留为可选失败对照，不切换默认方案。每组分别训练两轮，不是同一模型连续训练六轮。

- 13 项新增测试通过，最终全套 238 项通过（18.808s）；四个相关 Python 文件编译与 diff 检查通过。新增的全局验证组均值测试覆盖 batch_size=1/5，防止改变验证 batch 划分后组权重悄悄变化。
- 输入抽样：训练首序列 sample 1--5 形成查询 sample 3/4/5，history float32 `[3,20,5]`，candidates float32 `[3,50,4]`，mask bool，labels int64 `[1,1,1]`。沿用原数据格式和 y 坐标，不引入新标签变换。
- 前缀 25 条训练共 5,000 帧/10,246 查询，8 条验证共 1,600 帧/2,765 查询。joint 的模型参数逐张量完全复现旧 smoke，旧有全部数值字段相同；新字段是诊断扩展，不能直接按两个 JSON 字典严格相等判断。
- 第二个真实训练 batch 的前缀梯度审计：普通分离 rank 的 history/edge/reject 梯度 L2 约 `8.78e-7/.000410/0`，match 的对应梯度 `0/0/.08224`；分组平衡 rank 约 `.0000172/.02573/0`，match 为 `0/0/.08235`。这是单 batch 路由证据，不是梯度规模的全数据估计或效果结论。

## 完整数据结果

25 条训练共 59,525 帧、122,861 查询；8 条验证共 17,421 帧、31,184 查询，positive/null/competition 为 `24674/6510/958`。训练 positive 为 101,308，competition 为 2,548；无候选训练查询只有 2 个，matchability 分母为 122,859，验证该分母为 31,184。采样、标签、数据划分与模块 29 完全相同。

三组 query SHA-256 均相同，并复现模块 29：

```text
train      ab2397472c2713d9b14cc7f641b3d18bf3b94c55d2ddcfac20999d3185271ed3
validation 16c52a36ac892c1c7835f7a0c52cb7ce16ea8f3f29a50c752dd8195ca796d30e
```

原 joint 的初始及两轮旧有全部验证数值、train NLL 均相同，epoch 2 模型参数逐张量 `torch.equal=True`；checkpoint 新增 objective 字段，不能以文件 SHA 相同作为兼容要求。全量实际第二 batch 的 rank history/edge/reject 梯度 L2：普通分离 `2.22e-6/.003413/0`、平衡 `1.11e-5/.007916/0`；match 对应为 `0/0/.105183`、`0/0/.105240`，确认梯度路由生效。

下表共同 val NLL 始终按旧 joint CE；训练目标在各分支含义不同，不能横向按数值高低评优。竞争纯排序排除 null，仅用于诊断；“含拒绝”列使用实际 argmax 预测。

| 分支 | epoch | 实际训练目标 | 共同 val NLL | 全部正确率 | 竞争含拒绝正确/958 | 竞争纯排序正确/958 | null 正确率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 最近观测，未训练 | 0 | — | .365830 | .858164 | 821 | 821 | .369892 |
| joint | 1 | .250593 | .268045 | .897095 | 803 | 822 | .650077 |
| joint | 2 | .228123 | .263005 | .898955 | 788 | 822 | .685561 |
| separated | 1 | .270823 | .283182 | .896934 | 800 | 828 | .642243 |
| separated | 2 | .249388 | .285078 | .897095 | 793 | 825 | .657450 |
| separated_balanced | 1 | .393544 | .285584 | .895684 | 805 | 826 | .634255 |
| separated_balanced | 2 | .372146 | .287571 | .896197 | 802 | 829 | .649002 |

epoch 2 的错误类型及条件候选准确率：

| 分支 | positive 正确率 | positive 错选候选 | positive 误拒 | null 错接 | 竞争错选候选 | 竞争误拒 |
|---|---:|---:|---:|---:|---:|---:|
| 最近观测，未训练 | .986990 | 168 | 153 | 4102 | 137 | 0 |
| joint | .955257 | 124 | 980 | 2047 | 116 | 54 |
| separated | .960323 | 126 | 853 | 2230 | 117 | 48 |
| separated_balanced | .961417 | 126 | 826 | 2285 | 117 | 39 |

这里 positive 正确表示选中代理候选标签，不是坐标误差指标。新分支减少部分误拒，但同时增加 null 错接；竞争纯排序的小幅收益未形成总体净改善。balanced 的纯排序正确 829，比 joint 多 7，比最近观测多 8；含拒绝正确 802，低于最近观测 821。不能以错选数减少直接推断真实身份交换减少。

固定 epoch 2 逐序列竞争纯排序正确数如下，不能事后逐序列挑选分支：

| 序列 | 分母 | 最近观测 | joint | separated | separated_balanced |
|---|---:|---:|---:|---:|---:|
| 20240913_1_1 | 49 | 44 | 44 | 45 | 45 |
| 20240913_1_2 | 30 | 29 | 27 | 27 | 28 |
| 20240913_3_3 | 99 | 90 | 90 | 91 | 92 |
| 20240913_3_8 | 115 | 102 | 105 | 104 | 103 |
| 20240915_1_4 | 199 | 144 | 143 | 144 | 144 |
| 20240916_1_4 | 51 | 45 | 45 | 44 | 44 |
| 20240916_2_3 | 198 | 170 | 169 | 170 | 173 |
| 20240916_2_5 | 217 | 197 | 199 | 200 | 200 |

验证没有一致的逐序列收益。尤其困难三人纯排序 `105→104→103`，增加竞争组权重仍未改善该记录；单人 `20240915_1_4` 含 93 个几何歧义查询，不能将该序列的 199 个“竞争”全部解释为真实多目标交叉。

epoch 2 的全局训练诊断 rank/match 为：普通分离 `.015282/.234088`，balanced `.135597/.235583`；对应验证 rank/match 为 `.022239/.267881`、`.195353/.269861`。balanced 排序均值按两组等权定义，与普通均值不是同一尺度。epoch 1→2 训练目标降低，而共同验证 NLL 两个新分支均略升；两轮不足以断言泛化机制，但没有支持延长训练或立即推广的证据。

三组平均裁剪前梯度（epoch 1/2）：joint `.094764/.102715`，普通分离 `.078239/.083786`，balanced `.100391/.106936`，无 NaN/Inf。数据一次构造 train/val 耗时 `250.0/61.4s`；各轮训练加验证约 `3.2/3.9/4.0s`（三分支），这些是小型缓存查询模型的耗时，不能外推旧联合 tracker 的 epoch 时长。

完整产物位于 `original/my_track/runs/separated_assoc_full_20261024/{joint,separated,separated_balanced}/`，各含 manifest 与 epoch_1/2.pt。epoch 2 checkpoint 文件 SHA-256：

```text
joint              f7c33b15b67f67f827c4897fb25309a086a0ce41d77613197be11d7f4dbf7ef6
separated          65af23fc9e73841d7f27de5de136c4657911d81c7039793d9ca6a6ef3e493a34
separated_balanced 995e111a9b1839b2cea5059e2462fda0a7dea89a24db317145a1688aade83365
```

结论：假设 A/B 仅获得很弱的条件排序支持，均未达到含拒绝和总体无实质退化的验收条件。本轮不改默认 joint，不接入生命周期闭环，不扫描验证集阈值补救。下一步优先查清竞争代理标签歧义和真实多目标候选冲突的分布，再设计全局一对一的训练/推理对照；这只是后续方向，尚未实现或验证。

## 命令、字段与限制

```powershell
# 已执行：各序列前 200 帧，三个分支各两轮
python scripts/train_m1_short_window_association.py --epochs 2 --max-frames 200 --workers 4 --no-progress --objectives joint separated separated_balanced --output-dir original/my_track/runs/separated_assoc_smoke_20261024
# 完整数据：一次加载，三个分支各两轮
python scripts/train_m1_short_window_association.py --epochs 2 --workers 4 --no-progress --objectives joint separated separated_balanced --output-dir original/my_track/runs/separated_assoc_full_20261024
```

多分支各写子目录 `joint/separated/separated_balanced`，包含 manifest 与逐轮 checkpoint；已有目标目录拒绝覆盖。单分支不加参数仍为原 joint，显式选择单分支可用 `--objectives separated`，其输出直接写指定目录。没有自动选择最佳轮次，没有长训练或阈值扫描。

`train_objective` 是各训练 batch 实际总损失按查询数加权的在线均值；`train_components` 按各分项的各自样本计数汇总诊断均值。特别是 balanced 时，batch 内组均值的平均不等于全 epoch 的组均值，因而其 `total` 未必等于 `train_objective`，二者不能混用。验证 `objective_components` 按完整查询集的各组计数计算，对 batch 划分不敏感。共同 `validation.all.nll` 始终是原含 null 的联合 CE，不能拿新的训练总目标和它直接比较下降幅度。

梯度分离也不表示两个预测在数值上独立：拒绝 margin 仍读取候选最大值，拒绝头仍读取排序分支产生的历史表示，只是不由拒绝损失更新它们。候选排序变化仍可能影响接收结果。当前源代码针对冻结输入，未声称控制将来解冻 M1 时的输入梯度。

代理几何标签、条件历史、独立查询无全局一对一以及验证集多次探索的限制全部保留；本轮不产生闭环 IDF1、系统检测召回或最终测试集结论。
