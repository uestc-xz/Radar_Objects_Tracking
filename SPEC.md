# 穿墙雷达数据与原始方案说明

> 已裁剪：仅保留原始方案与数据说明部分，在线跟踪方案、模块实施表与实验记录已移除。

## 实验模块 34：低置信持续 coast 输出（已验证，未推广）

- 在模块 33 `lifecycle_known_no_death_coast` 基础上新增显式 opt-in `lifecycle_known_no_death_persistent_coast`：只把 coast 输出置信度下限设为 0，沿用 `output_max_missed=HIST_LEN`；已知人数/no-death 假设与其余设置不变。默认变体集合不变。
- 全 8 序列 coast 真阳 `1048→1149`（+101），coast 假阳 `1225→3403`（+2178）；P/R/F1 `.8980/.6646/.7639→.8234/.6674/.7373`，identity-F1 `.6053→.5837`，IDSW `153→145`，matched MAE `.2814→.2828m`。
- 困难三目标序列 F1/identity-F1 `.6632/.4012→.6374/.3853`，IDSW 均为 64。极少的额外召回不足以抵消低置信假阳，未推广。
- 该实验仅用于量化“身份继续存活”与“持续输出坐标”不同；低置信预测维持身份状态，不代表其坐标可作为可靠航迹点输出。详见 `experiments/m1_persistent_coast_20261024.md`。

## 实验模块 33：已知人数无消亡下延长 coast 输出（已验证，未推广）

- 显式 opt-in `lifecycle_known_no_death_coast` 保持原置信度门，仅将输出漏检窗口设为 `HIST_LEN`；其余固定人数/no-death 设置不变，默认不启用。
- 全量 identity-F1 `.6091→.6053`、F1 `.7684→.7639`、IDSW `155→153`，召回 `.6542→.6646` 同时 precision `.9309→.8980`；困难三人 F1/IDF1 微增但 IDSW `62→64`。
- 以更多 coast 假点换少量召回并未提高身份汇总指标，不推广。结果见 `experiments/m1_known_no_death_recovery_20261024.md`。

## 实验模块 32：已知人数、无消亡与身份恢复组合（已验证，未推广）

- 对照 `lifecycle_known_no_death` 与 opt-in `lifecycle_known_no_death_recovery`；已知每序列人数仅作 max_tracks 和恢复唯一缺失判据，不用于每帧 GT 或量测分配。
- 组合恢复仅追加身份别名；保留每序列原模型输入及坐标/分数证据，默认变体集合不变。
- 完整验证中组合与无恢复版预测/evidence digest 一致，recovered 恒为 0，aggregate F1/identity-F1/IDSW 均为 `.7684/.6091/155`；困难三目标 `.6551/.3932/62`。身份恢复在 no-death 生命周期下冗余，不推广。
- 结果见 `experiments/m1_known_no_death_recovery_20261024.md`，JSON 为 `known_no_death_recovery_{hard,full}_20261024.json`。仅作为无消亡理想化对照，不是自然人数跟踪泛化结论。

## 实验模块 31：代理竞争监督审计与冻结排序闭环（已验证，未推广）

- 基线 `d1d61b8`；用户确认剩余第一轮：先检查监督，再复用三个固定 epoch 2 checkpoint，不新增训练或扫描参数。
- 审计：沿原 .5m 配对/条件历史和查询指纹，拆分单/多目标竞争、当前及历史代理歧义、候选可同时靠近多个 GT、不同查询最近候选冲突；GT 仅用于离线审计和最终指标，不进入 tracker。
- 闭环：原生命周期已有门控匈牙利，本轮只增加默认等价的配对扩展点。独立子类以实际出生/消费量测维护 HIST_LEN 历史，不插入预测点；先查询、后更新，删除/reset 清理。至少两次过去观测才调用模型，否则保留原空间代价；网络读全部主/弱提议、无 GT/人数先验/128D feat。
- 固定对照：原 cascade、原 joint 空间、joint 最近观测，以及 joint 下三个冻结模型的候选排序；各自并列同一身份恢复。原门控/最大基数一对一、出生/确认/输出/删除公式保留；不使用学习 null，不在验证集选择 epoch、权重或门限。
- 验收：手算有符号代价与最大基数匹配、候选冲突/无候选/非法值；真实自主历史的因果/过期/重置/出生/删除，旧基类输出复现、GT 扰动不改推理。先前缀、困难序列，再完整八记录；并列 1m/.5m 检测、召回、身份、换 ID、管理数及预测指纹，比较学习与固定排序、原系统。不以局部排序代替闭环。
- 实施与结果：`experiments/m1_association_closed_loop_20261024.md`；默认 tracker/M1/训练/配置不切换，Git 由用户执行。

## 实验模块 30：短窗排序与可匹配性分项监督（已验证，未推广）

- 基线：`65998dc`，模块 29 已确认总体 CE 下降主要伴随拒绝增加，竞争纯排序只净改善 1/958。先做梯度与样本贡献对照，不接入闭环。
- 数据/模型：复用模块 29 全部查询、标签、HIST_LEN、候选、初始化、网络参数和优化器；GT 不进入前向。只新增显式损失选项和拒绝分支历史 detach，默认 joint 及既有 checkpoint 保持兼容。
- 损失：`separated` 的排序为 positive 查询候选内 CE 均值；可匹配性为有候选查询的 BCE，输入 `max(candidate_logits).detach() - null_logit`。拒绝分支对历史编码 detach，可匹配性梯度只更新拒绝头；排序只更新历史编码/候选头。无候选查询确定输出 null，无须拟合一个虚构边。
- 竞争对照：`separated_balanced` 仅把排序改为竞争/其余 positive 各组均值的等权平均；某组为空时使用存在的组。数据不重采样、不删歧义，不改验证预测或旧联合 CE 口径。
- 验收：手算分项 CE/BCE、padding/空组/无候选、真实参数梯度路由、默认与 detach 前向逐位一致、旧 checkpoint 和旧 joint 复现；三个固定分支各两轮、完整八序列比较，原 query SHA-256 必须一致。分组和纯排序、误拒/错接及共同 joint CE 并列，不比较不同目标的 train 总数值高低。
- 结果：238 项测试通过；完整三组各两轮完成，joint 参数逐张量与模块 29 相同。epoch 2 原 joint/分离/分离加平衡的竞争纯排序为 `822/825/829`（分母 958），含拒绝的竞争正确数 `788/793/802`，总体正确率 `.89895/.89709/.89620`；最近观测基线竞争正确数 821。拆分与平衡有小幅排序收益，但未达净收益门槛，不推广、不接入闭环。困难三人纯排序 `105/104/103`（分母 115），局部退化仍在。
- 文档：`experiments/m1_separated_association_20261024.md`；只实验短窗模块，Git 仍由用户执行。

## 实验模块 29：独立短窗候选排序（已验证，未推广）

- 依据：模块 24 的干净三元组检索提高未转化为困难多目标身份收益；模块 28 的运动规则对照失败。先验证同一候选集合里的排序和拒绝能力，再考虑闭环接入。
- 契约：冻结 M1；复用原主/弱聚类，所有当前提议共同竞争。输入最近 HIST_LEN 个 sample 步骤的历史量测相对位置、分数、年龄、有效标记，以及当前提议相对位置/分数/主弱标记；不输入 GT 坐标、槽号或 128 维特征。输出每条条件历史的候选及“不匹配” logits。
- 监督：离线逐帧 .5m 最大基数一对一 GT 配对，只用此前帧配对的量测构造同槽历史，至少两次观测；当前帧先出样本再更新历史，不跨文件夹，不把跳号压成一步。当前无配对仍保留 null 标签。几何标签歧义单列，不将代理标签等同真实身份。
- 模型/损失：小型历史 MLP + 共享候选 MLP + null 头；相对坐标平移不变、候选置换等变。最后层零初始化，初始行为等价于 1m 拒绝门的最近观测排序。仅一个 masked listwise CE，训练/验证使用相同条件历史定义。
- 验收：数值损失/梯度、padding、零候选、坐标平移/候选置换、时间因果/跳号/序列边界及文件读取测试；真实小样本审计、短训练 smoke、完整回归。输出每序列及 positive/null/近邻竞争/歧义分组，比较最近观测和最后两观测速度基线。条件排序不冒充闭环 IDF1。
- 计划：`experiments/m1_short_window_association_20261024.md`。默认 tracker、旧入口和配置不改；先完成可复现训练入口，Git 由用户执行。
- 结果：12,162 参数，完整训练 25 条/59,525 帧形成 122,861 查询；完整验证 8 条/17,421 帧形成 31,184 查询。实跑两轮 CUDA，train/val CE `.25059/.26804→.22812/.26300`；相对最近观测，整体准确率 `.85816→.89895`，null `.36989→.68556`，但 positive `.98699→.95526`、竞争 `.85699→.82255`。错误接收 `4102→2047`，误拒真候选 `153→980`。排除 null 的竞争纯排序仅 `821/958→822/958`，不证明身份收益；不接入默认系统。
- 验证：新增 15 项、全套 225 项回归通过；缓存/标签形状、因果/跳号/序列隔离、数值损失/梯度、真实 prefix 与全量两轮完成。后续先拆开候选排序和可匹配性监督，关注竞争样本只有训练查询约 2.07%，不继续靠降低总体 CE 判优；本轮不是闭环 IDF1 实验。

## 实验模块 28：训练集标定的卡尔曼关联预测（已验证，失败对照）

- 依据：模块 27 后沿完整时间线复查，存在量测归属持续偏离和交叉期间反复变化。量测直接覆盖关联位置是一个待验证因素；此前历史直线拟合失败，不叠加旧规则。
- 契约：独立 KalmanMotionTracker 维护 `[x,y,vx,vy]` 及协方差，只替换关联位置预测。沿用原主/弱提议、欧氏 1m 门、确认/删除/容量和原量测/外推输出；不改变默认类、M1、网络训练、损失和身份恢复。
- 参数：仅用完整训练集标定高分主聚类与 GT 1m 一对一配对的逐轴残差二阶矩 R、同槽连续三步 GT 二阶差分二阶矩 q、GT 一阶位移二阶矩初速度方差。时间为输入步骤，序列不拼接；方差最低 1e-6 仅用于数值正定。参数在验证前冻结，拒绝缺失标定参数，验证不扫描参数。
- 公式与实验：见 `experiments/m1_kalman_motion_20261024.md`。新生零速度，位置方差 R；按每步常速度 F 和独立加速度 Q 预测，Joseph 形式更新协方差。每帧多阶段预测不重复推进状态，漏检不当作量测，删除/reset 清理。
- 验收：手算滤波、跨多步预测、合成关联扰动、出生/清理/非法输入隔离、标定数值/序列边界及 GT 推理隔离测试；先困难序列，再完整八序列四组（原版、原版恢复、Kalman、Kalman恢复）。并列 1m/.5m 全部检测/身份/管理指标、旧预测指纹及恢复前后检测证据校验。无收益则保留独立失败对照，不继续扫参。
- Git 由用户执行；本轮是状态估计对照，不表示训练新网络或已改善身份。
- 结果：25 条训练序列 59,525 帧标定 R/q/V，验证前冻结，15 项新增测试及全套 210 项通过。困难序列 F1 `.74893→.70446` / 身份 F1 `.14807→.09930` / IDSW `76→99`；完整八序列 17,421 帧 F1 `.88564→.86429` / 身份 F1 `.39648→.29319` / IDSW `200→217`，组合身份恢复也退化（身份 F1 `.48847→.38609`）。旧两组八序列指纹/全部主指标复现，恢复不改任何检测证据。拒绝推广，不继续扫噪声或叠加规则；案例、同历史预测诊断和命令见实验文档。

## 实验模块 27：实际量测更新追踪（已验证，只读诊断）

- 目的：区分当前帧几何歧义与 tracker 实际消费的量测，继续审计已确认轨迹更新；仅扩展只读诊断，不改关联、训练、损失或指标。
- 契约：诊断器在 `_observe` 调用前记录底层片段、更新前身份别名/确认/漏记状态、预测、量测位置/分数及主弱来源，调用原方法后正常返回；`finally` 恢复原方法。新生不经过 `_observe`，不冒充量测更新；暂定片段的更新前别名可以为空。GT 只用于离线分析。
- 验收：合成双目标回放测试先失败再通过；困难序列空间/身份恢复各完整回放 2,470 帧，逐序列预测 SHA-256 和事件列表须与旧诊断一致；全套 my_track 回归、静态编译和 diff 检查。
- 已核对：空间/恢复的指纹与事件完全复现，事件仍为 76/73。空间版 69 次事件的新输出当帧获得更新，其中 67 次量测最近 GT 为该事件槽；11 次连续事件前后两 ID 当帧均获更新。几何最近关系不能确定真实身份，不能把这些事件直接判作关联正确或评估错误。
- 最终检查：诊断测试 7/7、全套 my_track 195/195（14.840s），脚本与测试静态编译及 diff 检查通过。schema 3 的空间/恢复新回放与旧结果的指纹、完整事件列表和管理统计相同。
- 文件与证据：`scripts/diagnose_m1_id_switches.py`、`tests/my_track/test_id_switch_diagnostics.py`、`experiments/m1_id_switch_diagnosis_20261023.md`。本模块无训练和性能收益声明；Git 由用户执行。

## 实验模块 26：暂定片段剩余弱提议续接（两轮全量已验证，仅可选）

- 依据：模块 25 的事件 222 诊断中，暂定片段 7 在第 183/184 帧有近距离弱提议，却因只能续接高分提议而再次删除。只检验这一条状态规则，不叠加历史拟合、活动优先或滞回代价。
- 契约：先执行原确认/暂定高分关联，再执行原确认弱分关联，最后允许未匹配暂定片段使用剩余弱提议；每个提议最多使用一次。弱提议不得新生，确认仍要求完整 HIST_LEN 窗口和原命中比例；原速度、置信度、删除、容量与输出规则保留。
- 实现：基类增加默认空阶段扩展点；独立 TentativeWeakLifecycleTracker 子类启用，只支持 cascade。原评估脚本新增两个显式 opt-in 变体及单序列筛选，默认变体集合和旧训练入口不变。
- 验收：行为测试先失败后实现；困难序列 2,470 帧后完整八序列四组对照（原版、原版+身份恢复、新规则、新规则+身份恢复），并列 1m/.5m 检测、MAE、IDSW、identity-F1、owner 切换及假输出/管理统计；旧对照逐序列预测指纹须复现。
- 计划与结果：experiments/m1_tentative_weak_20261024.md。GT 仅离线评估；Git 由用户执行。
- 第一版完整八记录 F1 `.8789` / 身份 F1 `.3800` / IDSW `209` 均退化，单人弱候选支持高分假出生形成长寿命假轨。第二版将弱命中仅供续命，确认要求同一 HIST_LEN 内高分命中达到原 .6 比例；无新增数值参数，单人额外 675 个 FP 消失且输出复现原版，完整 F1 `.88708` / 身份 F1 `.39431` / IDSW `205`，加恢复 `.46631` / `196`，仍弱于旧恢复组合 `.48847` / `187`。两版均不推广，不再调阈值；下一步聚焦已确认状态错误更新和身份恢复误接。完整两轮均为 17,421 帧回放，不涉及训练。

## 实验模块 25：事件级关联滞回与长间隔恢复对照（已验证，仅保留可选）

- 目的：把 IDSW 拆成连续帧歧义竞争与长间隔重捕获两类，分别验证可解释的状态规则；不修改默认生命周期 tracker、M1、训练入口或配置默认值。
- 连续竞争变体：对空间门内多选/多对一/一对多歧义有效边，仅对上一输入步仍被观测、且候选离上一观测位置较远的边增加非负代价；保持全局匈牙利最大基数匹配，不使用 GT 或固定人数。
- 长间隔变体：仅将新的滞回 tracker 与既有 `IdentityRecoveryTracker` 组合，单独报告恢复前后指标；不把恢复逻辑混入连续竞争代价。
- 评估：独立脚本在困难三目标序列和完整八序列并列 spatial、hysteresis、spatial+recovery、hysteresis+recovery；报告检测 P/R/F1、MAE、总/分 gap IDSW、identity-F1 及管理统计。默认生产路径不读取新模块。
- 结果：困难序列滞回使 F1 `.7489→.7396`、IDSW `76→75`、identity-F1 `.1481→.1322`、MAE `.3865→.3903m`；身份恢复单独提高 identity-F1 至 `.2051`。完整八序列滞回使 F1 `.8856→.8840`、IDSW `200→204`、identity-F1 `.3965→.3893`；恢复单独为 identity-F1 `.4885` / IDSW `187`，组合滞回+恢复为 `.4726` / `191`。事件拆分显示恢复减少长 gap（`gap>5: 85→72`），滞回未改善连续竞争（连续 IDSW `86→92`）。不推广滞回；恢复保持原 opt-in 结论。
- 文件和原始结果：`original/my_track/event_hysteresis.py`、`scripts/evaluate_m1_event_hysteresis.py`、`tests/my_track/test_event_hysteresis*.py`、`experiments/m1_event_hysteresis_20261023.md`；完整八序列全量 JSON 保留在 `original/my_track/runs/event_hysteresis_20261023_full.json`。GT 只参与离线指标；默认生产路径不读取新模块。
- 事件诊断补充（已验证）：`scripts/diagnose_m1_id_switches.py` 只读导出事件上下文、底层片段/恢复别名和完整失配区间候选支持。困难三目标 76 次事件含 24 次连续匹配、52 次存在间隔；52 个区间只有 6 个全程无 .8 门内候选。案例显示旧片段占容量、暂定片段不能续接弱提议以及满 HIST_LEN 才确认，会放大量测问题。不能将输出间隔等同于 M1 漏检，也不能仅由当前几何分类判定回切原因。完整证据和后续独立实验方向见 `experiments/m1_id_switch_diagnosis_20261023.md`；未改默认逻辑。

## 实验模块 24：独立 M1 特征身份投影（筛查中，仅可选）

- 目的：验证冻结 M1 的 128 维 embedding 是否含有可经轻量度量学习强化的短时序列内身份信号；不修改 M1、默认 tracker、现有训练入口或缓存格式。
- 训练标签：训练脚本只离线读取冻结缓存与 GT。每帧先用 `score>=0.8` 和 `0.5 m` 距离门，对候选与有效 GT 槽作最大基数一对一匹配；GT 槽只在单条序列内作伪身份。正样本是同序列同槽、相距 1–8 帧的匹配候选；负样本是相距至多 1 帧的不同槽候选。三元组绝不跨序列构造。
- 模型/损失：独立 `128→64→32` MLP，输出 L2 归一化 embedding；余弦 triplet hinge `max(0, s_neg-s_pos+0.2)`。训练仅处理特征张量，不运行 M1 或 JointTracker。
- 评估：先比较验证序列 raw/projected triplet pairwise accuracy；再用相同冻结 M1 候选完整回放纯空间、raw embedding 和 projected embedding 三种生命周期关联，逐序列重置状态，报告检测 P/R/F1、匹配 MAE、GT-IDSW、identity-F1。GT 不参与推理状态更新。
- 文件：`original/my_track/m1_feature_projection.py`、`scripts/train_m1_feature_projection.py`、`scripts/evaluate_m1_projected_feature_association.py`、`tests/my_track/test_m1_feature_projection.py`。
- 默认输出、缓存与 tracker 均不改。只在特征检索指标有提升且完整验证关联 ID 指标有净收益、检测和坐标无实质退化时考虑保留；否则记录失败并不推广。
- 当前状态：7 项单测和编译通过；25 条训练序列/8 条完整验证序列的两轮筛查完成。投影验证 pairwise accuracy `.9030→.9791`，但完整回放 IDSW `200→216`，identity-F1 `.3965→.4242` 的收益主要来自单目标记录，三目标交叉记录退化；不设为默认。结果与复现命令见 `experiments/m1_feature_projection_20261023.md` 和 `CODE_LOG.md`。

## 实验模块 23：短时 M1 外观辅助关联（已验证，仅可选）

- 目的：验证冻结 M1 128 维特征是否能在已有空间门内解决短时歧义关联；不修改默认 tracker、训练、M1 或缓存格式。
- 输入/输出：候选坐标 `[N,2]`、前景分数 `[N]`、可选特征 `[N,128]` → 与生命周期基线同契约的 `TemporalFrameOutput`。
- 规则：与坐标聚类成员严格一致地对特征做分数加权并归一化；仅对门内一对多/多对一有效边附加余弦代价；匹配后动量更新原型；超过 `max_feature_age` 后空间回退。
- 验收：合成行为覆盖无特征等价、歧义配对、超时回退、输入校验、候选出生和聚类成员对齐；全套 my_track 回归测试。评估器显式读取冻结 cache 的 `feat`，GT 只用于离线指标。
- 结果：困难三人序列 2,470 帧，默认特征权重 .35 下身份 F1=.1480 / IDSW=76，与空间基线 .1481 / 76 基本相同；检测 F1=.7498（基线 .7489），MAE=.38692m（基线 .38648m）。权重 .6 无身份提升，权重 1.0 使 F1/MAE 退化。未证明身份收益，不扩展八序列、不设为默认。
- 文件与完整结果：见 `experiments/feature_association_20261023.md`；评估入口 `scripts/evaluate_m1_feature_association.py`。

## 实验模块 22：持续关联诊断与历史/活动优先对照（已验证，均仅可选）

- 基线节点 `90e1c6d`；实施和诊断见 `experiments/m1_history_motion_20261023.md`。
- 最近 HIST_LEN 个输入步骤内的已观测位置做有界线性趋势拟合，仅用于候选关联预测；已观测输出继续使用量测坐标，漏检输出和生命周期公式不变。
- 基类只新增默认行为等价的关联预测扩展点；旧训练、M1、默认配置不改。自主原版、原版+身份恢复、历史运动、历史运动+身份恢复四组完整对照。
- 验收：数值手算、短历史回退、异常单帧对关联的影响、输出坐标/重置契约；真实缓存 smoke、完整八序列、旧基线指纹复现与身份/检测双指标。失败也记录。
- 第一轮历史拟合全量失败：F1=.8818、身份 F1=.3219，未推广。第二轮独立检验活动状态 .5m 内优先配对、剩余仍走原 1m 全体关联；不叠加历史拟合，细则与反事实证据见实验文档。
- 第二轮优先级单独 F1=.8864、身份 F1=.4036，配合身份恢复 .5312（上一轮 .4885），但 owner 切换 152（上一轮 147），最难三人身份 .2051→.1692。第三轮仅在有失联状态时启用优先级，全部八序列输出与第二轮完全相同，未解决上述风险。默认不切换；不把总体身份分数上涨等同于解决串人。
- 三轮均为完整八记录 17,421 帧的四组对照，不涉及真实模型训练；主 1m 门及旧默认比较集合保留，并列增加 .5m 诊断。160 项测试、七文件编译及 diff 检查通过；结果文件与命令见实验文档，Git 由用户执行。

## 实验模块 21：独立航迹片段身份恢复（已验证，保留可选）

- 基线节点：`8f434d1`；完成提交：`90e1c6d`。实施细则与验证记录：`experiments/m1_identity_recovery_20261023.md`。
- 只在已确认输出之后恢复身份，不改量测、坐标、分数、观测标记、确认/删除和检测指标。默认生命周期算法保持冻结。
- 自主对照：失联身份端点记忆、有限时间/空间可达门、双向歧义拒绝；已知固定人数下“唯一缺席身份”另列理想化对照。
- 验收：身份恢复行为测试、真实缓存小样本、完整八序列同帧对照；旧算法指纹复现，检测计数/坐标误差逐项不变。身份收益与错误接回均记录，失败变体不推广。
- 结果：自主可达门身份恢复 F1=.4885（原 .3965）、GT 换 ID=187（原 200）、输出 ID=54（原 82）；检测 F1=.8856、MAE=.2805m 与原输出逐点一致。但预测 ID 换 GT=147（原 138），三人误认风险增加，故仅可选，不替换默认。已知人数变体身份 F1=.5532，依赖显式条件，检测 F1=.8798。原始全量结果见 `original/my_track/runs/identity_recovery_full_20261023.json`。

## 实验模块 20：独立置信度生命周期管理（实现与全量对照完成）

- 基线 Git 节点：`335e652`；完成提交：`8f434d1`；实施与验证细则见 `experiments/m1_lifecycle_20261023.md`，用户已确认。
- 契约：只用 M1 坐标/前景概率；高分新生、弱分续轨、历史确认、置信度衰减、有限预测输出与独立 ID 保留。自主/已知人数/已知人数且无消亡分开评估。
- 默认确认窗口复用 HIST_LEN；无消亡是保留已确认 ID，不等于 max_missed=0。评估保留 GT 原始槽号，报告换 ID 和碎片化，避免只看最近 GT owner 的 IDSW。
- 保持现有网络、训练入口、配置不变；Git 提交由用户执行，提供中文备注。
- 推荐独立默认：cascade、HIST_LEN=20、弱量测仅续接确认航迹、有限输出/身份保留分离。8 条记录 17,421 帧：P=.9443、R=.8338、F1=.8856、匹配 MAE=.2805m；新诊断身份 F1=.3965、GT 换 ID=200、恢复片段=610。较强 simple miss0 为 F1=.8610、身份 F1=.1219；simple miss1 为 F1=.8328、身份 F1=.1171、GT 换 ID=1539。
- 失败对照保留：joint 统一关联 F1=.8766、身份 F1=.3106；真正不消亡条件下 recall 仅 .6542（cascade），不推广为默认。第二版重跑 cascade 的八序列输出指纹及全部指标与第一版完全一致。
- 限制：高分单帧聚类 F1=.8942 仍略高；三人交叉与长期失联再捕获尚未解决，身份分数仅 .3965，不代表已达到稳定系统效果。完整方法和两轮全量消融见实验文档。

## 实验模块 16：M1 逐帧候选聚类融合基线（已验证）

- 目的：不训练网络、不改动现有 tracker 训练逻辑，先测量冻结 M1 的候选集合经简单空间聚类和置信度加权后，能否形成准确的逐帧目标点。
- 输入：M1 冷缓存中的 `out[50,4]` 和 `conf[50,2]`，GT 标签平均后为米制 `[3,2]`，空槽 `(0,0)`。
- 预测：由 `softmax(conf)[...,1]` 阈值筛选候选；在给定空间半径内取连通分量；每个分量按 M1 score 加权平均坐标，作为一个预测点。不得使用 GT 选阈值、聚类或调整预测。
- 评估：每帧使用匈牙利算法匹配预测点与有效 GT；匹配距离不超过独立配置的评估半径才算 TP。汇总 precision、recall、F1、匹配点欧氏 MAE、FP/FN、预测点数和量测数。阈值/聚类半径可做网格扫描。
- 明确限制：逐帧匈牙利只用于无身份定位评估，不表示 ID 连续，也不计算 IDSW。模块放在独立实现和脚本下，不改动 M1、my_track 训练/验证行为或默认配置。
- 验证标准：合成点位测试覆盖加权融合、空候选、点簇分离、半径边界和非法输入；真实缓存至少运行一组小样本，再运行完整验证集的阈值/半径扫描。
- 实际结果：8 条验证记录完整 17,421 帧；置信度阈值 0.96 时原始候选 precision/recall/F1/匹配 MAE 分别为 0.9139/0.8629/0.8877/0.2767 m。聚类半径 0.75 m 时为 0.9283/0.8624/0.8942/0.2778 m。聚类主要降低重复候选 FP，坐标 MAE 和召回几乎不变；此结果支持先研究候选关联，不证明跨帧 ID 能力。阈值/半径网格在验证集上仅作探索，不据此宣称调参后的泛化性能。
- 代码：`original/my_track/m1_frame_baseline.py`、`scripts/evaluate_m1_cluster_baseline.py`、`tests/my_track/test_m1_frame_baseline.py`。完整输出：`test_results_tracker_screen_20261023/m1_cluster_baseline_full.json`（本地忽略产物）。

## 实验模块 18：单帧量测锚定残差与局部证据对照（已完成，未显示稳定优势）

- 依赖与证据：模块 17 的 12 轮训练已由用户中止；最佳逐轮 F1=0.8616，保存的 epoch 6 检查点 F1=0.8201，阈值扫描后约 0.856，仍低于聚类基线 0.8942。
- 范围：新增独立 `ClusterResidualCorrector`，旧单帧模型、旧 checkpoint、原 tracker/M1/配置均保留；仅独立单帧脚本新增可选模型类型。`cluster-residual` 的两轮结果与固定锚点评估写在实验文档中；位置纠偏同配对只降低约 1.2mm，门限扫描与 M1 聚类基线接近，未证明明确收益。
- 契约：主/补充聚类 proposal 不读取 GT；以量测坐标/置信度为直连，零初始化、每轴最多 0.5m 残差；原始锚点 1m 内 Hungarian 配对监督，远处缺失目标仍计 FN；最多 3 个无序输出，不赋予 ID 含义。
- 实验计划与验收：见 `experiments/single_frame_corrector_20261023.md`。可选 `cluster-evidence` 两轮结果没有超过基线；该单帧分支保留用于复现，不再扩充特征。

## 实验模块 19：冻结 M1 时序候选基线（已完成，未接入默认 tracker）

- 目的：验证简单历史关联/短时漏检保持能否补充单帧 M1 候选；不影响现有 `JointTracker`、训练逻辑或配置默认值。
- 输入/输出：每条记录独立读取 M1 冷缓存候选坐标与前景概率；候选聚类后作门控匈牙利关联，输出最多 3 个带序列内 track ID 的点；短时无观测用常速度外推并衰减分数。
- 评估：使用 8 条验证序列共 17,421 帧，报告检测定位指标、仅观测点指标及 IDSW。GT 不参与候选生成和状态更新。
- 结果：自主最多 3 条轨迹、`max_missed=1` 时 TP 从 29,452 增至 29,563，但 FP 从 6,633 增至 9,099，F1 从 0.8610 降至 0.8328，IDSW 从 191 增至 255；`max_missed=2` 更差。严格门限可降低 IDSW，但未显示综合 F1 优势，因此不接入默认路径。
- 条件对照：启用 `--known-target-count`，仅使用序列元数据中的目标数限制最多新生轨迹，F1=0.9032、P=0.9094、R=0.8970；再设 `max_missed=0` 即漏检立即删除，F1=0.9154、P=0.9286、R=0.9025、MAE=0.2871m。此前把后者解释为“不消亡”有误；这些数值只是已知数量先验下的参考，不是固定生命周期结果或数学上界。必须与自主模式分开报告。Git 节点：`335e652`。
- 后续：简单外推不是最终航迹管理方案。进一步方向应由状态置信度、历史预测与当前候选共同决定观测更新/暂时保持/新生，单独验证遮挡和近距离交叉；已知目标数量可作为显式实验条件保留。
- 文件：`original/my_track/m1_temporal_baseline.py`、`scripts/evaluate_m1_temporal_baseline.py`、`tests/my_track/test_m1_temporal_baseline.py`、`experiments/m1_temporal_baseline_20261023.md`。

## 实验模块 17：冻结 M1 单帧集合纠偏（训练 12 轮后中止）

- 目的：隔离评估轻量可学习纠偏器是否能把冻结 M1 候选集合转为更准确的单帧目标点；不涉及 ID、历史、128 维 M1 特征或现有 tracker。
- 输入/输出：缓存候选坐标 `[B,50,2]`（米）与前景概率 `[B,50]`；输出无序预测坐标 `[B,3,2]` 与存在 logits `[B,3]`。
- 模型：候选 `(x,y,score)` token 集合编码 + 3 个 learned queries cross-attention；没有候选位置编码，理论上对候选排列不变。
- 损失：逐样本预测/GT 有效点 Hungarian 配对，匹配点 Smooth-L1；存在 BCE 覆盖所有预测槽；`L=L_position+0.5*L_presence`。GT `(0,0)` 为空，GT 槽顺序不监督身份。
- 文件：`original/my_track/single_frame_corrector.py`、`scripts/train_single_frame_corrector.py`、`tests/my_track/test_single_frame_corrector.py`、`experiments/single_frame_corrector_20261023.md`。
- 验收：CPU 合成和缓存适配器单测通过；训练脚本不调用 M1 或现有 tracker；真实训练与阈值诊断见模块 18。
- 已知限制：只测逐帧定位/目标数，不含身份连续性；验证集用于每轮 best checkpoint 筛选，不能据此声称独立测试泛化。

## 当前实现模块：my_track 教师关联课程训练与稳定性诊断（真实训练复核中）

- 2026-10-23 模块 13-15 已验证数值保护与序列内 GT 槽规范化：修复空槽 ST 融合约 `3.60e35` 的异常梯度，增加非有限梯度拒绝/梯度裁剪，新槽以量测锚定；GT 仅按每窗/记录首次出现位置建立固定 x/y 排序，轨迹交叉不重排。用户缓存两轮筛查显示：默认强教师时多目标槽塌缩；过快退火到 0.5/0 虽达到覆盖率 1.0，却 MAE 10-29m 且约 51% 越界。按槽均衡 CE 已试验但无有效改善，未保留。M1、验证损失公式、默认学习率和输出目录不变；多目标自主跟踪尚未达标，详细证据见 `experiments/my_track_stability_20261023.md`。

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
