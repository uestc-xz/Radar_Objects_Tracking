# 代码变更日志

## 模块 19 扩展：已知序列目标数的理想化航迹管理对照 — 2026-10-23

- 目的：在不改变默认自主模式的前提下，验证“每条记录目标数固定且已知、目标不消亡”这一数据条件对航迹管理上限的影响。
- 修改文件：`scripts/evaluate_m1_temporal_baseline.py`、`tests/my_track/test_m1_temporal_baseline.py`、`experiments/m1_temporal_baseline_20261023.md`、`SPEC.md`。
- 核心逻辑：新增 `known_target_count`，从序列目录名 `<date>_<target_count>_<variant>` 读取数量；`--known-target-count` 只将数量传给 `TemporalCandidateTracker.max_tracks`，不读 GT 坐标、不读 GT 槽 ID、不替系统做关联。
- 验证：全量验证 8 条记录、17,421 帧。自主最多 3 条且 `max_missed=1` 为 P=0.7647/R=0.9144/F1=0.8328；已知数量为 P=0.9094/R=0.8970/F1=0.9032；已知数量且 `max_missed=0` 为 P=0.9286/R=0.9025/F1=0.9154、MAE=0.2871m、IDSW=189。
- 结论：目标数量与生命周期先验显著减少 FP，适合作为理想化上限而非默认能力；默认 tracker、训练逻辑和配置未改。完整结果见 `experiments/m1_temporal_baseline_20261023.md`。

## 模块 18：量测锚定残差与局部证据短训 — 2026-10-23

- 目的：在冻结 M1 的单帧候选上隔离检测/坐标纠偏问题，避免把阈值收益误判成定位收益。
- 修改文件：`original/my_track/cluster_residual_corrector.py`、`original/my_track/single_frame_evaluation.py`、`scripts/train_single_frame_corrector.py`、`scripts/evaluate_single_frame_corrector.py`、对应测试、`SPEC.md` 与实验记录。
- 核心逻辑：统一 proposal 分数比较为 float64，修复 float32 的 0.96 边界候选丢失；学习存在阈值默认 0.5，与 M1 proposal 门限解耦；新增固定原始锚点/GT 配对诊断；可选 `cluster-evidence` 为 proposal MLP 增加八维局部点簇证据，旧 `set`/`cluster-residual` 接口保持不变。
- 验证命令：`E:\Miniconda3\python.exe -m unittest discover -s tests\\my_track -v`；真实短训分别为 `cluster-residual` 和 `cluster-evidence` 两轮；阈值诊断使用 `scripts\\evaluate_single_frame_corrector.py`。
- 验证结果：全套 86 项单元测试通过。`cluster-residual` 第二轮固定阈值 0.5 为 F1=0.89245、MAE=0.28054m，阈值 0.6 为 F1=0.89496、MAE=0.27804m；`cluster-evidence` 第二轮固定阈值 0.5 为 F1=0.89030、MAE=0.28154m，阈值 0.6 约 F1=0.90，未显示稳定的基线级提升。固定同一 29,452 对后，residual 纠偏 MAE=0.28347m、anchor MAE=0.28463m；evidence 纠偏 MAE=0.28332m，实际坐标改善约 1mm 量级。
- 结论：M1 聚类基线 F1=0.89416、MAE=0.27779m 仍是当前可靠基准；短训主要校准 proposal 存在概率，不能据此宣称网络已经提升定位。proposal 召回上限和跨帧身份问题仍未解决。
- 已知限制：局部证据只作可选实验模型，未替换默认模型；真实结果来自验证集探索，不能代表独立测试泛化。未创建 Git 提交。

## 模块 17：冻结 M1 单帧集合纠偏器 — 2026-10-23

- 目的：用最小独立实验验证候选坐标与置信度能否学习为逐帧最终点；排除历史状态、M1 feature 和身份关联的影响。
- 依赖：冻结 M1 cache 的 `out/conf`、`TrackSequenceColdDataset` full-sequence 路径索引、三槽米制 GT。
- 修改文件：`original/my_track/single_frame_corrector.py`、`scripts/train_single_frame_corrector.py`、`tests/my_track/test_single_frame_corrector.py`、`experiments/single_frame_corrector_20261023.md`、SPEC、本日志。既有 tracker 训练代码和配置未改。
- 核心逻辑：集合 Transformer 编码 `[x,y,score]`，三个 query 输出坐标/存在 logits；每帧 Hungarian 匹配有效 GT，Smooth-L1 坐标损失 + 0.5 BCE 存在损失。候选和 GT 槽排列不影响语义；输出不含 ID。
- 输入/输出：坐标 `[B,50,2]`、前景概率 `[B,50]` -> 无序坐标 `[B,3,2]`、存在 logits `[B,3]`。训练数据只从 cache `out/conf` 与标签读取，忽略 `feat`。
- 验证：新增 5 项 CPU 测试覆盖形状、候选排列不变、GT 槽置换不变、空帧有限反传、缓存路径和 GT 均值。冷数据接线抽样为验证集 8 记录/17,421 帧，`out=(50,4)`、`conf=(50,2)`、`feat=(50,128)`、标签 `(3,2)`。训练尚未运行。
- 已知限制：逐帧实验不能衡量身份连续性；当前验证集指标用于模型选择，泛化性能尚未独立验证。
- Git 节点：未创建。

## 模块 16：M1 逐帧候选聚类融合基线 — 2026-10-23

- 目的：先隔离定位量测质量，判断冻结 M1 候选经简单空间融合后能否形成准确逐帧目标点；不训练网络、不评价身份连续性。
- 依赖：`TrackSequenceColdDataset` 的验证记录索引、已有 M1 `out/conf` 缓存和米制 GT 标签；GT 仅供评估期匈牙利匹配。
- 修改文件：`original/my_track/m1_frame_baseline.py`、`scripts/evaluate_m1_cluster_baseline.py`、`tests/my_track/test_m1_frame_baseline.py`、`SPEC.md`、本日志、`experiments/my_track_stability_20261023.md`。现有训练/验证逻辑与默认配置未改。
- 核心逻辑：softmax 取前景 score，按阈值选候选；以欧氏距离半径构造连通分量，簇内按 score 加权均值；评估用逐帧匈牙利匹配，半径外配对计作 FP/FN。提供阈值和聚类半径网格扫描。
- 输入/输出：候选位置 `[M,2]`、概率 `[M]`；输出簇中心 `[K,2]`、簇最高 score `[K]`、簇大小 `[K]`。逐帧评估返回 TP/FP/FN 和匹配距离和，脚本汇总 precision/recall/F1/MAE。
- 验证：7 项合成单测通过；完整 8 条验证记录 17,421 帧运行成功。score threshold 0.96、cluster radius 0.75m：P=0.9283、R=0.8624、F1=0.8942、matched MAE=0.2778m。原始未聚类高分候选 F1=0.8877、MAE=0.2767m。评估匹配优先最大化门限内 TP 数，再最小化有效匹配距离；该边界有专门回归测试。阈值网格仅探索，不视作验证集调参后的泛化结果。
- 已知限制：连通聚类有传递闭包效应；距离匹配半径 1m 是评估口径而非预测门限；当前指标没有 ID/遮挡/时序语义，不能代表多目标跟踪表现。
- Git 节点：未创建。

## 模块 12：按用户指定轮次回退训练路径 — 2026-10-23

- 触发：用户报告模块 10 修改后首轮 train=18.71803、val=139.98029、pos=135.69984、MAE=207.154m、oob=0.546，要求只撤销“OK你来修改我来执行”一轮的修改。单轮指标不能证明其中某个改动为唯一原因。
- 回退：删除双路径训练、教师状态库复制、辅助损失及其配置/检查点元数据；恢复每帧一次前向的单路径教师课程，沿用原预热和退火调度。融合分母恢复 `weight_sum.clamp_min(torch.finfo(pos.dtype).tiny)`。
- 保留：训练/验证独立损失权重，当前 LR=3e-5，输出目录，验证跨序列合批与逐帧归约，诊断字段，冻结 M1 与缓存冷路径。未改变网络参数形状，未删除或恢复已有日志/模型，未干预运行中的训练进程。
- 回归：先新增单路径训练入口检查和微小正置信度归一化检查，均在回退前失败、回退后通过。运行 `python -m unittest discover -s tests/my_track -v`：39 项中 38 项通过，唯一失败为既有测试要求 LR=3e-4，而用户当前配置为 3e-5；保留用户配置。冷/暖串并行验证一致性检查通过。未运行真实数据训练或效果对照。
- 使用：重启原训练入口后生效；当前进程已加载的代码不会自动回退。代码回退不会还原已训练权重，效果对照应重新初始化训练，而非续训本次退化试验的权重。
- Git 节点：未创建；只定点编辑，未对整个提交或工作区进行回退。

## 模块 13：空槽融合梯度与优化器保护 — 2026-10-23

- 目的：修复有限前向损失下空槽 ST 导数被 tiny 分母放大，导致 Adam 二阶状态溢出的问题。
- 数据审查：抽查 25 条训练、8 条验证记录的中间帧，缓存 out/conf/feat 为 float32、形状 50x4/50x2/50x128，GT 平均后 3x2；坐标同为米制，未发现全局 y 偏移。匹配误差多数分米级，个别超过 1m，不能据抽样认定全量 M1 可靠。
- 证据：单观测槽、两空槽的四损失反传产生约 3.60e35 梯度，float32 平方溢出；新增测试在旧实现上失败。
- 修改文件：joint_tracker.py、train.py、config.py、两项测试、SPEC 与实验计划。
- 核心逻辑：有观测槽严格除以真实权重和，空槽以 1 作无害分母并掩蔽其融合输出/导数；先归一化权重再融合，不使用坐标缩小 epsilon。梯度范数 float64 累加；非有限梯度在 step 前报错；有效梯度全局裁剪 5，记录裁剪前范数。
- 输入/输出：Measurements、TrackerOutput、四项损失定义均不变；checkpoint 增加裁剪阈值记录，参数名/形状不变。
- 验证：44 项 my_track 单测通过；涵盖空槽 Adam moments、零 score、微小正 score、3e30/4e30 范数、非有限梯度拒绝和裁剪。git diff --check 通过。删除旧测试对 LR=3e-4 的硬编码断言，未修改实际 LR=3e-5。
- 限制：只证明数值故障修复，不证明真实数据跟踪效果改善；旧 epoch-1 检查点实测 1/65 个 Adam 二阶张量非有限，已有 optimizer state 不可续用，应新建优化器。
- Git 节点：待提交；首次沙箱写入 .git/index.lock 被拒绝。

## 模块 14：量测锚定初始化与限额真实数据筛查 — 2026-10-23

- 目的：避免新航迹将米制量测与原点混合；用短实验筛查，而非继续盲目长训。
- 依赖：模块 13 的空槽导数修复。无修改固定槽契约、四项损失、验证归约或教师课程。
- 修改文件：joint_tracker.py、test_joint_tracker.py、scripts/screen_tracker.py、test_tracker_screening.py、实验文档和 SPEC。
- 核心逻辑：纠偏头最后一层零初始化；无历史槽的量测门为 1，已有历史槽保留可学习 sigmoid 门及自由残差。手算样例：新槽单量测 (2,4) 初始输出即 (2,4)。参数名和形状兼容旧检查点，但零初始化只影响新建模型。
- 筛查工具：强制指定全新输出目录，均匀抽取每条训练记录的窗口、保持验证前缀连续；复用正式训练/验证，保存样本 ID 清单和配置。默认正式输出目录与 LR 不修改。
- 验证：锚定新增两项先失败后通过，最终 my_track 全套 50 项通过；py_compile、git diff --check 通过。8 条验证中间帧、教师正确关联、未训练同 seed 比较，冷启动平均欧氏误差从 1.2776m 至 0.3237m；不是自主指标。
- 真实筛查：25 条训练各取 4 窗口，batch=4；8 条验证各前 200 帧。LR=3e-5 第 2 轮 train=2.06459、val=4.24746、MAE=0.89805m、coverage=0.50；LR=3e-4 第 2 轮 train=1.07513、val=2.53822、MAE=0.95665m、coverage=0.57143。两个新 checkpoint 均 0/65 个非有限 Adam 二阶张量，未干预已有 GPU 进程。
- 已知限制：较高 LR 的前景候选全部分给槽 1，观测帧数 [1600,0,0]，多目标关联未解决。仅诊断的软距离偏置可改善覆盖率但恶化验证损失/IDSW，未并入代码。没有完整数据新旧对照，不宣称最终效果改善；详情与全部数字见 experiments/my_track_stability_20261023.md。
- Git 节点：待提交。

## 模块 15：序列局部槽语义与关联退火筛查 — 2026-10-23

- 目的：检查跨序列 GT 槽号是否能作为全局身份类别，以及空槽初始化/教师课程对自主关联塌缩的影响。
- 依赖：模块 13/14 的空槽稳定性、量测锚定和限额筛查入口。
- 修改文件：`train_tracker.py`、`train.py`、`association.py`、`screen_tracker.py`、`test_tracker_training.py`、`test_tracker_screening.py`、实验文档与 SPEC。
- 核心逻辑：训练窗口和验证记录各自仅按目标首次有效坐标 `(x,y)` 建立一次稳定槽映射；不逐帧重排。候选关联标签、教师融合与坐标/存在监督使用同一重排 GT。筛查脚本可显式覆盖 teacher warmup/anneal，正式配置保持原值。按槽平衡前景 CE 原型未保留。
- 输入/输出：新增 `canonicalize_ground_truth_tracks([B,T,3,2] 或 [T,3,2])`，shape 不变、轨迹身份对应保持；空槽排后，轨迹交叉不改变槽序。屏幕清单记录每窗/记录样本 ID 和有效教师比例。
- 验证：GT 交叉、延迟首观测及教师调度覆盖单测；全量 `unittest discover -s tests/my_track -v` 53 项通过，相关文件 `py_compile` 与 `git diff --check` 通过。两种强教师筛查下 coverage 仍仅 0.50/0.571；快速退火 0.5→0 虽 coverage=1，但 val MAE 28.793m/10.314m，OOB=0.518/0.509。全流程数值不发散（但回归误差严重），无效的按槽均衡原型对相同筛查指标无可测收益。
- 已知限制：槽规范化只厘清标签语义，不解决空槽出生、误量测抑制、关联反馈稳定性。训练身份仍是窗口局部语义；验证 metrics 公式未改变。
- Git 节点：未创建；沙箱不允许写 `.git/index.lock`，提权审批服务两次连接失败。

## 模块 11：完整序列验证合批 — 2026-10-23

- 目的：将验证从单序列逐帧前向改为跨独立序列 batch 前向，提高 GPU 利用率；不打乱任何序列内部的时间顺序。
- 配置：新增 `VAL_SEQUENCE_BATCH_SIZE=8`，实际并行数不超过可用 `TrackBank` 数和剩余序列数；设为 1 可回到串行执行。冷缓存和冻结 M1 暖路径均支持。
- 状态隔离：每组序列各自绑定一个状态库，只在组开始重置；短序列结束后退出当前时间步的 batch，不填充或计算额外帧。MAE、IDSW、覆盖率等仍按独立序列计算后汇总。
- 损失口径：验证通过 `framewise_loss=True` 启用 `tracker_losses(..., framewise_position=True)`，先在每帧有效目标坐标内平均，再按真实帧数平均；空目标帧位置损失为零。其余三项本来已按样本等权，训练归约默认保持不变。
- 传输与观测：逐帧 CPU 回传改为整个 batch 回传；进度条每 20 个时间步刷新累计帧数与当前并行数。文件读取仍按需串行，没有新增 DataLoader worker 或跨帧异步状态更新。
- 验证：新增合批、冷/暖串并行一致性、不同长度/人数/空目标、空集/限制和逐帧损失归约测试。全套 38 项中 37 项通过；唯一失败仍为旧测试把学习率写死为 `3e-4`，当前用户配置 `3e-5` 未修改。未运行真实数据训练或 GPU 性能测量，不预估加速倍数。
- 修改文件：`config.py`、`train.py`、`train_tracker.py`、`association.py`、对应两项测试文件及实施记录。输出目录、已有 checkpoint 和正在运行的训练进程未改动；新代码在下次启动生效。未创建 Git 提交。

## 模块 10：训练自主路径与教师辅助损失 — 2026-10-23

- 状态：已按用户要求在模块 12 回退；以下为历史实现记录，不代表当前执行逻辑。
- 目的：消除训练前期 GT 教师融合与验证自主推理之间的路径差异，让第 1 轮就对自主关联、量测融合和最终位置回归提供监督。
- 核心逻辑：`_forward_sequences` 在教师比例大于 0 且辅助权重启用时维护两套独立 `TrackBank`；自主路径计算主损失并推进实际训练状态，教师路径计算 `TEACHER_AUX_LOSS_WEIGHT=0.5` 的辅助损失。验证不传教师比例和辅助权重，保持单自主路径。
- 数值保护：联合融合分母增加 `FUSION_DENOMINATOR_EPS=1e-3` 下限，避免低置信量测的归一化反向梯度爆炸。
- 修改文件：`config.py`、`joint_tracker.py`、`train_tracker.py`、`train.py`、`SPEC.md`、本模块训练测试。
- 验证：新增自主主路径/教师辅助路径测试通过；训练测试文件仍有一个既有学习率断言要求 `3e-4`，当前用户配置为 `1e-5`，因此该旧断言失败，未修改用户当前学习率。未运行真实数据训练。
- Git 节点：未创建。

## 模块 9：my_track 训练与验证损失权重分离 — 2026-10-23

- 训练权重：association=2、position=5、existence=0.25、confidence=0.1；验证保留原权重，position=1。
- 实现：`config.py` 新增独立训练配置；冷/暖前向新增可选 `loss_weights`，默认使用原验证权重；`train.py` 仅在训练调用中传入训练权重，并将其另存到 checkpoint。
- 保持不变：原始子损失及日志字段、验证指标、网络结构、教师课程、数据、当前学习率 `3e-5` 和输出目录；不干预已有训练进程。
- 状态：代码已修改，按用户要求不运行测试、静态编译、验证或训练；尚无效果结论，未创建 Git 提交。

## 模块 8：my_track 训练稳定性诊断与低学习率对照 — 2026-10-22

- 目的：把教师融合训练期的低损失与自主验证质量分开观测，定位关联退化、坐标失控或优化不稳定的来源；不改变联合跟踪器结构和训练课程。
- 首轮证据：冻结 M1、`Adam(lr=1e-3)` 的真实运行在 epoch 1--5 中，训练损失从 `0.70652` 降至 `0.09991`，但自主验证关联 CE 从 `3.55132` 升至 `12.11530`，位置 MAE 在 `46.876 m`、`236.377 m`、`84.260 m`、`116.328 m` 间剧烈波动。第 6 个 epoch（教师比例 0.9）中途终止，保留原日志供对照。
- 受控变更：`LR=3e-4`，并切换到独立的 `runs/teacher_lr3e4_diagnostics/`；M1 冻结、数据、损失权重、教师调度、推理和验证路径均未改变。
- 训练观测：训练端汇总 association/position/existence/confidence 四项损失和优化器更新前的全局梯度 L2 范数；进度条增加 association、position、gradient、LR、teacher。
- 验证观测：`association_diagnostics` 只统计总体/前景关联准确率和 GT 真类概率；整序列另统计有效航迹点超出 `x=[-2.7,2.7]`、`y=[0.8,6.2]` 物理区域的比例。上述统计均不参加反传、损失或状态更新。
- 修改文件：`config.py`、`association.py`、`train_tracker.py`、`train.py`、训练相关单元测试与实验文档。
- 验证：新增诊断单元测试后，`unittest discover -s tests/my_track -v` 共 33 项通过；相关模块 `py_compile` 通过。受控真实训练尚未产生结果时，不填写成功结论。
- 真实启动：沙箱内因 DataLoader 管道权限失败；获准在沙箱外启动后，首 batch 前向 CUDA OOM，尚未更新参数，进程已退出。保留两次启动日志，未自动更改 batch/worker 配置或再次重启。
- Git 节点：未创建。

## 模块 7：my_track 教师关联课程训练 — 2026-10-22

- 目的：在冻结 M1 时先稳定学习量测融合与坐标纠偏，随后将候选融合平滑过渡到模型自主关联。
- 依赖：固定 GT 槽标签 `build_assignment_targets`、JointTracker 的 ST 关联、现有冷/暖训练入口。
- 修改文件：`original/my_track/config.py`、`joint_tracker.py`、`train_tracker.py`、`train.py`、两项 my_track 测试及实施文档。
- 核心逻辑：训练融合使用 `A_effective=(1-r)*A_predicted_ST+r*one_hot(GT_label)`；模型 logits 仍始终计算关联 CE。默认第 1--5 轮 `r=1`，随后 10 轮线性退火，第 15 轮后 `r=0`。验证和推理不传教师标签。
- 输入/输出：新增可选 `teacher_assignment_labels[B,50]` 与 `teacher_assignment_ratio∈[0,1]`；输出字段保持兼容，`assignments` 仍表示模型硬预测，`assignment_onehot` 表示实际用于融合的分配。
- 验证命令：`E:\Miniconda3\python.exe -B -m unittest discover -s tests\my_track -v`；`E:\Miniconda3\python.exe -B -m py_compile ...`；`git diff --check`。
- 验证结果：31 项单元测试通过；新增测试确认全教师融合、零比例自主路径等价、正比例标签校验与调度边界；静态编译和 diff 检查通过。
- 已知限制：未运行真实缓存训练；教师阶段验证始终使用自主关联，早期验证损失不应被当作最终模型选择依据。
- Git 节点：未创建

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

## 模块 19：冻结 M1 时序候选基线 — 2026-10-23

- 目的：评估轻量时序关联和短时漏检保持能否补充逐帧 M1 候选；保持现有 `JointTracker` 与训练入口不变。
- 依赖：模块 16 的逐帧聚类与冷缓存数据适配器；每条序列独立创建状态。
- 修改文件：`original/my_track/m1_temporal_baseline.py`、`scripts/evaluate_m1_temporal_baseline.py`、`tests/my_track/test_m1_temporal_baseline.py`、`experiments/m1_temporal_baseline_20261023.md`、`SPEC.md`。
- 核心逻辑：候选聚类后门控匈牙利关联；匹配更新位置/速度，短时漏检用常速度外推并衰减分数；不跨序列共享 ID，不使用 GT 参与预测。
- 输入/输出：输入每帧缓存坐标 `[50,2]` 与前景分数 `[50]`；输出最多三条点位、序列内 ID、当前帧观测掩码。
- 验证命令：`E:\Miniconda3\python.exe -m unittest discover -s tests\my_track -p test_m1_temporal_baseline.py -v`；`E:\Miniconda3\python.exe scripts\evaluate_m1_temporal_baseline.py --max-frames 0 --max-missed 1 --output original\my_track\runs\m1_temporal_baseline_full.json`。
- 验证结果：4 项新测试通过；全量验证 17,421 帧。`score>=0.8, gate=1m, max_missed=1` 得 P/R/F1=0.7647/0.9144/0.8328、MAE=0.2876m、IDSW=255；`max_missed=0` 的 F1=0.8610、IDSW=191。无条件延续无综合收益；完整矩阵见实验文档与本地忽略输出 JSON。
- 已知限制：当前 IDSW 是按逐帧最近 GT 归属统计的诊断口径，不等价于 MOT 标准 IDF1/HOTA；简单常速度关联不是最终航迹管理器。用户同意下阶段可单独试验已知序列人数和序列内无消亡假设，但推理不得读 GT 坐标或 GT 槽 ID。
- Git 节点：未创建。

