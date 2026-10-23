# my_track 联合航迹网络实施计划

状态：代码实现与合成验证已完成。真实数据首轮已暴露自主关联不稳定，现进入保持结构不变的 `LR=3e-4` 受控对照。结构细节与可校准的工程初值见 [my_track_architecture_refinement.md](my_track_architecture_refinement.md)。

## 任务顺序

2026-10-23 定点回退补充：模块 10 的自主主损失/教师辅助双路径与融合分母 `1e-3` 下限已撤销，当前恢复单路径教师课程和 dtype tiny 分母保护。模块 9 独立训练权重、模块 11 验证合批、用户当前 `LR=3e-5` 与输出目录保留。合成测试 38/39 通过，唯一失败是旧学习率断言；未复测真实训练效果。详细范围见 `CODE_LOG.md` 模块 12。

2026-10-23 验证提速补充：`VAL_SEQUENCE_BATCH_SIZE=8` 控制跨独立记录的 batch 前向，序列内部保持递归和独立状态；不同长度按实际有效帧缩减 batch，不填充。位置损失启用逐帧等权归约，保证不同人数合批不改变验证口径。冷/暖合成串并行一致性已验证，真实 GPU 耗时待下次训练测量；文件加载仍按需串行。设为 1 可切回串行执行。详细变更与已知旧测试失败见 `CODE_LOG.md` 模块 11。

| 顺序 | 模块 | 交付与验证 | 状态 |
|---|---|---|---|
| 1 | `association.py`、`config.py` | 固定 GT 槽标签、可多候选同槽、空槽及四项损失；CPU 反传测试 | 已实现，6 项通过 |
| 2 | 新增 `joint_tracker.py` | 候选/航迹集合交互、ST 独占分配、多候选融合、最终点纠偏；形状/梯度测试 | 已实现，5 项通过 |
| 3 | `track_state.py` | initialized/confirmed/observed 分离，HIST_LEN 确认，漏检保持和 reset；状态测试 | 已实现，4 项通过 |
| 4 | `train_tracker.py`、`train.py` | 冷/暖适配，滑窗每帧损失，冻结 M1，tracker checkpoint；合成训练路径测试 | 已实现，4 项独立测试通过（整序列回放集成另计入任务 5） |
| 5 | `track_sequence_dataset.py`、新 `metrics.py` | full-sequence 路径索引与按需加载，固定 warm-up 映射、IDSW/误差/覆盖/漏检指标 | 已实现，8 项含回放集成通过 |
| 6 | 文档与汇总 | 全套单元测试、py_compile、diff check、接口核对；更新 CODE_LOG 与阶段状态 | 已完成，27 项全量测试通过 |
| 7 | 教师关联课程训练 | 训练期 GT 融合与线性退火；推理/验证保持自主关联；合成边界测试 | 已实现，新增 4 项、全量 31 项测试通过 |
| 8 | 训练稳定性诊断与低学习率对照 | 独立输出、四项训练损失、梯度范数、自主关联质量及坐标越界率 | 已实现，真实启动受 OOM 阻塞 |

## 关键接口

- `build_assignment_targets(candidate_pos[B,50,2], gt_pos[B,3,2], radius_m) -> labels[B,50]`
- `association_loss(logits[B,50,4], labels[B,50]) -> scalar`
- `tracker_losses(association_logits, labels, final_pos, existence_prob, track_confidence, gt_pos) -> dict[str, scalar]`
- `JointTracker.forward(measurements, state, dt) -> TrackerOutput`
- `JointTracker.forward(measurements, state, dt, teacher_assignment_labels=None, teacher_assignment_ratio=0.0) -> TrackerOutput`；教师标签仅训练使用，默认值保持现有推理语义。
- `TrackBank.as_tensors(device) -> dict[str, Tensor]`; `update(final_pos, fused_feat, observed, existence_prob, track_confidence, dt)`
- `_forward_sequences(coor[B,T,50,2], conf[B,T,50,2], feat[B,T,50,128], gt[B,T,3,2], banks, tracker, dt)` 对全部 B*T 帧产生损失和轨迹序列。
- `sequence_tracking_metrics(pred_pos[T,3,2], active[T,3], observed[T,3], gt_pos[T,3,2], warmup, match_gate_m)`；warm-up 汇总后仅做一次槽映射。若目标在 warm-up 中无观测，仅在初始化时按固定 GT 槽补齐映射，并报告 target coverage/untracked frames，防止漏目标看成零误差。

## 验证命令

```powershell
& 'E:\Miniconda3\python.exe' -B -m unittest discover -s tests\my_track -v
& 'E:\Miniconda3\python.exe' -B -m py_compile original\my_track\association.py original\my_track\joint_tracker.py original\my_track\track_state.py original\my_track\train_tracker.py original\my_track\train.py original\my_track\metrics.py original\my_dataset\track_sequence_dataset.py
git diff --check
```

未安装 pytest，使用仓库已有 Python 环境运行标准库测试；不安装依赖。当前代码验收结果为 33 项测试通过，`py_compile` 通过，其中包含训练/验证进度显示、教师调度和训练诊断测试。历史运行日志文件含用户保留的 CRLF 空白行，故仅对本轮源码、测试和文档运行 scoped `git diff --check`。真实 M1 缓存上的正样本率、阈值扫描和训练曲线仍需实际实验确认，不属于合成测试可以证明的事项。

## 实现边界与未校准项

- `CONF_THRESHOLD=0.96` 是旧脚本遗留配置，不参与新联合跟踪器的量测筛选。
- 1.0 m 关联半径、0.5 存在确认阈值、0.95 漏检置信度衰减和四项损失权重均是第一版初值。
- 当前测试证明接口、梯度、冷/暖冻结语义和合成状态行为；已有真实训练初步指标，但没有完成课程退火和自主阶段的最终验证。
- 旧 `FeatureMapper`、`ResidualGRU`、`ScoreFusion` 文件及导出为兼容保留；新训练只创建 `JointTracker`。
- 教师关联训练扩展已实现：关联 CE 始终监督模型 logits；仅候选融合使用从 `1` 线性退火至 `0` 的 GT/模型分配混合，验证与推理固定使用 `0`。

## 教师关联课程训练（已验证）

为将量测纠偏学习与早期错误关联解耦，训练期对每个候选使用已有
`build_assignment_targets` 的固定 GT 槽标签。模型仍始终输出并以关联 CE 监督
`association_logits`，但融合分配改为：

```text
A_effective = (1 - r) * A_predicted_ST + r * one_hot(GT_label)
```

- `r=1`：融合位置/特征和 `observed` 由教师标签决定，位置纠偏不会被模型早期误关联污染。
- `0<r<1`：线性混合教师分配与模型 ST 分配；模型关联对融合路径的梯度影响随 `r` 平滑增加。
- `r=0`：不向 `JointTracker` 传教师标签，严格使用原有自主 hard/ST 分配；整序列验证和实际推理始终采用该值。
- 默认调度为 `TEACHER_ASSOC_WARMUP_EPOCHS=5`、`TEACHER_ASSOC_ANNEAL_EPOCHS=10`：第 1--5 轮 `r=1`，第 6--14 轮依次为 `0.9` 至 `0.1`，第 15 轮及之后为 `0`。
- 每 epoch 的 `teacher_assignment_ratio` 写入 checkpoint，终端训练进度也显示 `teacher=...`；首轮训练 CSV 保持原表头。稳定性对照使用独立目录和扩展表头，不混写历史 CSV。

合成测试验证了：全教师比例可覆盖模型融合归属；零比例的融合坐标、最终坐标和观测掩码与原自主路径一致；正比例缺标签会被拒绝；调度预热与线性退火边界正确。前 14 个 epoch 的验证指标只作诊断，不应用于最终模型选择。

## 真实训练首轮与受控对照（进行中）

首轮真实训练使用冻结的 M1 特征缓存、既有数据划分、教师比例 `r=1.0` 和
`Adam(lr=1e-3)`；其第 6 个 epoch 在 `r=0.9` 的中途由人工中止，前五个完整
epoch 的摘要如下。训练期的 `r=1` 使位置融合直接使用 GT 分配，而验证始终
`r=0`，因此两者损失不能作同尺度比较。

| epoch | train loss | val assoc CE | val pos | IDSW | MAE | coverage |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.70652 | 3.55132 | 31.82730 | 472 | 54.194 m | 0.93 |
| 2 | 0.14447 | 3.81086 | 26.91844 | 378 | 46.876 m | 1.00 |
| 3 | 0.11330 | 9.87787 | 94.76063 | 321 | 236.377 m | 1.00 |
| 4 | 0.10525 | 11.70062 | 58.83780 | 257 | 84.260 m | 1.00 |
| 5 | 0.09991 | 12.11530 | 71.80819 | 239 | 116.328 m | 1.00 |

该趋势与自主关联退化、递归状态误差放大相符，但尚不能仅凭曲线确认因果。为隔离
学习率影响，当前受控实验从新初始化开始，仅作下列变化：

- `LR: 1e-3 -> 3e-4`，输出目录改为 `runs/teacher_lr3e4_diagnostics/`；不复用或覆盖首轮 checkpoint/CSV。
- 每个训练 batch 记录四项原始损失均值和优化器 `step()` 前的全局梯度 L2 范数。
- 验证记录总体/前景关联准确率、GT 关联类别概率均值，以及有效航迹点落在物理边界外的比例；这些值均只读，不参与损失或状态更新。

先完成前 3 个完整 epoch，再将 `val_association_foreground_true_probability`、关联 CE、
越界率和 MAE 与上表对照。若关联 CE 仍快速上升且前景真类概率持续下降，说明降低
学习率不足以解决问题；下一步再审查教师训练与自主关联的暴露差异，本轮不自动改动网络。

### 2026-10-22 启动记录

- 21:24 首次启动因沙箱禁止创建 DataLoader 多进程管道而退出；保留 `train_stdout.log`、`train_stderr.log`。
- 21:27 获准在沙箱外重新启动，确认 CUDA 可用，训练窗口数为 2340、验证整序列数为 8；21:31 在第一个 batch 的前向阶段发生 CUDA OOM 并退出。
- OOM 报告申请 2 MiB 失败，同时显示设备空闲 3.79 GiB、PyTorch 已分配 1.15 GiB；启动期间系统命令也曾出现进程初始化失败。因此宿主资源/多进程压力值得继续排查，不能仅据此认定模型占满物理显存。
- 没有完成任何优化器更新，未生成 CSV 或 checkpoint。完整失败日志保存在 `runs/teacher_lr3e4_diagnostics/train_stderr_full.log`；不写入对照指标，不自动缩批、改模型或重启。
- 建议下一次经确认后先使用 `num_workers=0` 做首批验证，以隔离 Windows 多进程与宿主内存压力；batch、学习率、网络和损失保持不变。
