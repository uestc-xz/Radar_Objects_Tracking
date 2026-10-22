# my_track 联合航迹网络实施计划

状态：代码实现与合成验证已完成，等待真实数据训练与阈值校准。结构细节与可校准的工程初值见 [my_track_architecture_refinement.md](my_track_architecture_refinement.md)。所有测试使用标准库 `unittest` 与合成数据；本轮不运行全量训练。

## 任务顺序

| 顺序 | 模块 | 交付与验证 | 状态 |
|---|---|---|---|
| 1 | `association.py`、`config.py` | 固定 GT 槽标签、可多候选同槽、空槽及四项损失；CPU 反传测试 | 已实现，6 项通过 |
| 2 | 新增 `joint_tracker.py` | 候选/航迹集合交互、ST 独占分配、多候选融合、最终点纠偏；形状/梯度测试 | 已实现，5 项通过 |
| 3 | `track_state.py` | initialized/confirmed/observed 分离，HIST_LEN 确认，漏检保持和 reset；状态测试 | 已实现，4 项通过 |
| 4 | `train_tracker.py`、`train.py` | 冷/暖适配，滑窗每帧损失，冻结 M1，tracker checkpoint；合成训练路径测试 | 已实现，4 项独立测试通过（整序列回放集成另计入任务 5） |
| 5 | `track_sequence_dataset.py`、新 `metrics.py` | full-sequence 路径索引与按需加载，固定 warm-up 映射、IDSW/误差/覆盖/漏检指标 | 已实现，8 项含回放集成通过 |
| 6 | 文档与汇总 | 全套单元测试、py_compile、diff check、接口核对；更新 CODE_LOG 与阶段状态 | 已完成，27 项全量测试通过 |

## 关键接口

- `build_assignment_targets(candidate_pos[B,50,2], gt_pos[B,3,2], radius_m) -> labels[B,50]`
- `association_loss(logits[B,50,4], labels[B,50]) -> scalar`
- `tracker_losses(association_logits, labels, final_pos, existence_prob, track_confidence, gt_pos) -> dict[str, scalar]`
- `JointTracker.forward(measurements, state, dt) -> TrackerOutput`
- `TrackBank.as_tensors(device) -> dict[str, Tensor]`; `update(final_pos, fused_feat, observed, existence_prob, track_confidence, dt)`
- `_forward_sequences(coor[B,T,50,2], conf[B,T,50,2], feat[B,T,50,128], gt[B,T,3,2], banks, tracker, dt)` 对全部 B*T 帧产生损失和轨迹序列。
- `sequence_tracking_metrics(pred_pos[T,3,2], active[T,3], observed[T,3], gt_pos[T,3,2], warmup, match_gate_m)`；warm-up 汇总后仅做一次槽映射。若目标在 warm-up 中无观测，仅在初始化时按固定 GT 槽补齐映射，并报告 target coverage/untracked frames，防止漏目标看成零误差。

## 验证命令

```powershell
& 'E:\Miniconda3\python.exe' -B -m unittest discover -s tests\my_track -v
& 'E:\Miniconda3\python.exe' -B -m py_compile original\my_track\association.py original\my_track\joint_tracker.py original\my_track\track_state.py original\my_track\train_tracker.py original\my_track\train.py original\my_track\metrics.py original\my_dataset\track_sequence_dataset.py
git diff --check
```

未安装 pytest，使用仓库已有 Python 环境运行标准库测试；不安装依赖。最终验收结果为 27 项测试通过，`py_compile` 与 `git diff --check` 通过，其中包含训练/验证进度显示的终端行为测试。真实 M1 缓存上的正样本率、阈值扫描和训练曲线尚待实际实验，不属于合成测试可以证明的事项。

## 实现边界与未校准项

- `CONF_THRESHOLD=0.96` 是旧脚本遗留配置，不参与新联合跟踪器的量测筛选。
- 1.0 m 关联半径、0.5 存在确认阈值、0.95 漏检置信度衰减和四项损失权重均是第一版初值。
- 当前测试证明接口、梯度、冷/暖冻结语义和合成状态行为；没有真实缓存正样本统计，也没有完整训练/验证指标。
- 旧 `FeatureMapper`、`ResidualGRU`、`ScoreFusion` 文件及导出为兼容保留；新训练只创建 `JointTracker`。
