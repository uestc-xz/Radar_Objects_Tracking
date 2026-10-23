# 事件级关联滞回与长间隔恢复对照（模块 25）

## 目的与边界

把 GT 槽位换 ID 按上一次成功匹配到本次匹配之间的空帧数拆分，分别观察连续帧竞争和漏检后重捕获。所有变体使用冻结 M1 的同一份坐标/置信度缓存，GT 只用于离线指标；不改 M1、默认 `LifecycleTracker`、训练器或训练配置。长间隔恢复复用既有 `IdentityRecoveryTracker`，与新滞回逻辑作为独立因子组合。

## 实现

- `original/my_track/event_hysteresis.py`：可选 `HysteresisLifecycleTracker`。只有空间门内存在多候选/多轨迹有效边歧义时，才对上一输入步仍被观测的已确认轨迹施加保持惩罚；惩罚对象是离其上一观测位置超过 `hold_gate_m` 的有效边。分配仍由全局门控匹配完成，惩罚非负，不会让门外边成为有效边。
- `scripts/evaluate_m1_event_hysteresis.py`：同序列独立重置，比较 `spatial`、`hysteresis`、`spatial_recovery`、`hysteresis_recovery`；报告 1m/0.5m 检测、坐标和身份指标，并把 IDSW 按 `consecutive`、`gap_1_2`、`gap_3_5`、`gap_gt5` 分箱。
- `tests/my_track/test_event_hysteresis.py`：零惩罚逐字段等价、合成交叉歧义的保持行为、仅门内歧义边加价、参数校验。
- `tests/my_track/test_event_hysteresis_evaluation.py`：验证连续 IDSW 与漏检后 IDSW 的分箱。
- `scripts/diagnose_m1_id_switches.py`：只读输出事件附近上下文，包括原始 M1 候选、分数、候选/GT 距离、更新前航迹状态和更新后输出；同时标记当前输出侧与候选侧是否存在门内歧义。

`hold_gate_m=.5`、`hold_penalty=.1`；恢复设置沿用已有局部可达门 `gate=1m`、增长 `.02m/step`、最大 `7.64m`、歧义边际 `.5m`、记忆 `HIST_LEN*50`。没有针对验证集扫参。

## 验证与结果

测试先在新评估函数不存在时按预期导入失败；模块和评估函数加入后，5 项新增测试通过。困难序列 `20240913_3_8` 共 2,470 帧；全验证集 8 条记录共 17,421 帧。四变体在同一回放中逐帧运行。

### 困难三目标序列

| 变体 | F1 | identity-F1 | GT-IDSW | MAE/m | 连续 | gap 1–2 | gap 3–5 | gap >5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| spatial | .7489 | .1481 | 76 | .3865 | 24 | 9 | 6 | 37 |
| hysteresis | .7396 | .1322 | 75 | .3903 | 25 | 8 | 5 | 37 |
| spatial + recovery | .7489 | .2051 | 73 | .3865 | 24 | 9 | 6 | 34 |
| hysteresis + recovery | .7396 | .1799 | 72 | .3903 | 25 | 8 | 5 | 34 |

### 完整八序列

| 变体 | F1 | identity-F1 | GT-IDSW | MAE/m | 连续 | gap 1–2 | gap 3–5 | gap >5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| spatial | .8856 | .3965 | 200 | .2805 | 86 | 18 | 11 | 85 |
| hysteresis | .8840 | .3893 | 204 | .2807 | 92 | 18 | 10 | 84 |
| spatial + recovery | .8856 | .4885 | 187 | .2805 | 86 | 18 | 11 | 72 |
| hysteresis + recovery | .8840 | .4726 | 191 | .2807 | 92 | 18 | 10 | 71 |

## 结论

此滞回规则没有达到预期：困难序列总 IDSW 只少 1，但连续换 ID 增 1，F1/身份 F1/MAE 均退化；完整八序列 IDSW 反增 4，连续换 ID `86→92`，identity-F1 `-.0072`。这说明“上一帧刚观测到”不是足够可靠的身份保持证据，简单惩罚候选远离上一坐标会损害正确的交叉/运动关联。**不设为默认，也不继续调这个惩罚强度。**

既有长间隔恢复效果再次复现：完整集 identity-F1 `.3965→.4885`、IDSW `200→187`，其中 `gap>5` 换 ID `85→72`；检测和 MAE 与空间基线逐点一致。该方法仍有双人/多人身份混淆风险，保持 opt-in。将滞回与恢复组合反而比单独恢复差（identity-F1 `.4885→.4726`），不组合推广。

原始结果 JSON：

- 困难序列：`original/my_track/runs/event_hysteresis_20261023_hard.json`
- 完整验证集：`original/my_track/runs/event_hysteresis_20261023_full.json`
- 困难序列事件证据：`original/my_track/runs/id_switch_diagnostic_20261023_hard_spatial_all_v2.json`、`id_switch_diagnostic_20261023_hard_recovery_all_v2.json`

复现命令：

```powershell
E:\Miniconda3\python.exe scripts\evaluate_m1_event_hysteresis.py --sequence 20240913_3_8 --output original\my_track\runs\event_hysteresis_20261023_hard.json --hold-gate-m 0.5 --hold-penalty 0.1
E:\Miniconda3\python.exe scripts\evaluate_m1_event_hysteresis.py --output original\my_track\runs\event_hysteresis_20261023_full.json --hold-gate-m 0.5 --hold-penalty 0.1
```

以上输出路径已存在；复跑需更换新 JSON 文件名。实验未创建 Git 提交，暂存/提交由用户自行操作。

## 事件证据结论（困难三目标）

早期仅看主聚类的当前帧分类为：连续帧且输出/主候选均歧义 12 次；连续帧仅输出歧义 4 次；连续帧均无当前歧义 8 次；有间隔且均无当前歧义 41 次；有间隔且输出/主候选均歧义 7 次；有间隔仅输出歧义 4 次。该分类只说明当帧几何关系，不能据此断言“无歧义必然是前序回切”或“有间隔就是 M1 漏检”。

进一步加入弱提议与完整 gap 覆盖率后的最终口径见 `experiments/m1_id_switch_diagnosis_20261023.md`：52 个有间隔事件仅 6 个全程无 .8 门内候选，确认与容量规则会放大输出缺口。`spatial_recovery` 将困难序列 IDSW `76→73`，管理统计中恢复 12 个片段、歧义拒绝 13 次；恢复片段数不等于修复的 IDSW 数。保留上述历史结果，不继续叠加通用连续帧滞回。
