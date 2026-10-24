# 已知人数、无消亡与身份恢复组合 — 2026-10-24

模块 32，依赖模块 20/21 已验证生命周期和恢复器。用户接受序列目标不消亡的理想化条件；本实验只检验旧轨迹不删除时是否仍有必要叠加恢复器。由用户手动提交。

## 假设与实现

困难三目标 ID 切换事件有较多长漏检间隔。若保持轨迹存活且上限等于已知目标数，重新观测可能直接回到旧片段，无须事后做片段 ID 继承。历史组合 `lifecycle_known_no_death` 已存在，但没有直接与身份恢复配对的完整比较。

新增评估选项 `lifecycle_known_no_death_recovery`，只在显式 `--variants` 指定时执行，不加入默认集合。固定人数仅用在每序列 `max_tracks` 和 `TrackletIdentityRecovery.known_count` 的唯一缺失条件；每帧检测/关联不读取 GT。其余跟踪参数与 `lifecycle_known_no_death` 一致。恢复器只改 fragment ID，不改坐标、分数或观测标记。

## 结果

| 数据 | 变体 | P | R | F1 | Identity-F1 | GT-IDSW |
|---|---|---:|---:|---:|---:|---:|
| 困难 20240913_3_8 | known no-death | .8655 | .5270 | .6551 | .3932 | 62 |
| 困难 20240913_3_8 | + recovery | .8655 | .5270 | .6551 | .3932 | 62 |
| 全部 8 序列 | known no-death | .9309 | .6542 | .7684 | .6091 | 155 |
| 全部 8 序列 | + recovery | .9309 | .6542 | .7684 | .6091 | 155 |

- 八序列逐样本 prediction digest 与原 `lifecycle_known_no_death` 实验完全相同；evidence digest 也相同，recovered 计数八条均为 0。
- 意味着该设置下恢复器并未触发：状态未被删除，旧片段在后续关联时继续沿用原 ID。相同结果不是恢复器额外收益。
- 理想化人数/不消亡模式的 Identity-F1 `.6091` 高于自主 cascade + reachable recovery 的 `.4885`，但总体召回仅 `.6542` 对 `.8338`，F1 `.7684` 对 `.8856`。短暂连续输出不能覆盖长无量测间隔；no-death 也没有恢复这些缺失坐标。
- 不推广为默认。它适合作为明确先验下的对照，不能说明未知目标数或目标会消亡时的效果。

## 复现与测试

```powershell
E:\Miniconda3\python.exe -B scripts\evaluate_m1_lifecycle.py --variants lifecycle_known_no_death lifecycle_known_no_death_recovery --sequence 20240913_3_8 --output original\my_track\runs\known_no_death_recovery_hard_20261024.json
E:\Miniconda3\python.exe -B scripts\evaluate_m1_lifecycle.py --variants lifecycle_known_no_death lifecycle_known_no_death_recovery --output original\my_track\runs\known_no_death_recovery_full_20261024.json
E:\Miniconda3\python.exe -B -m unittest discover -s tests\my_track -p test_lifecycle_evaluation.py -q
```

最终全 `tests/my_track` 252 项通过；新评估器没有改变默认变体集合。JSON 保存在 `original/my_track/runs/known_no_death_recovery_hard_20261024.json` 与 `original/my_track/runs/known_no_death_recovery_full_20261024.json`。

## 延长 coast 输出对照（模块 33）

在同一理想化 known/no-death 轨迹设置下，增加显式 opt-in `lifecycle_known_no_death_coast`，仅把 `output_max_missed` 从 2 改为 `HIST_LEN=20`，保留原输出置信度 `.5`。

| 数据 | 变体 | Precision | Recall | F1 | Identity-F1 | GT-IDSW |
|---|---|---:|---:|---:|---:|---:|
| 全部 8 序列 | known no-death | .9309 | .6542 | .7684 | .6091 | 155 |
| 全部 8 序列 | + coast20 | .8980 | .6646 | .7639 | .6053 | 153 |
| 困难 20240913_3_8 | known no-death | .8655 | .5270 | .6551 | .3932 | 62 |
| 困难 20240913_3_8 | + coast20 | .8242 | .5548 | .6632 | .4012 | 64 |

全局只增加约 1.0 个百分点召回，precision 降 3.3 点，identity-F1/F1 略降；困难序列 F1/identity-F1 小幅上升但 IDSW 增加。不能据困难单序列收益忽略 aggregate precision/ID 指标，故不推广。它量化了“继续输出预测坐标”对准确率的代价；HIST_LEN 是本次唯一变更，不进行同验证集扫 coast 时长。

命令：

```powershell
E:\Miniconda3\python.exe -B scripts\evaluate_m1_lifecycle.py --variants lifecycle_known_no_death lifecycle_known_no_death_coast --sequence 20240913_3_8 --output original\my_track\runs\known_no_death_coast_hard_20261024.json
E:\Miniconda3\python.exe -B scripts\evaluate_m1_lifecycle.py --variants lifecycle_known_no_death lifecycle_known_no_death_coast --output original\my_track\runs\known_no_death_coast_full_20261024.json
```
