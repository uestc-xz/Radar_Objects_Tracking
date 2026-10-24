# 低置信持续 coast 输出 — 2026-10-24

模块 34。依赖模块 33 的已知人数/no-death coast 对照。目的不是将 coast 输出视作正式方案，而是隔离回答一个问题：已有轨迹即使置信度持续衰减，是否还应继续输出其预测坐标？

## 假设与实现

新增显式评估变体 `lifecycle_known_no_death_persistent_coast`，固定人数仍只用于每序列 `max_tracks`；no-death 保留轨迹状态。相对于模块 33，仅改变 `output_confidence=.5` 为 `0`；`output_max_missed=HIST_LEN`、其它状态机、M1 缓存和坐标预测公式不变。变体未加入 `DEFAULT_VARIANTS`，没有改变默认行为。

注意，输出阈值设为 0 并不意味着模型置信度被提升；它只允许置信度衰减后的预测位置进入结果。GT 仍只用于离线评价，不参与逐帧推理。

## 结果

| 数据 | 变体 | Coast TP | Coast FP | Precision | Recall | F1 | Identity-F1 | IDSW | Matched MAE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 全 8 序列 | coast20，置信门 .5 | 1048 | 1225 | .8980 | .6646 | .7639 | .6053 | 153 | .2814m |
| 全 8 序列 | 持续 coast，置信门 0 | 1149 | 3403 | .8234 | .6674 | .7373 | .5837 | 145 | .2828m |
| 困难 20240913_3_8 | coast20，置信门 .5 | 序列明细见 JSON | 序列明细见 JSON | .8242 | .5548 | .6632 | .4012 | 64 | - |
| 困难 20240913_3_8 | 持续 coast，置信门 0 | 序列明细见 JSON | 序列明细见 JSON | .7383 | .5607 | .6374 | .3853 | 64 | - |

全量仅多 101 个 coast TP，却多 2178 个 coast FP；recall 增长 0.28 个百分点，precision 下降 7.46 点，F1 下降 2.66 点，identity-F1 下降 2.16 点。GT-IDSW 少 8 次不抵消输出中大量错误预测；matched MAE 轻微升高。困难序列同样是小幅增召回、较大精度损失并伴随 F1/identity-F1 下降。结论：低置信位置不能因为轨迹身份仍存活就默认作为有效坐标输出，不推广。

本结论受已知人数、目标不消亡这一理想化先验限制；数据是同一验证集八条序列。它不证明未来引入更好的运动模型、坐标不确定性或输出状态区分时也无收益。

## 复现

```powershell
E:\Miniconda3\python.exe -B scripts\evaluate_m1_lifecycle.py --sequence 20240913_3_8 --variants lifecycle_known_no_death lifecycle_known_no_death_coast lifecycle_known_no_death_persistent_coast --output original\my_track\runs\known_no_death_persistent_coast_hard_20261024.json
E:\Miniconda3\python.exe -B scripts\evaluate_m1_lifecycle.py --variants lifecycle_known_no_death lifecycle_known_no_death_coast lifecycle_known_no_death_persistent_coast --output original\my_track\runs\known_no_death_persistent_coast_full_20261024.json
E:\Miniconda3\python.exe -B -m unittest discover -s tests\my_track -p test_lifecycle_evaluation.py -q
```

JSON 保存在 `original/my_track/runs/known_no_death_persistent_coast_{hard,full}_20261024.json`。评估指标为当前项目的 gated 1-to-1 距离诊断，并非 MOTChallenge/HOTA。Git 由用户手动提交。
