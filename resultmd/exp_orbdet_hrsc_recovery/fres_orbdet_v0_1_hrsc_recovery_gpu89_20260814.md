# Orbdet-v0.1 HRSC 恢复合同严格对照结果

## 状态

`complete_algorithm_failed`：严格合同与回归测试、双卡冒泡、正式
30→60→200E、显式 epoch-200 val/train 评测均已完成。Candidate 最佳 val
`0.0916` 低于预注册 H2RBox reference `0.8710`，因此 Orbdet-v0.1 按规则判定
失败；本轮未评 held-out test。

- tmux：`orbdet_v01_recovery_staged200e_gpu89_20260814`
- launcher PID：`4189894`
- main rank PIDs：`4189897,4189898`
- launch log：
  `work_dirs/orbdet_v0_1_hrsc_recovery_staged200e_gpu89_20260814.launch.log`

首个稳定窗口已于 `18:29:33` 核验：两个主 rank 存活，GPU 8/9 各约
4038 MiB；epoch 1 iter 180/218 的 loss `1.9493`、loss_bbox_ss `0.1137`、
grad norm `17.5858`、q_mean `0.5702` 均为有限值，fatal scan 无匹配。

正式训练于 `21:49:45` 保存 epoch-200 checkpoint；显式 val/train 评测于
`21:51:15` 完成。tmux 已正常退出，精确 train/test 进程均不存在，GPU 8/9
各回到 17 MiB、0% utilization。

## 预注册判定

- reference：恢复合同 H2RBox，显式 epoch-200 val `dota/mAP=0.8710`。
- candidate：相同 seed 3407、相同恢复合同，只替换
  `OrbdetDetector + OrbdetHarmonicConsistencyLoss`。
- candidate 最佳 val `< 0.8710`：Orbdet-v0.1 失败，进入模型修改。
- candidate 最佳 val `>= 0.8710`：单种子门通过，后续再补多 seed。
- 本轮禁止 held-out test。

## 冻结合同

| item | value |
|---|---|
| dataset | HRSC2016，train/val/test = 436/181/453 |
| batch | 1/GPU，global 2 |
| optimizer | AdamW，lr 1e-4，weight decay 0.05 |
| warmup | 500 optimizer steps |
| scheduler | iteration-based；milestones 105060/144612 |
| seed | 3407，deterministic=False |
| execution | fresh 0→30E；resume 30→60E；resume 60→200E |
| validation | every 30E + explicit epoch-200 val |
| held-out test | forbidden |

合同测试把 candidate 的 detector/loss/work_dir 归一化回 reference 后做完整配置
比较。新合同、恢复基线合同与 Orbdet 回归测试合计 `16 passed`，另有两个既有依赖
warning。

## 唯一功能变量

| reference | candidate |
|---|---|
| `H2RBoxDetector` | `OrbdetDetector` |
| `H2RBoxConsistencyLoss` | `OrbdetHarmonicConsistencyLoss` |

Candidate 固定 `loss_weight=0.4`、`min_quality=0.25`、`gamma=2.0`、
`high_quality_thr=0.75`；center/shape/angle 配置与 reference 对齐。

## 双卡冒泡

冒泡时间：`2026-08-14 18:26:49–18:27:04 +08:00`。物理 GPU 8/9，四图，
global batch 2，共 2 optimizer steps：

| step | loss | loss_bbox_ss | grad_norm | q_mean | q_min | q_high_frac |
|---:|---:|---:|---:|---:|---:|---:|
| 1/2 | 7.7960 | 1.2953 | 263.1058 | 0.6138 | 0.3115 | 0.2222 |
| 2/2 | 6.3395 | 0.7569 | 197.1045 | 0.6192 | 0.2503 | 0.3333 |

- 所有记录值有限；配置的 gradient clipping max norm 为 35。
- MMEngine 峰值显存：2218 MiB/GPU。
- checkpoint：
  `work_dirs/smoke/orbdet_v0_1_hrsc_recovery_bs1_gpu89_20260814/epoch_1.pth`。
- fatal scan：无 traceback、CUDA OOM、NCCL failure、runtime error、NaN/Inf。

## Validation ledger

| epoch | val mAP | AP50 | recall | decision |
|---:|---:|---:|---:|---|
| 30 | 0.0450 | 0.0450 | 0.133 | continue |
| 60 | 0.0668 | 0.0670 | 0.216 | continue |
| 90 | 0.0399 | 0.0400 | 0.183 | continue |
| 120 | 0.0787 | 0.0790 | 0.235 | continue |
| **150** | **0.0916** | **0.0920** | **0.275** | **best candidate** |
| 180 | 0.0871 | 0.0870 | 0.283 | below best |
| 200 (explicit) | 0.0893 | 0.0890 | 0.266 | final |

自动保存的最佳 checkpoint 是 `best_dota_mAP_epoch_150.pth`，metadata 为
epoch 150 / iter 32700。

## Epoch-200 显式评测

| split | mAP | AP50 | recall | GT | detections |
|---|---:|---:|---:|---:|---:|
| val | 0.0893 | 0.0890 | 0.266 | 541 | 537 |
| train | 0.0949 | 0.0950 | 0.300 | 1207 | 1174 |

Train AP 同样只有 `0.0949`，证明不是单纯 val 泛化差，而是训练本身仍处于塌缩
区域。作为对照，恢复合同 H2RBox 的 epoch-200 train mAP 为 `0.9015`。

## 预注册结论

| method | selected epoch | best val mAP | delta vs reference |
|---|---:|---:|---:|
| recovered H2RBox | 200 | 0.8710 | 0 |
| Orbdet-v0.1 | 150 | 0.0916 | -0.7794 |

Orbdet-v0.1 比 reference 低 `0.7794` raw mAP，即 **77.94 AP points**。根据训练前
冻结的 `< 0.8710 -> failed` 规则，v0.1 正式失败，下一步应修改模型/损失，而不是
继续堆 epoch、换数据集或用 held-out test 调参。

Epoch 200 最后训练窗口仍显示 q_mean `0.9980`、q_min `0.9783`、q_high_frac
`1.0000`，但 train/val AP 同时塌缩。这说明当前 harmonic quality 很快饱和到 1，
没有提供有效的可靠性区分；这是下一版需要优先诊断的信号，而不是成功证据。

## 运行完整性

| item | value |
|---|---|
| formal start | 2026-08-14 18:28:25 +08:00 |
| epoch-200 checkpoint | 2026-08-14 21:49:45 +08:00 |
| explicit evaluation complete | 2026-08-14 21:51:15 +08:00 |
| total wall time | about 3:22:50 |
| MMEngine peak memory | 2219 MiB/GPU |
| observed process memory | about 4160 MiB/GPU |
| fatal scan | 0 matches |
| held-out test | not run |

## 产物

- design：
  `docs/superpowers/specs/2026-08-14-orbdet-v0-1-recovery-contract-200e-design.md`
- plan：
  `docs/superpowers/plans/2026-08-14-orbdet-v0-1-recovery-contract-200e.md`
- candidate config：
  `configs/orbdet/orbdet_v0_1_r50_hrsc_recovery_bs1_stepaligned_60e_gpu89.py`
- smoke config：
  `configs/orbdet/orbdet_v0_1_r50_hrsc_recovery_bs1_stepaligned_smoke_gpu89.py`
- contract test：`tests/test_orbdet_hrsc_recovery_contract.py`
- formal launcher：
  `scripts/formal/run_orbdet_v0_1_hrsc_recovery_staged_200e_gpu89_20260814.sh`
- formal work dir：
  `work_dirs/formal/orbdet_v0_1_hrsc_recovery_bs1_stepaligned_200e_gpu89_seed3407_20260814/`
- best checkpoint：
  `work_dirs/formal/orbdet_v0_1_hrsc_recovery_bs1_stepaligned_200e_gpu89_seed3407_20260814/best_dota_mAP_epoch_150.pth`
- epoch-200 val：
  `work_dirs/eval/orbdet_v0_1_hrsc_recovery_epoch200_val_gpu8_20260814/`
- epoch-200 train：
  `work_dirs/eval/orbdet_v0_1_hrsc_recovery_epoch200_train_gpu8_20260814/`
