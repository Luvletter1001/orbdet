# H2RBox HRSC optimizer-step 对齐恢复实验结果

## 状态与结论

`continuation_authorized_to_epoch_200`：恢复实验已按协议完成 30E/60E 两个检查点。
结果确认此前 clean H2RBox 的低 AP 是优化合同塌缩，而不是模型在当前数据上只能
达到约 0.13 mAP。用户随后明确授权从 60E 继续训满 200E。

30E 恢复门通过后才从 `epoch_30.pth` 续训；日志明确记录
`resumed epoch: 30, iter: 6540`。60E 的 train/val 均继续提升。200E 延长阶段保持
相同优化合同并按 30E 间隔做合理观察；最终只评 val/train，不运行 held-out test。

## 固定实验合同

| item | value |
|---|---|
| model | `H2RBoxDetector / R50-FPN / H2RBoxHead` |
| train / val / held-out test | 436 / 181 / 453 images |
| physical GPUs | 8,9 |
| batch | 1/GPU, global 2 |
| optimizer | AdamW, lr `1e-4`, weight decay `0.05` |
| warmup | 500 optimizer steps |
| LR milestones | 105060 / 144612 optimizer steps |
| aligned endpoint | 157590 optimizer steps |
| seed | 3407 |
| steps per epoch | 218 |
| 30E / 60E budget | 6540 / 13080 optimizer steps |
| held-out test | not evaluated |

两个恢复检查点都早于第一个 LR milestone，因此 LR 全程保持 `1e-4`；这正是本次
需要隔离验证的 batch、LR 与 optimizer-step 更新频率合同。

## 30E/60E 结果

| epoch | split | mAP | AP50 | recall | GT | detections |
|---:|---|---:|---:|---:|---:|---:|
| 30 | train | 0.7492 | 0.7490 | 0.854 | 1207 | 2786 |
| 30 | val | 0.7240 | 0.7240 | 0.819 | 541 | 1182 |
| 60 | train | 0.8565 | 0.8570 | 0.907 | 1207 | 2663 |
| 60 | val | 0.7660 | 0.7660 | 0.850 | 541 | 1230 |

从 30E 到 60E：train mAP 增加 `0.1073`，val mAP 增加 `0.0420`；train/val
recall 分别增加 `0.053` 和 `0.031`。60E train mAP 相比失败参考线 `0.1308`
高 `0.7257`，已经明确离开塌缩区。

## 恢复门与继续训练判断

- 30E 时 train mAP/recall 为 `0.7492/0.854`，val mAP/recall 为
  `0.7240/0.819`，满足“明显离开失败区且 train/val 一致恢复”的继续条件。
- 60E 时 train 和 val 同时继续上升，没有出现 train 上升而 val 下降的反向信号。
- 当前仅完成 `13080 / 157590 = 8.3%` 的历史 optimizer-step 对齐总预算，且
  `30E -> 60E` 尚未形成平台期。因此建议继续，但采用新的有界门：恢复到 120E，
  在 90E/120E 记录 val，并在 120E 复测 train；是否再延长应由这些结果决定。
- 用户已于 60E 门完成后明确授权续训至 200E；延长阶段必须在 200E 自动停止。

## 与历史 88.2 AP 的关系

历史 Orbdet-v0.1 的 `0.8820` 确实存在，但准确记录是 epoch 360 的最佳点，epoch
510 为 `0.8807`。该实验使用 617 张 `trainval` 训练，并在 453 张 `test` 上反复
验证和选 best；当前恢复实验使用 436 张 `train` 训练、181 张 `val` 检查，并保留
453 张 test 未触碰。因此 `0.8820` 不能与本次 60E val `0.7660` 直接相减。

## 运行与完整性证据

| item | evidence |
|---|---|
| stage 1 | 2026-08-14 11:58:25–12:21:17 +08:00, train 1–30E + val |
| stage 2 | 2026-08-14 12:24:39–12:44:38 +08:00, resume 31–60E + val |
| MMEngine peak memory | 2218 MiB/GPU |
| observed GPU process memory | about 4080 MiB/GPU |
| final train window | loss 0.8491, grad norm 16.6590, finite |
| contract/regression tests | 17 passed, 2 existing warnings |
| fatal-log scan | no traceback, CUDA OOM, NCCL failure, runtime error, NaN or Inf match |
| 60E gate process check | no matching train/test process before extension |
| 60E gate GPU check | GPU 8/9 each 17 MiB, 0% utilization before extension |

## 200E 延长阶段

- 恢复起点：`epoch_60.pth` / iter 13080。
- 终点：200E / 43600 optimizer steps；仍早于第一个 105060-step LR milestone。
- 中间 val：90E、120E、150E、180E；200E 结束后显式评 val 与 train。
- checkpoint 保留数提高到 7，仅影响产物保留，不改变模型、数据或优化过程。
- 后台入口：
  `scripts/formal/resume_h2rbox_hrsc_recovery_e60_to_e200_gpu89_20260814.sh`。
- tmux：`h2rbox_recovery_e60_to_e200_20260814`；启动时间
  `2026-08-14 12:52:50 +08:00`。
- 延长阶段主日志：
  `work_dirs/formal/h2rbox_r50_hrsc_recovery_bs1_stepaligned_gpu89_seed3407_20260814/20260814_125240/20260814_125240.log`。
- 聚合启动日志：
  `work_dirs/h2rbox_hrsc_recovery_e60_to_e200_gpu89_20260814.launch.log`。

60E checkpoint SHA256：

- `epoch_60.pth`：`ac120e79d70099605f4cabef3be0d3382ca461218e2752b33ef496c3aa8dbf40`
- `best_dota_mAP_epoch_60.pth`：`a6de0ccd4953b98772bf57451a6742b2594e169d59f49278e2e857f5e8a9eebf`

## 产物

- 配置：`configs/orbdet/h2rbox_r50_hrsc_recovery_bs1_stepaligned_60e_gpu89.py`
- 契约测试：`tests/test_h2rbox_hrsc_recovery_contract.py`
- 计划：`resultmd/exp_h2rbox_hrsc_recovery/fplan_h2rbox_hrsc_recovery_gpu89_20260814.md`
- checkpoint：
  `work_dirs/formal/h2rbox_r50_hrsc_recovery_bs1_stepaligned_gpu89_seed3407_20260814/epoch_60.pth`
- best-val checkpoint：
  `work_dirs/formal/h2rbox_r50_hrsc_recovery_bs1_stepaligned_gpu89_seed3407_20260814/best_dota_mAP_epoch_60.pth`
- 1–30E 日志：
  `work_dirs/formal/h2rbox_r50_hrsc_recovery_bs1_stepaligned_gpu89_seed3407_20260814/20260814_115819/20260814_115819.log`
- 31–60E 日志：
  `work_dirs/formal/h2rbox_r50_hrsc_recovery_bs1_stepaligned_gpu89_seed3407_20260814/20260814_122435/20260814_122435.log`
- 30E train eval：
  `work_dirs/eval/h2rbox_hrsc_recovery_epoch30_train_gpu8_20260814/20260814_122210/20260814_122210.log`
- 60E train eval：
  `work_dirs/eval/h2rbox_hrsc_recovery_epoch60_train_gpu8_20260814/20260814_124534/20260814_124534.log`
