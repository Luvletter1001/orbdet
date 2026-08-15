---
formal_train_status: running
formal_train_stage: dota1_msrr_gpu4567_stage1_3e
user_explicit_authorization_required: satisfied
authorized_scope: orbdet_v0.2_hrsc_clean_gpu89
authorized_additional_scope: orbdet_tasks_1_2_3_gpu89
authorized_gpus: 8,9
authorized_at: 2026-08-15T03:20:00+08:00
completed_at: 2026-08-15T12:45:31+08:00
baseline_seed3407_status: complete_epoch103
baseline_seed3407_best_epoch: 48
baseline_seed3407_best_val_mAP: 0.9029
baseline_seed42_status: complete_epoch103
baseline_seed42_best_epoch: 60
baseline_seed42_best_val_mAP: 0.8992
baseline_seed2026_status: complete_epoch103
baseline_seed2026_best_epoch: 72
baseline_seed2026_best_val_mAP: 0.8977
godc_code_status: tests_passed_68
godc_smoke_status: complete_2_steps
godc_formal_status: complete_epoch103
godc_best_epoch: 24
godc_best_val_mAP: 0.8980
dota_v1_code_status: tests_passed_68
dota_v1_smoke_status: complete_2_steps
dota_v1_formal_status: complete_epoch12
held_out_test_policy: not_run
dota1_msrr_gpu4567_audit_status: complete
dota1_msrr_gpu4567_smoke_status: complete_2_steps
dota1_msrr_gpu4567_stage1_status: running_epoch2_epoch1_verified
dota1_msrr_gpu4567_epoch1_checkpoint_bytes: 389144617
dota1_msrr_gpu4567_epoch1_checkpoint_sha256: 025ef114f7aaf0d9c65ef97547401bd5424c28b9cd7a6cba9344de9ee2115f94
dota1_msrr_gpu4567_posteval_status: armed_waiting_for_epoch3
dota1_msrr_gpu4567_posteval_deadline: 2026-08-16T09:50:00+08:00
dota1_msrr_gpu4567_posteval_tmux: orbdet_msrr4567_stage3_posteval_20260816
dota1_msrr_gpu4567_checkpoint_gate: epoch3_iter51246_state371_sha256
dota1_msrr_gpu4567_authorized_gpus: 4,5,6,7
dota1_msrr_gpu4567_authorized_at: 2026-08-16T03:59:00+08:00
dota1_msrr_gpu4567_started_at: 2026-08-16T04:39:47+08:00
dota1_msrr_gpu4567_deadline: 2026-08-16T09:55:00+08:00
dota1_msrr_gpu4567_tmux: orbdet_msrr4567_6h_20260816
v02_worktree: /data1/zcy/Orbdet/.worktrees/v02-stability
godc_worktree: /data1/zcy/Orbdet/.worktrees/godc-integration
dota_v1_worktree: /data1/zcy/Orbdet/.worktrees/dota-v1-formal
updated_at: 2026-08-16T06:09:01+08:00
---

# 正式训练状态

用户授权的任务 1、2、3 已全部完成；2026-08-16 新增的任务 4 正在运行。此前
正式训练使用物理 GPU 8/9，任务 4 使用 GPU 4–7。所有分布式运行均设置
`NCCL_P2P_DISABLE=1` 与 `NCCL_IB_DISABLE=1`。训练产物与 Git 历史只保存
在本地，没有推送远端。

> 文件名为早期安全门的兼容路径；本页 front matter 与正文是当前权威状态。

## 任务 1：Orbdet-v0.2 clean-HRSC 多 seed

| seed | schedule | best_epoch | val_mAP | final_ckpt |
|---:|---:|---:|---:|---|
| 3407 | 103E | 48 | 0.9029 | `work_dirs/formal/orbdet_v0_2_hrsc_clean_gpu89_seed3407_20260814/epoch_103.pth` |
| 42 | 103E | 60 | 0.8992 | `work_dirs/formal/orbdet_v0_2_hrsc_clean_gpu89_seed42_20260815/epoch_103.pth` |
| 2026 | 103E | 72 | 0.8977 | `work_dirs/formal/orbdet_v0_2_hrsc_clean_gpu89_seed2026_20260815/epoch_103.pth` |

三个 seed 的 best-val `mAP` 均值为 `0.8999`，最差为 `0.8977`，总体标准差
约为 `0.0022`。任务 1 的队列完成标记位于
`work_dirs/formal/orbdet_v0_2_hrsc_multiseed_gpu89_20260815/COMPLETE`。

## 任务 2：GODC HBox/FPN C2

- 双 rank、2 optimizer-step smoke 已完成并生成 `epoch_1.pth`。
- seed 3407 正式训练完成 103E，最佳 clean validation 为 epoch 24 的
  `mAP=0.8980`、`AP50=0.8980`。
- 最终 checkpoint：
  `work_dirs/formal/orbdet_godc_c2_hrsc_clean_gpu89_seed3407_20260815/epoch_103.pth`。
- 控制器完成标记：
  `work_dirs/formal/orbdet_godc_after_v02_gpu89_20260815/COMPLETE`。

## 任务 3：Orbdet-v0.2 DOTA-v1 1x

- 修复多类别 rotation-agnostic 标签在旧版 CUDA/PyTorch 上对 Long tensor
  执行 `index_reduce` 的兼容性问题后，双 rank、2 optimizer-step smoke
  完成并生成 `epoch_1.pth`。
- 正式训练于 2026-08-15 07:14 启动，12E 于 12:45 完成。
- checkpoint：`epoch_4.pth`、`epoch_8.pth`、`epoch_12.pth` 均已生成；
  最终 checkpoint 位于
  `work_dirs/formal/orbdet_v0_2_dota1_1x_gpu89_seed3407_20260815/epoch_12.pth`。
- 控制器完成标记：
  `work_dirs/formal/orbdet_v02_dota1_after_godc_gpu89_20260815/COMPLETE`。
- DOTA prepared trainval 没有独立 validation split，因此本次合同不运行
  validation，也不报告训练集伪验证 `mAP`；hidden test 未运行。

## 任务 4：Orbdet-v0.2 DOTA-v1 MS+RR GPU4567 stage 1

- 用户授权六小时 GPU 4–7 窗口；完整 12E 无法在该窗口内完成，因此本轮只运行
  可续训的 epoch 1–3，每 epoch 保存 checkpoint。
- 官方 SS/MS+RR checkpoint 审计已完成；SS trainval `mAP=0.8131`，两份
  submission ZIP 均通过 15 文件与压缩完整性检查。
- 四 rank 两步 smoke 已完成并生成 `epoch_1.pth`。
- stage 1 于 2026-08-16 04:39:47 启动；epoch 1 于 06:08:17 完整落盘并
  通过 epoch=1、iter=17,082、371 tensors 与 SHA256 验证，当前已进入
  epoch 2。控制器硬截止约为 09:55。
- 当前只计划在 epoch 3 完整结束且剩余至少 1,800 秒时运行 trainval/SS/MS
  阶段评测；评测绝对截止为 09:50，不启动 epoch 4。
- 有界 watcher 已于 04:55:21 启动，目前只轮询 stage `COMPLETE`，不占用 GPU。
- `tmux_session=orbdet_msrr4567_6h_20260816`；完整 12E 状态仍为
  `not_complete`，后续 resume 需要用户再次明确授权。

## 验收与安全

- 三项正式控制器均正常写出 `COMPLETE` 与合同要求的最终 checkpoint。
- 最终控制器日志未发现 `Traceback`、`RuntimeError`、NCCL error 或 `NaN`。
- DOTA 源数据保持只读；没有改写 annotation/image。
- 没有终止或修改其他用户的 GPU 进程。
- 详细结果记录：
  `resultmd/exp_orbdet_overnight_123_20260815/fres_orbdet_overnight_123_gpu89.md`。
