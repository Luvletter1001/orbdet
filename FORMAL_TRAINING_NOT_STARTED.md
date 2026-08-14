---
formal_train_status: running
formal_train_stage: orbdet_v0_2_hrsc_seed42_then_seed2026
user_explicit_authorization_required: satisfied
authorized_scope: orbdet_v0.2_hrsc_clean_gpu89
authorized_additional_scope: orbdet_tasks_1_2_3_priority_1_2
authorized_gpus: 8,9
authorized_at: 2026-08-15T03:20:00+08:00
baseline_seed3407_status: complete_epoch103
baseline_seed3407_best_epoch: 48
baseline_seed3407_best_val_mAP: 0.9029
baseline_seed42_status: running
baseline_seed42_started_at: 2026-08-15T03:26:55+08:00
baseline_seed2026_status: queued
godc_code_status: tests_passed_63
godc_smoke_status: armed_after_multiseed_success
godc_formal_status: armed_after_smoke_success
held_out_test_policy: forbidden_until_model_freeze
v02_worktree: /data1/zcy/Orbdet/.worktrees/v02-stability
godc_worktree: /data1/zcy/Orbdet/.worktrees/godc-integration
v02_tmux_session: orbdet_v02_multiseed_gpu89_20260815
godc_controller_tmux_session: orbdet_godc_after_v02_gpu89_20260815
updated_at: 2026-08-15T03:35:07+08:00
---

# 正式训练状态

用户已明确授权任务 1、2、3 的正式训练，并要求本夜优先完成任务 1、2；物理
GPU 8/9 的所有启动均固定设置 `NCCL_P2P_DISABLE=1` 与
`NCCL_IB_DISABLE=1`。所有工作仅保存在本地，不推送远端。

## 已完成的参考 seed

Orbdet-v0.2 clean-HRSC seed 3407 已在 2026-08-15 00:36:58 完整跑到
epoch 103。最佳 clean validation 是 epoch 48 的 mAP `0.9029`、AP50
`0.9030`。held-out test 仍未运行。

## 当前任务 1

- 冻结分支：`exp/v02-hrsc-stability`。
- 顺序：seed 42 正在运行，正常结束后自动启动 seed 2026。
- 队列采用独立 config、端口和 work dir；已有 checkpoint 时拒绝覆盖。
- 成功/失败标记：
  `work_dirs/formal/orbdet_v0_2_hrsc_multiseed_gpu89_20260815/`。

## 已布防的任务 2

GODC 的 HBox/FPN `C2` 接入已通过项目 `tests/` 全量 63 项测试。独立控制器
当前只等待任务 1 的 `COMPLETE` 标记，不占用 GPU。任务 1 成功后，它将：

1. 运行 8 图、双 rank、2 optimizer-step smoke；
2. 检查 smoke 生成 `epoch_1.pth`；
3. 仅在 smoke 成功后启动 seed 3407 的 103E clean-HRSC 正式训练；
4. 任一阶段失败即写 `FAILED` 并停止，不会结束其他用户进程。

任务 3（DOTA-v1）保持授权但优先级较低，不会插到任务 1、2 之间。
