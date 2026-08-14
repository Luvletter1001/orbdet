---
formal_train_status: running
formal_train_stage: orbdet_v0_2_anchored_symmetry_seed3407_post_epoch24_gate
baseline_train_status: h2rbox_recovery_epoch200_complete
baseline_val_mAP_epoch200: 0.8710
previous_candidate_status: orbdet_v0_1_algorithm_failed
candidate_train_status: running
candidate_test_status: held_out_forbidden_during_selection
smoke_train_status: passed_2_of_2_steps
user_explicit_authorization_required: satisfied
authorized_scope: orbdet_v0.2_hrsc_clean_gpu89
authorized_gpus: 8,9
authorized_at: 2026-08-14T23:34:54+08:00
started_at: 2026-08-14T23:36:51+08:00
target_train_epoch: 103
target_optimizer_steps: 11227
seed: 3407
early_gate_epoch: 24
early_gate_val_mAP: 0.70
early_gate_status: passed
epoch12_val_mAP: 0.7702
epoch24_val_mAP: 0.8946
epoch24_val_AP50: 0.8950
best_val_epoch: 24
best_val_mAP: 0.8946
target_val_mAP: 0.88
held_out_test_policy: forbidden_until_model_freeze
config: /data1/zcy/Orbdet/configs/orbdet/orbdet_v0_2_r50_hrsc_clean_gpu89.py
smoke_config: /data1/zcy/Orbdet/configs/orbdet/orbdet_v0_2_r50_hrsc_clean_gpu89_smoke.py
work_dir: /data1/zcy/Orbdet/work_dirs/formal/orbdet_v0_2_hrsc_clean_gpu89_seed3407_20260814
tmux_session: orbdet_v02_hrsc_clean_gpu89_seed3407_20260814
launcher_pid: 30007
rank_pids: 30010,30011
smoke_peak_memory_mib_per_gpu: 5607
updated_at: 2026-08-14T23:51:03+08:00
---

# 正式训练状态

用户已在当前回合明确授权实现 Orbdet-v0.2，并在物理 GPU 8/9 上训练。
代码、合同测试和双卡冒泡已经通过，正式 103E 训练已启动。

## 当前候选

- 方向锚点：官方 H2RBox-v2 original/rotated/flipped symmetry path。
- 边界处理：PSC coder + snap loss。
- 跨视图对应：相同目标使用固定 `bid` 聚合。
- Orbdet `q`：只记录 rotation/flip/HBox anchored diagnostics，不参与损失
  加权，也不参与推理打分。
- 数据：`train.txt` 训练、`val.txt` 选模、`test.txt` 保持封存。
- 计划：GPU 8/9 双 rank，batch 2/rank，seed 3407，103E。
- 提前停止门：epoch 24 validation mAP 小于 0.70 时停止候选。

epoch 12 clean validation 为 `0.7702`；epoch 24 升至 `0.8946`，AP50
`0.8950`、recall `0.919`。候选已通过非塌缩门与首 seed 的 0.88 目标，正式
训练继续到 103E。稳定性结论仍需 seed 42/2026，不以单 seed 代替。

上一轮 Orbdet-v0.1 最佳 validation mAP 为 `0.0916`，已经按预注册规则
判定算法失败；其 harmonic quality 不会用于本候选训练。
