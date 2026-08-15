# Orbdet-v0.2 DOTA-v1 R50 MS+RR 12E 执行日志

## 当前状态

`formal_status=running_stage1_3e`，完整 12E 仍为 `not_complete`。用户于
2026-08-16 授权 GPU 4–7 六小时正式训练窗口。由于完整四卡 12E 预计约 18 小时，
本窗口采用可续训 stage 1：先完成官方 checkpoint 审计与四 rank smoke，再训练
epoch 1–3；每 epoch 保存，后续必须经新授权从 `epoch_3.pth` 续到 12E。

当前等待条件：

- `official_audit=complete`
- `official_ss_trainval_mAP=0.8131`
- `official_ss_zip=validated_15_files`
- `official_msrr_zip=validated_15_files`
- `smoke=complete_2_steps`
- `formal_stage1=running_epoch1`
- `stage1_started_at=2026-08-16T04:39:47+08:00`
- `controller_deadline_about=2026-08-16T09:55:00+08:00`

没有终止、修改或抢占任何外部 GPU 进程。DOTA 源图像与 annotations 未改写。

## GPU4567 六小时阶段合同

| field | value |
|---|---|
| physical_gpus | 4,5,6,7 |
| batch_per_rank | 1 |
| global_batch | 4 |
| optimizer | AdamW |
| lr | `1e-4`，由官方 global batch 2 的 `5e-5` 线性缩放 |
| weight_decay | `0.005` |
| full_schedule | 12E |
| current_stage | epoch 1–3 |
| steps_per_epoch | 17,082 |
| checkpoint_interval | 1E |
| timeout | `335m`（本次重启后的剩余窗口） |
| seed | 3407 |

smoke 两步的 `loss=7.9213 / 3.9704`、`grad_norm=214.8736 / 132.2174`，
均为有限值。正式阶段在 step 160 时 `loss=2.4409`、`grad_norm=28.9547`，
稳定单步约 0.309 秒，初始 ETA 约 4 小时 42 分。

## 冻结训练合同

| field | value |
|---|---|
| config | `configs/orbdet/orbdet_v0_2_r50_dota1_ms_rr_1x_gpu89.py` |
| model | `OrbdetV02Detector` + `H2RBoxV2Head` |
| data | `/data/zcy/dataset/trainval_ms_full/` |
| raw_patches | 138,883 |
| effective_nonempty | 68,325 |
| physical_gpus | 8,9 |
| batch_per_rank | 1 |
| global_batch | 2 |
| optimizer | AdamW |
| lr | `5e-5` |
| weight_decay | `0.005` |
| RR | `prob=1`, `angle_range=180` |
| classes | 15 |
| rotation_agnostic_classes | `[1, 9, 11]` |
| agnostic_resize_classes | `[1]` |
| max_epochs | 12 |
| milestones | `[8, 11]` |
| seed | 3407 |
| validation | disabled on full trainval |
| expected_optimizer_steps | about 409,956 |
| estimated_wall_time | 50–65 hours, hardware/load dependent |

## 启动边界

正式 launcher 必须同时看到官方审计 `COMPLETE`、两 rank smoke `COMPLETE` 与
`epoch_1.pth`，并确认 GPU 8/9 没有外部 compute PID。正式运行不 resume、不覆盖
旧 checkpoint、不排队、不自动转为 36E。

可复现入口：

- smoke：`scripts/smoke/run_orbdet_v0_2_r50_dota1_ms_rr_gpu89_smoke.sh`
- formal：`scripts/formal/run_orbdet_v0_2_r50_dota1_ms_rr_1x_gpu89_seed3407.sh`
- formal work dir：
  `work_dirs/formal/orbdet_v0_2_r50_dota1_ms_rr_1x_gpu89_seed3407_20260815/`
- GPU4567 six-hour controller：
  `scripts/formal/run_orbdet_v0_2_r50_dota1_ms_rr_gpu4567_six_hour_window.sh`
- GPU4567 stage1 config：
  `configs/orbdet/orbdet_v0_2_r50_dota1_ms_rr_1x_gpu4567_stage1_3e.py`
- GPU4567 stage1 work dir：
  `work_dirs/formal/orbdet_v0_2_r50_dota1_ms_rr_gpu4567_seed3407_stage1_3e_20260816/`
