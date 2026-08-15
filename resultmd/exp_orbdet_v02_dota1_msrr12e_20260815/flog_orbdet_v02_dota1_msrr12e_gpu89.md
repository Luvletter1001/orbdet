# Orbdet-v0.2 DOTA-v1 R50 MS+RR 12E 执行日志

## 当前状态

`formal_status=not_started`。用户已于 2026-08-15 当前轮明确授权路线：先做官方
checkpoint 推理审计，然后运行 R50 MS+RR 12E；确认链路后，在线 80+ 目标转入
Swin-B，不运行 R50 36E。

当前等待条件：

- `official_audit=waiting_gpu_idle`
- `gpu_8_9=occupied_by_unidentified_external_compute`
- `smoke=not_started`
- `formal=not_started`

没有终止、修改或抢占任何外部 GPU 进程。DOTA 源图像与 annotations 未改写。

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
