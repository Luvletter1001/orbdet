# Orbdet-v0.2 DOTA-v1 MS+RR GPU4567 Stage 1 运行记录

## 当前状态

`run_status=running`。控制器于 2026-08-16 04:20:05 启动，官方审计与 smoke
均已通过，3E stage 1 于 04:39:47 开始。完整 12E 尚未完成。

| field | value |
|---|---|
| tmux_session | `orbdet_msrr4567_6h_20260816` |
| physical_gpus | 4,5,6,7 |
| nproc | 4 |
| NCCL_P2P_DISABLE | 1 |
| NCCL_IB_DISABLE | 1 |
| data_len | 68,325 |
| batch_per_rank | 1 |
| global_batch | 4 |
| lr | `1e-4` |
| weight_decay | `0.005` |
| stage_epochs | 3 |
| full_epochs | 12 |
| steps_per_epoch | 17,082 |
| seed | 3407 |
| hard_timeout | `335m` |
| expected_finish | about 09:23 CST |
| latest_observed | epoch 1, step 160 |
| latest_loss | `2.4409` |
| latest_grad_norm | `28.9547` |

## 前置门禁证据

- focused contract：`7 passed, 3 warnings`。
- official SS trainval：`mAP=0.8131`、`AP50=0.8130`。
- SS ZIP：15 files，`testzip=None`，SHA256
  `fd298f9028ac7805ddddf64d4515d6a3da56a9d418c6295893caa3517ced4fa7`。
- MS+RR ZIP：15 files，`testzip=None`，SHA256
  `75b76cb1d8a94192a41b6fbc3d5665018c573958f5b8a6c17abf0631047a6aab`。
- smoke：2 optimizer steps，有限 loss/grad，生成 366.7 MiB `epoch_1.pth`。

## 路径

- controller log：
  `work_dirs/controllers/orbdet_v0_2_dota1_ms_rr_gpu4567_six_hour_20260816/controller.log`
- stage work dir：
  `work_dirs/formal/orbdet_v0_2_r50_dota1_ms_rr_gpu4567_seed3407_stage1_3e_20260816/`
- smoke checkpoint：
  `work_dirs/smoke/orbdet_v0_2_r50_dota1_ms_rr_1x_gpu4567_20260816/epoch_1.pth`

## 解释边界

本轮 3E 只是可续训阶段，不是完整 12E 结果，不能用于最终在线 AP 结论。若控制器
提前触发硬截止，只报告最近完整 epoch checkpoint；不自动 resume，不启动 R50
36E，也不自动转入 Swin-B。
