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
| expected_finish | about 09:10 CST |
| latest_observed | epoch 1, step 2,600 at 04:53 |
| latest_loss | `1.6625` |
| latest_grad_norm | `9.9207` |

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

## 六小时窗口的阶段评测安排

3E 完成后不进入 epoch 4。若距离 09:50 仍有至少 1,800 秒，则四卡顺序执行：

1. raw trainval 20,995 patches 的诊断性 mAP（不是 held-out 指标）；
2. SS test 10,833 patches，生成 15 类 Task1 ZIP；
3. MS test 71,888 patches，生成 15 类 Task1 ZIP。

评测配置继承本次 MS+RR stage 模型，而不是旧 SS 训练模型；所有数据规模已由
`6 passed` 的新契约测试实际构建验证。watcher 的绝对截止为 09:50，达到截止会
向评测发送 INT，且不会终止其他用户进程。

watcher 已于 04:55:21 在
`orbdet_msrr4567_stage3_posteval_20260816` 中挂起等待；启动后确认仅有
`RUNNING` marker，没有提前创建推理进程或占用额外显存。联合回归为
`13 passed, 4 warnings`。

05:00 新增 checkpoint 强门禁：推理前必须反序列化并确认 epoch 3、iter
51,246、371 tensors、Orbdet/MS-trainval 配置 token，随后记录 SHA256。门禁 CLI
先以缺失实现得到预期 RED，再转为 `8 passed`；并用真实 384,530,537-byte smoke
checkpoint 验证通过（epoch 1、iter 2、371 tensors，SHA256
`874ec8f5bcd8d79b38297394f44a3934c9d1c79b788dd547777986c2857eb486`）。

## 解释边界

本轮 3E 只是可续训阶段，不是完整 12E 结果，不能用于最终在线 AP 结论。若控制器
提前触发硬截止，只报告最近完整 epoch checkpoint；不自动 resume，不启动 R50
36E，也不自动转入 Swin-B。
