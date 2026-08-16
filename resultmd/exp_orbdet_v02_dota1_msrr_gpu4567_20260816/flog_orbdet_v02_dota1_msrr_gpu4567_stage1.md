# Orbdet-v0.2 DOTA-v1 MS+RR GPU4567 Stage 1 运行记录

## 当前状态

`run_status=complete_authorized_stage`。控制器于 2026-08-16 04:20:05 启动，
官方审计与 smoke 均通过；3E stage 1 于 04:39:47 开始、09:05:00 完成，
有界后评测于 09:26:19 完成。完整 12E 尚未完成。

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
| stage_completed_at | `2026-08-16 09:05:00 +08:00` |
| posteval_completed_at | `2026-08-16 09:26:19 +08:00` |
| final_checkpoint_iter | 51,246 |
| final_logged_step | 51,244 |
| final_logged_loss | `1.0825` |
| final_logged_grad_norm | `3.6446` |

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
- post-eval root：
  `work_dirs/eval/orbdet_v0_2_dota1_ms_rr_gpu4567_stage1_epoch3_20260816/`
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

## Epoch 1 里程碑

`epoch_1.pth` 于 06:08:17 完整生成，大小 389,144,617 bytes。真实反序列化
验证为 epoch 1、iter 17,082、371 state tensors，嵌入配置含
`OrbdetV02Detector` 与 `trainval_ms_full`，SHA256 为
`025ef114f7aaf0d9c65ef97547401bd5424c28b9cd7a6cba9344de9ee2115f94`。

epoch 1 共记录 854 个 logger points，所有 loss/grad/time 均有限。首个与最后
2,000-step 窗口对比：median loss `2.0877 -> 1.2741`，median grad norm
`11.0235 -> 5.2044`，mean symmetry loss `0.2596 -> 0.0712`。末窗口 mean
`q_joint=0.7775`、`q_high_frac=0.8048`、`hbox_fidelity=0.7780`；这些是训练
诊断量，不作为泛化指标。训练随后自动进入 epoch 2。

## Epoch 2 里程碑

`epoch_2.pth` 于 07:36:24 完整生成，大小 393,763,241 bytes。真实反序列化
验证为 epoch 2、iter 34,164、371 state tensors，嵌入配置含
`OrbdetV02Detector` 与 `trainval_ms_full`，SHA256 为
`e3a07e758da448c03107070d61407887b5deed8ec8f9e4cdc135351deb13429e`。

epoch 2 同样记录 854 个 logger points，所有 loss/grad/time 均有限。首个与最后
2,000-step 窗口对比：median loss `1.2704 -> 1.2020`，median grad norm
`4.8829 -> 4.0881`，mean symmetry loss `0.0628 -> 0.0480`。末窗口 mean
`q_joint=0.8280`、`q_high_frac=0.8356`、`hbox_fidelity=0.7693`。训练随后
自动进入最终 epoch 3。

## Epoch 3 与训练终态

`epoch_3.pth` 于 09:05:00 完整生成，大小 398,382,889 bytes。真实反序列化
验证为 epoch 3、iter 51,246、371 state tensors，嵌入配置含
`OrbdetV02Detector` 与 `trainval_ms_full`，SHA256 为
`7847a8991984a87ae1545a1a04f26490bcfe16213603615c24a19a299740996b`。

三轮各记录 854 个 logger points，共 2,562 点，所有核心标量均有限。从首
2,000 optimizer steps 到最后 100 个记录点，median loss 为
`2.3652 -> 1.2010`，median grad norm 为 `12.2727 -> 3.3387`，median
symmetry loss 为 `0.3355 -> 0.0320`，median `q_joint` 为
`0.3542 -> 0.9683`。epoch 1/2/3 的 median loss 分别为 `1.4546`、
`1.2355`、`1.1730`，说明 3E 末仍在收敛，但不能据此外推在线 AP。

## Epoch 3 阶段评测

09:05 在 checkpoint 强门禁通过后依次完成三项评测，09:26:19 写出总
`COMPLETE`：

- trainval 20,995 patches：`mAP=0.6851`、`AP50=0.6850`。这是训练集诊断，
  不是 held-out 结果；同链路官方 checkpoint 为 `0.8131`，当前 3E 阶段低
  `0.1280`，因此不能声称达到 80+。
- 每类 AP：plane `0.895`、baseball-diamond `0.593`、bridge `0.435`、
  ground-track-field `0.571`、small-vehicle `0.740`、large-vehicle `0.766`、
  ship `0.798`、tennis-court `0.899`、basketball-court `0.723`、
  storage-tank `0.671`、soccer-ball-field `0.502`、roundabout `0.659`、
  harbor `0.619`、swimming-pool `0.740`、helicopter `0.663`。
- SS ZIP：17,377,563 bytes，根目录 15 个唯一 `Task1_*.txt`，CRC 正常，
  SHA256 `d513f4bf9a69e126edb6c19946a11df860d8c59037c802ca10e2bd0b314ab4b8`。
- MS+RR ZIP：35,980,142 bytes，根目录 15 个唯一 `Task1_*.txt`，CRC 正常，
  SHA256 `2768857944a36f27d9b5816de53c0d30141533bdc4e46e0ca7ea42abce3a85af`。

SS 与 MS+RR ZIP 分别位于：

- `work_dirs/eval/orbdet_v0_2_dota1_ms_rr_gpu4567_stage1_epoch3_20260816/ss_submission/orbdet_v0_2_msrr_stage1_epoch3_ss_task1/orbdet_v0_2_msrr_stage1_epoch3_ss_task1.zip`
- `work_dirs/eval/orbdet_v0_2_dota1_ms_rr_gpu4567_stage1_epoch3_20260816/ms_submission/orbdet_v0_2_msrr_stage1_epoch3_msrr_task1/orbdet_v0_2_msrr_stage1_epoch3_msrr_task1.zip`

最终审计未发现 Traceback、RuntimeError、NCCL error、OOM、NaN 或 Inf；无
Orbdet 训练/推理残留进程，GPU 4–7 已释放。未启动 epoch 4，也未推送远端。

## 解释边界

本轮 3E 只是可续训阶段，不是完整 12E 结果，不能用于最终在线 AP 结论。下一次
获得明确训练授权后，应从已验证的 `epoch_3.pth` 续训到 12E，而不是重跑前 3E；
本轮不自动 resume，不启动 R50 36E，也不自动转入 Swin-B。
