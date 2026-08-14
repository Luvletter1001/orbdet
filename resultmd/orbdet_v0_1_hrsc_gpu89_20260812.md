# Orbdet-v0.1 HRSC2016 GPU 8/9 正式实验记录

## 当前结论

Orbdet-v0.1 已完成实现、TDD、HRSC 数据合同、双卡训练冒泡和 510E 正式训练。
最佳点为 epoch 360：`dota/mAP=0.8820`、`AP50=0.8820`；epoch 510 为
`mAP=0.8807`、`AP50=0.8810`。最佳 checkpoint 的复测同样得到 0.8820。

该数值目前只定义为工程结果，不能直接作为论文 test 结果：本任务以
`trainval.txt` 训练，并在训练期间反复以 `test.txt` 验证和选择 best checkpoint，
因此 test 已参与模型选择。论文复现必须改为 train/val 选模，并对 held-out test
只评一次。

## 实验合同

| item | value |
|---|---|
| exp_id | `orbdet_v0_1_hrsc_gpu89_20260812` |
| model | `OrbdetDetector / R50-FPN / H2RBoxHead` |
| supervision | `qbox -> hbox -> rbox` during train |
| train_split | `HRSC2016 ImageSets/trainval.txt`, 617 images |
| eval_split | `HRSC2016 ImageSets/test.txt`, 453 images |
| input | `800 x 800` |
| physical_gpus | `8,9` |
| global_batch | `2` |
| optimizer | `AdamW, lr=1e-4, weight_decay=0.05` |
| max_epochs | `510` |
| val_interval | `30` |
| lr_milestones | `340,468` |
| target_wall_clock | `11–12 h`, hard cap `12.5 h` |
| stable_rule | last 3 AP points range `<=0.015`, recent gains `<0.005` |
| reasonable_AP50 | `0.80–0.92` |

## 创新点

`OrbdetHarmonicConsistencyLoss` 从原图与旋转视图的方向差计算二阶圆周谐波可靠度，
同时处理 `pi` 周期和宽高交换的 `pi/2` 等价轴。可靠度在加权前 detach，并保留
`q_min=0.25`，避免低置信样本在训练初期完全失去梯度。

推理仍使用成熟 CNN 密集候选与 rotated NMS。谐波质量不乘入分类分数，避免复现
fixed-query 线中候选绑定弱和 `class * quality` 二次抑制造成的排序塌缩。

## Preflight evidence

| check | evidence | status |
|---|---|---|
| unit/contracts | `18 passed`, 2 upstream warnings | pass |
| dataset build | train `617`, test `453` | pass |
| HBox privacy | sample `max_abs_sin_angle=0.0` | pass |
| model build | `OrbdetDetector`, 32,115,533 params | pass |
| ImageNet weight | SHA256 `0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a` | pass |
| DDP smoke | 8 images, 4/4 steps, checkpoint saved | pass |
| smoke memory | `2219 MiB/GPU` in MMEngine log | pass |

## Calibration

完整 617 图、每卡 batch 1 的 epoch-1 标定结果：

| metric | value |
|---|---:|
| iterations | 309 |
| steady_iter_time | `0.230–0.245 s` |
| train_epoch_wall_clock | about `78 s` |
| full_eval_steps | 114 |
| full_eval_wall_clock | about `15 s` |
| epoch1_AP50 | `0.0000` diagnostic only |
| final_window_loss | `1.7454` |
| final_window_loss_bbox_ss | `0.1196` |
| final_window_grad_norm | `15.8851` |

epoch-1 AP 为零不作为方法结论；它只表示单轮尚未形成有效检测器。

## 预测链故障与修复

第一次完整评测触发 `InstanceData` 长度断言。根因是 `H2RBoxHead` 仅在
`square_classes` 非空时刷新 NMS 后的 `bboxes` 局部变量；HRSC 船舶不属于 square
class，因此将 NMS 前 1000+ 框错误写回 NMS 后结果。

新增真实 head 回归测试后稳定复现 `72 boxes vs 10 scores`，随后做最小修复：NMS 后
无条件从 `results.bboxes` 读取当前框，再可选处理 square classes。修复后定向测试和
全套 18 项测试通过，epoch-1 全量评测正常完成。

## Checksums

| artifact | sha256 |
|---|---|
| formal config | `a959caa965d619e1f779414ca8084ae351c9ead9601d58511d027e2a6582b80a` |
| harmonic loss | `a93bcef930e7556118d17e90458daa121bc91cee1d7ee2cf4d51f199d45ed898` |
| Orbdet detector | `c567b4e8614f129fd00718345eb41bcd802538b815eb9bc362c3f54b499e0630` |
| H2RBox prediction fix | `e85be9ea59116981c506291b9b56557ef85276cfd7f10d50fa993d77131845af` |

## Live validation ledger

| epoch | AP50 | loss | loss_bbox_ss | q_mean | checkpoint | decision |
|---:|---:|---:|---:|---:|---|---|
| 1 | 0.0000 | 1.7454 | 0.1196 | 0.6438 | calibration `epoch_1.pth` | diagnostic only |
| 30 | 0.6250 | 1.1204 | 0.1239 | 0.9584 | `epoch_30.pth` | continue; below target band |
| 60 | 0.7770 | 0.8617 | 0.0951 | 0.9912 | `epoch_60.pth`, current best | continue; near target floor |

正式会话：foreground session `59514`，launcher PID `3270900`，初始 rank PID
`3270967,3270968`。首个日志窗口（epoch 1, iter 20/309）为：loss `3.3836`、
loss_bbox_ss `0.2743`、q_mean `0.4802`、q_min `0.2501`，均为有限值。

epoch 30 的 453 图完整评估得到 `dota/mAP=0.6248`、`AP50=0.6250`、recall
`0.726`（1228 GT，3152 detections）。该点已明显脱离失败区，但尚未进入预设
`0.80–0.92` 合理区间，因此不早停，继续观察 epoch 60/90。

epoch 60 得到 `dota/mAP=0.7775`、`AP50=0.7770`、recall `0.850`（1228 GT，
2067 detections），相对 epoch 30 提升约 `+0.153` AP。增长仍快，且只比目标下界
低约 2.3 AP，因此继续到 epoch 90；尚不能宣称稳定。

## GPU coexistence

正式任务与 `lzy` 的六个 THUMOS 进程共享 GPU 8/9；开跑前这些进程合计各占约
9.9 GiB，Orbdet 标定新增约 2.2 GiB/GPU。未终止、重启或修改任何其他用户进程。

## Final status

训练已正常完成 510E，GPU 8/9 不再由本任务占用。保存的最佳模型为
`work_dirs/formal/orbdet_v0_1_hrsc_gpu89/best_dota_mAP_epoch_360.pth`。后续所有
方法比较必须使用统一的干净 train/val/test 协议，不能把本实验的 88.2 与不同
split、batch 和更新预算下的 H2RBox 结果直接相减。
