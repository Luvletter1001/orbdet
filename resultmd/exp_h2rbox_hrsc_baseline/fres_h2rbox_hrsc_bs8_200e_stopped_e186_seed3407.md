# H2RBox HRSC 大批量诊断实验复盘（停止于 186E）

## 结论

本实验在工程上稳定运行，但不能作为论文 baseline，也不能用于证明 Orbdet-v0.1
优于 H2RBox。用户于 2026-08-13 18:10 主动要求停止；训练在 epoch 186
完成后被 SIGINT 正常终止，GPU 0–3 已释放，日志和 checkpoint 均保留。

最佳验证结果为 epoch 60 的 `dota/mAP=0.0798`、`AP50=0.0800`；最后一次
完整验证为 epoch 180 的 `dota/mAP=0.0728`、`AP50=0.0730`。130–180E
长期停留在 0.068–0.074 AP50，说明增加 epoch 没有修复当前优化协议。

## 实验合同

| item | value |
|---|---|
| model | pure `H2RBoxDetector / R50-FPN / H2RBoxHead` |
| supervision | HRSC qbox -> enclosing hbox -> zero-angle rbox |
| train / val / held-out test | 436 / 181 / 453 images |
| input | 800 x 800 |
| GPUs | physical 0,1,2,3 |
| batch | 8/GPU, global 32 |
| optimizer | AdamW, lr 2e-4, weight decay 0.05 |
| warmup | 100 iterations |
| schedule | 200E, milestones 133/184, gamma 0.1 |
| actual stop | completed epoch 186; 2,604 optimizer steps |
| seed | 3407 |
| wall clock | about 33m28s |

## 验证曲线

| epoch | AP50 | epoch | AP50 |
|---:|---:|---:|---:|
| 10 | 0.000 | 100 | 0.032 |
| 20 | 0.008 | 110 | 0.074 |
| 30 | 0.010 | 120 | 0.071 |
| 40 | 0.044 | 130 | 0.074 |
| 50 | 0.039 | 140 | 0.070 |
| 60 | **0.080** | 150 | 0.068 |
| 70 | 0.045 | 160 | 0.071 |
| 80 | 0.063 | 170 | 0.073 |
| 90 | 0.077 | 180 | 0.073 |

epoch 60 的 recall 为 0.224；epoch 180 的 recall 为 0.229。后半程分类损失
下降到约 5e-4，但 recall 和 AP 不增长，表现为“分类置信度收缩/饱和，而旋转几何
没有学好”。133E 与 184E 两次学习率衰减后均未出现 AP 恢复。

## 为什么不能和 Orbdet-v0.1 的 88.2 AP50 直接比较

1. Orbdet-v0.1 使用 617 张 trainval 训练，并在 453 张 test 上每 30E 验证和选择
   best checkpoint；本实验使用 436 张 train 训练、181 张 val 选模，并保留 test。
   数据量、验证域和模型选择协议不同。
2. Orbdet-v0.1 为 global batch 2、510E、约 157,590 次 optimizer update；本实验
   到停止时为 global batch 32、186E、仅 2,604 次 update，相差约 60.5 倍。
3. Orbdet-v0.1 使用 lr 1e-4、warmup 500；本实验使用 lr 2e-4、warmup 100。
4. 因此当前差距同时混入方法、数据、batch、更新次数、学习率、warmup 和 test
   选模偏差，不能归因于 harmonic consistency。

## 可复用经验

- 显存占满不是实验目标。弱监督旋转一致性依赖足够的参数更新和随机视图，大 batch
  会显著减少每个 epoch 的 update 数；不能只通过线性学习率缩放补偿。
- 记录训练预算时必须同时报告 epoch、optimizer steps、样本曝光量和 wall clock。
- baseline 与新方法必须共享同一 split、dataloader、batch、schedule、seed 和评测口径；
  唯一变量只能是待验证模块。
- test 不能参与 checkpoint 选择。当前 Orbdet-v0.1 的 88.2 只能视为工程结果，论文
  中需要用 train/val 选模后对 held-out test 只评一次。
- fixed-query 的可迁移经验是“先保证候选/视图对应可靠，再使用质量信号”；质量分数应
  用于训练期软加权与诊断，不应在推理时再次乘分类分数造成二次抑制。
- 当前结果是有价值的负例：`bs8/GPU + lr2e-4 + warmup100 + 200E` 不适合作为
  HRSC H2RBox baseline。

## 后续建议（本次不执行）

先冻结 Orbdet 创新，建立唯一可信对照：800x800、batch 2、AdamW lr 5e-5、
warmup 500、72E、milestones 48/66；H2RBox 与 Orbdet 只替换 consistency loss，
共同使用 train/val 选模和 held-out test 一次评测。至少运行 3 seeds，并报告
mean/std、AP50、recall、角度误差和质量分桶结果。

## 产物

- config: `configs/orbdet/h2rbox_r50_hrsc_200e_4gpu.py`
- launch log: `work_dirs/h2rbox_r50_hrsc_200e_bs8_gpu0123.launch.log`
- work dir: `work_dirs/formal/h2rbox_r50_hrsc_trainval_200e_gpu0123_bs8_seed3407/`
- best checkpoint: `best_dota_mAP_epoch_60.pth`
- retained periodic checkpoints: `epoch_10.pth` through `epoch_180.pth`

