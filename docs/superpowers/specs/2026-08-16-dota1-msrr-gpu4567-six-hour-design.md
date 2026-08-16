# DOTA-v1 MS+RR GPU 4–7 六小时阶段设计

## 目标

在 2026-08-16 约六小时的 GPU 4–7 独占窗口内，完成官方 H2RBox-v2
checkpoint 链路审计、四 rank 两步训练 smoke，并尽可能完成一个可续训的
Orbdet-v0.2 R50 MS+RR 阶段 checkpoint。

## 方案

使用四张 A40，每卡 batch 1，全局 batch 4。官方 checkpoint 的全局 batch 2
对应学习率为 `5e-5`，四卡阶段按线性缩放使用 `1e-4`；AdamW
`weight_decay=0.005`、12E 总计划、milestones 8/11、RR 与类别对称设置保持不变。

本窗口只运行 stage 1（epoch 1–3），每 epoch 保存 checkpoint。后续窗口使用完整
12E 配置从 `epoch_3.pth` 人工续训；本次不创建自动恢复任务，也不把 3E 阶段记为
12E 完成。

## 执行链

1. 四 rank 官方 SS trainval、SS test submission、MS+RR test submission 审计。
2. 8 个样本、四 rank、恰好两个 optimizer step 的 smoke。
3. 四 rank R50 MS+RR stage 1，训练到 epoch 3。
4. 当前控制器以 `335m` 启动，最晚约 09:55 中断并保留最近一个完整 epoch
   checkpoint；这比六小时上限额外保留约 25 分钟安全余量。
5. 若 epoch 3 正常完成且距离 09:50 至少还有 1,800 秒，则依次运行 raw
   trainval 诊断评测、single-scale test submission 与 multi-scale test
   submission。评测链有独立的 09:50 绝对截止，不够时间就跳过，绝不启动
   epoch 4，也不把 trainval 自评解释成 held-out 泛化结果。

所有分布式命令固定 `CUDA_VISIBLE_DEVICES=4,5,6,7`、
`NCCL_P2P_DISABLE=1`、`NCCL_IB_DISABLE=1`，且启动前拒绝已有 GPU compute
PID、重复 Orbdet 作业或覆盖既有 checkpoint。

## 验收

- 官方权重哈希与 strict-load 合同不变，审计产生 SS/MS 两个合法 15 文件 ZIP。
- smoke 产生 `epoch_1.pth`，日志无 traceback、NCCL error 或非有限 loss/grad。
- stage 1 正常完成时产生 `epoch_1.pth`、`epoch_2.pth`、`epoch_3.pth`；若受硬截止
  中断，则只报告最近完整 epoch，绝不报告 3E/12E 完成。
- 阶段评测仅在 3E `COMPLETE` 后启动，使用同一模型合同；数据全集大小固定为
  trainval 20,995、SS test 10,833、MS test 71,888。两份 ZIP 都必须含根目录
  15 个 `Task1_*.txt` 文件并通过 CRC 检查。
- 推理前必须完整反序列化 `epoch_3.pth`，并验证 metadata 为 epoch 3、iter
  51,246、371 个 state tensors，且嵌入配置含 `OrbdetV02Detector` 与
  `trainval_ms_full`；验证证据包含 checkpoint SHA256。
- 不改写 DOTA 源数据，不终止或共享其他用户 GPU 进程，不推送远端。

## 实际完成

- 3E stage 于 09:05:00 完成，最终 checkpoint 合同为 epoch 3、iter 51,246、
  371 tensors，SHA256
  `7847a8991984a87ae1545a1a04f26490bcfe16213603615c24a19a299740996b`。
- trainval 诊断为 `mAP=0.6851`、`AP50=0.6850`；该值明确标记为训练集自评。
- SS 与 MS+RR submission 均为根目录 15 个 Task1 文件且 CRC 正常，后评测于
  09:26:19 完成。
- 所有工作在硬截止前结束，未启动 epoch 4，GPU 4–7 已释放，完整 12E 仍为
  `not_complete`。
