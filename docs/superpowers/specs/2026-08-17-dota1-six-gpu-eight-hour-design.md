# DOTA-v1 GPU 4–9 八小时并行阶段设计

## 目标与授权

用户于 2026-08-17 明确批准在物理 GPU 4–9 空闲后使用约八小时，执行两项彼此
隔离的 Orbdet-v0.2 DOTA-v1 正式任务：

1. GPU 4–7 将现有 MS+RR checkpoint 从 epoch 3 续训至 epoch 8；
2. GPU 8–9 从头训练 SS seed 42 至完整 epoch 12。

所有训练、验证和收尾任务最晚于 2026-08-18 08:20 +08:00 退出。该授权不包括
抢占、终止或修改外部 GPU 进程，也不包括 epoch 9 之后的 MS+RR 自动续训。

## 为什么采用 4+2

现有 MS+RR epoch-3 checkpoint 的优化合同是四 rank、每 rank batch 1、全局
batch 4、AdamW learning rate `1e-4`。直接改用六 rank 会把全局 batch 改成 6，
使已有 optimizer/scheduler 轨迹失去可比性；按六卡吞吐也无法在八小时内完成剩余
9 个 epoch。因此主任务继续使用四卡，空出的双卡用于一个独立、可完整结束的 SS
重复种子。

## 任务 A：GPU 4–7 MS+RR epoch 3→8

### 冻结合同

- source checkpoint：
  `work_dirs/formal/orbdet_v0_2_r50_dota1_ms_rr_gpu4567_seed3407_stage1_3e_20260816/epoch_3.pth`
- checkpoint SHA256：
  `7847a8991984a87ae1545a1a04f26490bcfe16213603615c24a19a299740996b`
- model：`OrbdetV02Detector` + `H2RBoxV2Head`
- data：`trainval_ms_full`，1024×1024，`RandomRotate(prob=1, angle_range=180)`
- seed：3407
- ranks：4，physical GPUs 4,5,6,7
- batch：1/rank，global batch 4
- optimizer：AdamW，learning rate `1e-4`，weight decay `0.005`
- resume state：epoch 3，iter 51,246
- target：epoch 8，预计新增 85,410 optimizer steps
- validation：训练阶段关闭；每个 epoch 保存 checkpoint

四卡历史实测每 epoch 约 1 小时 28 分 24 秒，新增五个 epoch 预计约 7 小时
22 分。完成后必须验证 epoch、iter、state tensor 数、配置 token、文件大小和
SHA256。若 07:50 前完成并仍有足够窗口，则运行与 epoch-3 相同口径的 trainval
诊断、SS submission 和 MS+RR submission；否则只做 checkpoint 强验证并跳过
耗时后评测。完整 MS+RR 12E 在本阶段结束后仍为未完成状态。

## 任务 B：GPU 8–9 SS seed 42 完整 12E

### 冻结合同

- base config：`configs/orbdet/orbdet_v0_2_r50_dota1_1x_gpu89.py`
- model/data/pipeline：与已完成 seed 3407 SS 12E 完全一致
- seed：42
- ranks：2，physical GPUs 8,9
- batch：2/rank，global batch 4
- optimizer：AdamW，learning rate `1e-4`
- resume：false；从 ImageNet 初始化重新训练
- target：完整 epoch 12；milestones 8/11
- validation：训练阶段关闭；checkpoint 使用独立目录且不覆盖 seed 3407

双卡 seed 3407 的历史实测为 5 小时 31 分，训练结束后的 trainval 诊断与 SS
submission 约 14 分钟。seed 42 的 trainval 数值只能作为训练集诊断，不能当作
独立验证或在线测试结论；submission ZIP 只生成并校验，不自动上传。

## 调度与优先级

1. GPU 8/9 与 GPU 4–7 分别通过两次稳定采样确认无外部 compute PID。
2. 两个任务使用不同 master port、独立 work dir、独立控制器日志与状态标记。
3. 任务 A 是最高优先级。先确认其 resume 门禁与前 200 step 稳定，再启动任务 B。
4. 两项任务均设置 `NCCL_P2P_DISABLE=1`、`NCCL_IB_DISABLE=1`、
   `OMP_NUM_THREADS=1` 和 `MKL_NUM_THREADS=1`。
5. 若并发后任务 A 的稳定 step time 持续高于 `0.341 s`（四卡基线 `0.31 s`
   的 10% 退化），任务 B 在写出最近一个完整 epoch checkpoint 后停止；不得影响
   任务 A，也不得操作任何外部进程。
6. 08:20 为绝对截止。只向本阶段创建并核验过 PID 身份的进程发送终止信号；保留
   最近完整 checkpoint，不建立自动恢复或下一阶段队列。

## 启动门禁

- source checkpoint 可反序列化且强验证通过；
- 两项派生配置经合同测试证明只改变约定字段；
- smoke 各完成两个 optimizer steps，loss/grad 有限，无 NaN、OOM 或 NCCL 错误；
- 目标 work dir 不存在 checkpoint、`COMPLETE` 或冲突 marker；
- GPU 4–9 无外部 compute PID；任何外部占用均导致对应任务拒绝启动；
- 当前工作树的无关未提交内容保持不变。

## 失败与降级

- 任务 A 未到 epoch 8：保留最后完整 epoch，状态记为有界阶段未完成，不伪造
  `COMPLETE`。
- 任务 B 未到 epoch 12：保留最后完整 epoch，状态记为未完成，不将其纳入双 seed
  结论。
- 后评测时间不足：优先 checkpoint 强验证，跳过 submission，不延长 GPU 窗口。
- 任一任务出现非有限 loss、OOM、NCCL error 或 traceback：只停止本任务的匹配
  进程并写出失败收据，不终止另一任务或其他用户进程。

## 验收产物

- 任务 A：epoch 4–8（以实际完成上限为准）的 checkpoint、最终强验证、训练日志、
  状态收据；有时间时追加 epoch-8 trainval/SS/MS+RR 后评测。
- 任务 B：seed-42 epoch-12 checkpoint、强验证、trainval 诊断、SS Task1 ZIP、
  状态收据。
- 汇总记录必须写明真实开始/结束时间、GPU、world size、batch、step time、最终
  epoch/iter、SHA256、异常与降级动作。
- 不改写 DOTA 原始/链接数据，不覆盖既有 checkpoint，不推送远端。
