# 执行计划

## P0：读取现场，不重复启动

1. 检查当前时间、GPU 8/9、H2RBox tmux 和精确训练进程。
2. 读取 baseline 最新 epoch、ETA、val ledger 和错误扫描。
3. 如果现有 H2RBox 正常运行，只等待；禁止再启动第二份 baseline。
4. 所有 DDP 命令必须设置：

```text
CUDA_VISIBLE_DEVICES=8,9
NCCL_P2P_DISABLE=1
NCCL_IB_DISABLE=1
```

## P1：完成 H2RBox baseline 闭环

1. 等待现有 H2RBox 完成 200E。
2. 核验 `epoch_200.pth`、最后一次 val、best-val checkpoint 和无致命错误。
3. 使用 best-val checkpoint 对 `ImageSets/test.txt` 只评一次。
4. test 使用独立目录：
   `work_dirs/test/h2rbox_hrsc_clean_bestval_gpu89_20260814/`。
5. 保存 predictions 和 test log；不得根据 test 结果回选 epoch。

## P2：先写合同测试，再创建 clean Orbdet

先创建 `tests/test_hrsc_clean_comparison_contract.py`，并确认它因候选文件缺失而失败。
测试至少约束：

- candidate 与 baseline 的 train/val/test dataloader 完全相等；
- optimizer、scheduler、hooks、seed、200E 和 val interval 完全相等；
- 模型除 detector type、consistency loss 和 work_dir 外完全相等；
- candidate loss 参数等于 GOAL.md 中的固定值；
- launcher 固定 GPU 8/9、双 rank、NCCL 安全变量且无 `--resume`；
- deadline guard 使用绝对 epoch `1786667400`，只匹配精确 session/config。

然后创建：

- `configs/orbdet/orbdet_v0_1_r50_hrsc_clean_200e_gpu89_bs2.py`
- `configs/orbdet/orbdet_v0_1_r50_hrsc_clean_200e_gpu89_bs2_smoke.py`
- `scripts/smoke/run_orbdet_v0_1_hrsc_clean_gpu89_bs2_smoke.sh`
- `scripts/formal/run_orbdet_v0_1_hrsc_clean_200e_gpu89_bs2.sh`
- `scripts/formal/guard_orbdet_gpu89_until_0830_20260814.sh`

完成后运行新合同测试和现有 `tests/test_orbdet_v0_1.py`，必须全部通过。

## P3：deadline guard

在独立 tmux 中启动 guard：

```text
orbdet_gpu89_deadline_guard_0830_20260814
```

guard 每 30 秒检查时间。08:30 时只向以下本实验 session 发送 Ctrl-C：

- `h2rbox_hrsc200e_bs2_gpu89_20260814`
- `orbdet_hrsc_clean200e_bs2_gpu89_20260814`
- `orbdet_hrsc_clean_queue_gpu89_20260814`

宽限后仍存活时，只允许 signal 命令行中包含以下精确 config 的进程：

- `h2rbox_r50_hrsc_200e_2gpu_officiallike.py`
- `orbdet_v0_1_r50_hrsc_clean_200e_gpu89_bs2.py`

禁止使用泛化的 `pkill tools/train.py`、`killall python` 或按 GPU 全杀。

## P4：candidate 冒泡与正式训练

1. GPU 8/9 空闲后运行 8 图、2 step 双卡冒泡。
2. 要求 2/2 step、有限 loss/grad、保存 `epoch_1.pth`、无 OOM/NCCL。
3. 正式训练使用 tmux：
   `orbdet_hrsc_clean200e_bs2_gpu89_20260814`。
4. 正式训练开始前计算：

```text
remaining = 1786667400 - current_unix_time - 300
```

仅当 `remaining > 0` 才启动，并用 GNU `timeout --signal=INT --kill-after=60`
包裹正式 launcher，保留最后五分钟用于退出或 test。
5. 首个稳定窗口需验证两个主 rank、GPU 8/9 显存/利用率、有限 loss 与 quality
统计、无 fatal error。

## P5：candidate test 与归档

1. 若 200E 在截止前完成，使用最大 val mAP checkpoint 对 held-out test 只评一次。
2. 若训练被 08:30 截止，跳过 test，记录最后完整 epoch/checkpoint。
3. 结果写入：
   `resultmd/exp_orbdet_hrsc_clean/fres_orbdet_hrsc_clean_gpu89_20260814.md`。
4. 必须记录：协议、完整 val 曲线、best checkpoint、test、recall、GPU memory、
wall time、错误扫描和 deadline 状态。
5. 公平结论只比较同协议 H2RBox 与 clean Orbdet；早前 trainval→test 选模的 88.2
仅作为工程先验，不进入主对比表。

## 研究决策门

- clean Orbdet 若稳定显著超过 H2RBox：下一步补同协议 H2RBox-v2，然后冻结模型上
  DOTA-v1.0。
- clean Orbdet 若仅小幅提高且远落后 H2RBox-v2：继续改模型，不换数据集掩盖问题。
- clean Orbdet 若无提升：重新审视 harmonic reliability 假设，不追加大规模训练。

