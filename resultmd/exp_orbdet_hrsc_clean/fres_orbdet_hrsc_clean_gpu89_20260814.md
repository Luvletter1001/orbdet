# HRSC clean H2RBox vs Orbdet-v0.1 GPU 8/9 final result

## Status

`complete`：两种方法均完成 200E、validation-only checkpoint selection，且各自
只对 held-out test 评测一次。实验在 `2026-08-14 05:57:56 +08:00` 完成，早于
`08:30` 的 GPU 截止时间。

## Scientific contract

| item | fixed value |
|---|---|
| dataset | HRSC2016 |
| train / val / held-out test | 436 / 181 / 453 images |
| input | 800 x 800 |
| physical GPUs | 8,9 |
| batch | 2/GPU, global 4 |
| optimizer | AdamW, lr 5e-5, weight decay 0.05 |
| gradient clipping | max norm 35 |
| warmup | 500 optimizer iterations |
| schedule | 200E, milestones 133/184, gamma 0.1 |
| validation | every 10E on `ImageSets/val.txt` |
| checkpoint selection | maximum validation `dota/mAP` |
| held-out test | best-val checkpoint, exactly once per method |
| seed | 3407 |
| distributed safety | `CUDA_VISIBLE_DEVICES=8,9`, `NCCL_P2P_DISABLE=1`, `NCCL_IB_DISABLE=1` |

The only functional model variable was:

| baseline | candidate |
|---|---|
| `H2RBoxDetector` | `OrbdetDetector` |
| `H2RBoxConsistencyLoss` | `OrbdetHarmonicConsistencyLoss` |

Candidate parameters were fixed before training: `loss_weight=0.4`,
`min_quality=0.25`, `gamma=2.0`, and `high_quality_thr=0.75`. Center, shape,
and angle loss configurations matched the baseline. Candidate inference kept the
H2RBox head and rotated NMS path; harmonic quality was not multiplied into test
scores.

## Preflight evidence

- Contract test was first observed RED with five failures caused only by missing
  candidate artifacts.
- After the minimal implementation, the new contract tests and existing Orbdet
  suite reported `16 passed` in total; the two warnings were existing dependency
  deprecations.
- Candidate smoke used 8 images and global batch 4, producing exactly 2/2 DDP
  optimizer steps.
- Smoke step 1: loss 7.5211, loss_bbox_ss 1.1201, grad norm 436.3863,
  quality mean/min/high fraction 0.6532/0.2850/0.2778.
- Smoke step 2: loss 6.5376, loss_bbox_ss 0.7020, grad norm 297.3258,
  quality mean/min/high fraction 0.6806/0.2503/0.5238.
- All smoke values were finite; `epoch_1.pth` was saved and no OOM, traceback,
  non-finite value, or NCCL failure occurred.

## Validation ledger

All values below come from the formal launch logs. Bold marks the checkpoint
selected independently for each method before held-out test evaluation.

| epoch | H2RBox val mAP | Orbdet val mAP |
|---:|---:|---:|
| 10 | 0.0066 | 0.0075 |
| 20 | 0.0329 | 0.0652 |
| 30 | 0.0391 | 0.0759 |
| 40 | 0.0784 | **0.1020** |
| 50 | 0.0686 | 0.0688 |
| 60 | 0.0843 | 0.0920 |
| 70 | 0.0863 | 0.0599 |
| 80 | **0.1104** | 0.0796 |
| 90 | 0.0861 | 0.0395 |
| 100 | 0.0817 | 0.0797 |
| 110 | 0.0810 | 0.0661 |
| 120 | 0.0876 | 0.0792 |
| 130 | 0.0925 | 0.0898 |
| 140 | 0.0899 | 0.0842 |
| 150 | 0.0889 | 0.0849 |
| 160 | 0.0893 | 0.0858 |
| 170 | 0.0893 | 0.0869 |
| 180 | 0.0928 | 0.0857 |
| 190 | 0.0907 | 0.0840 |
| 200 | 0.0901 | 0.0851 |

Best-validation comparison:

| method | best epoch | val mAP | checkpoint |
|---|---:|---:|---|
| H2RBox | 80 | 0.1104 | `best_dota_mAP_epoch_80.pth` |
| clean Orbdet | 40 | 0.1020 | `best_dota_mAP_epoch_40.pth` |

Clean Orbdet is lower by `0.0084` raw mAP (0.84 AP points) at each method's
validation-selected optimum.

## One-shot held-out test

Both rows use `ImageSets/test.txt` after validation-only checkpoint selection.
No test result was used to reselect an epoch.

| method | selected epoch | test mAP | AP50 | recall | GT | detections |
|---|---:|---:|---:|---:|---:|---:|
| H2RBox | 80 | 0.0842 | 0.0840 | 0.298 | 1228 | 1835 |
| clean Orbdet | 40 | 0.0803 | 0.0800 | 0.273 | 1228 | 1365 |

Clean Orbdet is lower by `0.0039` raw test mAP (0.39 AP points) and `0.025`
recall (2.5 percentage points).

## Runtime and resource evidence

| item | H2RBox | clean Orbdet |
|---|---|---|
| formal start | 2026-08-14 01:58:09 +08:00 | 2026-08-14 04:08:51 +08:00 |
| final validation | 2026-08-14 03:44:55 +08:00 | 2026-08-14 05:56:18 +08:00 |
| formal wall time | about 1:46:46 | about 1:47:27 |
| MMEngine peak memory | 3790 MiB/GPU | 3790 MiB/GPU |
| observed process memory | about 5576 MiB/GPU | about 5576 MiB/GPU |
| final checkpoint | `epoch_200.pth` | `epoch_200.pth` |

Both formal logs were scanned for traceback, CUDA OOM, NCCL failure/timeout,
runtime error, segmentation fault, NaN, and Inf; no fatal match was found.
Dependency and deprecation warnings were present but did not affect completion.

## Deadline audit

- Absolute cutoff: `2026-08-14 08:30:00 +08:00` / epoch `1786667400`.
- A dedicated exact-scope guard was started before candidate smoke/formal work.
- Candidate formal training used an outer timeout computed from the absolute
  deadline with the final five minutes reserved.
- Both formal training and the one-shot test completed well before cutoff.
- At `2026-08-14 05:58:50 +08:00`, no exact comparison train/test process
  remained; GPU 8/9 each reported 17 MiB and 0% utilization.
- The now-unneeded experiment-owned guard was then closed by its exact tmux name.
- No unrelated process or tmux session was signaled or modified.

## Fair conclusion and research gate

Under the fixed clean protocol, `OrbdetHarmonicConsistencyLoss` did not improve
over original H2RBox: it was lower on both the validation-selected optimum and
the single held-out test. This run therefore follows the `no improvement`
decision branch: re-examine the harmonic reliability hypothesis and do not add
another large-scale training run merely to search for a favorable result.

The earlier trainval-to-test-selected Orbdet result around 88.2 is not protocol
matched and is excluded from this primary comparison.

## Artifacts

- Baseline config: `configs/orbdet/h2rbox_r50_hrsc_200e_2gpu_officiallike.py`
- Candidate config: `configs/orbdet/orbdet_v0_1_r50_hrsc_clean_200e_gpu89_bs2.py`
- Baseline formal directory:
  `work_dirs/formal/h2rbox_r50_hrsc_train_val_200e_gpu89_bs2_seed3407_20260814/`
- Candidate formal directory:
  `work_dirs/formal/orbdet_v0_1_hrsc_clean_200e_gpu89_bs2_seed3407_20260814/`
- Baseline formal log:
  `work_dirs/h2rbox_r50_hrsc_200e_gpu89_bs2_20260814.launch.log`
- Candidate formal log:
  `work_dirs/orbdet_v0_1_hrsc_clean_200e_gpu89_bs2_20260814.launch.log`
- Baseline test directory:
  `work_dirs/test/h2rbox_hrsc_clean_bestval_gpu89_20260814/`
- Candidate test directory:
  `work_dirs/test/orbdet_v0_1_hrsc_clean_bestval_gpu89_20260814/`
- Candidate smoke directory:
  `work_dirs/smoke/orbdet_v0_1_hrsc_clean_gpu89_bs2_20260814/`
- Deadline guard log:
  `work_dirs/orbdet_gpu89_deadline_guard_0830_20260814.log`
