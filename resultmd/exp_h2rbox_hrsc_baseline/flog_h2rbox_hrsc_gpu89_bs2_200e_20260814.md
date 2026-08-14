# H2RBox HRSC GPU 8/9 bs2 200E 过程记录

## Status

`complete`：正式 200E、best-val 选择与唯一一次 held-out test 均已完成。

## Experiment contract

| item | value |
|---|---|
| model | pure `H2RBoxDetector / R50-FPN / H2RBoxHead` |
| consistency | official `H2RBoxConsistencyLoss`, weight 0.4 |
| supervision | HRSC qbox -> enclosing hbox -> zero-angle rbox |
| train / val / held-out test | 436 / 181 / 453 images |
| input | 800 x 800 |
| physical GPUs | 8,9 |
| batch | 2/GPU, global 4 |
| optimizer | AdamW, lr 5e-5, weight decay 0.05, clip max norm 35 |
| warmup | 500 optimizer iterations |
| schedule | 200E, milestones 133/184, gamma 0.1 |
| validation | every 10E on `val.txt`; best by `dota/mAP` |
| seed | 3407 |

## Smoke evidence

Two-rank smoke completed at `2026-08-14 01:56 +08:00` on GPUs 8/9:

- 8 images, global batch 4, 2/2 optimizer steps completed.
- step 1: loss 8.3885, loss_bbox_ss 1.9875, grad_norm 564.1758.
- step 2: loss 7.1007, loss_bbox_ss 1.2453, grad_norm 367.2701.
- all logged values were finite; configured gradient clipping is max norm 35.
- MMEngine peak memory: 3790 MiB/GPU.
- checkpoint: `work_dirs/smoke/h2rbox_r50_hrsc_200e_gpu89_bs2_20260814/epoch_1.pth`.
- no OOM, non-finite loss, traceback, or NCCL failure.

## Formal launch evidence

| item | value |
|---|---|
| started_at | `2026-08-14 01:58:09 +08:00` |
| tmux | `h2rbox_hrsc200e_bs2_gpu89_20260814` |
| launcher PID | `3775327` |
| main rank PIDs | `3775335,3775336` |
| GPU process memory | about 5554 MiB/GPU |
| MMEngine peak memory | 3790 MiB/GPU |
| initial ETA | about 1:52:06 at epoch 2 iter 20 |

Epoch 1 training windows:

| iter | lr | loss | loss_bbox_ss | grad_norm |
|---:|---:|---:|---:|---:|
| 20/109 | 1.7936e-5 | 4.4233 | 0.5226 | 276.6481 |
| 40/109 | 1.9272e-5 | 3.2825 | 0.3617 | 148.6192 |
| 60/109 | 2.0608e-5 | 2.1889 | 0.2345 | 22.4448 |
| 80/109 | 2.1944e-5 | 1.9546 | 0.1962 | 16.0301 |
| 100/109 | 2.3280e-5 | 1.9364 | 0.1823 | 15.9169 |

Epoch 2 iter 20 reported loss 1.9739 and loss_bbox_ss 0.1971. Both DDP
ranks remained alive and GPU 8/9 were active. No OOM, non-finite loss,
traceback, or NCCL failure was present in the launch log at verification time.

## Validation ledger

| epoch | val mAP | AP50 |
|---:|---:|---:|
| 10 | 0.0066 | 0.0070 |
| 20 | 0.0329 | 0.0330 |
| 30 | 0.0391 | 0.0390 |
| 40 | 0.0784 | 0.0780 |
| 50 | 0.0686 | 0.0690 |
| 60 | 0.0843 | 0.0840 |
| 70 | 0.0863 | 0.0860 |
| **80** | **0.1104** | **0.1100** |
| 90 | 0.0861 | 0.0860 |
| 100 | 0.0817 | 0.0820 |
| 110 | 0.0810 | 0.0810 |
| 120 | 0.0876 | 0.0880 |
| 130 | 0.0925 | 0.0930 |
| 140 | 0.0899 | 0.0900 |
| 150 | 0.0889 | 0.0890 |
| 160 | 0.0893 | 0.0890 |
| 170 | 0.0893 | 0.0890 |
| 180 | 0.0928 | 0.0930 |
| 190 | 0.0907 | 0.0910 |
| 200 | 0.0901 | 0.0900 |

Training completed at `2026-08-14 03:44:55 +08:00`; wall time from tmux
launch was about `1:46:46`. `epoch_200.pth` exists and the best validation
checkpoint is `best_dota_mAP_epoch_80.pth`. The full log has no fatal error,
OOM, non-finite loss, traceback, or NCCL failure.

## Held-out test

`best_dota_mAP_epoch_80.pth` was evaluated exactly once on
`ImageSets/test.txt` after training and validation-only checkpoint selection:

| metric | value |
|---|---:|
| held-out test mAP | 0.0842 |
| AP50 | 0.0840 |
| recall | 0.298 |
| ground truths | 1228 |
| detections | 1835 |

Predictions and the test log are preserved under
`work_dirs/test/h2rbox_hrsc_clean_bestval_gpu89_20260814/`. The held-out result
was not used to reselect an epoch.

## Artifacts

- design: `docs/superpowers/specs/2026-08-14-h2rbox-hrsc-200e-gpu89-design.md`
- plan: `docs/superpowers/plans/2026-08-14-h2rbox-hrsc-200e-gpu89.md`
- formal config: `configs/orbdet/h2rbox_r50_hrsc_200e_2gpu_officiallike.py`
- formal launcher: `scripts/formal/run_h2rbox_r50_hrsc_200e_gpu89_bs2.sh`
- formal work dir: `work_dirs/formal/h2rbox_r50_hrsc_train_val_200e_gpu89_bs2_seed3407_20260814/`
- launch log: `work_dirs/h2rbox_r50_hrsc_200e_gpu89_bs2_20260814.launch.log`
