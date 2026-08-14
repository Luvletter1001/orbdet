# H2RBox HRSC clean recovery plan

## Goal

Recover a trustworthy H2RBox baseline on the clean HRSC train/val split before
making any further Orbdet comparison.  This run does not evaluate or select on
the held-out test split.

## Evidence motivating recovery

- The failed clean 200E H2RBox checkpoint reached only 0.1308 mAP on its own
  training split, so the previous comparison was made under a collapsed common
  optimization contract.
- The successful historical Orbdet run used batch 1/rank, global batch 2,
  AdamW at 1e-4, 309 optimizer steps/epoch, 500-step warmup, and LR milestones
  at epochs 340/468.

## Fixed recovery contract

| item | value |
|---|---|
| model | H2RBoxDetector / R50-FPN / H2RBoxHead |
| train / val / held-out test | 436 / 181 / 453 images |
| training target | enclosing HBox only |
| physical GPUs | 8,9 |
| batch | 1/GPU, global 2 |
| optimizer | AdamW, lr 1e-4, weight decay 0.05 |
| warmup | 500 optimizer steps |
| LR milestones | 105060 / 144612 optimizer steps |
| aligned endpoint | 157590 optimizer steps |
| seed | 3407 |
| validation | epochs 30 and 60 on val only |
| held-out test | forbidden during this recovery gate |

The aligned boundaries are copied from the successful schedule:

```text
340 * 309 = 105060 steps
468 * 309 = 144612 steps
510 * 309 = 157590 steps
```

With 436 training images and global batch 2, the recovery run has 218 optimizer
steps/epoch: 6540 steps at 30E and 13080 steps at 60E.  Therefore no LR decay
occurs in either recovery stage.

## Execution gates

1. Train from ImageNet initialization to 30E, then stop.
2. Evaluate `epoch_30.pth` separately on train and val.
3. Stop on fatal errors, non-finite values, or continued collapse.  The failed
   reference line is train mAP 0.1308 and recall 0.310.
4. Resume to 60E only if train AP/recall materially leave that failed region
   and validation behavior is consistent with recovery rather than collapse.
5. Evaluate `epoch_60.pth` on train and val, archive results, and stop.  Do not
   extend beyond 60E automatically.

## Artifacts

- Config:
  `configs/orbdet/h2rbox_r50_hrsc_recovery_bs1_stepaligned_60e_gpu89.py`
- Contract test: `tests/test_h2rbox_hrsc_recovery_contract.py`
- Work directory:
  `work_dirs/formal/h2rbox_r50_hrsc_recovery_bs1_stepaligned_gpu89_seed3407_20260814/`
- Process/final record:
  `resultmd/exp_h2rbox_hrsc_recovery/fres_h2rbox_hrsc_recovery_gpu89_20260814.md`
