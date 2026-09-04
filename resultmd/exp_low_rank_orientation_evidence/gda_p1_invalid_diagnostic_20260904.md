# GDA Plan-B P1 Invalid Diagnostic Run

Date: 2026-09-04
Disposition: **invalid-diagnostic; do not resume or use for method claims**

## Run identity

- Source worktree: `/data1/zcy/Orbdet/.worktrees/codex-hbox-angle-benchmark`
- Source commit at launch: `8492065`
- Config: `configs/orbdet/orbdet_gda_probe_r50_dota1_grouped_ss_e0_gpu4.py`
- Work directory: `/data1/zcy/Orbdet/work_dirs/formal/orbdet_gda_probe_dota1_grouped_ss_e0_gpu4_seed3407`
- Launch: 2026-09-04 03:20:03 +08:00
- User-authorized stop: SIGTERM sent only to process group `3079663` at
  2026-09-04 09:26:29 +08:00; final buffered log row is 09:26:35.
- Final position: epoch 7, iteration 220/5373.
- Last scheduled checkpoint: `epoch_4.pth` (407,201,833 bytes).

## Validation trajectory

| epoch | P1 mAP | P1 AP50 | historical E0 mAP |
|---:|---:|---:|---:|
| 1 | 0.0088 | 0.0090 | 0.2397 |
| 2 | 0.0240 | 0.0240 | 0.3796 |
| 3 | 0.0319 | 0.0320 | 0.4068 |
| 4 | 0.0419 | 0.0420 | 0.4789 |
| 5 | 0.0578 | 0.0580 | 0.5316 |
| 6 | 0.0756 | 0.0760 | 0.5014 |

These numbers are diagnostic only. P1 used one GPU with batch size 2 and
5373 optimizer updates per epoch; E0 used two GPUs with batch size 2 per rank
and 2687 updates per epoch. Their global batch, update trajectory, and warmup
sample exposure differ.

## Demonstrated implementation defects

1. Probe rows were paired across views by the rank of each view's surviving
   `bid` values. A real DOTA target-assignment scan reproduced missing middle
   objects and wrong physical-object triplets.
2. Exact-zero `u2` caused NaN gradients through `atan2(0, 0)`; sufficiently
   large raw `t` overflowed `exp(t)`.

The run also had auxiliary gradients coupled into the shared features. Every
logged epoch-1-to-5 gradient-norm window exceeded the configured clip threshold
of 35. Because correctness defects, optimization confounds, and shared-gradient
pressure coexist, this run cannot identify which factor caused the observed
degradation.

## Allowed use

- Allowed: regression-test motivation, runtime-cost accounting, failure-mode
  documentation.
- Forbidden: evidence that GDA works, evidence that GDA cannot work, formal
  baseline comparison, checkpoint resume.

