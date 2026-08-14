# Orbdet Clean HRSC Comparison Until 08:30 Design

## Objective

Use physical GPUs 8 and 9 to finish the current clean H2RBox HRSC baseline,
evaluate its best-validation checkpoint once on held-out test, and then run one
clean Orbdet-v0.1 comparison whose only functional model change is the
consistency loss. All GPU use owned by this experiment must stop no later than
`2026-08-14 08:30:00 +08:00`.

## Scientific contract

The comparison inherits the existing clean H2RBox protocol exactly:

| item | fixed value |
|---|---|
| train / val / test | 436 / 181 / 453 HRSC images |
| input | 800 x 800 |
| GPUs | 8,9 |
| batch | 2/GPU, global 4 |
| optimizer | AdamW, lr 5e-5, weight decay 0.05 |
| warmup | 500 optimizer iterations |
| schedule | 200E, milestones 133/184 |
| validation | every 10E on val |
| seed | 3407 |
| test | one evaluation after best-val selection |

The only functional comparison variable is:

- baseline: `H2RBoxDetector` + `H2RBoxConsistencyLoss`;
- candidate: `OrbdetDetector` + `OrbdetHarmonicConsistencyLoss` with
  `min_quality=0.25`, `gamma=2.0`, and `high_quality_thr=0.75`.

`OrbdetDetector` only exposes training diagnostics; inference remains the same
H2RBox head and rotated NMS path. No checkpoint resume, test-time score
reweighting, new augmentation, or data-split change is allowed.

## Execution sequence

1. Keep the already-running H2RBox job unchanged until it completes.
2. Select the H2RBox checkpoint using validation mAP only and evaluate held-out
   test once.
3. Parse and contract-test the clean Orbdet config.
4. Run a bounded two-step Orbdet DDP smoke job on GPUs 8/9.
5. Start the clean Orbdet 200E job if smoke passes and the time budget remains.
6. If Orbdet training completes before the deadline, evaluate its best-val
   checkpoint once on held-out test.
7. Preserve logs, checkpoints, predictions, and a comparison ledger.

No H2RBox-v2 implementation is included in this window. It remains the next
strong baseline after the clean H2RBox-versus-Orbdet causal comparison.

## Deadline enforcement

The absolute deadline is Unix epoch `1786667400`, corresponding to
`2026-08-14 08:30:00 +08:00`.

- A dedicated guard checks the clock at intervals no longer than 30 seconds.
- At the deadline it sends Ctrl-C only to the named H2RBox and Orbdet tmux
  sessions created for this experiment.
- It then targets only processes whose command lines contain the exact Orbdet
  project configs for this comparison; unrelated GPU processes are untouched.
- Formal candidate training reserves five minutes for orderly shutdown and an
  optional final test. If less than five minutes remain, no new stage starts.
- A stage interrupted by the deadline is recorded as time-capped, not failed.

## Isolation and artifacts

- candidate config:
  `configs/orbdet/orbdet_v0_1_r50_hrsc_clean_200e_gpu89_bs2.py`
- candidate smoke config:
  `configs/orbdet/orbdet_v0_1_r50_hrsc_clean_200e_gpu89_bs2_smoke.py`
- candidate launcher:
  `scripts/formal/run_orbdet_v0_1_hrsc_clean_200e_gpu89_bs2.sh`
- deadline guard:
  `scripts/formal/guard_orbdet_gpu89_until_0830_20260814.sh`
- candidate work directory:
  `work_dirs/formal/orbdet_v0_1_hrsc_clean_200e_gpu89_bs2_seed3407_20260814/`
- experiment record:
  `resultmd/exp_orbdet_hrsc_clean/fres_orbdet_hrsc_clean_gpu89_20260814.md`

All new artifacts use distinct paths. Existing H2RBox and earlier Orbdet-v0.1
results are read-only inputs.

## Gates and failure handling

- Contract tests must prove matching splits, optimizer, schedule, batch, seed,
  validation policy, and test policy before the candidate starts.
- Smoke must complete both steps with finite loss and save `epoch_1.pth`.
- Formal training must show both ranks, finite losses, and no OOM/NCCL error.
- A non-finite loss, config mismatch, OOM, or NCCL failure stops the candidate
  stage and preserves evidence; parameters are not silently changed.
- Held-out test is never used for checkpoint selection.

## Completion states

- `complete`: both methods have best-val one-shot held-out test results.
- `candidate_complete_test_pending`: Orbdet trained but the deadline prevented
  its test; checkpoint is preserved for later test without retraining.
- `time_capped`: the 08:30 guard stopped an unfinished owned stage.
- `failed`: an engineering error, non-finite optimization, OOM, or NCCL failure
  stopped the stage before the deadline.

