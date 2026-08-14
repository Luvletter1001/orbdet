# Orbdet-v0.2 Anchored-Symmetry Stability Design

## Objective

Replace the self-referential Orbdet-v0.1 harmonic gate with an image-symmetry
orientation anchor and reproduce a clean HRSC validation result near or above
88 AP50 on physical GPUs 8/9.  The official HRSC test split remains untouched
while the candidate is developed and selected.

## Evidence and failure being fixed

Orbdet-v0.1 computes `q` from the same pairwise angle residual that it weights.
On the clean recovery contract it reached `q_mean=0.9980` while validation AP
was only `0.0893`; therefore `q` is not a correctness signal.  The otherwise
identical H2RBox contract reached `0.8710`, locating the brittle behavior in the
quality-weighted consistency loss.

H2RBox-v2 provides the closest validated correction.  Its original/rotated/
vertically-flipped views impose both rotation equivariance and reflection
symmetry.  PSC and snap loss remove angular boundary discontinuities, while
per-object `bid` aggregation gives a fixed correspondence across all three
views.  The official implementation reports 89.66 AP50 on HRSC.

## Architecture

1. Port the official H2RBox-v2 detector, dense head and consistency loss from
   `yuyi1005/mmrotate` dev-1.x commit
   `97b793577199e80f85f199b6bffb99b03017c4f4` with attribution preserved.
2. Add `OrbdetAnchoredSymmetryLoss`, whose optimization value is exactly the
   official `H2RBoxV2ConsistencyLoss`:

   `L_ss = L_snap(rotation) + 0.05 * L_snap(vertical_flip)`.

3. Compute detached diagnostics `q_rot`, `q_flip`, and
   `q_joint=sqrt(q_rot*q_flip)` from the two independent constraint residuals.
   These values never multiply classification, box, centerness, or symmetry
   losses.
4. Add `OrbdetV02Detector` as a thin subclass of `H2RBoxV2Detector`.  It exposes
   the detached diagnostics and combines `q_joint` with
   `exp(-L_hbox)` only for the logged `q_anchored` statistic.
5. Preserve inference exactly as `H2RBoxV2Detector`; no quality score is used
   in NMS or test-time ranking.

This is a stability recovery candidate, not a claim that H2RBox-v2 is an
Orbdet novelty.  EMA teachers and a zero-initialized residual angle head are
deferred until this independent anchor passes the clean multi-seed gate; adding
them now would prevent causal attribution.

## Data and evaluation contract

- Training input: HRSC `ImageSets/train.txt` (436 images).
- Model supervision: `qbox -> hbox -> rbox`, so true OBB angles are discarded.
- Validation: `ImageSets/val.txt` (181 images), OBB annotations retained only
  for evaluation.
- Held-out test: `ImageSets/test.txt` (453 images), never evaluated or selected
  during implementation and validation.
- Backbone: R50-FPN with the existing local ImageNet checkpoint.
- Per-rank batch: 2; two ranks on physical GPUs 8/9.
- Optimizer: official AdamW, lr `5e-5`, weight decay `0.005`.
- Clean-data schedule: 103 epochs, matching the official 72-epoch trainval run
  in optimizer updates.  LR drops are expressed at 7,440 and 10,230 optimizer
  steps; warmup remains 500 steps.
- Seed for the first causal run: 3407.  Seeds 42 and 2026 are run only after
  the first run clears the non-collapse gate.

## Gates

- Unit gate: anchored loss equals the official symmetry objective exactly and
  catches the two-view-consistent-but-flip-wrong construction.
- Config gate: clean split, HBox-only training, PSC, snap loss, physical GPU
  isolation, and NCCL transport settings are asserted automatically.
- Smoke gate: two DDP ranks complete finite forward/backward steps with all
  diagnostic values in `[0,1]`.
- Early experiment gate: by epoch 24 validation AP50 must be at least 0.70.
  Failure stops the candidate before full compute is spent.
- First-seed target: best clean validation AP50 at least 0.88.  Values in
  `[0.87,0.88)` are informative but do not meet the target.
- Stability target after three seeds: mean at least 0.88, worst at least 0.87,
  and standard deviation at most 0.005 raw AP.

## Safety

Every run has a unique work directory.  Launch scripts pin
`CUDA_VISIBLE_DEVICES=8,9`, `NCCL_P2P_DISABLE=1`, and `NCCL_IB_DISABLE=1`.
They reject duplicate matching jobs and never terminate unrelated processes.

