# GDA Plan-B Correctness Hardening Design

Date: 2026-09-04

## Goal

Repair the demonstrated correctness and numerical defects in the GDA Plan-B
prototype, establish honest experiment controls, and prevent a new formal run
until the repaired implementation passes bounded correctness and gradient
gates.

The stopped P1 run is diagnostic-only evidence. Its checkpoints and logs must
not be used to claim that GDA is effective or ineffective.

## Proven failures

1. Cross-view compaction matches objects by the rank of each view's surviving
   `bid` values. A missing middle object shifts later ranks and pairs different
   physical objects. The production DOTA pipeline reproduces this failure.
2. The reconstructed flip view uses a different integer `bid` range from the
   parent detector. A fix based only on integer intersections would therefore
   produce an empty intersection unless flip numbering is fixed at the same
   time.
3. `atan2(0, 0)` in the probe angle decode has finite forward output but NaN
   backward gradients. Unbounded `exp(t)` also has a demonstrated overflow
   boundary.
4. The stopped P1 and historical E0 baseline differ in world size, global
   batch, optimizer updates per epoch, and warmup samples. They are not a
   single-variable comparison.
5. The current auxiliary losses are locally averaged. A future DDP run would
   weight ranks rather than objects, and a zero-object rank can leave probe
   parameters outside the autograd graph.

## Scope

### In scope

- Restore the parent's `.2/.4/.6` `bid` contract and compact by true integer
  object identity.
- Keep empty local batches connected to every probe output.
- Make the `(t, u2)` decode finite in forward and backward at registered
  numerical boundaries.
- Define DDP-correct weighted means for object and valid-bit denominators,
  including zero-object ranks.
- Skip probe compute during ordinary prediction while retaining an explicit
  analysis-only probe path.
- Apply the baseline rotation-agnostic policy to GDA supervision.
- Add diagnostics for effective object/bit counts, `u2` radius, `t` range,
  gate statistics, and auxiliary/main gradient calibration inputs.
- Add a controlled single-GPU configuration whose global batch and updates per
  epoch match E0, subject to a bounded memory smoke before use.
- Correct documentation: the present method is an auxiliary regularizer, not
  a replacement for or direct denoising of the main angle loss.

### Out of scope

- Starting or resuming formal training.
- Choosing new auxiliary loss weights from the failed P1 trajectory.
- Replacing the production angle head with a GDA head.
- Claiming that correctness repairs will recover mAP.

## Design

### Object identity

Flip reconstruction will reproduce the parent detector exactly: each view
starts integer numbering at one, while the fractional suffix identifies the
view. Compaction will pool each view by `bid.long()`, compute the actual
three-way key intersection on device, and gather pooled rows by those keys.
Objects absent from any view are dropped without shifting later identities.

The helper will return a tensor of true object keys rather than a synthetic
rank list. Production code does not consume the third return value; tests use
it to pin identity semantics.

### Numerical representation

The probe decode will have an explicit safe region:

- `t` will be transformed through a documented bounded parameterization whose
  range covers all legal boxes in the 1024-pixel training frame with margin.
- `u2` values below a small radius will use a finite canonical-angle fallback
  with zero local angle gradient instead of evaluating `atan2(0, 0)`.
- diagnostics will expose the fallback rate and raw `t` range so the safety
  mechanism cannot silently dominate training.

The exact limits are test-derived from the legal data geometry, not selected
to improve the stopped run's metrics.

### Distributed reduction

Each loss term will first produce a local sum and a local denominator. Every
rank will participate in the same count collective. With DDP gradient
averaging, the differentiable local loss is divided by the global mean count,
`global_count / world_size`, not by the global sum. Envelope and cross-view
terms use object count; the anchored bit term uses its own valid-mask count.

For zero local objects, the numerator will be a zero-valued expression that
depends on all probe outputs, keeping probe parameters in the graph. Global
diagnostics will be reduced using sums and matching denominators.

### Prediction and analysis

Ordinary `predict`/evaluation will not execute the probe tower. A separate,
explicit analysis flag or method will enable probe emission for evidence
collection. Tests will spy on the tower, rather than inferring behavior only
from unchanged detector outputs.

### Experiment controls

No repaired formal P1 will be launched as part of this change. The next
experiment must use:

- the same global batch, optimizer updates per epoch, LR milestones, warmup
  sample count, dataset split, and seed policy as its control;
- an explicitly shared initialization snapshot for baseline and GDA arms;
- baseline, auxiliary-zero, detached-probe, and coupled arms on identical
  bounded data before any 12-epoch run;
- per-loss shared-gradient norms and cosine similarity against the main loss;
- a pre-registered epoch-1 stop rule.

Single-GPU batch four is only a candidate control configuration. It must pass
a bounded memory smoke and be paired with a single-GPU baseline/null arm.

## Test gates

1. Middle-object and multi-image `bid` loss tests fail on commit `8492065` and
   pass after the repair.
2. Exact-zero, near-zero `u2`, and extreme `t` forward/backward tests contain
   no NaN or inf.
3. A tiny end-to-end detector loss has finite gradients for all participating
   parameters and respects rotation-agnostic masks.
4. A two-process CPU/Gloo test matches a concatenated single-process reference
   for unequal counts and one zero-object rank.
5. An inference spy proves ordinary prediction skips the probe; explicit
   analysis mode produces probe rows.
6. Shared initialization is loaded from one snapshot in controlled arms.
7. Configuration tests pin global batch, updates per epoch, and warmup sample
   equivalence.
8. The focused suite, relevant regression suite, and `git diff --check` pass.

## Stopped P1 disposition

The process started on 2026-09-04 at 03:20 and was terminated with explicit
user authorization on 2026-09-04 at approximately 09:26, during epoch 7. The
last scheduled checkpoint is `epoch_4.pth`; epoch-6 validation reached
`mAP=0.0756`. The run is `invalid-diagnostic` because of demonstrated object
mis-pairing and experiment-control confounds.

