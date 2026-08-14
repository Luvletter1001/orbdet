# Orbdet Overnight GPU 8/9 Tasks 1–2 Design

## Status and authorization

Approved in conversation on 2026-08-15. The user explicitly authorized
formal overnight training on physical GPUs 8 and 9, asked to prioritize tasks
1 and 2, accepted the recommended design, and stated that no further approval
would be available while asleep. The work remains local and is not pushed.

## Outcomes

1. Complete the missing Orbdet-v0.2 clean-HRSC stability seeds 42 and 2026
   from a frozen source snapshot, sequentially on GPUs 8/9.
2. Integrate the existing group-orbit determinantal cluster (GODC) core into
   an HBox/FPN instance path, pass unit and two-rank smoke gates, then run one
   formal clean-HRSC seed on GPUs 8/9.

Task 3 (DOTA-v1) remains authorized by the earlier “123 都跑完” instruction,
but it is lower priority and must not delay or contaminate tasks 1 and 2.

## Alternatives considered

- **Selected: isolated staged queues.** Freeze task 1 in a dedicated Git
  worktree; develop task 2 on `main`; use unique work directories; execute
  task 1 seeds serially, then task 2 smoke and formal training. This preserves
  a causal baseline while keeping GPUs occupied.
- A single mutable checkout is simpler, but later imports or restarts could
  observe GODC changes during baseline replication.
- Finishing all implementation before any training gives one static tree, but
  wastes already-idle GPUs and delays the higher-priority stability seeds.

## Task 1: frozen v0.2 stability queue

- Branch/worktree: `exp/v02-hrsc-stability` at a frozen pre-GODC-integration
  commit under `.worktrees/v02-stability`.
- The seed configs inherit the proven 103-epoch clean-HRSC v0.2 config and
  override only `randomness.seed` and `work_dir`.
- Seeds run in order 42, then 2026, each with global batch 4 on two ranks.
- Every launcher pins `CUDA_VISIBLE_DEVICES=8,9`, `NCCL_P2P_DISABLE=1`, and
  `NCCL_IB_DISABLE=1`, and uses the Orbdet conda interpreter.
- The queue is fail-fast. Existing checkpoints cause refusal rather than
  overwrite. Completion/failure markers make the unattended outcome auditable.
- Training and validation use `train.txt` and `val.txt`; the held-out test set
  remains unused during selection.

## Task 2: HBox/FPN GODC adapter

### Architecture

The existing H2RBox-v2 original/rotated/flipped training path remains the
primary objective. A protected auxiliary-loss hook exposes the already
computed FPN features and cropped original-view GT instances without a second
backbone pass. The default hook is a no-op, preserving all existing models.

A new `OrbdetGODCDetector` subclasses `OrbdetV02Detector` and installs a
registered `HBoxFPNGroupOrbitLoss` adapter. The adapter:

1. converts each original-view GT box to an axis-aligned HBox;
2. filters non-finite or degenerate boxes;
3. uses the standard multi-level `SingleRoIExtractor`/RoIAlign to produce
   square 7×7 instance features from FPN levels;
4. constructs the exact nontrivial `C2={e,r_π}` orbit;
5. applies the existing determinantal, fixed-space, energy, and variance
   terms; and
6. records detached ROI-count, determinantal, fixed-space, guard, and spectral
   gap diagnostics.

`C2` is deliberately fixed for this first detector experiment. It is
orientation-independent under 180-degree rotation and is plausible for ships;
automatic group selection would add a separate assignment collapse mode.
This stage tests whether instance-orbit structure helps detection; it does not
claim continuous angle recovery.

### Optimization contract

- The GODC term is additive under the explicit key `loss_godc` with a small,
  fixed weight of 0.02.
- Diagnostics are detached and never self-weight the objective.
- Empty-ROI batches return a graph-connected zero and finite diagnostics.
- Prediction, NMS, score computation, HBox-only training annotations, data
  split, optimizer, schedule, and seed remain identical to v0.2.
- The formal comparison uses seed 3407 and a unique work directory.

## Gates and unattended orchestration

1. Contract/unit tests must pass before launch.
2. Task 1 queue starts first and owns GPUs 8/9 until both seeds complete.
3. Task 2 controller waits for the task-1 success marker.
4. Task 2 runs a bounded two-rank, eight-image/two-step smoke test.
5. Only a successful smoke test may start the 103-epoch formal run.
6. Any failure writes a failure marker and stops the downstream chain.
7. No controller terminates unrelated GPU processes.

## Acceptance criteria

- Seed configs differ from the frozen formal config only in seed/work dir.
- Queue/launch contracts pin GPU/NCCL/env/ports and prevent overwrite.
- Default H2RBox-v2 behavior is unchanged when no auxiliary hook is configured.
- ROI adapter tests cover level extraction, invalid/empty boxes, gradients,
  diagnostics, and registry construction.
- GODC smoke produces finite nonzero `loss_godc`, finite gradients, and no
  distributed/runtime error.
- Formal jobs create unique logs/checkpoints and the status marker reports the
  actual active/completed stage.

## Safety

Source DOTA/HRSC data are read-only. Existing checkpoints are never deleted or
overwritten. Other users' processes are neither signalled nor modified. All
training artifacts stay local under `work_dirs/` and all Git history stays
local without a remote push.
