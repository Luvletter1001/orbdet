# Orbdet-v0.2 DOTA-v1 Formal GPU 8/9 Design

## Status and authorization

Approved under the user's earlier “123 都跑完”, “按推荐方案”, formal GPU
8/9 authorization, and instruction to continue without further overnight
approvals. Task 3 is lower priority and starts only after tasks 1 and 2 finish
successfully. Everything stays local and is not pushed.

## Decision

Use **Orbdet-v0.2 without GODC** for the first DOTA-v1 formal run. This keeps
the dataset-scale transfer result separate from the experimental GODC ablation
already scheduled on HRSC.

Alternatives rejected for this run:

- GODC-enabled DOTA would confound dataset transfer with a new auxiliary loss.
- Running both variants would roughly double a long DOTA schedule and is not
  necessary to establish the first formal contract.
- Reusing the existing 256×256 H2RBox smoke config would not constitute a
  meaningful 1024-resolution formal experiment.

## Data contract

- Read-only source: `/data1/zcy/Orbdet/data/DOTA-v1.0/`.
- Prepared trainval: 20,995 image/annotation pairs at 1024-patch scale;
  12,757 contain at least one object and 8,238 are empty patches.
- Hidden test: 10,833 image patches with no local labels.
- Supervision irreversibly converts `qbox -> hbox -> rbox` before the model.
- The prepared tree has no independent validation split. Formal training uses
  all trainval data with validation disabled rather than reporting leaked
  train-set mAP. Hidden test inference/submission is outside this run.

## Model and optimization

- Detector/head: `OrbdetV02Detector` + `H2RBoxV2Head` + PSC.
- Classes: all 15 DOTA-v1 categories.
- Rotation-agnostic square classes: storage tank (9) and roundabout (11).
- Input/crop: 1024×1024; R50-FPN initialized from the local ResNet-50 weight.
- Two ranks on physical GPUs 8/9, batch 2/rank, global batch 4.
- AdamW, lr `1e-4`, weight decay `0.05`, gradient clip 35.
- Official 1× duration: 12 epochs; epoch milestones 8 and 11; seed 3407.
- Checkpoints at epochs 4, 8, and 12, with last checkpoint retained. No
  `save_best` is used because there is no independent validation loop.

## Gates and queue

1. Build a frozen `exp/orbdet-v02-dota1-formal` worktree.
2. Unit contracts prove HBox-only supervision, 15 classes, 12E schedule,
   disabled validation/test execution, local weights, GPU/NCCL settings,
   unique work dirs, and fail-fast ordering.
3. A bounded two-rank smoke uses eight samples and exactly two optimizer steps.
4. The DOTA controller waits for the task-2 GODC `COMPLETE` marker.
5. It refuses an upstream `FAILED`, occupied GPUs, duplicate processes, or
   existing checkpoints.
6. Successful smoke must produce `epoch_1.pth` before formal 12E starts.
7. Every stage writes durable local markers and stops on failure.

## Acceptance criteria

- The formal config builds through the MMRotate registry.
- Dataloader initialization sees exactly 12,757 effective samples under the
  official `filter_empty_gt=True` policy, while the source remains 20,995
  read-only pairs.
- All project tests pass before arming the controller.
- Smoke reports finite losses/gradients on both ranks with no traceback/NCCL
  error and writes `epoch_1.pth`.
- Formal training starts only after tasks 1 and 2 complete and writes to its
  unique DOTA work directory.
