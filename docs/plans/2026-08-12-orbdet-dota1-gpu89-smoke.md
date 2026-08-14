# Orbdet DOTA-v1 GPU 8/9 Smoke-Test Implementation Plan

> **Execution scope:** Build an isolated Orbdet foundation, create the `orbdet`
> Conda environment, run only a bounded two-GPU smoke train/test, and stop before
> any formal training.

**Goal:** Create `/data1/zcy/Orbdet` as a clean CNN-based horizontal-box-to-OBB
research foundation and prove that its DOTA-v1 data, model, optimizer, NCCL/DDP,
checkpoint, and evaluation paths work on physical GPUs 8 and 9.

**Architecture:** Vendor the clean OpenRSD/MMRotate-family source snapshot at
commit `12d3fd8b75e8b64ec53fded9cf035a2306d58874`, rather than copying the dirty
OpenRSD worktree. Use the MMRotate H2RBox R50-FPN parent because it directly
matches horizontal-box-supervised oriented detection. Link, rather than copy,
the verified 15-class DOTA-v1 1024 single-scale data. Clone the known-working
`openrsd` environment into a separate named environment `orbdet`, then bind the
new source tree editable and run a one-epoch, eight-image smoke configuration.

**Tech stack:** Python 3.10, PyTorch 1.12.1+cu113, MMCV 2.2.0, MMEngine 0.10.3,
MMDetection 3.3.0, MMRotate 1.0.0rc1, NCCL DDP, DOTA-v1 `DOTAMetric`.

**Hard stop:** No full dataset schedule, queue, `tmux` session, resume job, or
formal-training launcher is created or started in this plan.

---

## Task 1: Create an isolated, provenance-pinned project foundation

**Files/directories:**

- Create: `/data1/zcy/Orbdet/`
- Copy from clean snapshot: `mmrotate/`, `mmdet/`, `mmengine/`,
  `mmrotate_configs/`, `mmdet_configs/`, `tools/`, `requirements/`
- Copy root packaging metadata: `setup.py`, `setup.cfg`, `README.md`, `LICENSE`,
  `model-index.yml`, `dataset-index.yml`, `pytest.ini`
- Create data links:
  `/data1/zcy/Orbdet/data/DOTA-v1.0/trainval -> /data/zcy/dataset/trainval_ss`
  and `/data1/zcy/Orbdet/data/DOTA-v1.0/test -> /data/zcy/dataset/test_ss`

1. Reconfirm that `/data1/zcy/Orbdet` does not already exist.
2. Create the target and copy only the explicit clean-source allowlist.
3. Record the source commit and component versions in the project README.
4. Verify that the data link exposes 20,995 matched trainval image/annotation
   pairs and exactly the 15 DOTA-v1 classes.

## Task 2: Create and verify the isolated environment

**Environment:** `/data/zcy/anaconda3/envs/orbdet`

1. Reconfirm that no `orbdet` environment exists; never overwrite an existing
   environment.
2. Clone `/data/zcy/anaconda3/envs/openrsd` locally with Conda (no network
   dependency).
3. Install `/data1/zcy/Orbdet` editable with `--no-deps` so imports resolve to
   the new tree rather than `/data1/zcy/OpenRSD`.
4. From outside the source directory, assert the interpreter path and exact
   versions, and assert that `mmrotate.__file__` is under `/data1/zcy/Orbdet`.
5. Assert CUDA/NCCL availability outside the filesystem sandbox.

## Task 3: Define the smoke safety contract test first

**Files:**

- Create: `/data1/zcy/Orbdet/tests/test_dota1_smoke_contract.py`
- Later create: `/data1/zcy/Orbdet/configs/orbdet/h2rbox_r50_dota1_smoke.py`
- Later create: `/data1/zcy/Orbdet/scripts/smoke/run_dota1_gpu89.sh`
- Later create: `/data1/zcy/Orbdet/scripts/smoke/test_dota1_gpu89.sh`

1. Write tests that require the smoke config and launchers to exist.
2. Require H2RBox, HBB conversion, `batch_size=1`, `num_workers=0`, at most eight
   train samples, one epoch, local pretrained initialization disabled, and an
   explicit checkpoint interval of one.
3. Require both launchers to pin `CUDA_VISIBLE_DEVICES=8,9`, two processes,
   `NCCL_P2P_DISABLE=1`, `NCCL_IB_DISABLE=1`, and the `orbdet` interpreter.
4. Require the train launcher to reference only the smoke config and reject
   resume/formal schedule tokens.
5. Run the test and observe the expected RED failure because implementation
   files are absent.

## Task 4: Implement the smallest bounded smoke configuration and launchers

**Config behavior:**

- Parent: MMRotate H2RBox R50-FPN for DOTA-v1
- Supervision: quadrilateral annotation -> horizontal box -> rotated box
- Train subset: at most 8 images; validation/test subset: at most 4 images
- Resolution: 256 x 256; per-GPU batch: 1; workers: 0
- Schedule: exactly 1 epoch; validation disabled during training
- Initialization: no remote/pretrained download
- Output: `/data1/zcy/Orbdet/work_dirs/smoke/dota1_h2rbox_gpu89`

1. Add the config and fixed-purpose train/test launchers.
2. Make launchers executable.
3. Re-run the contract test and require GREEN.
4. Run `py_compile`, `Config.fromfile`, registry import, dataset build, model
   build, and dataloader single-batch preflight checks before using GPUs.

## Task 5: Run the physical GPU 8/9 bubble only

1. Capture fresh GPU 8/9 occupancy. Existing foreign jobs are never killed.
2. Run the fixed smoke train launcher in the foreground with two DDP ranks and
   a dedicated master port.
3. Require both ranks to initialize, losses to be finite, one epoch to finish,
   and `epoch_1.pth` plus `last_checkpoint` to be written.
4. Run the fixed two-rank smoke test launcher against `epoch_1.pth` and require
   the evaluator to complete and emit DOTA mAP metrics. The score itself is not
   a quality claim because the model was intentionally trained from scratch for
   only a few iterations.

## Task 6: Verify stopping state and hand off

**Files:**

- Create after successful execution:
  `/data1/zcy/Orbdet/docs/smoke/DOTA1_GPU89_SMOKE_RECEIPT.md`
- Create:
  `/data1/zcy/Orbdet/FORMAL_TRAINING_NOT_STARTED.md`

1. Run all contract tests again and inspect logs/checkpoint paths.
2. Confirm no process whose command contains `/data1/zcy/Orbdet` remains.
3. Record exact commands, versions, data inventory, GPU mapping, outputs, and
   the explicit statement that formal training was not started.
4. Do not create or execute any formal training command after this point.

