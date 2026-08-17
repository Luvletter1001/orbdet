# DOTA-v1 Six-GPU Eight-Hour Stage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Safely run an eight-hour 4+2 GPU stage that resumes DOTA-v1 MS+RR from epoch 3 to epoch 8 on GPUs 4–7 and independently trains DOTA-v1 SS seed 42 for 12 epochs on GPUs 8–9.

**Architecture:** Two isolated MMEngine/DDP jobs inherit frozen, already-audited contracts and write to new work directories. Focused contract tests validate the only allowed config differences, launchers enforce checkpoint/GPU/deadline gates, and bounded post-evaluation reuses existing test configs with unique CLI output overrides.

**Tech Stack:** Python 3.10, PyTorch 1.12, MMEngine, MMRotate, four-rank/two-rank DDP, Bash, pytest, NVIDIA A40.

---

## File map

| File | Responsibility |
|---|---|
| `tests/test_orbdet_v02_dota1_six_gpu_eight_hour_contract.py` | Frozen config, resume, GPU, deadline, output-isolation, and no-overwrite contracts |
| `configs/orbdet/orbdet_v0_2_r50_dota1_ms_rr_gpu4567_resume_e3_to_e8.py` | Four-rank MS+RR continuation target |
| `configs/orbdet/orbdet_v0_2_r50_dota1_ms_rr_gpu4567_resume_e3_to_e4_smoke.py` | Two-step real-resume smoke |
| `configs/orbdet/orbdet_v0_2_r50_dota1_ss_seed42_gpu89.py` | Independent SS seed-42 12E run |
| `configs/orbdet/orbdet_v0_2_r50_dota1_ss_seed42_gpu89_smoke.py` | Two-step seed-42 smoke |
| `scripts/smoke/run_orbdet_v0_2_dota1_msrr_resume_e3_to_e4_gpu4567_smoke.sh` | Validate checkpoint resume on four ranks |
| `scripts/smoke/run_orbdet_v0_2_dota1_ss_seed42_gpu89_smoke.sh` | Validate seed-42 SS config on two ranks |
| `scripts/formal/run_orbdet_v0_2_dota1_msrr_resume_e3_to_e8_gpu4567.sh` | Bounded formal MS+RR continuation |
| `scripts/formal/run_orbdet_v0_2_dota1_ss_seed42_gpu89.sh` | Bounded formal SS seed-42 run |
| `scripts/eval/run_orbdet_v0_2_dota1_msrr_epoch8_posteval_gpu4567.sh` | Epoch-8 trainval/SS/MS post-evaluation |
| `scripts/eval/run_orbdet_v0_2_dota1_ss_seed42_epoch12_posteval_gpu89.sh` | Seed-42 trainval/SS post-evaluation |
| `scripts/formal/run_orbdet_v0_2_dota1_six_gpu_eight_hour_20260818.sh` | Ordered idle gate, smoke, parallel launch, cutoff, and receipts |
| `resultmd/exp_orbdet_v02_dota1_six_gpu_8h_20260818/flog_orbdet_v02_dota1_six_gpu_8h.md` | Immutable execution receipt |

### Task 1: Add the focused contract test

**Files:**
- Create: `tests/test_orbdet_v02_dota1_six_gpu_eight_hour_contract.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

from mmengine.config import Config


ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / 'configs/orbdet'
SCRIPTS = ROOT / 'scripts'
MS_BASE = CFG / 'orbdet_v0_2_r50_dota1_ms_rr_1x_gpu4567.py'
MS_RESUME = CFG / 'orbdet_v0_2_r50_dota1_ms_rr_gpu4567_resume_e3_to_e8.py'
MS_SMOKE = CFG / 'orbdet_v0_2_r50_dota1_ms_rr_gpu4567_resume_e3_to_e4_smoke.py'
SS_BASE = CFG / 'orbdet_v0_2_r50_dota1_1x_gpu89.py'
SS42 = CFG / 'orbdet_v0_2_r50_dota1_ss_seed42_gpu89.py'
SS42_SMOKE = CFG / 'orbdet_v0_2_r50_dota1_ss_seed42_gpu89_smoke.py'
DEADLINE = '2026-08-18 08:20:00 +0800'


def test_ms_resume_preserves_global_batch_and_targets_epoch8():
    base = Config.fromfile(MS_BASE)
    cfg = Config.fromfile(MS_RESUME)
    assert cfg.model == base.model
    assert cfg.train_dataloader == base.train_dataloader
    assert cfg.train_dataloader.batch_size == 1
    assert cfg.optim_wrapper.optimizer.lr == 1e-4
    assert cfg.optim_wrapper.optimizer.weight_decay == 0.005
    assert cfg.train_cfg.max_epochs == 8
    assert cfg.train_cfg.val_interval == 999
    assert cfg.randomness.seed == 3407
    assert cfg.default_hooks.checkpoint.interval == 1
    assert 'resume_e3_to_e8_20260818' in cfg.work_dir


def test_ms_resume_smoke_is_exactly_two_steps_after_epoch3():
    cfg = Config.fromfile(MS_SMOKE)
    assert cfg.train_dataloader.dataset.indices == 8
    assert cfg.train_dataloader.num_workers == 0
    assert cfg.train_dataloader.persistent_workers is False
    assert cfg.train_cfg.max_epochs == 4
    assert cfg.default_hooks.logger.interval == 1


def test_ss42_changes_only_seed_hooks_and_work_dir():
    base = Config.fromfile(SS_BASE)
    cfg = Config.fromfile(SS42)
    assert cfg.model == base.model
    assert cfg.train_dataloader == base.train_dataloader
    assert cfg.optim_wrapper == base.optim_wrapper
    assert cfg.param_scheduler == base.param_scheduler
    assert cfg.train_cfg == base.train_cfg
    assert cfg.randomness.seed == 42
    assert cfg.default_hooks.checkpoint.interval == 1
    assert 'seed42_20260818' in cfg.work_dir


def test_ss42_smoke_is_two_global_steps():
    cfg = Config.fromfile(SS42_SMOKE)
    assert cfg.train_dataloader.dataset.indices == 8
    assert cfg.train_dataloader.num_workers == 0
    assert cfg.train_cfg.max_epochs == 1
    assert cfg.param_scheduler[0].end == 2


def test_launchers_pin_resources_resume_and_deadline():
    cases = (
        ('formal/run_orbdet_v0_2_dota1_msrr_resume_e3_to_e8_gpu4567.sh',
         'CUDA_VISIBLE_DEVICES=4,5,6,7', '--nproc_per_node=4',
         '--resume="${source_checkpoint}"'),
        ('formal/run_orbdet_v0_2_dota1_ss_seed42_gpu89.sh',
         'CUDA_VISIBLE_DEVICES=8,9', '--nproc_per_node=2', '--resume'),
    )
    for rel, gpu, ranks, resume in cases:
        text = (SCRIPTS / rel).read_text()
        assert 'set -euo pipefail' in text
        assert gpu in text
        assert ranks in text
        assert 'NCCL_P2P_DISABLE=1' in text
        assert 'NCCL_IB_DISABLE=1' in text
        assert DEADLINE in text
        assert 'existing_checkpoints' in text
        assert 'gpu_processes' in text
        if resume.startswith('--resume='):
            assert resume in text
        else:
            assert resume not in text


def test_controller_orders_primary_before_secondary_and_never_overreaches():
    path = SCRIPTS / 'formal/run_orbdet_v0_2_dota1_six_gpu_eight_hour_20260818.sh'
    text = path.read_text()
    assert DEADLINE in text
    assert 'nvidia-smi -i 4,5,6,7' in text
    assert 'nvidia-smi -i 8,9' in text
    assert text.index('msrr_resume_e3_to_e8') < text.index('ss_seed42')
    assert 'epoch_8.pth' in text
    assert 'epoch_12.pth' in text
    assert 'epoch_9.pth' in text
    assert 'TIME_LIMIT_REACHED' in text
    assert 'rm -' not in text
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
rtk env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
  /data/zcy/anaconda3/envs/orbdet/bin/python -m pytest -q \
  -p no:cacheprovider tests/test_orbdet_v02_dota1_six_gpu_eight_hour_contract.py
```

Expected: FAIL because the four configs and five launcher/controller paths do not exist.

### Task 2: Add the four frozen training configs

**Files:**
- Create: `configs/orbdet/orbdet_v0_2_r50_dota1_ms_rr_gpu4567_resume_e3_to_e8.py`
- Create: `configs/orbdet/orbdet_v0_2_r50_dota1_ms_rr_gpu4567_resume_e3_to_e4_smoke.py`
- Create: `configs/orbdet/orbdet_v0_2_r50_dota1_ss_seed42_gpu89.py`
- Create: `configs/orbdet/orbdet_v0_2_r50_dota1_ss_seed42_gpu89_smoke.py`

- [ ] **Step 1: Add the MS continuation config**

```python
_base_ = './orbdet_v0_2_r50_dota1_ms_rr_1x_gpu4567.py'

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=8, val_interval=999)
default_hooks = dict(
    checkpoint=dict(
        _delete_=True, type='CheckpointHook', interval=1,
        max_keep_ckpts=6, save_last=True))
work_dir = (
    '/data1/zcy/Orbdet/work_dirs/formal/'
    'orbdet_v0_2_dota1_ms_rr_gpu4567_seed3407_resume_e3_to_e8_20260818')
```

- [ ] **Step 2: Add the two-step real-resume smoke config**

```python
_base_ = './orbdet_v0_2_r50_dota1_ms_rr_gpu4567_resume_e3_to_e8.py'

train_dataloader = dict(
    num_workers=0, persistent_workers=False, dataset=dict(indices=8))
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=4, val_interval=999)
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=1),
    checkpoint=dict(
        _delete_=True, type='CheckpointHook', interval=1,
        max_keep_ckpts=1, save_last=True))
work_dir = (
    '/data1/zcy/Orbdet/work_dirs/smoke/'
    'orbdet_v0_2_dota1_ms_rr_gpu4567_resume_e3_to_e4_20260818')
```

- [ ] **Step 3: Add SS seed-42 formal and smoke configs**

```python
# orbdet_v0_2_r50_dota1_ss_seed42_gpu89.py
_base_ = './orbdet_v0_2_r50_dota1_1x_gpu89.py'

randomness = dict(seed=42, deterministic=False)
default_hooks = dict(
    checkpoint=dict(
        _delete_=True, type='CheckpointHook', interval=1,
        max_keep_ckpts=4, save_last=True))
work_dir = (
    '/data1/zcy/Orbdet/work_dirs/formal/'
    'orbdet_v0_2_dota1_ss_gpu89_seed42_20260818')
```

```python
# orbdet_v0_2_r50_dota1_ss_seed42_gpu89_smoke.py
_base_ = './orbdet_v0_2_r50_dota1_ss_seed42_gpu89.py'

train_dataloader = dict(
    num_workers=0, persistent_workers=False, dataset=dict(indices=8))
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=1, val_interval=999)
param_scheduler = [dict(
    type='LinearLR', start_factor=1.0 / 3, by_epoch=False,
    begin=0, end=2)]
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=1),
    checkpoint=dict(
        _delete_=True, type='CheckpointHook', interval=1,
        max_keep_ckpts=1, save_last=True))
work_dir = (
    '/data1/zcy/Orbdet/work_dirs/smoke/'
    'orbdet_v0_2_dota1_ss_gpu89_seed42_20260818')
```

- [ ] **Step 4: Run config-focused tests**

Run the Task 1 pytest command. Expected: config tests PASS; launcher tests still FAIL.

### Task 3: Add smoke and formal launchers

**Files:**
- Create: `scripts/smoke/run_orbdet_v0_2_dota1_msrr_resume_e3_to_e4_gpu4567_smoke.sh`
- Create: `scripts/smoke/run_orbdet_v0_2_dota1_ss_seed42_gpu89_smoke.sh`
- Create: `scripts/formal/run_orbdet_v0_2_dota1_msrr_resume_e3_to_e8_gpu4567.sh`
- Create: `scripts/formal/run_orbdet_v0_2_dota1_ss_seed42_gpu89.sh`

- [ ] **Step 1: Implement shared launcher gates in each script**

Each launcher must use these exact constants and gate shape, with its own config,
work directory, GPUs, rank count, and port:

```bash
#!/usr/bin/env bash
set -euo pipefail

repo_root="$(rtk realpath "$(rtk dirname "${BASH_SOURCE[0]}")/../..")"
python_bin=/data/zcy/anaconda3/envs/orbdet/bin/python
deadline='2026-08-18 08:20:00 +0800'
deadline_epoch="$(rtk date -d "${deadline}" +%s)"

on_error() {
  exit_code=$?
  rtk mkdir -p "${work_dir}"
  rtk touch "${work_dir}/FAILED"
  exit "${exit_code}"
}
on_signal() {
  trap - ERR INT TERM
  rtk mkdir -p "${work_dir}"
  rtk touch "${work_dir}/INTERRUPTED"
  exit 130
}
trap on_error ERR
trap on_signal INT TERM

if (( $(rtk date +%s) >= deadline_epoch )); then
  rtk echo 'The authorized GPU window has expired.' >&2
  exit 2
fi
gpu_processes="$(rtk nvidia-smi -i "${physical_gpus}" \
  --query-compute-apps=pid --format=csv,noheader,nounits)"
if [[ -n "${gpu_processes}" ]]; then
  rtk echo "GPU ${physical_gpus} are not idle; refusing to start." >&2
  exit 3
fi
shopt -s nullglob
existing_checkpoints=("${work_dir}"/*.pth)
if (( ${#existing_checkpoints[@]} > 0 )) || \
   [[ -e "${work_dir}/RUNNING" || -e "${work_dir}/COMPLETE" ]]; then
  rtk echo "Refusing to overwrite outputs in ${work_dir}" >&2
  exit 4
fi
```

The command block in every launcher must be:

```bash
remaining_seconds=$(( deadline_epoch - $(rtk date +%s) ))
rtk mkdir -p "${work_dir}"
rtk touch "${work_dir}/RUNNING"
rtk timeout --foreground --signal=TERM --kill-after=30s \
  "${remaining_seconds}" \
  rtk env PYTHONNOUSERSITE=1 PYTHONPATH="${repo_root}" \
  MPLCONFIGDIR=/tmp/zcy-codex/mplconfig OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  CUDA_VISIBLE_DEVICES="${physical_gpus}" \
  NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
  "${python_bin}" -m torch.distributed.launch \
  --nproc_per_node="${nproc}" --master_port="${master_port}" \
  "${repo_root}/tools/train.py" "${config}" \
  --launcher=pytorch --work-dir="${work_dir}" "${resume_args[@]}"
```

Use the following exact per-script values:

| launcher | GPUs | ranks | port | final checkpoint | resume |
|---|---:|---:|---:|---|---|
| MS smoke | 4,5,6,7 | 4 | 29670 | `epoch_4.pth` | verified epoch-3 source |
| SS smoke | 8,9 | 2 | 29671 | `epoch_1.pth` | none |
| MS formal | 4,5,6,7 | 4 | 29672 | `epoch_8.pth` | verified epoch-3 source |
| SS formal | 8,9 | 2 | 29673 | `epoch_12.pth` | none |

MS scripts must validate the immutable source SHA256 and run:

```bash
rtk env PYTHONNOUSERSITE=1 "${python_bin}" \
  "${repo_root}/tools/analysis_tools/validate_checkpoint_contract.py" \
  "${source_checkpoint}" --expected-epoch 3 --expected-iter 51246 \
  --expected-state-tensors 371 --config-token OrbdetV02Detector \
  --config-token trainval_ms_full > "${work_dir}/source_checkpoint_contract.json"
```

Formal MS completion must require epoch 8 / iter 136656 and reject epoch 9.
Formal SS completion must require epoch 12 / iter 38280 and reject epoch 13.
Move `RUNNING` to `COMPLETE` only after the final validator exits zero.

- [ ] **Step 2: Mark scripts executable**

```bash
rtk chmod +x \
  scripts/smoke/run_orbdet_v0_2_dota1_msrr_resume_e3_to_e4_gpu4567_smoke.sh \
  scripts/smoke/run_orbdet_v0_2_dota1_ss_seed42_gpu89_smoke.sh \
  scripts/formal/run_orbdet_v0_2_dota1_msrr_resume_e3_to_e8_gpu4567.sh \
  scripts/formal/run_orbdet_v0_2_dota1_ss_seed42_gpu89.sh
```

- [ ] **Step 3: Run focused tests**

Expected: config and four launcher tests PASS; controller test still FAIL.

### Task 4: Add bounded post-evaluation launchers

**Files:**
- Create: `scripts/eval/run_orbdet_v0_2_dota1_msrr_epoch8_posteval_gpu4567.sh`
- Create: `scripts/eval/run_orbdet_v0_2_dota1_ss_seed42_epoch12_posteval_gpu89.sh`

- [ ] **Step 1: Implement epoch-8 MS post-evaluation**

Use the verified `epoch_8.pth`, existing epoch-3 trainval/SS/MS configs, four
ranks on GPUs 4–7, ports 29674–29676, and unique CLI overrides:

```bash
--cfg-options \
  work_dir="${eval_root}/trainval" \
  test_evaluator.outfile_prefix="${eval_root}/trainval/result"
```

For SS and MS format-only runs override only `work_dir` and
`test_evaluator.outfile_prefix` to unique epoch-8 paths. Validate both ZIPs
contain exactly 15 root-level `Task1_*.txt` files and have no CRC error.
Skip the entire launcher when fewer than 1,500 seconds remain before 08:20.

- [ ] **Step 2: Implement seed-42 SS post-evaluation**

Use the verified seed-42 `epoch_12.pth`, existing SS trainval/submission configs,
two ranks on GPUs 8–9, ports 29677–29678, and unique `work_dir` and
`test_evaluator.outfile_prefix` CLI overrides. Validate the Task1 ZIP exactly as
above. Never upload it.

- [ ] **Step 3: Mark launchers executable and extend the focused test**

Assert GPU pins, ranks, unique ports, absolute checkpoint paths, checkpoint
validator arguments, unique output roots, deadline, ZIP validation, and absence
of `tools/train.py` or `--resume`.

### Task 5: Add the bounded controller

**Files:**
- Create: `scripts/formal/run_orbdet_v0_2_dota1_six_gpu_eight_hour_20260818.sh`

- [ ] **Step 1: Implement stable idle waiting**

The controller must wait only until the approved absolute deadline and require
two consecutive empty samples 30 seconds apart:

```bash
wait_for_idle() {
  local gpu_list="$1"
  local stable=0
  while (( $(rtk date +%s) < deadline_epoch )); do
    pids="$(rtk nvidia-smi -i "${gpu_list}" \
      --query-compute-apps=pid --format=csv,noheader,nounits)"
    if [[ -z "${pids}" ]]; then
      stable=$(( stable + 1 ))
      (( stable >= 2 )) && return 0
    else
      stable=0
    fi
    rtk sleep 30
  done
  return 1
}
```

- [ ] **Step 2: Implement ordered smoke and parallel launch**

Run MS resume smoke, then start MS formal in a background process and record
its PID. Poll its scalars until step 51,446 (200 resumed steps) appears with
finite loss/time and no traceback/OOM/NCCL error. Then run SS smoke and start
SS formal with a separate PID. `wait` for both wrappers and record each exit
code without masking the other.

- [ ] **Step 3: Implement cutoff and evaluation policy**

If `epoch_12.pth` is verified, run seed-42 post-evaluation. If `epoch_8.pth` is
verified and at least 1,500 seconds remain, run MS post-evaluation. Otherwise
write `POSTEVAL_SKIPPED_INSUFFICIENT_WINDOW`. Reject an MS `epoch_9.pth` or SS
`epoch_13.pth`. At the deadline, touch `TIME_LIMIT_REACHED`; do not create a
resume queue.

- [ ] **Step 4: Run the complete focused contract**

Expected: all focused tests PASS.

### Task 6: Run regression checks and commit infrastructure

**Files:** all Task 1–5 files.

- [ ] **Step 1: Run syntax and focused tests**

```bash
rtk bash -n scripts/smoke/run_orbdet_v0_2_dota1_msrr_resume_e3_to_e4_gpu4567_smoke.sh
rtk bash -n scripts/smoke/run_orbdet_v0_2_dota1_ss_seed42_gpu89_smoke.sh
rtk bash -n scripts/formal/run_orbdet_v0_2_dota1_msrr_resume_e3_to_e8_gpu4567.sh
rtk bash -n scripts/formal/run_orbdet_v0_2_dota1_ss_seed42_gpu89.sh
rtk bash -n scripts/eval/run_orbdet_v0_2_dota1_msrr_epoch8_posteval_gpu4567.sh
rtk bash -n scripts/eval/run_orbdet_v0_2_dota1_ss_seed42_epoch12_posteval_gpu89.sh
rtk bash -n scripts/formal/run_orbdet_v0_2_dota1_six_gpu_eight_hour_20260818.sh
rtk env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
  /data/zcy/anaconda3/envs/orbdet/bin/python -m pytest -q \
  -p no:cacheprovider \
  tests/test_orbdet_v02_dota1_six_gpu_eight_hour_contract.py \
  tests/test_orbdet_v02_dota1_formal_contract.py \
  tests/test_orbdet_v02_dota1_msrr_gpu4567_stage1_posteval_contract.py
```

Expected: all tests PASS; shell syntax checks exit zero.

- [ ] **Step 2: Review only intended files and commit**

```bash
rtk git diff --check
rtk git status --short
rtk git add tests/test_orbdet_v02_dota1_six_gpu_eight_hour_contract.py \
  configs/orbdet/orbdet_v0_2_r50_dota1_ms_rr_gpu4567_resume_e3_to_e8.py \
  configs/orbdet/orbdet_v0_2_r50_dota1_ms_rr_gpu4567_resume_e3_to_e4_smoke.py \
  configs/orbdet/orbdet_v0_2_r50_dota1_ss_seed42_gpu89.py \
  configs/orbdet/orbdet_v0_2_r50_dota1_ss_seed42_gpu89_smoke.py \
  scripts/smoke/run_orbdet_v0_2_dota1_msrr_resume_e3_to_e4_gpu4567_smoke.sh \
  scripts/smoke/run_orbdet_v0_2_dota1_ss_seed42_gpu89_smoke.sh \
  scripts/formal/run_orbdet_v0_2_dota1_msrr_resume_e3_to_e8_gpu4567.sh \
  scripts/formal/run_orbdet_v0_2_dota1_ss_seed42_gpu89.sh \
  scripts/eval/run_orbdet_v0_2_dota1_msrr_epoch8_posteval_gpu4567.sh \
  scripts/eval/run_orbdet_v0_2_dota1_ss_seed42_epoch12_posteval_gpu89.sh \
  scripts/formal/run_orbdet_v0_2_dota1_six_gpu_eight_hour_20260818.sh
rtk git commit -m "feat: add DOTA six-GPU bounded stage"
```

### Task 7: Execute the authorized stage and verify launch

**Files:** runtime outputs only.

- [ ] **Step 1: Revalidate source checkpoint and GPU ownership**

Run the validator from Task 3 and inspect `nvidia-smi` for GPUs 4–9. Do not
terminate any foreign PID. Confirm the current commit and preserve unrelated
dirty files.

- [ ] **Step 2: Start one bounded controller session**

```bash
rtk tmux new-session -d -s orbdet_dota1_sixgpu_8h_20260818 \
  "rtk bash /data1/zcy/Orbdet/scripts/formal/run_orbdet_v0_2_dota1_six_gpu_eight_hour_20260818.sh"
```

Expected: the session waits for stable GPU idleness, runs smoke gates, then
launches MS first and SS after the MS 200-step health gate.

- [ ] **Step 3: Confirm live health**

Inspect only the new controller/work directories. Require correct physical
GPU mapping, finite loss/grad, expected memory, expected world sizes, no OOM,
NCCL error, NaN, traceback, or foreign PID modification.

### Task 8: Monitor, evaluate, and record the receipt

**Files:**
- Create: `resultmd/exp_orbdet_v02_dota1_six_gpu_8h_20260818/flog_orbdet_v02_dota1_six_gpu_8h.md`
- Modify: `FORMAL_TRAINING_NOT_STARTED.md`

- [ ] **Step 1: Record epoch milestones as they land**

For every completed checkpoint, record timestamp, bytes, epoch, iter, state
tensor count, SHA256, recent median step time, and finite loss/grad evidence.

- [ ] **Step 2: Run bounded post-evaluation policy**

Run only the evaluations admitted by the controller's remaining-time gate.
Record trainval as diagnostic, not held-out validation; validate ZIP structure
and SHA256 without uploading.

- [ ] **Step 3: Update formal status truthfully**

Mark MS complete only through the actual last verified epoch and retain
`full_12e=not_complete` unless epoch 12 genuinely exists. Mark SS seed 42
complete only if epoch 12 passes the validator. Record deadline, interruptions,
and skipped evaluations exactly.

- [ ] **Step 4: Run final integrity checks and commit the receipt**

```bash
rtk rg -n 'Traceback|RuntimeError|NCCL.*error|OutOfMemory|NaN|Inf' \
  work_dirs/controllers/orbdet_dota1_sixgpu_8h_20260818 \
  work_dirs/formal/orbdet_v0_2_dota1_ms_rr_gpu4567_seed3407_resume_e3_to_e8_20260818 \
  work_dirs/formal/orbdet_v0_2_dota1_ss_gpu89_seed42_20260818
rtk git diff --check
rtk git add FORMAL_TRAINING_NOT_STARTED.md \
  resultmd/exp_orbdet_v02_dota1_six_gpu_8h_20260818/flog_orbdet_v02_dota1_six_gpu_8h.md
rtk git commit -m "docs: record DOTA six-GPU bounded stage"
```

Expected: receipt matches filesystem/checkpoint evidence; no source data or
foreign process was modified.
