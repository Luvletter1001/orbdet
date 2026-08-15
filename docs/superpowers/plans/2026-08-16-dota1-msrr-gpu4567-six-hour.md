# DOTA-v1 MS+RR GPU 4–7 Six-Hour Stage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在六小时 GPU 4–7 窗口内完成官方审计、四 rank smoke 和可续训的 R50 MS+RR 3E 阶段。

**Architecture:** 新增 GPU4567 专用配置与 fail-fast launcher，不修改既有 GPU89 复现合同。四卡正式配置保留 12E 总合同，stage1 配置只把本次上限设为 3E；外层控制器提供 5h45m 硬截止。

**Tech Stack:** MMRotate 1.0.0rc1, MMEngine 0.10.3, PyTorch 1.12.1+cu113, Bash, pytest, four NVIDIA A40 GPUs.

---

### Task 1: Add RED GPU4567 contract

**Files:**
- Modify: `tests/test_h2rbox_v2_dota1_official_audit_msrr_contract.py`

- [ ] Add exact paths and assertions for four-card configs and launchers.
- [ ] Require global batch 4, LR `1e-4`, WD `0.005`, stage1 3E, smoke 8 samples,
  checkpoint interval 1, four NCCL ranks and `345m` (`5h45m`) deadline.
- [ ] Run the focused test and observe failure because GPU4567 files are absent.

### Task 2: Implement four-card configs and launchers

**Files:**
- Create: `configs/orbdet/orbdet_v0_2_r50_dota1_ms_rr_1x_gpu4567.py`
- Create: `configs/orbdet/orbdet_v0_2_r50_dota1_ms_rr_1x_gpu4567_smoke.py`
- Create: `configs/orbdet/orbdet_v0_2_r50_dota1_ms_rr_1x_gpu4567_stage1_3e.py`
- Create: `scripts/eval/run_h2rbox_v2_dota1_official_checkpoint_audit_gpu4567.sh`
- Create: `scripts/smoke/run_orbdet_v0_2_r50_dota1_ms_rr_gpu4567_smoke.sh`
- Create: `scripts/formal/run_orbdet_v0_2_r50_dota1_ms_rr_gpu4567_stage1_3e.sh`
- Create: `scripts/formal/run_orbdet_v0_2_r50_dota1_ms_rr_gpu4567_six_hour_window.sh`

- [ ] Implement minimal inheritance overrides and isolated work directories.
- [ ] Implement idle-GPU, duplicate-job, marker and non-overwrite gates.
- [ ] Apply GPU4567 plus NCCL P2P/IB disables to every distributed command.
- [ ] Run focused tests, py_compile, bash syntax, config parse and model/dataset build.

### Task 3: Execute the six-hour chain

- [ ] Launch one named tmux controller with a `5h45m` timeout.
- [ ] Verify four audit ranks and nonzero GPU utilization.
- [ ] After audit, validate mAP/logs/ZIP contents before smoke can pass its marker gate.
- [ ] Verify smoke finite losses/gradients and `epoch_1.pth`.
- [ ] Verify stage1 four ranks, dataset length 68,325 and finite initial losses.

### Task 4: Record local evidence

- [ ] Update the audit and formal Chinese result records with actual timestamps,
  markers, metrics, checkpoint paths and timeout state.
- [ ] Run `git diff --check`, the focused test, and inspect runtime logs for fatal errors.
- [ ] Commit code and records locally without pushing.
