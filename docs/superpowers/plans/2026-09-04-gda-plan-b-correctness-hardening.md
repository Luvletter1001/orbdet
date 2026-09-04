# GDA Plan-B Correctness Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair demonstrated GDA object-matching and numerical failures, make training/inference and DDP behavior explicit, and create honest controlled experiment configurations without starting formal training.

**Architecture:** Preserve the existing probe/head/detector split. Restore the parent detector's integer object-identity contract, move all loss reduction into a DDP-aware tensor helper, and isolate ordinary prediction from explicit probe analysis. Correctness and gradient contracts are enforced with focused tests before any production edit.

**Tech Stack:** Python 3.10, PyTorch 1.12, MMEngine 0.10, MMDetection/MMRotate, pytest, CPU/Gloo for distributed tests.

---

### Task 1: Archive the invalid P1 evidence

**Files:**
- Create: `resultmd/exp_low_rank_orientation_evidence/gda_p1_invalid_diagnostic_20260904.md`
- Create: `/data1/zcy/Orbdet/work_dirs/formal/orbdet_gda_probe_dota1_grouped_ss_e0_gpu4_seed3407/INVALID_DIAGNOSTIC`

- [x] **Step 1: Write the immutable result note**

Record the exact launch and stop timestamps, final process position, retained
checkpoint, epoch-1-to-6 validation values, comparison confounds, and the two
demonstrated correctness defects. State that the run cannot support positive
or negative method claims.

- [x] **Step 2: Write the work-directory marker**

The marker must contain:

```text
status: invalid-diagnostic
stopped_at: 2026-09-04T09:26:35+08:00
last_log_position: epoch_7_iter_220
last_scheduled_checkpoint: epoch_4.pth
formal_resume_allowed: false
```

- [x] **Step 3: Verify the evidence against the preserved log**

Run:

```bash
rtk rg -n 'dota/mAP:|Epoch\(train\).*\[7\]\[ 220/5373\]' /data1/zcy/Orbdet/work_dirs/formal/orbdet_gda_probe_dota1_grouped_ss_e0_gpu4_seed3407/launch.log
```

Expected: six validation rows ending at `0.0756` and the epoch-7 iteration-220 row.

- [x] **Step 4: Commit only the repository result note**

```bash
rtk git add resultmd/exp_low_rank_orientation_evidence/gda_p1_invalid_diagnostic_20260904.md
rtk git commit -m "docs: invalidate confounded GDA P1 run"
```

### Task 2: Restore exact cross-view object identity

**Files:**
- Modify: `tests/test_gda_plan_b.py`
- Modify: `mmrotate/models/detectors/orbdet_gda.py:32-90`
- Modify: `mmrotate/models/detectors/orbdet_gda.py:110-126`

- [x] **Step 1: Replace B-T7 with production-identity regression tests**

Use real parent-style IDs (`1.2/2.2/3.2`, `1.4/2.4/3.4`,
`1.6/2.6/3.6`). Add a middle-missing case whose expected keys are `[1, 3]`
and whose second triplet is object 3 in every view. Add a multi-image case
whose keys remain globally unique.

```python
def test_b_t7_compacts_by_true_bid_identity_when_middle_object_is_missing():
    rows3, _, keys = compact_probe_by_object(rows_v, bids_v, extras_v)
    assert torch.equal(keys, torch.tensor([1, 3]))
    assert torch.equal(rows3[:, :, 0],
                       torch.tensor([[10., 100., 1000.],
                                     [30., 300., 3000.]]))
```

Add a detector helper test asserting rebuilt flip IDs are `[1.6, 2.6, ...]`
rather than continuing after ori/rot IDs.

- [x] **Step 2: Run the new tests and verify RED**

Run:

```bash
rtk env PYTHONNOUSERSITE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES='' /data/zcy/anaconda3/envs/orbdet/bin/python -s -m pytest tests/test_gda_plan_b.py -k 'b_t7 or flip_bid' -q
```

Expected: FAIL showing rank-based `C/B/C` pairing and incorrect flip integers.

- [x] **Step 3: Implement on-device key intersection and parent flip IDs**

Pool by `bids.long()` and retain sorted keys. Compute membership with
`torch.searchsorted`; gather each pooled tensor using the common true keys.
For an empty view, construct the empty pooled tensor from
`rows.sum(dim=0, keepdim=True)[:0]` so it stays connected to autograd.

Reset flip `offset = 1`, increment across images exactly like
`H2RBoxV2Detector.loss`, and allocate each `bid` on that image's bbox device.

- [x] **Step 4: Run B-T7 tests and the existing Plan-B suite**

Run:

```bash
rtk env PYTHONNOUSERSITE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES='' /data/zcy/anaconda3/envs/orbdet/bin/python -s -m pytest tests/test_gda_plan_b.py -q
```

Expected: all Plan-B tests pass.

- [x] **Step 5: Commit**

```bash
rtk git add mmrotate/models/detectors/orbdet_gda.py tests/test_gda_plan_b.py
rtk git commit -m "fix: preserve GDA cross-view object identity"
```

### Task 3: Make the probe representation finite in forward and backward

**Files:**
- Modify: `tests/test_gda_plan_b.py`
- Modify: `mmrotate/models/losses/orbdet_gda_probe_losses.py:70-105`

- [x] **Step 1: Add exact-zero, near-zero, and extreme-scale tests**

Test `rows[..., 2:4] == 0`, radii on both sides of the safety threshold, and
raw `t` values `[-100, -20, 0, 20, 100]`. Backpropagate the sum of all losses
and require every loss, Sigma, and row gradient to be finite.

```python
total = sum(losses.values())
total.backward()
assert all(torch.isfinite(v).all() for v in losses.values())
assert torch.isfinite(rows.grad).all()
```

- [x] **Step 2: Verify RED**

Run the new test alone. Expected: FAIL because zero `u2` has NaN gradients
and extreme positive `t` overflows.

- [x] **Step 3: Implement bounded `t` and safe `u2` decode**

Add constructor parameters `t_min=-6.0`, `t_max=14.0`, and
`u2_min_radius=1e-4`. Preserve local identity around zero with a two-sided
smooth bound:

```python
t = torch.where(raw_t >= 0,
                t_max * torch.tanh(raw_t / t_max),
                -t_min * torch.tanh(raw_t / (-t_min)))
r2 = u2x.square() + u2y.square()
valid = r2 >= u2_min_radius ** 2
safe_x = torch.where(valid, u2x, torch.ones_like(u2x))
safe_y = torch.where(valid, u2y, torch.zeros_like(u2y))
psi = torch.atan2(safe_y, safe_x)
```

Expose detached `raw_t_min`, `raw_t_max`, `u2_radius_mean`, and
`u2_fallback_frac` diagnostics from `forward`.

- [x] **Step 4: Verify GREEN and representation regressions**

Run `tests/test_gda_plan_b.py` and `tests/test_gda_gate_a.py`. Expected: all pass.

- [x] **Step 5: Commit**

```bash
rtk git add mmrotate/models/losses/orbdet_gda_probe_losses.py tests/test_gda_plan_b.py
rtk git commit -m "fix: make GDA probe decode numerically finite"
```

### Task 4: Correct loss masks, empty graphs, and DDP weighting

**Files:**
- Modify: `tests/test_gda_plan_b.py`
- Create: `tests/test_gda_ddp_reduction.py`
- Modify: `mmrotate/models/detectors/orbdet_gda.py:128-205`
- Modify: `mmrotate/models/losses/orbdet_gda_probe_losses.py:124-228`

- [x] **Step 1: Add loss-contract tests**

Add tests for an all-empty local `rows3` tensor that requires gradients, an
orientation-agnostic object that contributes zero to all GDA terms, and
unequal local object/bit counts.

The two-process CPU/Gloo test must compare DDP-averaged local gradients with a
single-process concatenated reference. One rank uses zero objects. Use a
temporary file rendezvous and `torch.multiprocessing.spawn`; no CUDA or NCCL.

- [x] **Step 2: Verify RED**

Expected failures: empty losses are disconnected, agnostic objects contribute,
and rank-local means disagree with the global reference.

- [x] **Step 3: Pass object masks from detector to loss**

Append an `is_orientation_agnostic` extra per positive point using the head's
existing `_get_rotation_agnostic_mask`. After identity compaction, form one
object mask and pass it as `valid_object_mask` to `OrbdetGDAProbeLoss`.

- [x] **Step 4: Implement DDP-aware sum/count reduction**

Use `mmdet.utils.reduce_mean` on detached denominators. The differentiable
formula is:

```python
def _ddp_weighted_mean(local_sum, local_count):
    mean_count = reduce_mean(local_count.detach().to(local_sum))
    return local_sum / mean_count.clamp_min(1.0)
```

DDP subsequently averages rank gradients, yielding a true global item mean.
Object terms and cross-view bit use valid-object count; anchored bit uses its
own valid element count. Eliminate Python `bool(mask.any())` and
`int(mask.sum())` from the CUDA path by flattening all view logits/masks.

Add `rows3.sum() * 0` to every local numerator so all probe outputs remain in
the graph on an empty rank. All ranks execute identical count collectives.

- [x] **Step 5: Verify GREEN**

Run:

```bash
rtk env PYTHONNOUSERSITE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES='' /data/zcy/anaconda3/envs/orbdet/bin/python -s -m pytest tests/test_gda_plan_b.py tests/test_gda_ddp_reduction.py -q
```

Expected: all tests pass without hangs.

- [x] **Step 6: Commit**

```bash
rtk git add mmrotate/models/detectors/orbdet_gda.py mmrotate/models/losses/orbdet_gda_probe_losses.py tests/test_gda_plan_b.py tests/test_gda_ddp_reduction.py
rtk git commit -m "fix: normalize GDA losses across distributed objects"
```

### Task 5: Separate ordinary prediction from explicit probe analysis

**Files:**
- Modify: `tests/test_gda_plan_b.py`
- Modify: `mmrotate/models/dense_heads/h2rbox_gda_head.py:49-146`

- [x] **Step 1: Add inference-spy and initialization-parity tests**

Patch `gda_probe_tower.forward` with a counting wrapper. In `eval()` call the
ordinary head forward and assert count zero and parent outputs bit-identical.
Then call `forward_gda_probe(feats)` explicitly and assert one tower call per
feature level.

Reset the same seed before constructing a parent and child head without
loading a state dict; require every shared state tensor to be equal.

- [x] **Step 2: Verify RED**

Expected: ordinary eval executes the probe and same-seed shared initialization
differs.

- [x] **Step 3: Preserve RNG and add the explicit analysis method**

Construct probe modules inside `torch.random.fork_rng(devices=[])` so adding
the branch does not advance the shared initialization stream. Only execute the
probe inside `forward_single` while training.

Add:

```python
def forward_gda_probe(self, x: Tuple[Tensor]) -> List[Tensor]:
    if not self.gda_enabled:
        return []
    outputs = [self._forward_gda_probe_single(feat) for feat in x]
    self._gda_probe_out = outputs
    return outputs
```

This method is the only eval-time evidence path.

- [x] **Step 4: Verify GREEN and commit**

Run `tests/test_gda_plan_b.py`, then commit the head and tests with message
`fix: isolate GDA probe analysis from prediction`.

### Task 6: Create honest control configurations

**Files:**
- Create: `configs/orbdet/orbdet_v0_2_r50_dota1_grouped_ss_e0_gpu4_control.py`
- Modify: `configs/orbdet/orbdet_gda_probe_r50_dota1_grouped_ss_e0_gpu4.py`
- Create: `tests/test_gda_control_contract.py`

- [x] **Step 1: Add a failing configuration contract**

Load the E0, single-GPU control, and GDA configs with `mmengine.Config`.
Require batch four for both single-GPU arms, identical optimizer/scheduler/data
settings, and equal updates per epoch to the two-GPU E0 contract. Permit only
model type/probe config and work directory to differ between control and GDA.

- [x] **Step 2: Verify RED because the control is absent and GDA uses batch two**

- [x] **Step 3: Add the control and make GDA inherit it**

The control inherits E0, overrides `train_dataloader.batch_size=4`, records
`intended_world_size=1`, `global_batch_size=4`, and a distinct work directory.
The GDA config inherits the control and changes only detector/head/probe fields
and work directory. Do not change loss weights.

- [x] **Step 4: Verify GREEN and commit**

Run `tests/test_gda_control_contract.py`, then commit configs and test with
message `test: freeze honest GDA single-GPU controls`.

### Task 7: Final verification and handoff

**Files:**
- Modify: `resultmd/exp_low_rank_orientation_evidence/gda_hardening_20260904/progress.md`
- Modify: `resultmd/exp_low_rank_orientation_evidence/gda_hardening_20260904/findings.md`

- [x] **Step 1: Run the complete focused suite**

```bash
rtk env PYTHONNOUSERSITE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES='' /data/zcy/anaconda3/envs/orbdet/bin/python -s -m pytest tests/test_gda_gate_a.py tests/test_gda_plan_b.py tests/test_gda_ddp_reduction.py tests/test_gda_control_contract.py -q
```

- [x] **Step 2: Run the existing relevant regression suite**

```bash
rtk env PYTHONNOUSERSITE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES='' /data/zcy/anaconda3/envs/orbdet/bin/python -s -m pytest tests/test_low_rank_orientation_evidence.py tests/test_collect_low_rank_orientation_evidence.py tests/test_report_low_rank_orientation_evidence.py tests/test_eval_filtered_orientation_map.py -q
```

- [x] **Step 3: Run static and repository checks**

```bash
rtk git diff --check
rtk git status --short
```

Expected: no whitespace errors; unrelated pre-existing untracked planning
files remain untouched and are listed separately from task changes.

- [x] **Step 4: Inspect the live system without starting training**

```bash
rtk ps aux | rtk rg 'orbdet_gda_probe|tools/train.py' | rtk rg -v 'rg '
rtk nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader
```

Expected: no GDA P1 process. Other users' processes, if any, are not modified.

- [x] **Step 5: Update task records and commit**

Record exact test counts and remaining risks. Commit only this task's
`gda_hardening_20260904` records with message
`docs: record GDA correctness hardening verification`.

- [x] **Step 6: Report residual limits**

State explicitly that no GPU memory smoke, gradient-calibration experiment, or
formal training ran. Request separate authorization before any such run.
