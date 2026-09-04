# SCQO le90 Diagnostic Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make SCQO angle-error labels invariant to the equivalent rotated-box parameterizations `(w, h, theta)` and `(h, w, theta + pi/2)`.

**Architecture:** Keep rotated-IoU matching on the untouched prediction and GT boxes. Immediately after matching, clone both box tensors, regularize the clones with MMRotate's existing `RotatedBoxes.regularize_boxes('le90')`, and compute/log both angle errors from those canonical copies. The model, prediction path, evidence adapter, source samples, matching, JSON field set, and training code remain unchanged.

**Tech Stack:** Python 3.10, PyTorch, MMRotate `RotatedBoxes`, pytest, existing frozen-checkpoint SCQO collector/reporter.

---

### Task 1: Reproduce the width-height gauge bug

**Files:**
- Modify: `tests/test_scqo_plan_a_tools.py`
- Test: `tests/test_scqo_plan_a_tools.py`

- [x] **Step 1: Write the failing regression test**

Add a test that creates two geometrically identical boxes represented as
`(w=4, h=8, theta=0)` and `(w=8, h=4, theta=pi/2)`, calls the real
`collector._match_by_gt`, and asserts:

```python
assert match['rotated_iou'] == pytest.approx(1.0)
assert match['angle_error_deg'] == pytest.approx(0.0, abs=1e-5)
assert match['angle_error_c4_deg'] == pytest.approx(0.0, abs=1e-5)
assert match['pred_angle'] == pytest.approx(match['gt_angle'], abs=1e-5)
```

Also clone the input tensors before the call and assert that the collector did
not mutate either input box container.

- [x] **Step 2: Run the new test and verify RED**

Run:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/Orbdet/.worktrees/scqo-plan-a \
/data/zcy/anaconda3/envs/orbdet/bin/python -m pytest -q \
tests/test_scqo_plan_a_tools.py::test_match_by_gt_regularizes_equivalent_boxes_to_le90
```

Expected: FAIL because the current collector reports approximately 90 degrees
for the pi-periodic error even though rotated IoU is 1.

### Task 2: Canonicalize diagnostic angle inputs only

**Files:**
- Modify: `tools/analysis_tools/scqo_collect_evidence.py`
- Test: `tests/test_scqo_plan_a_tools.py`

- [x] **Step 1: Reuse MMRotate's canonical box implementation**

Import `RotatedBoxes` from `mmrotate.structures` and add the focused helper:

```python
def _regularize_le90(boxes: torch.Tensor) -> torch.Tensor:
    return RotatedBoxes(boxes.clone()).regularize_boxes('le90')
```

The clone is mandatory because frozen input samples and model predictions must
remain untouched.

- [x] **Step 2: Compute logged angles from canonical copies**

In `_match_by_gt`, keep `match_rotated_predictions` on the original tensors.
After matching, create canonical prediction and GT tensors once, then take
`pred_angle`, `gt_angle`, `angle_error_deg`, and `angle_error_c4_deg` from those
canonical tensors. Do not change match indices, scores, IoU, evidence, or row
geometry.

- [x] **Step 3: Run the focused test and verify GREEN**

Run the Task 1 command again.

Expected: PASS, with raw input tensors unchanged.

- [x] **Step 4: Run collector and diagnostics regression tests**

Run:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/Orbdet/.worktrees/scqo-plan-a \
/data/zcy/anaconda3/envs/orbdet/bin/python -m pytest -q \
tests/test_scqo_plan_a_tools.py tests/test_scqo_diagnostics.py
```

Expected: all tests pass. Existing square-box C4 behavior remains unchanged.

### Task 3: Verify the corrected scientific label

**Files:**
- Create: `resultmd/exp_scqo_plan_a_hrsc_le90_audit_20260817/` artifacts through the existing immutable CLIs
- Preserve: `resultmd/exp_scqo_plan_a_hrsc_audit_20260817/`

- [x] **Step 1: Run the complete SCQO unit suite**

Run all five `tests/test_scqo_*.py` files in the Orbdet conda environment with
`CUDA_VISIBLE_DEVICES=`.

Expected: zero failures.

- [x] **Step 2: Run a two-image frozen-checkpoint vertical slice**

Collect v0.2 evidence with `--max-images 2` into a new temporary debug result
directory. Verify that equivalent width-height box parameterizations no longer
produce a 90-degree label.

- [x] **Step 3: Run the full frozen v0.2 and GODC audits**

Use the existing best checkpoints and exact frozen HRSC validation configs.
Publish to the new immutable `exp_scqo_plan_a_hrsc_le90_audit_20260817`
directory. Never overwrite or delete the previous audit.

- [x] **Step 4: Generate and inspect the new report**

Run `scqo_report_evidence.py` on the corrected JSONL files. Verify hashes,
181-image counts, 541 rows per run, match counts, eligible rows, class balance,
and Gate B. `INSUFFICIENT` or `FAIL` is a valid scientific outcome.

- [x] **Step 5: Commit**

Stage only the plan, regression test, collector fix, and corrected evidence
artifacts. Run `git diff --check`, commit with a message that names le90
canonicalization, and verify a clean `feat/scqo-plan-a` worktree.
