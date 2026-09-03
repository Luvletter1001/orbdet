# Low-Rank Orientation Evidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a pure PyTorch diagnostic that estimates a pi-periodic visual axis and low-rank channel-consensus evidence from ROI features.

**Architecture:** A new loss-adjacent functional module consumes `[N, C, H, W]` features. It uses centered spatial gradients to build one p=2 axial vector per channel, obtains the dominant phase and singular spectrum of the resulting `[C, 2]` matrix, and returns tensors without detector integration. Focused synthetic tests lock down orientation convention, ambiguity, rejection, and autograd.

**Tech Stack:** Python, PyTorch 1.12, pytest, MMRotate repository layout.

---

## File structure

- Create: `mmrotate/models/losses/low_rank_orientation_evidence.py` — pure evidence function and input validation.
- Modify: `mmrotate/models/losses/__init__.py` — export the function for reuse by a future detached ROI adapter.
- Create: `tests/test_low_rank_orientation_evidence.py` — synthetic functional tests.

### Task 1: Define the public behavior with failing tests

**Files:**
- Create: `tests/test_low_rank_orientation_evidence.py`

- [ ] **Step 1: Write failing tests for visual-axis decoding and low-rank evidence**

```python
import math

import torch

from mmrotate.models.losses.low_rank_orientation_evidence import (
    low_rank_channel_orientation_evidence)


def _axis_delta(prediction, target):
    return torch.abs(torch.remainder(prediction - target + math.pi / 2,
                                     math.pi) - math.pi / 2)


def _linear_texture(angle, channels=(1.0, 2.0, 3.0), size=17):
    coordinates = torch.linspace(-1.0, 1.0, size)
    yy, xx = torch.meshgrid(coordinates, coordinates, indexing='ij')
    normal = angle - math.pi / 2
    plane = xx * math.cos(normal) + yy * math.sin(normal)
    return torch.stack([amplitude * plane for amplitude in channels]).unsqueeze(0)


def test_decodes_tangent_axis_from_rank_one_channel_evidence():
    output = low_rank_channel_orientation_evidence(_linear_texture(0.35))
    assert float(_axis_delta(output['axis_angle'], torch.tensor([0.35]))) < 1e-4
    assert float(output['confidence']) > 0.999
    assert float(output['anisotropy']) > 0.999
    assert bool(output['valid'])


def test_orthogonal_channel_populations_are_marked_ambiguous():
    features = _linear_texture(0.0, channels=(1.0,))
    features = torch.cat([features, _linear_texture(math.pi / 2, channels=(1.0,))], dim=1)
    output = low_rank_channel_orientation_evidence(features, min_anisotropy=0.1)
    assert float(output['anisotropy']) < 1e-5
    assert not bool(output['valid'])


def test_constant_features_are_finite_and_invalid():
    output = low_rank_channel_orientation_evidence(torch.ones(2, 3, 9, 9))
    assert torch.isfinite(output['axis_angle']).all()
    assert torch.isfinite(output['confidence']).all()
    assert torch.equal(output['valid'], torch.zeros(2, dtype=torch.bool))


def test_result_preserves_autograd_connectivity():
    features = torch.randn(2, 4, 9, 9, requires_grad=True)
    output = low_rank_channel_orientation_evidence(features)
    (output['axis_angle'].sum() + output['confidence'].sum() + output['anisotropy'].sum()).backward()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()
```

- [ ] **Step 2: Run the test file and verify import failure**

Run: `PYTHONNOUSERSITE=1 /data/zcy/anaconda3/envs/orbdet/bin/python -m pytest tests/test_low_rank_orientation_evidence.py -q`

Expected: collection fails because `mmrotate.models.losses.low_rank_orientation_evidence` does not exist.

- [ ] **Step 3: Commit the failing test**

```bash
git add tests/test_low_rank_orientation_evidence.py
git commit -m "test: specify low-rank orientation evidence"
```

### Task 2: Implement the evidence primitive

**Files:**
- Create: `mmrotate/models/losses/low_rank_orientation_evidence.py`
- Modify: `mmrotate/models/losses/__init__.py`
- Test: `tests/test_low_rank_orientation_evidence.py`

- [ ] **Step 1: Implement the exact public function**

```python
import math
from typing import Dict

import torch
from torch import Tensor


def low_rank_channel_orientation_evidence(
        features: Tensor,
        min_energy: float = 1e-6,
        min_anisotropy: float = 0.1,
        eps: float = 1e-6) -> Dict[str, Tensor]:
    if not isinstance(features, Tensor):
        raise TypeError('features must be a torch.Tensor')
    if features.ndim != 4:
        raise ValueError('features must have shape [N, C, H, W]')
    if features.shape[1] < 1 or features.shape[-2] < 3 or features.shape[-1] < 3:
        raise ValueError('features require C >= 1 and H, W >= 3')
    if not features.is_floating_point():
        raise TypeError('features must use a floating-point dtype')
    if not bool(torch.isfinite(features).all()):
        raise ValueError('features must be finite')
    if not math.isfinite(min_energy) or min_energy < 0.0:
        raise ValueError('min_energy must be finite and non-negative')
    if not math.isfinite(min_anisotropy) or not 0.0 <= min_anisotropy <= 1.0:
        raise ValueError('min_anisotropy must be in [0, 1]')
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError('eps must be finite and positive')

    work = features.float() if features.dtype in (torch.float16, torch.bfloat16) else features
    gx = 0.5 * (work[:, :, 1:-1, 2:] - work[:, :, 1:-1, :-2])
    gy = 0.5 * (work[:, :, 2:, 1:-1] - work[:, :, :-2, 1:-1])
    height, width = gx.shape[-2:]
    wy = torch.hann_window(height + 2, periodic=False, device=work.device, dtype=work.dtype)[1:-1]
    wx = torch.hann_window(width + 2, periodic=False, device=work.device, dtype=work.dtype)[1:-1]
    support = wy[:, None] * wx[None, :]
    axial_x = (support * (gx.square() - gy.square())).sum(dim=(-2, -1))
    axial_y = (support * (2.0 * gx * gy)).sum(dim=(-2, -1))
    moments = torch.stack((axial_x, axial_y), dim=-1)
    mean_moment = moments.mean(dim=1)
    _, singular_values, vh = torch.linalg.svd(moments, full_matrices=False)
    direction = vh[:, 0, :]
    sign = torch.where(
        (direction * mean_moment).sum(dim=-1, keepdim=True) < 0.0,
        -torch.ones_like(direction[:, :1]), torch.ones_like(direction[:, :1]))
    direction = direction * sign
    gradient_phase = 0.5 * torch.atan2(direction[:, 1], direction[:, 0])
    axis_angle = torch.remainder(gradient_phase + math.pi, math.pi) - math.pi / 2
    sigma1 = singular_values[:, 0]
    sigma2 = singular_values[:, 1] if singular_values.shape[1] == 2 else torch.zeros_like(sigma1)
    confidence = sigma1 / (sigma1 + sigma2 + eps)
    anisotropy = mean_moment.norm(dim=-1) / (moments.norm(dim=-1).mean(dim=1) + eps)
    energy = ((gx.square() + gy.square()) * support).sum(dim=(-2, -1)).mean(dim=1) / (support.sum() + eps)
    valid = (energy >= min_energy) & (anisotropy >= min_anisotropy)
    return dict(
        axis_angle=axis_angle,
        direction=direction,
        singular_values=singular_values,
        confidence=confidence,
        anisotropy=anisotropy,
        energy=energy,
        valid=valid)
```

The angle convention is:

```python
gradient_phase = 0.5 * torch.atan2(direction[..., 1], direction[..., 0])
axis_angle = torch.remainder(
    gradient_phase + math.pi / 2 + math.pi / 2, math.pi) - math.pi / 2
```

The second `+ pi / 2` in the remainder expression is its wrapping offset; the
first rotates the gradient normal into the texture tangent.

- [ ] **Step 2: Export the function**

```python
from .low_rank_orientation_evidence import low_rank_channel_orientation_evidence

# In the existing __all__ list, insert the string immediately after
# 'HBoxFPNGroupOrbitLoss':
# 'low_rank_channel_orientation_evidence',
```

Follow the existing explicit export-list style in `mmrotate/models/losses/__init__.py`; do not register a loss class because this increment has no optimizer-facing objective.

- [ ] **Step 3: Run the focused tests and verify they pass**

Run: `PYTHONNOUSERSITE=1 /data/zcy/anaconda3/envs/orbdet/bin/python -m pytest tests/test_low_rank_orientation_evidence.py -q`

Expected: `4 passed`.

- [ ] **Step 4: Run the neighboring GODC regression tests**

Run: `PYTHONNOUSERSITE=1 /data/zcy/anaconda3/envs/orbdet/bin/python -m pytest tests/test_group_orbit_determinantal_cluster.py tests/test_hbox_fpn_group_orbit_loss.py tests/test_orbdet_godc.py -q`

Expected: all selected tests pass, proving that the new diagnostic does not alter the legacy GODC path.

- [ ] **Step 5: Commit the implementation**

```bash
git add mmrotate/models/losses/low_rank_orientation_evidence.py mmrotate/models/losses/__init__.py tests/test_low_rank_orientation_evidence.py
git commit -m "feat: add low-rank orientation evidence"
```

### Task 3: Verify package integrity

**Files:**
- Modify: none
- Test: `tests/test_low_rank_orientation_evidence.py`

- [ ] **Step 1: Compile the changed Python modules**

Run: `PYTHONNOUSERSITE=1 /data/zcy/anaconda3/envs/orbdet/bin/python -m py_compile mmrotate/models/losses/low_rank_orientation_evidence.py tests/test_low_rank_orientation_evidence.py`

Expected: exits with status 0 and produces no output.

- [ ] **Step 2: Inspect the final diff and worktree**

Run: `git diff HEAD~1..HEAD --check && git status --short`

Expected: no whitespace errors and a clean worktree.
