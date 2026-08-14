# Group-Orbit Determinantal Cluster Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a tested, registry-buildable group-orbit determinantal loss and exact finite planar group-orbit builder without changing Orbdet-v0.2 training behavior.

**Architecture:** A standalone utility constructs exact `C2`, `C4`, `D1`, and `D2` feature orbits with tensor permutations. A registered loss flattens any precomputed orbit, forms a small float32 Gram matrix, computes determinantal, spectral-tail, fixed-space, guard, and stability quantities, and reduces the per-instance objective using MMEngine conventions. The module is exported but not connected to any detector or config.

**Tech Stack:** Python 3.10, PyTorch 1.12, MMEngine/MMRotate registry, pytest, YAPF/isort conventions.

---

## File map

| File | Responsibility |
|---|---|
| `tests/test_group_orbit_determinantal_cluster.py` | Analytic group, loss, guard, gradient, precision, and registry contracts |
| `mmrotate/models/losses/group_orbit_determinantal_cluster_loss.py` | Exact orbit builder and registered GODC loss |
| `mmrotate/models/losses/__init__.py` | Public package export |
| `docs/future_work/inspirations/2026-08-15-group-orbit-determinantal-cluster.md` | Research provenance and future integration boundary |

### Task 1: Exact finite planar group orbits

**Files:**

- Create: `tests/test_group_orbit_determinantal_cluster.py`
- Create: `mmrotate/models/losses/group_orbit_determinantal_cluster_loss.py`

- [ ] **Step 1: Write failing group-builder tests**

Create tests that import the absent utility and assert exact element ordering:

```python
import pytest
import torch

from mmrotate.models.losses.group_orbit_determinantal_cluster_loss import (
    build_planar_group_orbit)


def test_planar_group_orbits_have_exact_elements():
    x = torch.arange(1, 10, dtype=torch.float32).reshape(1, 1, 3, 3)

    c2 = build_planar_group_orbit(x, 'c2')
    assert torch.equal(c2[:, 0], x)
    assert torch.equal(c2[:, 1], torch.rot90(x, 2, (-2, -1)))

    c4 = build_planar_group_orbit(x, 'c4')
    assert c4.shape == (1, 4, 1, 3, 3)
    for index in range(4):
        assert torch.equal(c4[:, index],
                           torch.rot90(x, index, (-2, -1)))

    d1 = build_planar_group_orbit(x, 'd1')
    assert torch.equal(d1[:, 0], x)
    assert torch.equal(d1[:, 1], torch.flip(x, (-1, )))

    d2 = build_planar_group_orbit(x, 'd2')
    expected = (x, torch.rot90(x, 2, (-2, -1)),
                torch.flip(x, (-1, )), torch.flip(x, (-2, )))
    for index, element in enumerate(expected):
        assert torch.equal(d2[:, index], element)


@pytest.mark.parametrize('group', ['c1', 'unknown'])
def test_planar_group_orbit_rejects_trivial_or_unknown_groups(group):
    with pytest.raises(ValueError):
        build_planar_group_orbit(torch.ones(1, 1, 3, 3), group)


def test_c4_requires_square_support():
    with pytest.raises(ValueError, match='square'):
        build_planar_group_orbit(torch.ones(1, 1, 2, 3), 'c4')
```

- [ ] **Step 2: Run the focused test and observe RED**

Run:

```bash
rtk env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
  /data/zcy/anaconda3/envs/orbdet/bin/python -m pytest -q \
  -p no:cacheprovider tests/test_group_orbit_determinantal_cluster.py
```

Expected: collection fails with `ModuleNotFoundError` for
`group_orbit_determinantal_cluster_loss`.

- [ ] **Step 3: Implement the exact builder**

Create the module with validation and exact transforms:

```python
from typing import Dict, Tuple

import torch
from torch import Tensor


def build_planar_group_orbit(features: Tensor, group: str) -> Tensor:
    if not isinstance(features, Tensor):
        raise TypeError('features must be a torch.Tensor')
    if features.ndim != 4:
        raise ValueError('features must have shape [N, C, H, W]')
    if features.shape[-2] == 0 or features.shape[-1] == 0:
        raise ValueError('features must have non-empty spatial dimensions')
    if not isinstance(group, str):
        raise TypeError('group must be a string')

    group = group.lower()
    if group == 'c2':
        elements = (features, torch.rot90(features, 2, (-2, -1)))
    elif group == 'c4':
        if features.shape[-2] != features.shape[-1]:
            raise ValueError('c4 requires square spatial support')
        elements = tuple(
            torch.rot90(features, k, (-2, -1)) for k in range(4))
    elif group == 'd1':
        elements = (features, torch.flip(features, (-1, )))
    elif group == 'd2':
        elements = (features, torch.rot90(features, 2, (-2, -1)),
                    torch.flip(features, (-1, )),
                    torch.flip(features, (-2, )))
    elif group == 'c1':
        raise ValueError('the trivial group c1 is not a valid hypothesis')
    else:
        raise ValueError(f'unsupported planar group: {group!r}')
    return torch.stack(elements, dim=1)
```

- [ ] **Step 4: Run the builder tests and observe GREEN**

Run the Task 1 pytest command again. Expected: builder tests pass; later loss
imports may remain absent until Task 2.

- [ ] **Step 5: Commit the builder slice**

```bash
rtk git add tests/test_group_orbit_determinantal_cluster.py \
  mmrotate/models/losses/group_orbit_determinantal_cluster_loss.py
rtk git commit -m "feat: add exact planar group orbit builder"
```

### Task 2: Gram, determinantal, spectral, and fixed-space core

**Files:**

- Modify: `tests/test_group_orbit_determinantal_cluster.py`
- Modify: `mmrotate/models/losses/group_orbit_determinantal_cluster_loss.py`

- [ ] **Step 1: Add failing analytic loss tests**

Add imports and tests for invariant, asymmetric, and scaled-rank-one orbits:

```python
from mmrotate.models.losses.group_orbit_determinantal_cluster_loss import (
    GroupOrbitDeterminantalClusterLoss, build_planar_group_orbit)


def _loss_without_guards(**kwargs):
    return GroupOrbitDeterminantalClusterLoss(
        energy_guard_weight=0.0,
        variance_guard_weight=0.0,
        **kwargs)


def test_invariant_orbit_has_zero_rank_and_fixed_residuals():
    base = torch.tensor([[[[1., 2., 2., 1.],
                           [3., 4., 4., 3.]]]])
    orbit = build_planar_group_orbit(base, 'd1')
    loss_fn = _loss_without_guards(spectral_tail_weight=1.0)
    value = loss_fn(orbit)

    assert float(value) < 1e-6
    assert float(loss_fn.last_determinantal) < 1e-6
    assert float(loss_fn.last_spectral_tail) < 1e-6
    assert float(loss_fn.last_fixed_space) < 1e-6


def test_asymmetric_orbit_costs_more_than_symmetric_orbit():
    symmetric = torch.tensor([[[[1., 2., 2., 1.],
                                [3., 4., 4., 3.]]]])
    asymmetric = torch.tensor([[[[1., 2., 3., 4.],
                                 [2., 5., 7., 11.]]]])
    loss_fn = _loss_without_guards()

    symmetric_loss = loss_fn(build_planar_group_orbit(symmetric, 'd1'))
    asymmetric_loss = loss_fn(build_planar_group_orbit(asymmetric, 'd1'))
    assert float(asymmetric_loss) > float(symmetric_loss) + 1e-4


def test_fixed_space_rejects_scaled_rank_one_orbit():
    base = torch.tensor([[1., 2., 4., 8.]])
    orbit = torch.stack((base, 2.0 * base), dim=1)
    loss_fn = _loss_without_guards(
        determinantal_weight=1.0,
        spectral_tail_weight=0.0,
        fixed_space_weight=1.0)

    loss_fn(orbit)
    assert float(loss_fn.last_determinantal) < 1e-6
    assert float(loss_fn.last_fixed_space) > 1e-3
```

- [ ] **Step 2: Run only the new tests and observe RED**

Run:

```bash
rtk env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
  /data/zcy/anaconda3/envs/orbdet/bin/python -m pytest -q \
  -p no:cacheprovider tests/test_group_orbit_determinantal_cluster.py
```

Expected: import or construction fails because the loss class is absent.

- [ ] **Step 3: Implement the registered mathematical core**

Add `@MODELS.register_module()` class
`GroupOrbitDeterminantalClusterLoss(torch.nn.Module)` with constructor weights,
positive `eps`, positive energy/variance floors, `reduction`, and detached
diagnostic tensors. Its per-sample computation must use:

```python
matrix = masked_orbit.reshape(batch_size, group_order, -1)
gram = matrix @ matrix.transpose(-1, -2)
trace = gram.diagonal(dim1=-2, dim2=-1).sum(-1)
trace_g2 = gram.square().sum(dim=(-2, -1))
determinantal = (
    (trace.square() - trace_g2).clamp_min(0.0) /
    (2.0 * trace.square() + self.eps))
eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(0.0)
spectral_tail = (
    1.0 - eigenvalues[:, -1] / (trace + self.eps)).clamp_min(0.0)
orbit_mean = matrix.mean(dim=1, keepdim=True)
fixed_space = (matrix - orbit_mean).square().sum(dim=(1, 2)) / (
    matrix.square().sum(dim=(1, 2)) + self.eps)
sigma_1 = eigenvalues[:, -1].sqrt()
sigma_2 = eigenvalues[:, -2].sqrt()
q_gap = (sigma_1 - sigma_2) / (sigma_1 + self.eps)
```

Promote float16/bfloat16 inputs to float32 before Gram construction. Reject
non-finite input and any orbit with `K < 2` or an empty flattened dimension.

- [ ] **Step 4: Run analytic tests and observe GREEN**

Run the Task 2 pytest command. Expected: all current tests pass.

- [ ] **Step 5: Commit the mathematical core**

```bash
rtk git add tests/test_group_orbit_determinantal_cluster.py \
  mmrotate/models/losses/group_orbit_determinantal_cluster_loss.py
rtk git commit -m "feat: add group orbit determinantal loss"
```

### Task 3: Degeneracy guards, masking, reduction, gradients, and registry

**Files:**

- Modify: `tests/test_group_orbit_determinantal_cluster.py`
- Modify: `mmrotate/models/losses/group_orbit_determinantal_cluster_loss.py`
- Modify: `mmrotate/models/losses/__init__.py`

- [ ] **Step 1: Add failing robustness and integration tests**

Cover the remaining contracts with concrete assertions:

```python
def test_zero_constant_and_empty_support_trigger_guards():
    loss_fn = GroupOrbitDeterminantalClusterLoss(
        determinantal_weight=0.0,
        fixed_space_weight=0.0,
        spectral_tail_weight=0.0,
        min_energy=0.5,
        min_variance=0.5)
    zero = torch.zeros(1, 2, 1, 2, 2)
    constant = torch.ones(1, 2, 1, 2, 2)

    assert float(loss_fn(zero)) > 0.0
    assert float(loss_fn.last_energy_guard) > 0.0
    assert float(loss_fn(constant)) > 0.0
    assert float(loss_fn.last_variance_guard) > 0.0
    assert float(loss_fn(constant, support_mask=torch.zeros(1, 2, 2))) > 0.0


def test_reduction_weights_mask_and_empty_batch_are_finite():
    orbit = torch.randn(2, 2, 3, 4, 4)
    mask = torch.ones(2, 4, 4)
    none_fn = _loss_without_guards(reduction='none')
    values = none_fn(orbit, support_mask=mask,
                     weight=torch.tensor([1.0, 0.0]))
    assert values.shape == (2, )
    assert values[1] == 0

    empty = orbit[:0]
    mean_fn = _loss_without_guards(reduction='mean')
    assert mean_fn(empty).shape == ()
    assert torch.isfinite(mean_fn(empty))


def test_asymmetric_orbit_has_finite_nonzero_gradient():
    features = torch.randn(2, 3, 5, 5, requires_grad=True)
    loss_fn = _loss_without_guards()
    loss = loss_fn(build_planar_group_orbit(features, 'd1'))
    loss.backward()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()
    assert float(features.grad.abs().sum()) > 0.0


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_low_precision_input_uses_finite_float32_gram(dtype):
    orbit = torch.randn(2, 2, 3, 4, 4).to(dtype)
    value = _loss_without_guards()(orbit)
    assert value.dtype == torch.float32
    assert torch.isfinite(value)


def test_loss_builds_from_mmrotate_registry():
    from mmrotate.registry import MODELS
    from mmrotate.utils import register_all_modules

    register_all_modules()
    module = MODELS.build(dict(type='GroupOrbitDeterminantalClusterLoss'))
    assert isinstance(module, GroupOrbitDeterminantalClusterLoss)
```

- [ ] **Step 2: Run the robustness tests and observe RED**

Run the full focused test file. Expected: failures locate missing masking,
guards, reductions, low-precision promotion, or package export.

- [ ] **Step 3: Complete the robustness contract**

Implement shared-support normalization by inserting the group axis after the
batch axis and leading singleton feature axes until the mask broadcasts to the
orbit. Validate finite nonnegative masks and weights. Compute:

```python
energy = matrix.square().mean(dim=(1, 2))
variance = matrix.var(dim=-1, unbiased=False).mean(dim=1)
energy_guard = (self.min_energy - energy).clamp_min(0.0)
variance_guard = (self.min_variance - variance).clamp_min(0.0)
```

Combine configured component weights, apply sample weights, then implement
`none`, `sum`, and `mean`; when `avg_factor` is supplied, divide the weighted
sum by a strictly positive factor. Reset all diagnostics to finite zeros on an
empty batch.

Export both public names:

```python
from .group_orbit_determinantal_cluster_loss import (
    GroupOrbitDeterminantalClusterLoss, build_planar_group_orbit)
```

and append them to `__all__`.

- [ ] **Step 4: Run focused tests and observe GREEN**

Run:

```bash
rtk env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
  /data/zcy/anaconda3/envs/orbdet/bin/python -m pytest -q \
  -p no:cacheprovider tests/test_group_orbit_determinantal_cluster.py
```

Expected: all GODC tests pass on CPU.

- [ ] **Step 5: Commit the robust public module**

```bash
rtk git add mmrotate/models/losses/__init__.py \
  mmrotate/models/losses/group_orbit_determinantal_cluster_loss.py \
  tests/test_group_orbit_determinantal_cluster.py
rtk git commit -m "test: harden group orbit loss contracts"
```

### Task 4: Regression, formatting, and documentation verification

**Files:**

- Modify only if checks reveal a scoped issue:
  `mmrotate/models/losses/group_orbit_determinantal_cluster_loss.py`
- Modify only if checks reveal a scoped issue:
  `tests/test_group_orbit_determinantal_cluster.py`

- [ ] **Step 1: Run syntax and formatting checks**

```bash
rtk env PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
  /data/zcy/anaconda3/envs/orbdet/bin/python -m py_compile \
  mmrotate/models/losses/group_orbit_determinantal_cluster_loss.py \
  tests/test_group_orbit_determinantal_cluster.py
rtk /data/zcy/anaconda3/envs/orbdet/bin/python -m yapf --diff \
  mmrotate/models/losses/group_orbit_determinantal_cluster_loss.py \
  tests/test_group_orbit_determinantal_cluster.py
```

Expected: syntax succeeds and YAPF prints no diff.

- [ ] **Step 2: Run focused and Orbdet regression tests**

```bash
rtk env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
  /data/zcy/anaconda3/envs/orbdet/bin/python -m pytest -q \
  -p no:cacheprovider \
  tests/test_group_orbit_determinantal_cluster.py \
  tests/test_orbdet_v0_1.py tests/test_orbdet_v0_2.py
```

Expected: all selected tests pass; no GPU is allocated and no launcher runs.

- [ ] **Step 3: Verify the training safety boundary and Git diff**

```bash
rtk git diff -- FORMAL_TRAINING_NOT_STARTED.md configs scripts
rtk git status --short --branch
rtk git diff --check
```

Expected: no diff in the formal-training marker, configs, or scripts; only the
planned module, export, test, and plan are changed or committed.

- [ ] **Step 4: Commit any verification-only corrections**

If formatting or a scoped defect required edits, stage only the files listed
in this task and commit:

```bash
rtk git add mmrotate/models/losses/group_orbit_determinantal_cluster_loss.py \
  tests/test_group_orbit_determinantal_cluster.py
rtk git commit -m "style: finalize group orbit loss"
```

If no correction was needed, do not create an empty commit.

- [ ] **Step 5: Record final evidence**

Capture the branch, commit IDs, exact passing test count, and the explicit fact
that no formal training was started in the handoff response.
