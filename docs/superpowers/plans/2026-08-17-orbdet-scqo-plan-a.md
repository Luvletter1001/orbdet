# Orbdet SCQO Plan A Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the read-only SCQO mathematical and diagnostic stack that can decide whether existing Orbdet-v0.2/GODC features contain instance-level orientation-identifiability evidence, without changing model predictions or starting training.

**Architecture:** Refactor the existing GODC loss to expose per-instance statistics, add exact \(O(2)\) harmonic primitives, and build a detached ROI evidence extractor. A single-GPU audit tool reads frozen HRSC validation data and existing checkpoints, matches predictions to OBB ground truth, and writes immutable JSONL evidence; a separate reporting tool runs grouped cross-validated linear probes and evaluates the pre-registered Gate B without persisting a trainable model.

**Tech Stack:** Python 3.10, PyTorch 1.12, NumPy, MMCV rotated IoU/RoIAlign, MMEngine Runner, MMRotate registries, pytest, JSONL/Markdown reports.

---

## Scope and hard boundaries

This plan implements **Plan A only** from the approved SCQO design.

Included:

- per-instance GODC statistics;
- \(p=2/4\) harmonic actions, decoding, and analytic tests;
- detached C2/C4 ROI evidence, Hann support, \(p\)-atic amplitudes, and a
  histogram-preserving negative control;
- square GT-HBox-to-FPN evidence extraction;
- (e_2/e_4) angle summaries, one-to-one rotated matching, AUROC/AUPRC/ECE,
  reliability bins, geometry/FPN stratification, and a grouped
  cross-validated diagnostic linear probe;
- read-only audits of the frozen best v0.2 and GODC HRSC checkpoints;
- a machine-readable summary and Chinese evidence receipt.

Excluded:

- no SCQO dense head;
- no \(q_2/q_4\) output added to Orbdet;
- no learned or serialized stabilizer calibrator;
- no model/config change affecting prediction;
- no smoke or formal training;
- no DOTA run;
- no test-set evaluation;
- no update to <code>FORMAL_TRAINING_NOT_STARTED.md</code>.

The cross-validated logistic probe in this plan is an **evaluation instrument**.
It must never be saved into a checkpoint or reused as the future SCQO gate.

## File map

| File | Responsibility |
|---|---|
| <code>mmrotate/models/losses/group_orbit_determinantal_cluster_loss.py</code> | Expose differentiable per-instance GODC statistics while preserving the existing forward result |
| <code>tests/test_group_orbit_determinantal_cluster.py</code> | Numerical parity, per-instance shapes, empty and gradient contracts |
| <code>mmrotate/models/losses/scqo_harmonic_equivariance_loss.py</code> | Exact \(O(2)\rightarrow O(2)\) harmonic action, normalization, decoding, and bounded equivariance loss |
| <code>mmrotate/models/losses/__init__.py</code> | Export the SCQO harmonic primitives |
| <code>tests/test_scqo_harmonic_equivariance_loss.py</code> | Rotation, reflection, periodic decoding, collapse, precision, and registry tests |
| <code>mmrotate/models/task_modules/scqo_stabilizer_evidence.py</code> | Detached ROI-level C2/C4, \(p\)-atic, support, and negative-control evidence |
| <code>mmrotate/models/task_modules/scqo_fpn_evidence_adapter.py</code> | Convert GT HBoxes to square FPN RoIs and preserve image/instance/FPN identity |
| <code>mmrotate/models/task_modules/__init__.py</code> | Register the two SCQO evidence modules |
| <code>tests/test_scqo_stabilizer_evidence.py</code> | Synthetic regular/C4/reject evidence and detach contracts |
| <code>tests/test_scqo_fpn_evidence_adapter.py</code> | Square RoI, edge support, identity mapping, empty input, and registry tests |
| <code>mmrotate/evaluation/functional/scqo_diagnostics.py</code> | Angle errors, rotated matching, binary metrics, calibration, and grouped linear probe |
| <code>mmrotate/evaluation/functional/__init__.py</code> | Export diagnostic functions |
| <code>tests/test_scqo_diagnostics.py</code> | Boundary, matching, metric, no-leakage, and determinism tests |
| <code>tools/analysis_tools/scqo_collect_evidence.py</code> | Frozen-checkpoint/FPN evidence collection to JSONL |
| <code>tools/analysis_tools/scqo_report_evidence.py</code> | Gate B comparison and JSON/Markdown reporting |
| <code>tests/test_scqo_plan_a_tools.py</code> | CLI, overwrite protection, row schema, report, and no-training contract |
| <code>resultmd/exp_scqo_plan_a_hrsc_audit_20260817/</code> | Immutable manifests, JSONL evidence, summary JSON, and Chinese receipt |

## Task 1: Expose per-instance GODC statistics without changing the loss

**Files:**

- Modify: <code>mmrotate/models/losses/group_orbit_determinantal_cluster_loss.py</code>
- Modify: <code>tests/test_group_orbit_determinantal_cluster.py</code>

- [ ] **Step 1: Write failing per-instance statistics tests**

Append these tests:

~~~python
def test_statistics_expose_one_value_per_orbit_and_match_forward():
    loss_fn = GroupOrbitDeterminantalClusterLoss(
        determinantal_weight=1.0,
        spectral_tail_weight=0.5,
        fixed_space_weight=0.25,
        energy_guard_weight=0.0,
        variance_guard_weight=0.0,
        reduction='none')
    features = torch.tensor(
        [[[[1., 2.], [2., 1.]]],
         [[[1., 2.], [3., 5.]]]],
        requires_grad=True)
    orbit = build_planar_group_orbit(features, 'c2')

    stats = loss_fn.statistics(orbit)
    actual = loss_fn(orbit)
    expected = (stats['determinantal']
                + 0.5 * stats['spectral_tail']
                + 0.25 * stats['fixed_space'])

    assert set(stats) == {
        'determinantal', 'spectral_tail', 'fixed_space', 'q_gap',
        'energy', 'variance', 'energy_guard', 'variance_guard'
    }
    assert all(value.shape == (2, ) for value in stats.values())
    assert torch.allclose(actual, expected, atol=1e-7)

    actual.sum().backward()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()


def test_statistics_empty_batch_returns_empty_finite_vectors():
    loss_fn = GroupOrbitDeterminantalClusterLoss(reduction='none')
    orbit = torch.empty(0, 2, 3, 4, 4)

    stats = loss_fn.statistics(orbit)

    assert all(value.shape == (0, ) for value in stats.values())
    assert all(torch.isfinite(value).all() for value in stats.values())


def test_statistics_support_mask_matches_forward_masking():
    loss_fn = GroupOrbitDeterminantalClusterLoss(
        energy_guard_weight=0.0,
        variance_guard_weight=0.0,
        reduction='none')
    features = torch.arange(32, dtype=torch.float32).reshape(2, 1, 4, 4)
    orbit = build_planar_group_orbit(features, 'c2')
    mask = torch.zeros(2, 1, 4, 4)
    mask[:, :, 1:3, 1:3] = 1

    stats = loss_fn.statistics(orbit, support_mask=mask)
    value = loss_fn(orbit, support_mask=mask)

    expected = stats['determinantal'] + stats['fixed_space']
    assert torch.allclose(value, expected, atol=1e-7)
~~~

- [ ] **Step 2: Run the focused test and verify RED**

~~~bash
rtk env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
  /data/zcy/anaconda3/envs/orbdet/bin/python -m pytest -q \
  -p no:cacheprovider tests/test_group_orbit_determinantal_cluster.py
~~~

Expected: the three new tests fail with
<code>AttributeError: ... has no attribute 'statistics'</code>.

- [ ] **Step 3: Add the public statistics method**

Inside <code>GroupOrbitDeterminantalClusterLoss</code>, factor validation into
the following method and add <code>statistics</code> exactly:

~~~python
from typing import Dict, Optional, Union


def _validate_orbit(self, orbit: Tensor) -> None:
    if not isinstance(orbit, Tensor):
        raise TypeError('orbit must be a torch.Tensor')
    if orbit.ndim < 3:
        raise ValueError('orbit must have shape [N, K, ...]')
    if orbit.shape[1] < 2:
        raise ValueError('orbit group order K must be at least 2')
    if any(size == 0 for size in orbit.shape[2:]):
        raise ValueError('orbit members must have non-empty features')
    if not orbit.is_floating_point():
        raise TypeError('orbit must have a floating-point dtype')
    if not bool(torch.isfinite(orbit).all()):
        raise ValueError('orbit must contain only finite values')


def statistics(self,
               orbit: Tensor,
               support_mask: Optional[Tensor] = None) -> Dict[str, Tensor]:
    """Return differentiable per-instance orbit statistics."""
    self._validate_orbit(orbit)
    work_orbit = orbit
    if orbit.dtype in (torch.float16, torch.bfloat16):
        work_orbit = orbit.float()
    batch_size, group_order = work_orbit.shape[:2]
    if batch_size == 0:
        empty = work_orbit.new_empty((0, ))
        return {
            name: empty.clone()
            for name in (
                'determinantal', 'spectral_tail', 'fixed_space', 'q_gap',
                'energy', 'variance', 'energy_guard', 'variance_guard')
        }

    work_orbit = self._apply_support_mask(work_orbit, support_mask)
    matrix = work_orbit.reshape(batch_size, group_order, -1)
    gram = matrix @ matrix.transpose(-1, -2)
    trace = gram.diagonal(dim1=-2, dim2=-1).sum(-1)
    trace_g2 = gram.square().sum(dim=(-2, -1))
    determinantal = (
        (trace.square() - trace_g2).clamp_min(0.0)
        / (2.0 * trace.square() + self.eps))
    eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(0.0)
    spectral_tail = (
        1.0 - eigenvalues[:, -1] / (trace + self.eps)).clamp_min(0.0)
    orbit_mean = matrix.mean(dim=1, keepdim=True)
    fixed_space = (
        (matrix - orbit_mean).square().sum(dim=(1, 2))
        / (matrix.square().sum(dim=(1, 2)) + self.eps))
    sigma_1 = eigenvalues[:, -1].sqrt()
    sigma_2 = eigenvalues[:, -2].sqrt()
    q_gap = (sigma_1 - sigma_2) / (sigma_1 + self.eps)
    energy = matrix.square().mean(dim=(1, 2))
    variance = matrix.var(dim=-1, unbiased=False).mean(dim=1)
    return dict(
        determinantal=determinantal,
        spectral_tail=spectral_tail,
        fixed_space=fixed_space,
        q_gap=q_gap,
        energy=energy,
        variance=variance,
        energy_guard=(self.min_energy - energy).clamp_min(0.0),
        variance_guard=(self.min_variance - variance).clamp_min(0.0))
~~~

Replace the duplicated Gram/statistics block in <code>forward</code> with:

~~~python
self._validate_orbit(orbit)
reduction = reduction_override or self.reduction
if reduction not in ('none', 'mean', 'sum'):
    raise ValueError("reduction must be 'none', 'mean', or 'sum'")

stats = self.statistics(orbit, support_mask=support_mask)
batch_size = orbit.shape[0]
if batch_size == 0:
    self._reset_diagnostics(orbit)
    if reduction == 'none':
        return orbit.new_empty((0, ))
    return orbit.sum() * 0.0

determinantal = stats['determinantal']
spectral_tail = stats['spectral_tail']
fixed_space = stats['fixed_space']
energy_guard = stats['energy_guard']
variance_guard = stats['variance_guard']
q_gap = stats['q_gap']
self._record_diagnostics(determinantal, spectral_tail, fixed_space,
                         energy_guard, variance_guard, q_gap)

loss = determinantal.new_zeros((batch_size, ))
for component_weight, component in (
        (self.determinantal_weight, determinantal),
        (self.spectral_tail_weight, spectral_tail),
        (self.fixed_space_weight, fixed_space),
        (self.energy_guard_weight, energy_guard),
        (self.variance_guard_weight, variance_guard)):
    if component_weight > 0.0:
        loss = loss + component_weight * component
sample_weight = self._prepare_weight(weight, loss, batch_size)
return self._reduce(loss, sample_weight, reduction, avg_factor)
~~~

- [ ] **Step 4: Run GODC tests and verify GREEN**

~~~bash
rtk env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
  /data/zcy/anaconda3/envs/orbdet/bin/python -m pytest -q \
  -p no:cacheprovider \
  tests/test_group_orbit_determinantal_cluster.py \
  tests/test_hbox_fpn_group_orbit_loss.py \
  tests/test_orbdet_godc.py
~~~

Expected: all selected tests pass and the original GODC forward contract is
unchanged.

- [ ] **Step 5: Commit the statistics slice**

~~~bash
rtk git add \
  mmrotate/models/losses/group_orbit_determinantal_cluster_loss.py \
  tests/test_group_orbit_determinantal_cluster.py
rtk git commit -m "refactor: expose per-instance group orbit statistics"
~~~

## Task 2: Add exact \(O(2)\) harmonic primitives

**Files:**

- Create: <code>mmrotate/models/losses/scqo_harmonic_equivariance_loss.py</code>
- Modify: <code>mmrotate/models/losses/__init__.py</code>
- Create: <code>tests/test_scqo_harmonic_equivariance_loss.py</code>

- [ ] **Step 1: Write failing rotation, reflection, and collapse tests**

Create the test file:

~~~python
import math

import pytest
import torch

from mmrotate.models.losses.scqo_harmonic_equivariance_loss import (
    SCQOHarmonicEquivarianceLoss, decode_harmonic_angle,
    induced_harmonic_action)
from mmrotate.registry import MODELS
from mmrotate.utils import register_all_modules


def _rotation(phi):
    return torch.tensor(
        [[math.cos(phi), -math.sin(phi)],
         [math.sin(phi), math.cos(phi)]],
        dtype=torch.float32)


@pytest.mark.parametrize('order', [2, 4])
def test_harmonic_action_matches_rotation_and_reflection(order):
    theta = 0.31
    phi = 0.47
    q = torch.tensor([[math.cos(order * theta),
                       math.sin(order * theta)]])
    rotation = _rotation(phi).unsqueeze(0)
    reflected = torch.tensor([[[1., 0.], [0., -1.]]])

    rot_action = induced_harmonic_action(rotation, order)
    flp_action = induced_harmonic_action(reflected, order)
    expected_rot = torch.tensor(
        [[math.cos(order * (theta + phi)),
          math.sin(order * (theta + phi))]])
    expected_flp = torch.tensor(
        [[math.cos(-order * theta), math.sin(-order * theta)]])

    assert torch.allclose(
        torch.einsum('nij,nj->ni', rot_action, q),
        expected_rot,
        atol=1e-6)
    assert torch.allclose(
        torch.einsum('nij,nj->ni', flp_action, q),
        expected_flp,
        atol=1e-6)


def test_equivariance_loss_is_zero_for_correct_views_and_penalizes_constant():
    phi = torch.tensor([0.37, 0.61])
    transforms = torch.stack(tuple(_rotation(float(item)) for item in phi))
    q_ref = torch.tensor([[1., 0.], [1., 0.]], requires_grad=True)
    action = induced_harmonic_action(transforms, order=2)
    q_correct = torch.einsum('nij,nj->ni', action, q_ref.detach())
    loss_fn = SCQOHarmonicEquivarianceLoss(order=2, reduction='none')

    correct = loss_fn(q_ref, q_correct, transforms)
    constant = loss_fn(q_ref, q_ref.detach(), transforms)

    assert torch.allclose(correct, torch.zeros_like(correct), atol=1e-6)
    assert torch.all(constant > 0.1)
    correct.sum().backward()
    assert q_ref.grad is not None
    assert torch.isfinite(q_ref.grad).all()


def test_zero_carrier_has_finite_unit_penalty():
    loss_fn = SCQOHarmonicEquivarianceLoss(order=2)
    q = torch.zeros(3, 2)
    transform = torch.eye(2).expand(3, 2, 2)

    value = loss_fn(q, q, transform, reduction_override='none')

    assert torch.equal(value, torch.ones(3))


def test_half_precision_is_promoted_to_finite_float32():
    loss_fn = SCQOHarmonicEquivarianceLoss(order=4, reduction='none')
    reference = torch.tensor([[1., 0.]], dtype=torch.float16)
    transform = torch.eye(2, dtype=torch.float16).unsqueeze(0)

    value = loss_fn(reference, reference, transform)

    assert value.dtype == torch.float32
    assert torch.isfinite(value).all()


def test_decode_harmonic_angle_respects_quotient_period():
    angles = torch.tensor([-1.2, -0.2, 0.7])
    for order in (2, 4):
        q = torch.stack(
            (torch.cos(order * angles), torch.sin(order * angles)), dim=-1)
        decoded = decode_harmonic_angle(q, order)
        residual = torch.remainder(
            decoded - angles + math.pi / order,
            2 * math.pi / order) - math.pi / order
        assert torch.allclose(residual, torch.zeros_like(residual), atol=1e-6)


def test_harmonic_loss_builds_from_registry_and_rejects_non_orthogonal_matrix():
    register_all_modules()
    loss_fn = MODELS.build(
        dict(type='SCQOHarmonicEquivarianceLoss', order=4))
    bad = torch.tensor([[[1., 1.], [0., 1.]]])

    assert isinstance(loss_fn, SCQOHarmonicEquivarianceLoss)
    with pytest.raises(ValueError, match='orthogonal'):
        induced_harmonic_action(bad, order=4)
~~~

- [ ] **Step 2: Run the new file and verify RED**

~~~bash
rtk env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
  /data/zcy/anaconda3/envs/orbdet/bin/python -m pytest -q \
  -p no:cacheprovider tests/test_scqo_harmonic_equivariance_loss.py
~~~

Expected: collection fails because the SCQO harmonic module does not exist.

- [ ] **Step 3: Implement the harmonic module**

Create the complete module:

~~~python
# Copyright (c) OpenMMLab. All rights reserved.
"""Exact planar harmonic actions and equivariance loss for SCQO."""

import math
from typing import Optional

import torch
from torch import Tensor

from mmrotate.registry import MODELS


def _validate_order(order: int) -> None:
    if not isinstance(order, int) or order <= 0:
        raise ValueError('order must be a positive integer')


def normalize_harmonic(vector: Tensor, eps: float = 1e-8) -> Tensor:
    if not isinstance(vector, Tensor) or vector.shape[-1:] != (2, ):
        raise ValueError('harmonic vector must have shape [..., 2]')
    if not vector.is_floating_point() or not bool(torch.isfinite(vector).all()):
        raise ValueError('harmonic vector must be finite floating point')
    work = vector.float() if vector.dtype in (
        torch.float16, torch.bfloat16) else vector
    return work / (work.square().sum(-1, keepdim=True).sqrt() + eps)


def induced_harmonic_action(transform: Tensor,
                            order: int,
                            atol: float = 1e-4) -> Tensor:
    _validate_order(order)
    if not isinstance(transform, Tensor) or transform.shape[-2:] != (2, 2):
        raise ValueError('transform must have shape [..., 2, 2]')
    if not transform.is_floating_point():
        raise TypeError('transform must be floating point')
    if not bool(torch.isfinite(transform).all()):
        raise ValueError('transform must contain finite values')
    work = transform.float() if transform.dtype in (
        torch.float16, torch.bfloat16) else transform
    identity = torch.eye(2, device=work.device, dtype=work.dtype)
    gram = work.transpose(-1, -2) @ work
    if not bool(torch.allclose(
            gram, identity.expand_as(gram), atol=atol, rtol=atol)):
        raise ValueError('transform must be orthogonal')
    determinant = torch.linalg.det(work)
    if not bool(torch.allclose(
            determinant.abs(), torch.ones_like(determinant),
            atol=atol, rtol=atol)):
        raise ValueError('transform determinant must have magnitude one')

    phi = torch.atan2(work[..., 1, 0], work[..., 0, 0])
    cosine = torch.cos(order * phi)
    sine = torch.sin(order * phi)
    rotation = torch.stack(
        (cosine, -sine, sine, cosine), dim=-1).reshape(
            *phi.shape, 2, 2)
    reflection = torch.diag(
        work.new_tensor([1.0, -1.0])).expand_as(rotation)
    action = torch.where(
        (determinant < 0)[..., None, None],
        rotation @ reflection,
        rotation)
    return action


def decode_harmonic_angle(vector: Tensor,
                          order: int,
                          eps: float = 1e-8) -> Tensor:
    _validate_order(order)
    unit = normalize_harmonic(vector, eps=eps)
    return torch.atan2(unit[..., 1], unit[..., 0]) / float(order)


@MODELS.register_module()
class SCQOHarmonicEquivarianceLoss(torch.nn.Module):

    def __init__(self,
                 order: int,
                 eps: float = 1e-8,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0) -> None:
        super().__init__()
        _validate_order(order)
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError('eps must be positive and finite')
        if reduction not in ('none', 'mean', 'sum'):
            raise ValueError("reduction must be 'none', 'mean', or 'sum'")
        if not math.isfinite(loss_weight) or loss_weight < 0:
            raise ValueError('loss_weight must be finite and non-negative')
        self.order = order
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                reference: Tensor,
                view: Tensor,
                transform: Tensor,
                weight: Optional[Tensor] = None,
                reduction_override: Optional[str] = None) -> Tensor:
        if reference.shape != view.shape or reference.shape[-1:] != (2, ):
            raise ValueError('reference and view must share shape [N, 2]')
        action = induced_harmonic_action(transform, self.order)
        if action.shape[:-2] != reference.shape[:-1]:
            raise ValueError('transform batch shape must match harmonic vectors')
        reference_unit = normalize_harmonic(reference, self.eps)
        view_unit = normalize_harmonic(view, self.eps)
        compute_dtype = torch.promote_types(
            action.dtype,
            torch.promote_types(reference_unit.dtype, view_unit.dtype))
        action = action.to(dtype=compute_dtype)
        reference_unit = reference_unit.to(dtype=compute_dtype)
        view_unit = view_unit.to(dtype=compute_dtype)
        expected = torch.einsum('...ij,...j->...i', action, reference_unit)
        loss = 1.0 - (view_unit * expected).sum(-1)
        loss = loss.clamp(0.0, 2.0)
        if weight is not None:
            weight = torch.as_tensor(
                weight, device=loss.device, dtype=loss.dtype)
            if weight.numel() not in (1, loss.numel()):
                raise ValueError('weight must be scalar or one per sample')
            if not bool(torch.isfinite(weight).all()) or bool((weight < 0).any()):
                raise ValueError('weight must be finite and non-negative')
            if weight.numel() == 1:
                weight = weight.expand_as(loss)
            else:
                weight = weight.reshape_as(loss)
            loss = loss * weight
        reduction = reduction_override or self.reduction
        if reduction == 'mean':
            loss = loss.mean() if loss.numel() else loss.sum()
        elif reduction == 'sum':
            loss = loss.sum()
        elif reduction != 'none':
            raise ValueError('invalid reduction override')
        return loss * self.loss_weight
~~~

Export these names from <code>mmrotate/models/losses/__init__.py</code>:

~~~python
from .scqo_harmonic_equivariance_loss import (
    SCQOHarmonicEquivarianceLoss, decode_harmonic_angle,
    induced_harmonic_action, normalize_harmonic)
~~~

and add the same four names to <code>__all__</code>.

- [ ] **Step 4: Run the harmonic and baseline tests**

~~~bash
rtk env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
  /data/zcy/anaconda3/envs/orbdet/bin/python -m pytest -q \
  -p no:cacheprovider \
  tests/test_scqo_harmonic_equivariance_loss.py \
  tests/test_orbdet_v0_2.py
~~~

Expected: all tests pass; no detector/config file changes.

- [ ] **Step 5: Commit the harmonic core**

~~~bash
rtk git add \
  mmrotate/models/losses/scqo_harmonic_equivariance_loss.py \
  mmrotate/models/losses/__init__.py \
  tests/test_scqo_harmonic_equivariance_loss.py
rtk git commit -m "feat: add exact SCQO harmonic geometry"
~~~

## Task 3: Build detached ROI-level stabilizer evidence

**Files:**

- Create: <code>mmrotate/models/task_modules/scqo_stabilizer_evidence.py</code>
- Modify: <code>mmrotate/models/task_modules/__init__.py</code>
- Create: <code>tests/test_scqo_stabilizer_evidence.py</code>

- [ ] **Step 1: Write failing synthetic evidence tests**

Create deterministic patterns and contracts:

~~~python
import pytest
import torch

from mmrotate.models.task_modules.scqo_stabilizer_evidence import (
    SCQOStabilizerEvidence)
from mmrotate.registry import MODELS
from mmrotate.utils import register_all_modules


def _cross(size=14):
    value = torch.zeros(1, 1, size, size)
    value[:, :, size // 2 - 1:size // 2 + 1, 2:-2] = 1
    value[:, :, 2:-2, size // 2 - 1:size // 2 + 1] = 1
    return value


def _bar(size=14):
    value = torch.zeros(1, 1, size, size)
    value[:, :, size // 2 - 1:size // 2 + 1, 2:-2] = 1
    return value


def _disk(size=14):
    coordinate = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
    y, x = torch.meshgrid(coordinate, coordinate, indexing='ij')
    return ((x.square() + y.square()) <= 16).float()[None, None]


def test_c4_cross_has_lower_c4_residual_than_regular_bar():
    module = SCQOStabilizerEvidence(
        roi_size=14, channels=1, negative_seed=3407)
    cross = module(_cross())
    bar = module(_bar())

    assert float(cross['c4_fixed_space']) < 1e-6
    assert float(bar['c4_fixed_space']) > 1e-3
    assert float(cross['c4_negative_margin']) > 0


def test_patic_and_residual_evidence_separate_axis_disk_noise_and_occlusion():
    module = SCQOStabilizerEvidence(
        roi_size=14, channels=1, negative_seed=3407)
    generator = torch.Generator().manual_seed(19)
    noise = torch.randn(1, 1, 14, 14, generator=generator)
    occluded = _cross()
    occluded[:, :, :7, :7] = 0

    bar = module(_bar())
    disk = module(_disk())
    random_texture = module(noise)
    partial_cross = module(occluded)
    cross = module(_cross())

    assert float(disk['c4_fixed_space']) < 1e-6
    assert float(bar['a2']) > float(disk['a2']) + 0.1
    assert float(bar['a2']) > float(random_texture['a2'])
    assert float(partial_cross['c4_fixed_space']) > float(
        cross['c4_fixed_space']) + 1e-4


def test_constant_and_empty_information_are_invalid():
    module = SCQOStabilizerEvidence(
        roi_size=14, channels=1, negative_seed=3407)
    result = module(torch.ones(2, 1, 14, 14))

    assert not bool(result['valid'].any())
    assert torch.all(result['variance_guard'] > 0)

    empty_support = module(
        torch.randn(1, 1, 14, 14),
        valid_support=torch.zeros(1, 1, 14, 14))
    assert not bool(empty_support['valid'])


def test_half_precision_is_finite_and_nonfinite_input_is_rejected():
    module = SCQOStabilizerEvidence(
        roi_size=14, channels=1, negative_seed=3407)
    result = module(torch.randn(2, 1, 14, 14, dtype=torch.float16))

    assert all(torch.isfinite(value).all() for value in result.values())
    with pytest.raises(ValueError, match='finite'):
        module(torch.full((1, 1, 14, 14), float('nan')))


def test_evidence_is_detached_and_negative_control_preserves_energy():
    module = SCQOStabilizerEvidence(
        roi_size=14, channels=1, negative_seed=3407)
    features = torch.randn(3, 1, 14, 14, requires_grad=True)
    result = module(features)

    assert all(not value.requires_grad for value in result.values())
    assert torch.allclose(
        result['negative_energy'], result['energy'], rtol=1e-5, atol=1e-6)
    assert features.grad is None


def test_registry_and_empty_batch_contract():
    register_all_modules()
    module = MODELS.build(
        dict(
            type='SCQOStabilizerEvidence',
            roi_size=14,
            channels=4,
            negative_seed=3407))
    result = module(torch.empty(0, 4, 14, 14))

    assert isinstance(module, SCQOStabilizerEvidence)
    assert all(value.shape == (0, ) for value in result.values())
~~~

- [ ] **Step 2: Run the focused file and verify RED**

~~~bash
rtk env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
  /data/zcy/anaconda3/envs/orbdet/bin/python -m pytest -q \
  -p no:cacheprovider tests/test_scqo_stabilizer_evidence.py
~~~

Expected: collection fails because the evidence module is absent.

- [ ] **Step 3: Implement Hann support, \(p\)-atic amplitudes, and negative control**

Create the module with this public contract:

~~~python
# Copyright (c) OpenMMLab. All rights reserved.
"""Detached instance-level stabilizer evidence for SCQO."""

from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor

from mmrotate.models.losses.group_orbit_determinantal_cluster_loss import (
    GroupOrbitDeterminantalClusterLoss, build_planar_group_orbit)
from mmrotate.registry import MODELS


@MODELS.register_module()
class SCQOStabilizerEvidence(torch.nn.Module):

    _STAT_NAMES = (
        'determinantal', 'spectral_tail', 'fixed_space', 'q_gap')

    def __init__(self,
                 roi_size: int = 14,
                 channels: int = 256,
                 block_size: int = 2,
                 negative_seed: int = 3407,
                 min_energy: float = 1e-4,
                 min_variance: float = 1e-4,
                 min_support_fraction: float = 0.95) -> None:
        super().__init__()
        if roi_size <= 0 or roi_size % block_size:
            raise ValueError('roi_size must be positive and divisible by block_size')
        if channels <= 0:
            raise ValueError('channels must be positive')
        if not 0 < min_support_fraction <= 1:
            raise ValueError('min_support_fraction must be in (0, 1]')
        self.roi_size = roi_size
        self.channels = channels
        self.block_size = block_size
        self.min_support_fraction = min_support_fraction
        self.statistics_module = GroupOrbitDeterminantalClusterLoss(
            determinantal_weight=1.0,
            spectral_tail_weight=0.0,
            fixed_space_weight=0.0,
            energy_guard_weight=0.0,
            variance_guard_weight=0.0,
            min_energy=min_energy,
            min_variance=min_variance,
            reduction='none')
        hann = torch.hann_window(roi_size, periodic=False)
        self.register_buffer('hann', torch.outer(hann, hann)[None, None])
        blocks = (roi_size // block_size) ** 2
        generator = torch.Generator().manual_seed(negative_seed)
        self.register_buffer(
            'block_permutation', torch.randperm(blocks, generator=generator))
        sobel_x = torch.tensor(
            [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]) / 8.0
        self.register_buffer('sobel_x', sobel_x[None, None])
        self.register_buffer(
            'sobel_y', sobel_x.t().contiguous()[None, None])

    def _negative_control(self, features: Tensor) -> Tensor:
        n, c, h, w = features.shape
        b = self.block_size
        gh, gw = h // b, w // b
        blocks = features.reshape(n, c, gh, b, gw, b)
        blocks = blocks.permute(0, 1, 2, 4, 3, 5)
        blocks = blocks.reshape(n, c, gh * gw, b, b)
        blocks = blocks.index_select(2, self.block_permutation)
        blocks = blocks.reshape(n, c, gh, gw, b, b)
        return blocks.permute(0, 1, 2, 4, 3, 5).reshape(n, c, h, w)

    def _patic(self, features: Tensor, support: Tensor,
               order: int) -> Tensor:
        n, c, h, w = features.shape
        flat = features.reshape(n * c, 1, h, w)
        gx = F.conv2d(flat, self.sobel_x, padding=1).reshape(n, c, h, w)
        gy = F.conv2d(flat, self.sobel_y, padding=1).reshape(n, c, h, w)
        magnitude = torch.sqrt(gx.square() + gy.square() + 1e-12)
        angle = torch.atan2(gy, gx)
        weight = magnitude * support
        real = (weight * torch.cos(order * angle)).sum((1, 2, 3))
        imag = (weight * torch.sin(order * angle)).sum((1, 2, 3))
        denom = weight.sum((1, 2, 3)) + 1e-8
        return torch.sqrt(real.square() + imag.square()) / denom

    def forward(self,
                roi_features: Tensor,
                valid_support: Tensor = None) -> Dict[str, Tensor]:
        if roi_features.ndim != 4:
            raise ValueError('roi_features must have shape [N, C, H, W]')
        if roi_features.shape[1:] != (
                self.channels, self.roi_size, self.roi_size):
            raise ValueError('roi feature shape does not match configured contract')
        if not roi_features.is_floating_point():
            raise TypeError('roi_features must be floating point')
        if not bool(torch.isfinite(roi_features).all()):
            raise ValueError('roi_features must be finite')
        features = roi_features.detach().float()
        n = features.shape[0]
        if valid_support is None:
            valid_support = features.new_ones((n, 1, self.roi_size,
                                               self.roi_size))
        if valid_support.shape != (n, 1, self.roi_size, self.roi_size):
            raise ValueError('valid_support must have shape [N, 1, H, W]')
        valid_support = valid_support.detach().float().clamp(0.0, 1.0)
        support_fraction = valid_support.mean((1, 2, 3))
        support = valid_support * self.hann

        energy = features.square().mean((1, 2, 3))
        variance = features.var((1, 2, 3), unbiased=False)
        scale = energy.sqrt().clamp_min(1e-8)[:, None, None, None]
        normalized = features / scale

        c2 = self.statistics_module.statistics(
            build_planar_group_orbit(normalized, 'c2'), support)
        c4 = self.statistics_module.statistics(
            build_planar_group_orbit(normalized, 'c4'), support)
        negative = self._negative_control(features)
        negative_c4 = self.statistics_module.statistics(
            build_planar_group_orbit(negative / scale, 'c4'), support)
        energy_guard = (
            self.statistics_module.min_energy - energy).clamp_min(0.0)
        variance_guard = (
            self.statistics_module.min_variance - variance).clamp_min(0.0)

        result = {}
        for name in self._STAT_NAMES:
            result[f'c2_{name}'] = c2[name].detach()
            result[f'c4_{name}'] = c4[name].detach()
        result.update(
            energy=energy.detach(),
            variance=variance.detach(),
            energy_guard=energy_guard.detach(),
            variance_guard=variance_guard.detach(),
            support_fraction=support_fraction.detach(),
            a2=self._patic(normalized, support, 2).detach(),
            a4=self._patic(normalized, support, 4).detach(),
            negative_energy=negative.square().mean((1, 2, 3)).detach(),
            c4_negative_margin=(
                negative_c4['fixed_space'] - c4['fixed_space']).detach())
        result['valid'] = (
            (result['energy_guard'] == 0)
            & (result['variance_guard'] == 0)
            & (support_fraction >= self.min_support_fraction)).detach()
        return result
~~~

Add to <code>mmrotate/models/task_modules/__init__.py</code>:

~~~python
from .scqo_stabilizer_evidence import SCQOStabilizerEvidence
~~~

- [ ] **Step 4: Run synthetic evidence and GODC regression tests**

~~~bash
rtk env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
  /data/zcy/anaconda3/envs/orbdet/bin/python -m pytest -q \
  -p no:cacheprovider \
  tests/test_scqo_stabilizer_evidence.py \
  tests/test_group_orbit_determinantal_cluster.py
~~~

Expected: all tests pass and every evidence tensor is detached.

- [ ] **Step 5: Commit the ROI evidence module**

~~~bash
rtk git add \
  mmrotate/models/task_modules/scqo_stabilizer_evidence.py \
  mmrotate/models/task_modules/__init__.py \
  tests/test_scqo_stabilizer_evidence.py
rtk git commit -m "feat: add detached SCQO stabilizer evidence"
~~~

## Task 4: Add the square GT-HBox/FPN evidence adapter

**Files:**

- Create: <code>mmrotate/models/task_modules/scqo_fpn_evidence_adapter.py</code>
- Modify: <code>mmrotate/models/task_modules/__init__.py</code>
- Create: <code>tests/test_scqo_fpn_evidence_adapter.py</code>

- [ ] **Step 1: Write failing identity, square-RoI, edge, and empty tests**

Create the adapter tests:

~~~python
import torch
from mmengine.structures import InstanceData

from mmrotate.models.task_modules.scqo_fpn_evidence_adapter import (
    SCQOFPNInstanceEvidence)
from mmrotate.registry import MODELS
from mmrotate.structures import RotatedBoxes
from mmrotate.utils import register_all_modules


def _instances(boxes, labels):
    value = InstanceData()
    value.bboxes = RotatedBoxes(torch.tensor(boxes, dtype=torch.float32))
    value.labels = torch.tensor(labels, dtype=torch.long)
    return value


def _cfg():
    return dict(
        type='SCQOFPNInstanceEvidence',
        min_box_size=2.0,
        roi_extractor=dict(
            type='mmdet.SingleRoIExtractor',
            roi_layer=dict(
                type='RoIAlign', output_size=14, sampling_ratio=2),
            out_channels=4,
            featmap_strides=[1]),
        evidence=dict(
            type='SCQOStabilizerEvidence',
            roi_size=14,
            channels=4,
            negative_seed=3407))


def test_adapter_preserves_batch_instance_identity_and_square_boxes():
    register_all_modules()
    module = MODELS.build(_cfg())
    features = (torch.randn(2, 4, 32, 32, requires_grad=True), )
    instances = [
        _instances([[10, 10, 8, 4, 0]], [3]),
        _instances([[20, 18, 6, 10, 0], [5, 5, 1, 1, 0]], [4, 5])
    ]
    metas = [dict(img_shape=(32, 32)), dict(img_shape=(32, 32))]

    result = module(features, instances, metas)

    assert torch.equal(result['batch_index'], torch.tensor([0, 1]))
    assert torch.equal(result['instance_index'], torch.tensor([0, 0]))
    assert torch.equal(result['label'], torch.tensor([3, 4]))
    assert torch.equal(result['fpn_level'], torch.tensor([0, 0]))
    widths = result['square_hbox'][:, 2] - result['square_hbox'][:, 0]
    heights = result['square_hbox'][:, 3] - result['square_hbox'][:, 1]
    assert torch.allclose(widths, heights)
    assert all(not value.requires_grad for value in result.values())
    assert features[0].grad is None


def test_edge_box_records_partial_support_and_empty_batch_is_finite():
    register_all_modules()
    module = MODELS.build(_cfg())
    features = (torch.randn(1, 4, 20, 20), )
    edge = [_instances([[1, 1, 8, 4, 0]], [0])]
    result = module(features, edge, [dict(img_shape=(20, 20))])
    empty = module(
        features,
        [_instances(torch.empty(0, 5), torch.empty(0, dtype=torch.long))],
        [dict(img_shape=(20, 20))])
    empty_batch = module((torch.empty(0, 4, 20, 20), ), [], [])

    assert float(result['support_fraction']) < 1.0
    assert not bool(result['valid'])
    assert all(value.shape[0] == 0 for value in empty.values())
    assert all(value.shape[0] == 0 for value in empty_batch.values())
~~~

- [ ] **Step 2: Run the adapter test and verify RED**

~~~bash
rtk env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
  /data/zcy/anaconda3/envs/orbdet/bin/python -m pytest -q \
  -p no:cacheprovider tests/test_scqo_fpn_evidence_adapter.py
~~~

Expected: collection fails because the FPN adapter is absent.

- [ ] **Step 3: Implement square RoIs and analytic valid-image support**

Create <code>SCQOFPNInstanceEvidence</code> with the following methods and
forward contract:

~~~python
# Copyright (c) OpenMMLab. All rights reserved.
"""GT-HBox/FPN adapter for read-only SCQO evidence."""

from typing import Dict, List, Sequence, Tuple

import torch
from mmdet.structures.bbox import bbox2roi
from mmdet.utils import ConfigType, InstanceList
from torch import Tensor

from mmrotate.registry import MODELS


@MODELS.register_module()
class SCQOFPNInstanceEvidence(torch.nn.Module):

    def __init__(self,
                 roi_extractor: ConfigType,
                 evidence: ConfigType,
                 min_box_size: float = 2.0) -> None:
        super().__init__()
        if min_box_size <= 0:
            raise ValueError('min_box_size must be positive')
        self.roi_extractor = MODELS.build(roi_extractor)
        self.evidence = MODELS.build(evidence)
        self.min_box_size = min_box_size

    def _square_boxes(
            self, batch_gt_instances: InstanceList
    ) -> Tuple[List[Tensor], Tensor, Tensor, Tensor]:
        boxes_per_image = []
        batch_indices = []
        instance_indices = []
        labels = []
        for batch_index, instances in enumerate(batch_gt_instances):
            hboxes = instances.bboxes.convert_to('hbox').tensor
            if hboxes.numel() == 0:
                boxes_per_image.append(hboxes.reshape(0, 4))
                continue
            width = hboxes[:, 2] - hboxes[:, 0]
            height = hboxes[:, 3] - hboxes[:, 1]
            valid = (
                torch.isfinite(hboxes).all(1)
                & (width >= self.min_box_size)
                & (height >= self.min_box_size))
            original_index = valid.nonzero().reshape(-1)
            hboxes = hboxes[valid]
            center = (hboxes[:, :2] + hboxes[:, 2:]) / 2
            side = torch.maximum(
                hboxes[:, 2] - hboxes[:, 0],
                hboxes[:, 3] - hboxes[:, 1])
            square = torch.cat(
                (center - side[:, None] / 2,
                 center + side[:, None] / 2), dim=1)
            boxes_per_image.append(square)
            batch_indices.append(
                original_index.new_full(original_index.shape, batch_index))
            instance_indices.append(original_index)
            labels.append(instances.labels[valid])
        reference = batch_gt_instances[0].bboxes.tensor
        empty_long = reference.new_empty((0, ), dtype=torch.long)
        return (
            boxes_per_image,
            torch.cat(batch_indices) if batch_indices else empty_long,
            torch.cat(instance_indices) if instance_indices else empty_long,
            torch.cat(labels) if labels else empty_long)

    def _support(self, rois: Tensor,
                 batch_img_metas: Sequence[dict]) -> Tensor:
        size = self.evidence.roi_size
        grid = (
            torch.arange(size, device=rois.device, dtype=rois.dtype) + 0.5
        ) / size
        x = rois[:, 1, None] + grid[None] * (rois[:, 3] - rois[:, 1])[:, None]
        y = rois[:, 2, None] + grid[None] * (rois[:, 4] - rois[:, 2])[:, None]
        masks = []
        for row, x_row, y_row in zip(rois, x, y):
            image_index = int(row[0])
            image_height, image_width = batch_img_metas[
                image_index]['img_shape'][:2]
            valid_x = (x_row >= 0) & (x_row < image_width)
            valid_y = (y_row >= 0) & (y_row < image_height)
            masks.append(valid_y[:, None] & valid_x[None, :])
        return torch.stack(masks).unsqueeze(1).to(dtype=rois.dtype)

    def _empty_result(self, reference: Tensor) -> Dict[str, Tensor]:
        empty_features = reference.new_empty(
            (0, self.evidence.channels, self.evidence.roi_size,
             self.evidence.roi_size))
        empty_long = reference.new_empty((0, ), dtype=torch.long)
        result = self.evidence(empty_features)
        result.update(
            batch_index=empty_long,
            instance_index=empty_long.clone(),
            label=empty_long.clone(),
            fpn_level=empty_long.clone(),
            square_hbox=reference.new_empty((0, 4)))
        return result

    def forward(self,
                features: Tuple[Tensor, ...],
                batch_gt_instances: InstanceList,
                batch_img_metas: Sequence[dict]) -> Dict[str, Tensor]:
        if not features or len(features) < self.roi_extractor.num_inputs:
            raise ValueError('insufficient FPN features')
        if len(batch_gt_instances) != len(batch_img_metas):
            raise ValueError('instances and image metadata must align')
        if not batch_gt_instances:
            return self._empty_result(features[0])
        boxes, batch_index, instance_index, labels = self._square_boxes(
            batch_gt_instances)
        rois = bbox2roi(boxes)
        if rois.shape[0] == 0:
            return self._empty_result(features[0])
        roi_features = self.roi_extractor(
            tuple(item.detach()
                  for item in features[:self.roi_extractor.num_inputs]),
            rois)
        result = self.evidence(roi_features, self._support(rois, batch_img_metas))
        fpn_level = self.roi_extractor.map_roi_levels(
            rois, self.roi_extractor.num_inputs)
        result.update(
            batch_index=batch_index.detach(),
            instance_index=instance_index.detach(),
            label=labels.detach(),
            fpn_level=fpn_level.detach(),
            square_hbox=rois[:, 1:].detach())
        return result
~~~

Export the class from <code>mmrotate/models/task_modules/__init__.py</code>:

~~~python
from .scqo_fpn_evidence_adapter import SCQOFPNInstanceEvidence
~~~

- [ ] **Step 4: Run adapter, evidence, and registry tests**

~~~bash
rtk env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
  /data/zcy/anaconda3/envs/orbdet/bin/python -m pytest -q \
  -p no:cacheprovider \
  tests/test_scqo_fpn_evidence_adapter.py \
  tests/test_scqo_stabilizer_evidence.py \
  tests/test_hbox_fpn_group_orbit_loss.py
~~~

Expected: all tests pass; no gradients reach FPN inputs.

- [ ] **Step 5: Commit the FPN adapter**

~~~bash
rtk git add \
  mmrotate/models/task_modules/scqo_fpn_evidence_adapter.py \
  mmrotate/models/task_modules/__init__.py \
  tests/test_scqo_fpn_evidence_adapter.py
rtk git commit -m "feat: add SCQO HBox FPN evidence adapter"
~~~

## Task 5: Add angle, matching, calibration, and grouped-probe diagnostics

**Files:**

- Create: <code>mmrotate/evaluation/functional/scqo_diagnostics.py</code>
- Modify: <code>mmrotate/evaluation/functional/__init__.py</code>
- Create: <code>tests/test_scqo_diagnostics.py</code>

- [ ] **Step 1: Write failing diagnostic tests**

Create tests covering periodic seams, one-to-one matching, exact metrics, and
grouped folds:

~~~python
import math

import numpy as np
import torch

from mmrotate.evaluation.functional.scqo_diagnostics import (
    binary_average_precision, binary_auroc, cross_validated_logistic_probe,
    expected_calibration_error, match_rotated_predictions,
    periodic_angle_error)


def test_periodic_angle_error_uses_shortest_pi_path():
    pred = torch.tensor([math.pi / 2 - 0.01, -math.pi / 2 + 0.02])
    target = torch.tensor([-math.pi / 2 + 0.01, math.pi / 2 - 0.01])

    error = periodic_angle_error(pred, target, period=math.pi)

    assert torch.allclose(error, torch.tensor([0.02, 0.03]), atol=1e-6)


def test_matching_is_score_ordered_one_to_one_and_class_aware():
    gt = torch.tensor([[10., 10., 8., 4., 0.], [10., 10., 8., 4., 0.]])
    gt_labels = torch.tensor([0, 1])
    pred = gt.repeat_interleave(2, dim=0)
    pred_labels = torch.tensor([0, 0, 1, 1])
    scores = torch.tensor([0.9, 0.8, 0.7, 0.6])

    result = match_rotated_predictions(
        pred, scores, pred_labels, gt, gt_labels,
        score_threshold=0.05, iou_threshold=0.5)

    assert torch.equal(result['gt_index'], torch.tensor([0, 1]))
    assert torch.equal(result['pred_index'], torch.tensor([0, 2]))
    assert torch.allclose(result['iou'], torch.ones(2))


def test_binary_metrics_have_known_perfect_values():
    target = np.array([0, 0, 1, 1])
    score = np.array([0.1, 0.2, 0.8, 0.9])

    assert binary_auroc(target, score) == 1.0
    assert binary_average_precision(target, score) == 1.0
    assert expected_calibration_error(target, score, bins=2) < 0.2


def test_grouped_probe_is_deterministic_and_separates_signal():
    groups = np.repeat(np.arange(30), 4)
    target = np.tile(np.array([0, 0, 1, 1]), 30)
    signal = target.astype(np.float64) * 2 - 1
    features = np.stack(
        (signal + np.linspace(-0.1, 0.1, signal.size),
         np.sin(np.arange(signal.size))), axis=1)

    first = cross_validated_logistic_probe(
        features, target, groups, folds=3, l2=1e-2, max_iter=100)
    second = cross_validated_logistic_probe(
        features, target, groups, folds=3, l2=1e-2, max_iter=100)

    assert np.allclose(first, second)
    assert binary_auroc(target, first) > 0.99
    for group in np.unique(groups):
        assert np.all(np.isfinite(first[groups == group]))
~~~

- [ ] **Step 2: Run diagnostics and verify RED**

~~~bash
rtk env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
  /data/zcy/anaconda3/envs/orbdet/bin/python -m pytest -q \
  -p no:cacheprovider tests/test_scqo_diagnostics.py
~~~

Expected: collection fails because <code>scqo_diagnostics</code> is absent.

- [ ] **Step 3: Implement periodic errors and rotated matching**

Create these functions in the new module:

~~~python
import hashlib
import math
from typing import Dict, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from mmrotate.structures.bbox import rbbox_overlaps


def periodic_angle_error(prediction: Tensor,
                         target: Tensor,
                         period: float = math.pi) -> Tensor:
    if prediction.shape != target.shape:
        raise ValueError('prediction and target shapes must match')
    if not math.isfinite(period) or period <= 0:
        raise ValueError('period must be positive and finite')
    delta = prediction - target
    return torch.remainder(delta + period / 2, period).sub(period / 2).abs()


def match_rotated_predictions(pred_boxes: Tensor,
                              pred_scores: Tensor,
                              pred_labels: Tensor,
                              gt_boxes: Tensor,
                              gt_labels: Tensor,
                              score_threshold: float = 0.05,
                              iou_threshold: float = 0.5
                              ) -> Dict[str, Tensor]:
    keep = pred_scores >= score_threshold
    original_index = keep.nonzero().reshape(-1)
    order = torch.argsort(pred_scores[keep], descending=True)
    selected = original_index[order]
    matched_gt = set()
    gt_indices = []
    pred_indices = []
    matched_ious = []
    for pred_index in selected.tolist():
        candidates = (
            (gt_labels == pred_labels[pred_index]).nonzero().reshape(-1))
        candidates = torch.tensor(
            [int(item) for item in candidates if int(item) not in matched_gt],
            dtype=torch.long,
            device=gt_boxes.device)
        if candidates.numel() == 0:
            continue
        overlaps = rbbox_overlaps(
            pred_boxes[pred_index:pred_index + 1],
            gt_boxes[candidates])[0]
        best_value, best_offset = overlaps.max(dim=0)
        if float(best_value) < iou_threshold:
            continue
        gt_index = int(candidates[int(best_offset)])
        matched_gt.add(gt_index)
        gt_indices.append(gt_index)
        pred_indices.append(pred_index)
        matched_ious.append(float(best_value))
    device = gt_boxes.device
    return dict(
        gt_index=torch.tensor(gt_indices, device=device, dtype=torch.long),
        pred_index=torch.tensor(pred_indices, device=device, dtype=torch.long),
        iou=torch.tensor(matched_ious, device=device, dtype=torch.float32))
~~~

- [ ] **Step 4: Implement exact binary metrics and grouped linear probe**

Append these complete NumPy/PyTorch utilities:

~~~python
def _binary_arrays(target, score):
    target = np.asarray(target, dtype=np.int64).reshape(-1)
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    if target.shape != score.shape or target.size == 0:
        raise ValueError('target and score must be non-empty and aligned')
    if not np.isin(target, [0, 1]).all() or not np.isfinite(score).all():
        raise ValueError('binary targets and finite scores are required')
    if np.unique(target).size != 2:
        raise ValueError('both binary classes are required')
    return target, score


def binary_auroc(target, score) -> float:
    target, score = _binary_arrays(target, score)
    order = np.argsort(score, kind='mergesort')
    sorted_score = score[order]
    ranks = np.empty(score.size, dtype=np.float64)
    start = 0
    while start < score.size:
        stop = start + 1
        while stop < score.size and sorted_score[stop] == sorted_score[start]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    positives = target == 1
    n_pos = int(positives.sum())
    n_neg = target.size - n_pos
    rank_sum = ranks[positives].sum()
    return float(
        (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def binary_average_precision(target, score) -> float:
    target, score = _binary_arrays(target, score)
    order = np.argsort(-score, kind='mergesort')
    sorted_target = target[order]
    true_positive = np.cumsum(sorted_target)
    precision = true_positive / np.arange(1, target.size + 1)
    return float((precision * sorted_target).sum() / sorted_target.sum())


def expected_calibration_error(target, probability, bins=10) -> float:
    target, probability = _binary_arrays(target, probability)
    if np.any((probability < 0) | (probability > 1)):
        raise ValueError('probabilities must lie in [0, 1]')
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = target.size
    value = 0.0
    for index in range(bins):
        right_closed = index == bins - 1
        mask = (
            (probability >= edges[index])
            & ((probability <= edges[index + 1]) if right_closed
               else (probability < edges[index + 1])))
        if mask.any():
            value += (
                mask.sum() / total
                * abs(target[mask].mean() - probability[mask].mean()))
    return float(value)


def _fold_for_group(group, folds):
    digest = hashlib.sha256(str(group).encode('utf-8')).digest()
    return int.from_bytes(digest[:8], 'big') % folds


def cross_validated_logistic_probe(features,
                                    target,
                                    groups: Sequence,
                                    folds=3,
                                    l2=1e-2,
                                    max_iter=100):
    features = np.asarray(features, dtype=np.float64)
    target = np.asarray(target, dtype=np.int64).reshape(-1)
    groups = np.asarray(groups)
    if features.ndim != 2 or features.shape[0] != target.size:
        raise ValueError('features must have shape [N, D]')
    if groups.shape != target.shape or folds < 2:
        raise ValueError('groups must align and folds must be at least two')
    assignments = np.array(
        [_fold_for_group(item, folds) for item in groups], dtype=np.int64)
    probability = np.full(target.size, np.nan, dtype=np.float64)
    for fold in range(folds):
        test = assignments == fold
        train = ~test
        if not test.any() or np.unique(target[train]).size != 2:
            raise ValueError('every fold needs test rows and two train classes')
        median = np.median(features[train], axis=0)
        mad = np.median(np.abs(features[train] - median), axis=0)
        scale = np.maximum(1.4826 * mad, 1e-6)
        x_train = torch.tensor(
            (features[train] - median) / scale, dtype=torch.float64)
        y_train = torch.tensor(target[train], dtype=torch.float64)
        x_test = torch.tensor(
            (features[test] - median) / scale, dtype=torch.float64)
        weight = torch.zeros(
            features.shape[1], dtype=torch.float64, requires_grad=True)
        bias = torch.zeros((), dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.LBFGS(
            [weight, bias],
            lr=1.0,
            max_iter=max_iter,
            line_search_fn='strong_wolfe')

        def closure():
            optimizer.zero_grad()
            logits = x_train @ weight + bias
            loss = (
                F.binary_cross_entropy_with_logits(logits, y_train)
                + 0.5 * l2 * weight.square().sum())
            loss.backward()
            return loss

        optimizer.step(closure)
        with torch.no_grad():
            probability[test] = torch.sigmoid(
                x_test @ weight + bias).cpu().numpy()
    if not np.isfinite(probability).all():
        raise RuntimeError('grouped probe left non-finite predictions')
    return probability
~~~

Export the six public functions from
<code>mmrotate/evaluation/functional/__init__.py</code>.

- [ ] **Step 5: Run diagnostics and existing evaluator tests**

~~~bash
rtk env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
  /data/zcy/anaconda3/envs/orbdet/bin/python -m pytest -q \
  -p no:cacheprovider \
  tests/test_scqo_diagnostics.py \
  tests/test_orbdet_v02_dota1_epoch12_eval_contract.py
~~~

Expected: all tests pass without adding scikit-learn or another dependency.

- [ ] **Step 6: Commit the diagnostic functions**

~~~bash
rtk git add \
  mmrotate/evaluation/functional/scqo_diagnostics.py \
  mmrotate/evaluation/functional/__init__.py \
  tests/test_scqo_diagnostics.py
rtk git commit -m "feat: add SCQO orientation diagnostics"
~~~

## Task 6: Build the frozen-checkpoint evidence collector

**Files:**

- Create: <code>tools/analysis_tools/scqo_collect_evidence.py</code>
- Create: <code>tests/test_scqo_plan_a_tools.py</code>

- [ ] **Step 1: Write failing CLI and row-schema tests**

Create a tool contract test that imports only pure helpers and invokes
<code>--help</code>:

~~~python
import argparse
import json
from pathlib import Path
import subprocess
import sys

import pytest

from tools.analysis_tools.scqo_collect_evidence import (
    build_evidence_row, collect)


ROOT = Path(__file__).resolve().parents[1]
COLLECTOR = ROOT / 'tools/analysis_tools/scqo_collect_evidence.py'
REPORTER = ROOT / 'tools/analysis_tools/scqo_report_evidence.py'


def test_collector_help_and_row_schema():
    completed = subprocess.run(
        [sys.executable, str(COLLECTOR), '--help'],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True)
    assert completed.returncode == 0
    assert '--checkpoint' in completed.stdout
    row = build_evidence_row(
        run_name='v02',
        image_id='100000001',
        gt_index=0,
        label=0,
        evidence={'energy': 1.0, 'valid': True},
        geometry={
            'gt_width': 8.0,
            'gt_height': 4.0,
            'gt_area': 32.0,
            'gt_aspect_ratio': 2.0,
        },
        match=None)
    assert row == {
        'run_name': 'v02',
        'image_id': '100000001',
        'gt_index': 0,
        'label': 0,
        'evidence_available': True,
        'matched': False,
        'energy': 1.0,
        'valid': True,
        'gt_width': 8.0,
        'gt_height': 4.0,
        'gt_area': 32.0,
        'gt_aspect_ratio': 2.0,
    }


def test_collector_refuses_existing_output_before_loading_model(tmp_path):
    config = tmp_path / 'config.py'
    checkpoint = tmp_path / 'checkpoint.pth'
    output = tmp_path / 'evidence.jsonl'
    config.write_text('model = {}\n')
    checkpoint.write_bytes(b'not-loaded')
    output.write_text('immutable\n')
    args = argparse.Namespace(
        config=config, checkpoint=checkpoint, output=output)

    with pytest.raises(FileExistsError, match='must not already exist'):
        collect(args)


def test_collector_cannot_start_training_or_overwrite_status():
    text = COLLECTOR.read_text()
    forbidden = (
        'tools/train.py', 'train_step(', 'optim_wrapper',
        'tmux', 'nohup', 'FORMAL_TRAINING_NOT_STARTED.md')
    assert all(token not in text for token in forbidden)
~~~

- [ ] **Step 2: Run the tool test and verify RED**

~~~bash
rtk env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
  PYTHONPATH=/data1/zcy/Orbdet \
  /data/zcy/anaconda3/envs/orbdet/bin/python -m pytest -q \
  -p no:cacheprovider tests/test_scqo_plan_a_tools.py
~~~

Expected: collection fails because the collector is absent.

- [ ] **Step 3: Implement immutable row construction and manifest helpers**

Start the collector with these exact helpers:

~~~python
#!/usr/bin/env python3
"""Collect per-instance SCQO evidence from a frozen checkpoint."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import tempfile
from typing import Dict, Optional

import torch
from mmengine.config import Config
from mmengine.runner import Runner

from mmrotate.evaluation.functional.scqo_diagnostics import (
    match_rotated_predictions, periodic_angle_error)
from mmrotate.registry import MODELS
from mmrotate.utils import register_all_modules


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _scalar(value):
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError('row evidence must be scalar')
        value = value.item()
    if isinstance(value, bool):
        return value
    return float(value)


def build_evidence_row(run_name: str,
                       image_id: str,
                       gt_index: int,
                       label: int,
                       evidence: Optional[Dict],
                       geometry: Optional[Dict],
                       match: Optional[Dict]) -> Dict:
    row = dict(
        run_name=run_name,
        image_id=str(image_id),
        gt_index=int(gt_index),
        label=int(label),
        evidence_available=evidence is not None,
        matched=match is not None)
    if evidence is not None:
        row.update({key: _scalar(value) for key, value in evidence.items()})
    if geometry is not None:
        row.update({key: _scalar(value) for key, value in geometry.items()})
    if match is not None:
        row.update(
            pred_index=int(match['pred_index']),
            pred_score=float(match['pred_score']),
            rotated_iou=float(match['rotated_iou']),
            pred_angle=float(match['pred_angle']),
            gt_angle=float(match['gt_angle']),
            angle_error_deg=float(match['angle_error_deg']),
            angle_error_c4_deg=float(match['angle_error_c4_deg']),
            large_angle_error=bool(match['angle_error_deg'] > 15.0))
    return row


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=Path)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-images', type=int)
    parser.add_argument('--score-threshold', type=float, default=0.05)
    parser.add_argument('--iou-threshold', type=float, default=0.5)
    return parser.parse_args()
~~~

- [ ] **Step 4: Implement the frozen Runner and collection loop**

Complete the collector with:

~~~python
def _evidence_cfg():
    return dict(
        type='SCQOFPNInstanceEvidence',
        min_box_size=2.0,
        roi_extractor=dict(
            type='mmdet.SingleRoIExtractor',
            roi_layer=dict(
                type='RoIAlign', output_size=14, sampling_ratio=2),
            out_channels=256,
            featmap_strides=[8, 16, 32, 64, 128]),
        evidence=dict(
            type='SCQOStabilizerEvidence',
            roi_size=14,
            channels=256,
            block_size=2,
            negative_seed=3407,
            min_support_fraction=0.95))


def _row_evidence(result, row_index):
    ignored = {'batch_index', 'instance_index', 'label', 'square_hbox'}
    return {
        key: value[row_index]
        for key, value in result.items()
        if key not in ignored
    }


def collect(args) -> Dict:
    config_path = args.config.resolve()
    checkpoint_path = args.checkpoint.resolve()
    output_path = args.output.resolve()
    manifest_path = output_path.with_suffix('.manifest.json')
    if not config_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError('config and checkpoint must exist')
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError('output and manifest must not already exist')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    register_all_modules()
    cfg = Config.fromfile(config_path)
    cfg.load_from = None
    cfg.resume = False
    runner_workspace = tempfile.TemporaryDirectory(
        prefix=f'orbdet_scqo_{args.run_name}_')
    cfg.work_dir = runner_workspace.name
    runner = Runner.from_cfg(cfg)
    runner.load_checkpoint(str(checkpoint_path))
    model = runner.model
    model.eval()
    evidence_module = MODELS.build(_evidence_cfg()).to(
        next(model.parameters()).device)
    evidence_module.eval()

    row_count = 0
    image_count = 0
    matched_count = 0
    with output_path.open('x', encoding='utf-8') as stream:
        with torch.inference_mode():
            for data in runner.val_dataloader:
                processed = model.data_preprocessor(data, training=False)
                inputs = processed['inputs']
                samples = processed['data_samples']
                features = model.extract_feat(inputs)
                predictions = model.predict(inputs, samples, rescale=False)
                raw_evidence = evidence_module(
                    features,
                    [sample.gt_instances for sample in samples],
                    [sample.metainfo for sample in samples])

                for batch_index, (sample, prediction) in enumerate(
                        zip(samples, predictions)):
                    gt = sample.gt_instances
                    pred = prediction.pred_instances
                    matches = match_rotated_predictions(
                        pred.bboxes.tensor,
                        pred.scores,
                        pred.labels,
                        gt.bboxes.tensor,
                        gt.labels,
                        score_threshold=args.score_threshold,
                        iou_threshold=args.iou_threshold)
                    match_by_gt = {}
                    for offset in range(matches['gt_index'].numel()):
                        gt_index = int(matches['gt_index'][offset])
                        pred_index = int(matches['pred_index'][offset])
                        error = periodic_angle_error(
                            pred.bboxes.tensor[pred_index, 4],
                            gt.bboxes.tensor[gt_index, 4])
                        error_c4 = periodic_angle_error(
                            pred.bboxes.tensor[pred_index, 4],
                            gt.bboxes.tensor[gt_index, 4],
                            period=math.pi / 2)
                        match_by_gt[gt_index] = dict(
                            pred_index=pred_index,
                            pred_score=float(pred.scores[pred_index]),
                            rotated_iou=float(matches['iou'][offset]),
                            pred_angle=float(pred.bboxes.tensor[pred_index, 4]),
                            gt_angle=float(gt.bboxes.tensor[gt_index, 4]),
                            angle_error_deg=float(torch.rad2deg(error)),
                            angle_error_c4_deg=float(
                                torch.rad2deg(error_c4)))
                    evidence_by_gt = {}
                    rows = (raw_evidence['batch_index'] == batch_index).nonzero()
                    for row_index in rows.reshape(-1).tolist():
                        instance_index = int(
                            raw_evidence['instance_index'][row_index])
                        evidence_by_gt[instance_index] = _row_evidence(
                            raw_evidence, row_index)
                    for gt_index, label in enumerate(gt.labels.tolist()):
                        match = match_by_gt.get(gt_index)
                        gt_box = gt.bboxes.tensor[gt_index]
                        width = float(gt_box[2])
                        height = float(gt_box[3])
                        short_side = max(min(width, height), 1e-8)
                        row = build_evidence_row(
                            args.run_name,
                            sample.metainfo['img_id'],
                            gt_index,
                            label,
                            evidence_by_gt.get(gt_index),
                            dict(
                                gt_width=width,
                                gt_height=height,
                                gt_area=width * height,
                                gt_aspect_ratio=max(width, height) / short_side),
                            match)
                        stream.write(json.dumps(
                            row, ensure_ascii=False, sort_keys=True) + '\n')
                        row_count += 1
                        matched_count += int(match is not None)
                    image_count += 1
                if args.max_images is not None and image_count >= args.max_images:
                    break

    manifest = dict(
        schema_version=1,
        run_name=args.run_name,
        config=str(config_path),
        config_sha256=sha256(config_path),
        checkpoint=str(checkpoint_path),
        checkpoint_sha256=sha256(checkpoint_path),
        output=str(output_path),
        image_count=image_count,
        row_count=row_count,
        matched_count=matched_count,
        score_threshold=args.score_threshold,
        iou_threshold=args.iou_threshold)
    with manifest_path.open('x', encoding='utf-8') as stream:
        stream.write(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
            + '\n')
    runner_workspace.cleanup()
    return manifest


def main():
    args = parse_args()
    manifest = collect(args)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


if __name__ == '__main__':
    main()
~~~

- [ ] **Step 5: Run CLI/unit checks without loading a checkpoint**

~~~bash
rtk env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
  PYTHONPATH=/data1/zcy/Orbdet \
  /data/zcy/anaconda3/envs/orbdet/bin/python -m pytest -q \
  -p no:cacheprovider tests/test_scqo_plan_a_tools.py
~~~

Expected: all collector CLI, schema, immutability, and no-training contract
tests pass without loading a checkpoint. The reporter tests are added only in
Task 7.

- [ ] **Step 6: Commit the collector**

~~~bash
rtk git add \
  tools/analysis_tools/scqo_collect_evidence.py \
  tests/test_scqo_plan_a_tools.py
rtk git commit -m "feat: add frozen SCQO evidence collector"
~~~

## Task 7: Build the Gate B report

**Files:**

- Create: <code>tools/analysis_tools/scqo_report_evidence.py</code>
- Modify: <code>tests/test_scqo_plan_a_tools.py</code>

- [ ] **Step 1: Add failing report and gate tests**

Append a synthetic JSONL fixture test:

~~~python
def test_reporter_writes_pass_fail_evidence_without_persisting_probe(tmp_path):
    rows = []
    for image_index in range(30):
        for local in range(4):
            target = local >= 2
            rows.append(dict(
                run_name='synthetic',
                image_id=str(image_index),
                gt_index=local,
                label=0,
                evidence_available=True,
                matched=True,
                rotated_iou=0.8,
                angle_error_deg=20.0 if target else 2.0,
                angle_error_c4_deg=5.0 if target else 2.0,
                large_angle_error=target,
                c2_determinantal=0.1,
                c2_spectral_tail=0.1,
                c2_fixed_space=0.1,
                c2_q_gap=0.9,
                c4_determinantal=float(target),
                c4_spectral_tail=float(target),
                c4_fixed_space=float(target),
                c4_q_gap=float(not target),
                energy=1.0,
                variance=0.5 + 0.001 * image_index,
                energy_guard=0.0,
                variance_guard=0.0,
                support_fraction=1.0,
                fpn_level=image_index % 3,
                gt_width=8.0 + image_index,
                gt_height=4.0,
                gt_area=(8.0 + image_index) * 4.0,
                gt_aspect_ratio=(8.0 + image_index) / 4.0,
                a2=float(not target),
                a4=float(target),
                negative_energy=1.0,
                c4_negative_margin=float(target),
                valid=True))
    source = tmp_path / 'rows.jsonl'
    source.write_text(
        ''.join(json.dumps(row) + '\n' for row in rows),
        encoding='utf-8')
    output = tmp_path / 'report'
    completed = subprocess.run(
        [
            sys.executable, str(REPORTER), str(source),
            '--output-dir', str(output)
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True)

    assert completed.returncode == 0
    summary = json.loads((output / 'summary.json').read_text())
    assert summary['runs']['synthetic']['gate_b'] == 'PASS'
    assert (output / 'fres_scqo_plan_a_hrsc_audit.md').is_file()
    assert not list(output.glob('*.pth'))
    assert not list(output.glob('*.pkl'))

    repeated = subprocess.run(
        [
            sys.executable, str(REPORTER), str(source),
            '--output-dir', str(output)
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True)
    assert repeated.returncode != 0
    assert 'already exist' in repeated.stderr


def test_all_plan_a_tools_cannot_start_training_or_overwrite_status():
    text = COLLECTOR.read_text() + REPORTER.read_text()
    forbidden = (
        'tools/train.py', 'train_step(', 'optim_wrapper',
        'tmux', 'nohup', 'FORMAL_TRAINING_NOT_STARTED.md')
    assert all(token not in text for token in forbidden)
~~~

- [ ] **Step 2: Run the report test and verify RED**

~~~bash
rtk env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
  PYTHONPATH=/data1/zcy/Orbdet \
  /data/zcy/anaconda3/envs/orbdet/bin/python -m pytest -q \
  -p no:cacheprovider tests/test_scqo_plan_a_tools.py
~~~

Expected: the reporter test fails because the reporter is absent.

- [ ] **Step 3: Implement row loading, grouped probes, and exact Gate B**

Create the reporter:

~~~python
#!/usr/bin/env python3
"""Report the pre-registered SCQO Plan A evidence gate."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from mmrotate.evaluation.functional.scqo_diagnostics import (
    binary_average_precision, binary_auroc, cross_validated_logistic_probe,
    expected_calibration_error)


FEATURES = (
    'c2_determinantal', 'c2_spectral_tail', 'c2_fixed_space', 'c2_q_gap',
    'c4_determinantal', 'c4_spectral_tail', 'c4_fixed_space', 'c4_q_gap',
    'a2', 'a4', 'c4_negative_margin', 'energy', 'variance',
    'support_fraction')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('evidence', nargs='+', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    return parser.parse_args()


def load_rows(paths):
    runs = defaultdict(list)
    for path in paths:
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                row = json.loads(line)
                runs[row['run_name']].append(row)
    return runs


def _probe(rows, feature_names):
    x = np.asarray(
        [[float(row[name]) for name in feature_names] for row in rows])
    y = np.asarray(
        [float(row['angle_error_deg']) > 15.0 for row in rows],
        dtype=np.int64)
    groups = np.asarray([row['image_id'] for row in rows])
    probability = cross_validated_logistic_probe(
        x, y, groups, folds=3, l2=1e-2, max_iter=100)
    return dict(
        probability=probability,
        auroc=binary_auroc(y, probability),
        auprc=binary_average_precision(y, probability),
        ece=expected_calibration_error(y, probability, bins=10),
        reliability=_reliability_curve(y, probability, bins=10))


def _reliability_curve(target, probability, bins=10):
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = []
    for index in range(bins):
        mask = (
            (probability >= edges[index])
            & ((probability <= edges[index + 1]) if index == bins - 1
               else (probability < edges[index + 1])))
        result.append(dict(
            lower=float(edges[index]),
            upper=float(edges[index + 1]),
            count=int(mask.sum()),
            confidence=float(probability[mask].mean()) if mask.any() else None,
            error_rate=float(target[mask].mean()) if mask.any() else None))
    return result


def _masked_auroc(target, probability, mask):
    if int(mask.sum()) < 10 or np.unique(target[mask]).size != 2:
        return None
    return binary_auroc(target[mask], probability[mask])


def _tertile_aurocs(rows, target, probability, field):
    values = np.asarray([float(row[field]) for row in rows])
    edges = np.quantile(values, [1.0 / 3.0, 2.0 / 3.0])
    index = np.digitize(values, edges, right=True)
    return [
        _masked_auroc(target, probability, index == bucket)
        for bucket in range(3)
    ]


def _category_aurocs(rows, target, probability, field):
    values = np.asarray([row[field] for row in rows])
    return {
        str(value): _masked_auroc(target, probability, values == value)
        for value in sorted(set(values.tolist()), key=str)
    }


def _angle_summary(values):
    values = np.asarray(values, dtype=np.float64)
    return dict(
        mean=float(values.mean()),
        median=float(np.median(values)),
        p90=float(np.quantile(values, 0.9)),
        within_5=float((values <= 5.0).mean()),
        within_10=float((values <= 10.0).mean()),
        within_15=float((values <= 15.0).mean()))


def _reject_by_fpn(rows):
    available = [row for row in rows if row.get('evidence_available')]
    result = {}
    for level in sorted({int(row['fpn_level']) for row in available}):
        selected = [row for row in available if int(row['fpn_level']) == level]
        result[str(level)] = dict(
            count=len(selected),
            reject_fraction=float(np.mean(
                [not bool(row.get('valid', False)) for row in selected])))
    return result


def summarize_run(rows):
    eligible = [
        row for row in rows
        if row.get('evidence_available')
        and row.get('matched')
        and float(row.get('rotated_iou', 0.0)) >= 0.5
        and bool(row.get('valid', False))
    ]
    y = np.asarray(
        [float(row['angle_error_deg']) > 15.0 for row in eligible],
        dtype=np.int64)
    if len(eligible) < 60 or y.sum() < 15 or (1 - y).sum() < 15:
        return dict(
            gate_b='INSUFFICIENT',
            total_rows=len(rows),
            eligible_rows=len(eligible),
            reject_by_fpn=_reject_by_fpn(rows),
            reason='need at least 60 eligible rows and 15 examples per class')

    combined = _probe(eligible, FEATURES)
    energy = _probe(eligible, ('energy', ))
    variance = _probe(eligible, ('variance', ))
    aspect_ratio = _probe(eligible, ('gt_aspect_ratio', ))
    best_baseline = max(energy['auroc'], variance['auroc'])

    bin_aurocs = _tertile_aurocs(
        eligible, y, combined['probability'], 'variance')
    if any(value is None for value in bin_aurocs):
        bin_status = 'INSUFFICIENT'
    elif all(value > 0.5 for value in bin_aurocs):
        bin_status = 'PASS'
    else:
        bin_status = 'FAIL'

    numeric_pass = (
        combined['auroc'] >= 0.65
        and combined['auroc'] >= best_baseline + 0.05)
    if bin_status == 'INSUFFICIENT':
        gate = 'INSUFFICIENT'
    elif numeric_pass and bin_status == 'PASS':
        gate = 'PASS'
    else:
        gate = 'FAIL'
    return dict(
        gate_b=gate,
        total_rows=len(rows),
        eligible_rows=len(eligible),
        large_error_rows=int(y.sum()),
        combined_auroc=combined['auroc'],
        combined_auprc=combined['auprc'],
        combined_ece=combined['ece'],
        combined_reliability=combined['reliability'],
        energy_auroc=energy['auroc'],
        variance_auroc=variance['auroc'],
        aspect_ratio_auroc=aspect_ratio['auroc'],
        best_baseline_auroc=best_baseline,
        auroc_gain=combined['auroc'] - best_baseline,
        angle_error_e2=_angle_summary(
            [row['angle_error_deg'] for row in eligible]),
        angle_error_e4=_angle_summary(
            [row['angle_error_c4_deg'] for row in eligible]),
        stratified_aurocs=dict(
            texture=bin_aurocs,
            aspect_ratio=_tertile_aurocs(
                eligible, y, combined['probability'], 'gt_aspect_ratio'),
            area=_tertile_aurocs(
                eligible, y, combined['probability'], 'gt_area'),
            label=_category_aurocs(
                eligible, y, combined['probability'], 'label'),
            fpn_level=_category_aurocs(
                eligible, y, combined['probability'], 'fpn_level')),
        reject_by_fpn=_reject_by_fpn(rows))
~~~

- [ ] **Step 4: Implement immutable JSON and Chinese Markdown output**

Append:

~~~python
def _markdown(summary):
    lines = [
        '# SCQO Plan A HRSC 只读证据审计',
        '',
        '本报告只读取冻结 checkpoint 和 HRSC validation；没有训练、调参或测试集评估。',
        '',
        '| run | Gate B | eligible | AUROC | baseline | gain | e2 median | e2 P90 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for name, item in summary['runs'].items():
        lines.append(
            f"| {name} | {item['gate_b']} | "
            f"{item.get('eligible_rows', 0)} | "
            f"{item.get('combined_auroc', float('nan')):.4f} | "
            f"{item.get('best_baseline_auroc', float('nan')):.4f} | "
            f"{item.get('auroc_gain', float('nan')):.4f} | "
            f"{item.get('angle_error_e2', {}).get('median', float('nan')):.2f} | "
            f"{item.get('angle_error_e2', {}).get('p90', float('nan')):.2f} |")
    lines.extend([
        '',
        'Gate B 要求：combined AUROC ≥ 0.65、相对最佳能量/方差单变量基线'
        '提高 ≥ 0.05，并且低、中、高三个纹理桶的 AUROC 均 > 0.5。',
        '',
        'PASS 只授权继续讨论 Plan B；FAIL/INSUFFICIENT 均不得启动 E4。',
        ''
    ])
    return '\n'.join(lines)


def main():
    args = parse_args()
    output_dir = args.output_dir.resolve()
    summary_path = output_dir / 'summary.json'
    markdown_path = output_dir / 'fres_scqo_plan_a_hrsc_audit.md'
    if summary_path.exists() or markdown_path.exists():
        raise FileExistsError('report outputs already exist')
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = load_rows(args.evidence)
    summary = dict(
        schema_version=1,
        evidence=[str(path.resolve()) for path in args.evidence],
        runs={name: summarize_run(rows) for name, rows in sorted(runs.items())})
    with summary_path.open('x', encoding='utf-8') as stream:
        stream.write(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
            + '\n')
    with markdown_path.open('x', encoding='utf-8') as stream:
        stream.write(_markdown(summary))
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == '__main__':
    main()
~~~

- [ ] **Step 5: Run all Plan A tool tests**

~~~bash
rtk env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
  PYTHONPATH=/data1/zcy/Orbdet \
  /data/zcy/anaconda3/envs/orbdet/bin/python -m pytest -q \
  -p no:cacheprovider tests/test_scqo_plan_a_tools.py
~~~

Expected: all CLI, schema, overwrite, gate, and no-training tests pass.

- [ ] **Step 6: Commit the reporter**

~~~bash
rtk git add \
  tools/analysis_tools/scqo_report_evidence.py \
  tests/test_scqo_plan_a_tools.py
rtk git commit -m "feat: add SCQO evidence gate report"
~~~

## Task 8: Verify the full code path and run the authorized read-only audit

**Files:**

- Create:
  <code>resultmd/exp_scqo_plan_a_hrsc_audit_20260817/v02_seed3407_best.jsonl</code>
- Create:
  <code>resultmd/exp_scqo_plan_a_hrsc_audit_20260817/v02_seed3407_best.manifest.json</code>
- Create:
  <code>resultmd/exp_scqo_plan_a_hrsc_audit_20260817/godc_seed3407_best.jsonl</code>
- Create:
  <code>resultmd/exp_scqo_plan_a_hrsc_audit_20260817/godc_seed3407_best.manifest.json</code>
- Create:
  <code>resultmd/exp_scqo_plan_a_hrsc_audit_20260817/summary.json</code>
- Create:
  <code>resultmd/exp_scqo_plan_a_hrsc_audit_20260817/fres_scqo_plan_a_hrsc_audit.md</code>

- [ ] **Step 1: Format only the Plan A source files**

~~~bash
rtk env PYTHONNOUSERSITE=1 \
  /data/zcy/anaconda3/envs/orbdet/bin/python -m isort \
  mmrotate/models/losses/group_orbit_determinantal_cluster_loss.py \
  mmrotate/models/losses/scqo_harmonic_equivariance_loss.py \
  mmrotate/models/task_modules/scqo_stabilizer_evidence.py \
  mmrotate/models/task_modules/scqo_fpn_evidence_adapter.py \
  mmrotate/evaluation/functional/scqo_diagnostics.py \
  tools/analysis_tools/scqo_collect_evidence.py \
  tools/analysis_tools/scqo_report_evidence.py \
  tests/test_scqo_harmonic_equivariance_loss.py \
  tests/test_scqo_stabilizer_evidence.py \
  tests/test_scqo_fpn_evidence_adapter.py \
  tests/test_scqo_diagnostics.py \
  tests/test_scqo_plan_a_tools.py
~~~

~~~bash
rtk env PYTHONNOUSERSITE=1 \
  /data/zcy/anaconda3/envs/orbdet/bin/python -m yapf -ir \
  mmrotate/models/losses/group_orbit_determinantal_cluster_loss.py \
  mmrotate/models/losses/scqo_harmonic_equivariance_loss.py \
  mmrotate/models/task_modules/scqo_stabilizer_evidence.py \
  mmrotate/models/task_modules/scqo_fpn_evidence_adapter.py \
  mmrotate/evaluation/functional/scqo_diagnostics.py \
  tools/analysis_tools/scqo_collect_evidence.py \
  tools/analysis_tools/scqo_report_evidence.py \
  tests/test_scqo_harmonic_equivariance_loss.py \
  tests/test_scqo_stabilizer_evidence.py \
  tests/test_scqo_fpn_evidence_adapter.py \
  tests/test_scqo_diagnostics.py \
  tests/test_scqo_plan_a_tools.py
~~~

Expected: only named Plan A files change.

- [ ] **Step 2: Run the complete targeted regression suite**

~~~bash
rtk env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
  PYTHONPATH=/data1/zcy/Orbdet \
  /data/zcy/anaconda3/envs/orbdet/bin/python -m pytest -q \
  -p no:cacheprovider \
  tests/test_group_orbit_determinantal_cluster.py \
  tests/test_hbox_fpn_group_orbit_loss.py \
  tests/test_orbdet_godc.py \
  tests/test_orbdet_v0_2.py \
  tests/test_scqo_harmonic_equivariance_loss.py \
  tests/test_scqo_stabilizer_evidence.py \
  tests/test_scqo_fpn_evidence_adapter.py \
  tests/test_scqo_diagnostics.py \
  tests/test_scqo_plan_a_tools.py
~~~

Expected: all selected tests pass with no CUDA allocation and no training
process.

- [ ] **Step 3: Check that physical GPU 8 is safe to use**

~~~bash
rtk nvidia-smi -i 8 \
  --query-compute-apps=pid,process_name,used_memory \
  --format=csv,noheader
~~~

Expected: no foreign compute process is using GPU 8. If a foreign process is
listed, stop before inference and ask the user for a different GPU; never
terminate, signal, or modify that process.

- [ ] **Step 4: Run a two-image bounded collector check on one physical GPU**

Use the explicit bounded output path below only if it does not already exist.
If it exists, choose a new explicit path; do not delete or overwrite it:

~~~bash
rtk env CUDA_VISIBLE_DEVICES=8 PYTHONNOUSERSITE=1 \
  PYTHONPATH=/data1/zcy/Orbdet \
  /data/zcy/anaconda3/envs/orbdet/bin/python \
  tools/analysis_tools/scqo_collect_evidence.py \
  configs/orbdet/orbdet_v0_2_r50_hrsc_clean_gpu89.py \
  --checkpoint \
  work_dirs/formal/orbdet_v0_2_hrsc_clean_gpu89_seed3407_20260814/best_dota_mAP_epoch_48.pth \
  --run-name v02_seed3407_bounded \
  --output /tmp/orbdet_scqo_plan_a_bounded/v02.jsonl \
  --max-images 2
~~~

Expected: exits zero, writes a JSONL and manifest, reports exactly two images,
and does not create a checkpoint. If the explicit temporary directory already
exists, choose a new explicit path rather than overwriting it.

- [ ] **Step 5: Collect the full frozen v0.2 best-checkpoint evidence**

~~~bash
rtk env CUDA_VISIBLE_DEVICES=8 PYTHONNOUSERSITE=1 \
  PYTHONPATH=/data1/zcy/Orbdet \
  /data/zcy/anaconda3/envs/orbdet/bin/python \
  tools/analysis_tools/scqo_collect_evidence.py \
  configs/orbdet/orbdet_v0_2_r50_hrsc_clean_gpu89.py \
  --checkpoint \
  work_dirs/formal/orbdet_v0_2_hrsc_clean_gpu89_seed3407_20260814/best_dota_mAP_epoch_48.pth \
  --run-name v02_seed3407_best \
  --output \
  resultmd/exp_scqo_plan_a_hrsc_audit_20260817/v02_seed3407_best.jsonl
~~~

Expected: exits zero and manifest records 181 HRSC validation images, a
non-zero row count, a non-zero matched count, and the checkpoint SHA-256.

- [ ] **Step 6: Collect the full frozen GODC best-checkpoint evidence**

~~~bash
rtk env CUDA_VISIBLE_DEVICES=8 PYTHONNOUSERSITE=1 \
  PYTHONPATH=/data1/zcy/Orbdet \
  /data/zcy/anaconda3/envs/orbdet/bin/python \
  tools/analysis_tools/scqo_collect_evidence.py \
  configs/orbdet/orbdet_godc_c2_r50_hrsc_clean_gpu89.py \
  --checkpoint \
  work_dirs/formal/orbdet_godc_c2_hrsc_clean_gpu89_seed3407_20260815/best_dota_mAP_epoch_24.pth \
  --run-name godc_seed3407_best \
  --output \
  resultmd/exp_scqo_plan_a_hrsc_audit_20260817/godc_seed3407_best.jsonl
~~~

Expected: exits zero with the same schema and frozen-checkpoint manifest.

- [ ] **Step 7: Produce the Gate B report**

~~~bash
rtk env CUDA_VISIBLE_DEVICES= PYTHONNOUSERSITE=1 \
  PYTHONPATH=/data1/zcy/Orbdet \
  /data/zcy/anaconda3/envs/orbdet/bin/python \
  tools/analysis_tools/scqo_report_evidence.py \
  resultmd/exp_scqo_plan_a_hrsc_audit_20260817/v02_seed3407_best.jsonl \
  resultmd/exp_scqo_plan_a_hrsc_audit_20260817/godc_seed3407_best.jsonl \
  --output-dir resultmd/exp_scqo_plan_a_hrsc_audit_20260817
~~~

Expected: writes <code>summary.json</code> and the Chinese receipt. A scientific
<code>FAIL</code> or <code>INSUFFICIENT</code> is a valid successful program
execution and must not be rewritten as <code>PASS</code>.

- [ ] **Step 8: Verify safety, schemas, and unchanged training status**

~~~bash
rtk env PYTHONNOUSERSITE=1 \
  /data/zcy/anaconda3/envs/orbdet/bin/python -c \
  "import json, pathlib; p=pathlib.Path('resultmd/exp_scqo_plan_a_hrsc_audit_20260817'); s=json.loads((p/'summary.json').read_text()); assert set(s['runs']) == {'godc_seed3407_best','v02_seed3407_best'}; assert not list(p.glob('*.pth')); print({k:v['gate_b'] for k,v in s['runs'].items()})"
~~~

~~~bash
rtk git status --short
~~~

Expected: the first command prints both run statuses and finds no checkpoint;
the second command shows only the Plan A artifacts plus pre-existing user
changes. In particular, the pre-existing modification to
<code>FORMAL_TRAINING_NOT_STARTED.md</code> remains unstaged and untouched.

- [ ] **Step 9: Commit formatting and the immutable audit receipt**

First commit any formatting-only changes:

~~~bash
rtk git add \
  mmrotate/models/losses/group_orbit_determinantal_cluster_loss.py \
  mmrotate/models/losses/scqo_harmonic_equivariance_loss.py \
  mmrotate/models/task_modules/scqo_stabilizer_evidence.py \
  mmrotate/models/task_modules/scqo_fpn_evidence_adapter.py \
  mmrotate/evaluation/functional/scqo_diagnostics.py \
  tools/analysis_tools/scqo_collect_evidence.py \
  tools/analysis_tools/scqo_report_evidence.py \
  tests/test_scqo_harmonic_equivariance_loss.py \
  tests/test_scqo_stabilizer_evidence.py \
  tests/test_scqo_fpn_evidence_adapter.py \
  tests/test_scqo_diagnostics.py \
  tests/test_scqo_plan_a_tools.py
rtk git commit -m "style: finalize SCQO plan A diagnostics"
~~~

If no formatting changes remain, skip that commit. Then commit only the audit
artifacts:

~~~bash
rtk git add resultmd/exp_scqo_plan_a_hrsc_audit_20260817
rtk git commit -m "docs: record SCQO plan A HRSC evidence"
~~~

## Final acceptance checklist

- [ ] Existing GODC forward values remain numerically identical to the
  per-instance weighted statistics.
- [ ] Harmonic rotations and reflections are exact for \(p=2\) and \(p=4\).
- [ ] Zero harmonic vectors receive finite non-zero loss; constant carriers
  cannot satisfy non-trivial rotations.
- [ ] ROI evidence is detached and returns C2/C4, \(A_2/A_4\), guards,
  support, and negative-control margin per instance.
- [ ] Square HBox RoIs retain batch/instance identity; edge support is explicit.
- [ ] Prediction matching is score-ordered, one-to-one, class-aware, and uses
  rotated IoU \(\ge 0.5\).
- [ ] The diagnostic probe groups folds by image ID and never persists weights.
- [ ] Gate B is exactly: combined AUROC \(\ge 0.65\), gain over the best
  energy/variance probe \(\ge 0.05\), and all evaluable texture bins have
  AUROC \(>0.5\).
- [ ] The audit reads HRSC validation and the two frozen best checkpoints only.
- [ ] No model prediction, training config, dataset, checkpoint, GPU process,
  test split, or formal-training status is modified.
- [ ] The report faithfully records PASS, FAIL, or INSUFFICIENT.

## Handoff after execution

- If both runs are FAIL or INSUFFICIENT, cancel E4 and retain only the
  independent \(q_2\) representation question for a possible Plan B.
- If at least one run passes Gate B, report which evidence features and texture
  bins carry the signal; this authorizes discussion of Plan B, not training.
- Plan B still requires a new user approval and a separate implementation plan.
