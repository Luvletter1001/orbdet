# Copyright (c) OpenMMLab. All rights reserved.
"""Group-orbit determinantal clustering primitives for Orbdet."""

import math
from typing import Optional, Union

import torch
from torch import Tensor

from mmrotate.registry import MODELS


def build_planar_group_orbit(features: Tensor, group: str) -> Tensor:
    """Build an exact orbit for a finite planar symmetry group.

    Args:
        features: Feature tensor with shape ``[N, C, H, W]``.
        group: One of ``'c2'``, ``'c4'``, ``'d1'``, or ``'d2'``.

    Returns:
        Tensor: Orbit tensor with shape ``[N, |group|, C, H, W]``.
    """
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
        elements = tuple(torch.rot90(features, k, (-2, -1)) for k in range(4))
    elif group == 'd1':
        elements = (features, torch.flip(features, (-1, )))
    elif group == 'd2':
        elements = (features, torch.rot90(features, 2, (-2, -1)),
                    torch.flip(features, (-1, )), torch.flip(features, (-2, )))
    elif group == 'c1':
        raise ValueError('the trivial group c1 is not a valid hypothesis')
    else:
        raise ValueError(f'unsupported planar group: {group!r}')
    return torch.stack(elements, dim=1)


@MODELS.register_module()
class GroupOrbitDeterminantalClusterLoss(torch.nn.Module):
    """Rank-one and fixed-space loss for a precomputed group orbit.

    The group axis must be dimension one. All remaining dimensions are
    flattened before constructing the small group-order Gram matrix.
    """

    def __init__(self,
                 determinantal_weight: float = 1.0,
                 spectral_tail_weight: float = 0.0,
                 fixed_space_weight: float = 1.0,
                 energy_guard_weight: float = 1.0,
                 variance_guard_weight: float = 1.0,
                 min_energy: float = 1e-4,
                 min_variance: float = 1e-4,
                 eps: float = 1e-8,
                 reduction: str = 'mean') -> None:
        super().__init__()
        component_weights = {
            'determinantal_weight': determinantal_weight,
            'spectral_tail_weight': spectral_tail_weight,
            'fixed_space_weight': fixed_space_weight,
            'energy_guard_weight': energy_guard_weight,
            'variance_guard_weight': variance_guard_weight,
        }
        for name, value in component_weights.items():
            if not math.isfinite(value):
                raise ValueError(f'{name} must be finite')
            if value < 0.0:
                raise ValueError(f'{name} must be non-negative')
        if not any(value > 0.0 for value in component_weights.values()):
            raise ValueError('at least one component weight must be positive')
        if not math.isfinite(min_energy) or min_energy <= 0.0:
            raise ValueError('min_energy must be positive')
        if not math.isfinite(min_variance) or min_variance <= 0.0:
            raise ValueError('min_variance must be positive')
        if not math.isfinite(eps) or eps <= 0.0:
            raise ValueError('eps must be positive')
        if reduction not in ('none', 'mean', 'sum'):
            raise ValueError("reduction must be 'none', 'mean', or 'sum'")

        self.determinantal_weight = determinantal_weight
        self.spectral_tail_weight = spectral_tail_weight
        self.fixed_space_weight = fixed_space_weight
        self.energy_guard_weight = energy_guard_weight
        self.variance_guard_weight = variance_guard_weight
        self.min_energy = min_energy
        self.min_variance = min_variance
        self.eps = eps
        self.reduction = reduction

        self.last_determinantal = torch.tensor(0.0)
        self.last_spectral_tail = torch.tensor(0.0)
        self.last_fixed_space = torch.tensor(0.0)
        self.last_energy_guard = torch.tensor(0.0)
        self.last_variance_guard = torch.tensor(0.0)
        self.last_q_gap = torch.tensor(0.0)

    def _reset_diagnostics(self, reference: Tensor) -> None:
        zero = reference.new_zeros(())
        self.last_determinantal = zero
        self.last_spectral_tail = zero
        self.last_fixed_space = zero
        self.last_energy_guard = zero
        self.last_variance_guard = zero
        self.last_q_gap = zero

    def _record_diagnostics(self, determinantal: Tensor, spectral_tail: Tensor,
                            fixed_space: Tensor, energy_guard: Tensor,
                            variance_guard: Tensor, q_gap: Tensor) -> None:
        self.last_determinantal = determinantal.detach().mean()
        self.last_spectral_tail = spectral_tail.detach().mean()
        self.last_fixed_space = fixed_space.detach().mean()
        self.last_energy_guard = energy_guard.detach().mean()
        self.last_variance_guard = variance_guard.detach().mean()
        self.last_q_gap = q_gap.detach().mean()

    def _apply_support_mask(self, orbit: Tensor,
                            support_mask: Optional[Tensor]) -> Tensor:
        if support_mask is None:
            return orbit
        if not isinstance(support_mask, Tensor):
            raise TypeError('support_mask must be a torch.Tensor')
        if not bool(torch.isfinite(support_mask).all()):
            raise ValueError('support_mask must contain only finite values')
        if bool((support_mask < 0).any()):
            raise ValueError('support_mask must be non-negative')

        mask = support_mask.to(device=orbit.device, dtype=orbit.dtype)
        if mask.ndim == orbit.ndim:
            broadcast_mask = mask
        elif mask.ndim == 0:
            broadcast_mask = mask
            while broadcast_mask.ndim < orbit.ndim:
                broadcast_mask = broadcast_mask.unsqueeze(0)
        else:
            if mask.ndim > orbit.ndim - 1:
                raise ValueError('support_mask is not broadcastable to orbit')
            if mask.shape[0] in (1, orbit.shape[0]):
                broadcast_mask = mask.unsqueeze(1)
            else:
                broadcast_mask = mask.unsqueeze(0).unsqueeze(0)
            while broadcast_mask.ndim < orbit.ndim:
                broadcast_mask = broadcast_mask.unsqueeze(2)

        try:
            return orbit * broadcast_mask
        except RuntimeError as error:
            raise ValueError(
                'support_mask is not broadcastable to orbit') from error

    def _prepare_weight(self, weight: Optional[Tensor], reference: Tensor,
                        batch_size: int) -> Optional[Tensor]:
        if weight is None:
            return None
        if not isinstance(weight, Tensor):
            weight = reference.new_tensor(weight)
        else:
            weight = weight.to(device=reference.device, dtype=reference.dtype)
        if weight.ndim == 0:
            weight = weight.expand(batch_size)
        elif weight.numel() == batch_size:
            weight = weight.reshape(batch_size)
        else:
            raise ValueError(
                'weight must be scalar or have one value per sample')
        if not bool(torch.isfinite(weight).all()):
            raise ValueError('weight must contain only finite values')
        if bool((weight < 0).any()):
            raise ValueError('weight must be non-negative')
        return weight

    def _reduce(self, loss: Tensor, weight: Optional[Tensor], reduction: str,
                avg_factor: Optional[Union[float, Tensor]]) -> Tensor:
        if weight is not None:
            loss = loss * weight
        if reduction == 'none':
            if avg_factor is not None:
                raise ValueError(
                    'avg_factor can only be used with mean reduction')
            return loss
        if reduction == 'sum':
            if avg_factor is not None:
                raise ValueError(
                    'avg_factor can only be used with mean reduction')
            return loss.sum()
        if avg_factor is None:
            return loss.mean()

        factor = loss.new_tensor(avg_factor)
        if factor.numel() != 1 or not bool(torch.isfinite(factor).all()):
            raise ValueError('avg_factor must be one finite scalar')
        if bool(factor <= 0):
            raise ValueError('avg_factor must be positive')
        return loss.sum() / factor

    def forward(self,
                orbit: Tensor,
                support_mask: Optional[Tensor] = None,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[Union[float, Tensor]] = None,
                reduction_override: Optional[str] = None) -> Tensor:
        """Compute the loss for an orbit shaped ``[N, K, ...]``."""
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
        reduction = reduction_override or self.reduction
        if reduction not in ('none', 'mean', 'sum'):
            raise ValueError("reduction must be 'none', 'mean', or 'sum'")

        work_orbit = orbit
        if orbit.dtype in (torch.float16, torch.bfloat16):
            work_orbit = orbit.float()
        batch_size, group_order = work_orbit.shape[:2]
        if batch_size == 0:
            self._reset_diagnostics(work_orbit)
            if reduction == 'none':
                return work_orbit.new_empty((0, ))
            return work_orbit.sum() * 0.0

        work_orbit = self._apply_support_mask(work_orbit, support_mask)
        matrix = work_orbit.reshape(batch_size, group_order, -1)

        gram = matrix @ matrix.transpose(-1, -2)
        trace = gram.diagonal(dim1=-2, dim2=-1).sum(-1)
        trace_g2 = gram.square().sum(dim=(-2, -1))
        determinantal = ((trace.square() - trace_g2).clamp_min(0.0) /
                         (2.0 * trace.square() + self.eps))

        eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(0.0)
        spectral_tail = (1.0 - eigenvalues[:, -1] /
                         (trace + self.eps)).clamp_min(0.0)

        orbit_mean = matrix.mean(dim=1, keepdim=True)
        fixed_space = (matrix - orbit_mean).square().sum(dim=(1, 2)) / (
            matrix.square().sum(dim=(1, 2)) + self.eps)

        sigma_1 = eigenvalues[:, -1].sqrt()
        sigma_2 = eigenvalues[:, -2].sqrt()
        q_gap = (sigma_1 - sigma_2) / (sigma_1 + self.eps)

        energy = matrix.square().mean(dim=(1, 2))
        variance = matrix.var(dim=-1, unbiased=False).mean(dim=1)
        energy_guard = (self.min_energy - energy).clamp_min(0.0)
        variance_guard = (self.min_variance - variance).clamp_min(0.0)
        self._record_diagnostics(determinantal, spectral_tail, fixed_space,
                                 energy_guard, variance_guard, q_gap)

        loss = matrix.new_zeros((batch_size, ))
        components = (
            (self.determinantal_weight, determinantal),
            (self.spectral_tail_weight, spectral_tail),
            (self.fixed_space_weight, fixed_space),
            (self.energy_guard_weight, energy_guard),
            (self.variance_guard_weight, variance_guard),
        )
        for component_weight, component in components:
            if component_weight > 0.0:
                loss = loss + component_weight * component

        sample_weight = self._prepare_weight(weight, matrix, batch_size)
        return self._reduce(loss, sample_weight, reduction, avg_factor)
