# Copyright (c) OpenMMLab. All rights reserved.
"""Group-orbit determinantal clustering primitives for Orbdet."""

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
            if value < 0.0:
                raise ValueError(f'{name} must be non-negative')
        if not any(value > 0.0 for value in component_weights.values()):
            raise ValueError('at least one component weight must be positive')
        if min_energy <= 0.0:
            raise ValueError('min_energy must be positive')
        if min_variance <= 0.0:
            raise ValueError('min_variance must be positive')
        if eps <= 0.0:
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

    def _record_diagnostics(self, determinantal: Tensor,
                            spectral_tail: Tensor, fixed_space: Tensor,
                            q_gap: Tensor) -> None:
        self.last_determinantal = determinantal.detach().mean()
        self.last_spectral_tail = spectral_tail.detach().mean()
        self.last_fixed_space = fixed_space.detach().mean()
        self.last_q_gap = q_gap.detach().mean()

    def forward(self, orbit: Tensor) -> Tensor:
        """Compute the loss for an orbit shaped ``[N, K, ...]``."""
        if not isinstance(orbit, Tensor):
            raise TypeError('orbit must be a torch.Tensor')
        if orbit.ndim < 3:
            raise ValueError('orbit must have shape [N, K, ...]')
        if orbit.shape[1] < 2:
            raise ValueError('orbit group order K must be at least 2')
        if orbit.numel() == 0 and orbit.shape[0] != 0:
            raise ValueError('orbit members must have non-empty features')
        if not bool(torch.isfinite(orbit).all()):
            raise ValueError('orbit must contain only finite values')

        work_orbit = orbit
        if orbit.dtype in (torch.float16, torch.bfloat16):
            work_orbit = orbit.float()
        batch_size, group_order = work_orbit.shape[:2]
        matrix = work_orbit.reshape(batch_size, group_order, -1)

        gram = matrix @ matrix.transpose(-1, -2)
        trace = gram.diagonal(dim1=-2, dim2=-1).sum(-1)
        trace_g2 = gram.square().sum(dim=(-2, -1))
        determinantal = (
            (trace.square() - trace_g2).clamp_min(0.0) /
            (2.0 * trace.square() + self.eps))

        eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(0.0)
        spectral_tail = (
            1.0 - eigenvalues[:, -1] /
            (trace + self.eps)).clamp_min(0.0)

        orbit_mean = matrix.mean(dim=1, keepdim=True)
        fixed_space = (matrix - orbit_mean).square().sum(dim=(1, 2)) / (
            matrix.square().sum(dim=(1, 2)) + self.eps)

        sigma_1 = eigenvalues[:, -1].sqrt()
        sigma_2 = eigenvalues[:, -2].sqrt()
        q_gap = (sigma_1 - sigma_2) / (sigma_1 + self.eps)
        self._record_diagnostics(determinantal, spectral_tail, fixed_space,
                                 q_gap)

        loss = (self.determinantal_weight * determinantal +
                self.spectral_tail_weight * spectral_tail +
                self.fixed_space_weight * fixed_space)
        if self.reduction == 'none':
            return loss
        if self.reduction == 'sum':
            return loss.sum()
        return loss.mean()
