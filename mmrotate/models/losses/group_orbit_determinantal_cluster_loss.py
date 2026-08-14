# Copyright (c) OpenMMLab. All rights reserved.
"""Group-orbit determinantal clustering primitives for Orbdet."""

import torch
from torch import Tensor


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
