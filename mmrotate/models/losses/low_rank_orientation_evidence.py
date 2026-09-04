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
