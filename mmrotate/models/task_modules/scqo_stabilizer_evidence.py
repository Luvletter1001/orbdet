# Copyright (c) OpenMMLab. All rights reserved.
"""Detached instance-level stabilizer evidence for SCQO."""

import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from mmrotate.models.losses import (GroupOrbitDeterminantalClusterLoss,
                                    build_planar_group_orbit)
from mmrotate.registry import MODELS


@MODELS.register_module()
class SCQOStabilizerEvidence(torch.nn.Module):
    """Extract non-learned, detached stabilizer evidence from square ROIs."""

    _STAT_NAMES = ('determinantal', 'spectral_tail', 'fixed_space', 'q_gap')

    def __init__(self,
                 roi_size: int = 14,
                 channels: int = 256,
                 block_size: int = 2,
                 negative_seed: int = 3407,
                 min_energy: float = 1e-4,
                 min_variance: float = 1e-4,
                 min_support_fraction: float = .95) -> None:
        super().__init__()
        if not isinstance(roi_size, int) or roi_size <= 0:
            raise ValueError('roi_size must be positive')
        if not isinstance(block_size, int) or block_size <= 0:
            raise ValueError('block_size must be positive')
        if roi_size % block_size:
            raise ValueError('roi_size must be divisible by block_size')
        if not isinstance(channels, int) or channels <= 0:
            raise ValueError('channels must be positive')
        if not 0.0 < min_support_fraction <= 1.0:
            raise ValueError('min_support_fraction must be in (0, 1]')
        if not math.isfinite(min_energy) or min_energy <= 0.0:
            raise ValueError('min_energy must be positive')
        if not math.isfinite(min_variance) or min_variance <= 0.0:
            raise ValueError('min_variance must be positive')

        self.roi_size = roi_size
        self.channels = channels
        self.block_size = block_size
        self.min_energy = min_energy
        self.min_variance = min_variance
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

        hann_1d = torch.hann_window(roi_size, periodic=False)
        self.register_buffer('hann', torch.outer(hann_1d, hann_1d)[None, None])
        blocks_per_side = roi_size // block_size
        generator = torch.Generator().manual_seed(negative_seed)
        permutation = torch.randperm(blocks_per_side**2, generator=generator)
        self.register_buffer('block_permutation', permutation)
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]
                                ]) / 8.0
        self.register_buffer('sobel_x', sobel_x[None, None])
        self.register_buffer('sobel_y', sobel_x.t().contiguous()[None, None])

    def _negative_control(self, features: Tensor) -> Tensor:
        """Shuffle non-overlapping blocks with the fixed diagnostic order."""
        batch_size, channels, _, _ = features.shape
        blocks_per_side = self.roi_size // self.block_size
        blocks = features.reshape(batch_size, channels, blocks_per_side,
                                  self.block_size, blocks_per_side,
                                  self.block_size)
        blocks = blocks.permute(0, 1, 2, 4, 3,
                                5).reshape(batch_size, channels,
                                           blocks_per_side**2, self.block_size,
                                           self.block_size)
        blocks = blocks.index_select(2, self.block_permutation)
        blocks = blocks.reshape(batch_size, channels, blocks_per_side,
                                blocks_per_side, self.block_size,
                                self.block_size)
        return blocks.permute(0, 1, 2, 4, 3, 5).reshape_as(features)

    def _patic(self, features: Tensor, support: Tensor, order: int) -> Tensor:
        """Compute a support-weighted p-atic gradient orientation amplitude."""
        batch_size, channels, height, width = features.shape
        if batch_size == 0:
            return features.new_empty((0, ))
        flattened = features.reshape(batch_size * channels, 1, height, width)
        grad_x = F.conv2d(flattened, self.sobel_x, padding=1)
        grad_y = F.conv2d(flattened, self.sobel_y, padding=1)
        magnitude = torch.sqrt(grad_x.square() + grad_y.square() + 1e-12)
        angle = torch.atan2(grad_y, grad_x)
        weight = magnitude.reshape(batch_size, channels, height,
                                   width) * support
        angle = angle.reshape(batch_size, channels, height, width)
        real = (weight * torch.cos(order * angle)).sum(dim=(1, 2, 3))
        imaginary = (weight * torch.sin(order * angle)).sum(dim=(1, 2, 3))
        return torch.sqrt(real.square() + imaginary.square()) / (
            weight.sum(dim=(1, 2, 3)) + 1e-8)

    def _validate_features(self, roi_features: Tensor) -> None:
        if not isinstance(roi_features, Tensor):
            raise TypeError('roi_features must be a torch.Tensor')
        if not roi_features.is_floating_point():
            raise TypeError('roi_features must be floating point')
        expected_shape = (self.channels, self.roi_size, self.roi_size)
        if roi_features.ndim != 4 or roi_features.shape[1:] != expected_shape:
            raise ValueError('roi_features must have shape [N, C, H, W]')
        if not bool(torch.isfinite(roi_features).all()):
            raise ValueError('roi_features must contain only finite values')

    def forward(self,
                roi_features: Tensor,
                valid_support: Optional[Tensor] = None) -> Dict[str, Tensor]:
        """Return detached per-ROI diagnostic evidence."""
        self._validate_features(roi_features)
        features = roi_features.detach().float()
        batch_size = features.shape[0]
        support_shape = (batch_size, 1, self.roi_size, self.roi_size)
        if valid_support is None:
            valid_support = features.new_ones(support_shape)
        else:
            if not isinstance(valid_support, Tensor):
                raise TypeError('valid_support must be a torch.Tensor')
            if valid_support.shape != support_shape:
                raise ValueError('valid_support must have shape [N, 1, H, W]')
            valid_support = valid_support.detach().to(
                device=features.device, dtype=features.dtype)
            if not bool(torch.isfinite(valid_support).all()):
                raise ValueError(
                    'valid_support must contain only finite values')
            valid_support = valid_support.clamp(0.0, 1.0)

        support_fraction = valid_support.mean(dim=(1, 2, 3))
        support = valid_support * self.hann
        energy = features.square().mean(dim=(1, 2, 3))
        variance = features.var(dim=(1, 2, 3), unbiased=False)
        scale = torch.sqrt(energy).clamp_min(1e-8).reshape(batch_size, 1, 1, 1)
        normalized = features / scale
        c2 = self.statistics_module.statistics(
            build_planar_group_orbit(normalized, 'c2'), support)
        c4 = self.statistics_module.statistics(
            build_planar_group_orbit(normalized, 'c4'), support)

        negative = self._negative_control(features)
        normalized_negative = negative / scale
        negative_c4 = self.statistics_module.statistics(
            build_planar_group_orbit(normalized_negative, 'c4'), support)
        energy_guard = (self.min_energy - energy).clamp_min(0.0)
        variance_guard = (self.min_variance - variance).clamp_min(0.0)
        valid = ((energy_guard == 0) & (variance_guard == 0) &
                 (support_fraction >= self.min_support_fraction))
        result = {
            'energy':
            energy.detach(),
            'variance':
            variance.detach(),
            'energy_guard':
            energy_guard.detach(),
            'variance_guard':
            variance_guard.detach(),
            'support_fraction':
            support_fraction.detach(),
            'a2':
            self._patic(normalized, support, 2).detach(),
            'a4':
            self._patic(normalized, support, 4).detach(),
            'negative_energy':
            negative.square().mean(dim=(1, 2, 3)).detach(),
            'c4_negative_margin':
            (negative_c4['fixed_space'] - c4['fixed_space']).detach(),
            'valid':
            valid.detach(),
        }
        for name in self._STAT_NAMES:
            result[f'c2_{name}'] = c2[name].detach()
            result[f'c4_{name}'] = c4[name].detach()
        return result
