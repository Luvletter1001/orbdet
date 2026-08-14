# Copyright (c) OpenMMLab. All rights reserved.
"""Quality-aware consistency loss for Orbdet.

The reliability term uses the second circular harmonic of two orientation
predictions.  It is invariant to the pi-periodicity of an oriented axis and
also accepts the pi/2 width/height exchange used by H2RBox.
"""

import math
from typing import Optional

import torch
from mmdet.utils import ConfigType
from torch import Tensor

from mmrotate.registry import MODELS


def axis_harmonic_reliability(angle_delta: Tensor,
                              min_quality: float = 0.25,
                              gamma: float = 2.0) -> Tensor:
    """Measure agreement between two unoriented box axes.

    Args:
        angle_delta: Difference between the predicted and target angles in
            radians.
        min_quality: Gradient-preserving reliability floor.
        gamma: Exponent controlling how sharply ambiguous axes are down-
            weighted.

    Returns:
        A tensor with the same shape as ``angle_delta`` and values in
        ``[min_quality, 1]``.
    """
    if not 0.0 <= min_quality <= 1.0:
        raise ValueError('min_quality must be in [0, 1]')
    if gamma <= 0.0:
        raise ValueError('gamma must be positive')

    # |cos(delta)| is the resultant agreement of the second circular
    # harmonic. |sin(delta)| represents its width/height-swapped axis.
    raw = torch.maximum(angle_delta.cos().abs(), angle_delta.sin().abs())
    ambiguous_axis = math.sqrt(0.5)
    normalized = ((raw - ambiguous_axis) /
                  (1.0 - ambiguous_axis)).clamp(0.0, 1.0)
    return min_quality + (1.0 - min_quality) * normalized.pow(gamma)


@MODELS.register_module()
class OrbdetHarmonicConsistencyLoss(torch.nn.Module):
    """H2RBox consistency weighted by detached harmonic reliability."""

    def __init__(self,
                 center_loss_cfg: ConfigType = dict(
                     type='mmdet.L1Loss', loss_weight=0.0),
                 shape_loss_cfg: ConfigType = dict(
                     type='mmdet.IoULoss', loss_weight=1.0),
                 angle_loss_cfg: ConfigType = dict(
                     type='mmdet.L1Loss', loss_weight=1.0),
                 min_quality: float = 0.25,
                 gamma: float = 2.0,
                 high_quality_thr: float = 0.75,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0) -> None:
        super().__init__()
        if not 0.0 <= min_quality <= 1.0:
            raise ValueError('min_quality must be in [0, 1]')
        if gamma <= 0.0:
            raise ValueError('gamma must be positive')
        if not 0.0 <= high_quality_thr <= 1.0:
            raise ValueError('high_quality_thr must be in [0, 1]')

        self.center_loss = MODELS.build(center_loss_cfg)
        self.shape_loss = MODELS.build(shape_loss_cfg)
        self.angle_loss = MODELS.build(angle_loss_cfg)
        self.min_quality = min_quality
        self.gamma = gamma
        self.high_quality_thr = high_quality_thr
        self.reduction = reduction
        self.loss_weight = loss_weight

        # Diagnostics deliberately stay out of state_dict.
        self.last_quality_mean = torch.tensor(0.0)
        self.last_quality_min = torch.tensor(0.0)
        self.last_quality_high_frac = torch.tensor(0.0)

    def _record_quality(self, quality: Tensor) -> None:
        quality = quality.detach()
        if quality.numel() == 0:
            zero = quality.new_zeros(())
            self.last_quality_mean = zero
            self.last_quality_min = zero
            self.last_quality_high_frac = zero
            return
        self.last_quality_mean = quality.mean()
        self.last_quality_min = quality.min()
        self.last_quality_high_frac = (
            quality >= self.high_quality_thr).float().mean()

    def forward(self,
                pred: Tensor,
                target: Tensor,
                weight: Tensor,
                avg_factor: Optional[int] = None,
                reduction_override: Optional[str] = None) -> Tensor:
        """Calculate detached quality-weighted two-view box consistency."""
        if reduction_override not in (None, 'none', 'mean', 'sum'):
            raise ValueError(f'invalid reduction: {reduction_override}')
        reduction = reduction_override or self.reduction

        angle_delta = pred[..., 4] - target[..., 4]
        quality = axis_harmonic_reliability(
            angle_delta, self.min_quality, self.gamma).detach()
        self._record_quality(quality)

        if pred.numel() == 0:
            return pred.sum() * self.loss_weight

        quality_weight = weight * quality
        xy_pred = pred[..., :2]
        xy_target = target[..., :2]
        hbb_pred1 = torch.cat([-pred[..., 2:4], pred[..., 2:4]], dim=-1)
        hbb_pred2 = hbb_pred1[..., [1, 0, 3, 2]]
        hbb_target = torch.cat(
            [-target[..., 2:4], target[..., 2:4]], dim=-1)

        center_loss = self.center_loss(
            xy_pred,
            xy_target,
            weight=quality_weight[:, None],
            reduction_override=reduction,
            avg_factor=avg_factor)
        shape_loss1 = self.shape_loss(
            hbb_pred1,
            hbb_target,
            weight=quality_weight,
            reduction_override=reduction,
            avg_factor=avg_factor) + self.angle_loss(
                angle_delta.sin(),
                torch.zeros_like(angle_delta),
                weight=quality_weight,
                reduction_override=reduction,
                avg_factor=avg_factor)
        shape_loss2 = self.shape_loss(
            hbb_pred2,
            hbb_target,
            weight=quality_weight,
            reduction_override=reduction,
            avg_factor=avg_factor) + self.angle_loss(
                angle_delta.cos(),
                torch.zeros_like(angle_delta),
                weight=quality_weight,
                reduction_override=reduction,
                avg_factor=avg_factor)

        return self.loss_weight * (center_loss +
                                   torch.minimum(shape_loss1, shape_loss2))
