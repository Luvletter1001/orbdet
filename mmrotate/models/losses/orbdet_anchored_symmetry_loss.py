# Copyright (c) OpenMMLab. All rights reserved.
"""Independent symmetry-anchor diagnostics for Orbdet-v0.2.

The returned objective is exactly :class:`H2RBoxV2ConsistencyLoss`.  Quality
values are detached monitoring signals and never reweight a training loss.
"""

from typing import Optional, Tuple

import torch
from mmdet.utils import ConfigType
from torch import Tensor

from mmrotate.models.losses.h2rbox_v2_consistency_loss import (
    H2RBoxV2ConsistencyLoss)
from mmrotate.registry import MODELS


@MODELS.register_module()
class OrbdetAnchoredSymmetryLoss(H2RBoxV2ConsistencyLoss):
    """Official symmetry loss plus non-self-weighting quality diagnostics."""

    def __init__(
            self,
            loss_rot: ConfigType = dict(
                type='mmdet.SmoothL1Loss', loss_weight=1.0, beta=0.1),
            loss_flp: ConfigType = dict(
                type='mmdet.SmoothL1Loss', loss_weight=0.05, beta=0.1),
            use_snap_loss: bool = True,
            reduction: str = 'mean',
            quality_temperature: float = 0.25,
            high_quality_thr: float = 0.75) -> None:
        super().__init__(
            loss_rot=loss_rot,
            loss_flp=loss_flp,
            use_snap_loss=use_snap_loss,
            reduction=reduction)
        if quality_temperature <= 0.0:
            raise ValueError('quality_temperature must be positive')
        if not 0.0 <= high_quality_thr <= 1.0:
            raise ValueError('high_quality_thr must be in [0, 1]')
        self.quality_temperature = quality_temperature
        self.high_quality_thr = high_quality_thr

        # Diagnostics deliberately stay out of state_dict.
        self.last_quality_rot = torch.tensor(0.0)
        self.last_quality_flip = torch.tensor(0.0)
        self.last_quality_joint = torch.tensor(0.0)
        self.last_quality_high_frac = torch.tensor(0.0)

    def _constraint_residuals(
            self, pred_ori: Tensor, pred_rot: Tensor, pred_flp: Tensor,
            target_ori: Tensor, target_rot: Tensor) -> Tuple[Tensor, Tensor]:
        d_ang_rot = (pred_ori - pred_rot) - (target_ori - target_rot)
        d_ang_flp = pred_ori + pred_flp
        if self.use_snap_loss:
            d_ang_rot = (d_ang_rot + torch.pi / 2) % torch.pi - torch.pi / 2
            d_ang_flp = (d_ang_flp + torch.pi / 2) % torch.pi - torch.pi / 2
        return d_ang_rot, d_ang_flp

    def _record_quality(self, rot_quality: Tensor,
                        flip_quality: Tensor) -> None:
        if rot_quality.numel() == 0:
            zero = rot_quality.new_zeros(())
            self.last_quality_rot = zero
            self.last_quality_flip = zero
            self.last_quality_joint = zero
            self.last_quality_high_frac = zero
            return
        joint_quality = (rot_quality * flip_quality).sqrt()
        self.last_quality_rot = rot_quality.mean()
        self.last_quality_flip = flip_quality.mean()
        self.last_quality_joint = joint_quality.mean()
        self.last_quality_high_frac = (
            joint_quality >= self.high_quality_thr).float().mean()

    def forward(self,
                pred_ori: Tensor,
                pred_rot: Tensor,
                pred_flp: Tensor,
                target_ori: Tensor,
                target_rot: Tensor,
                agnostic_mask: Optional[Tensor] = None,
                avg_factor: Optional[int] = None,
                reduction_override: Optional[str] = None) -> Tensor:
        loss = super().forward(
            pred_ori,
            pred_rot,
            pred_flp,
            target_ori,
            target_rot,
            agnostic_mask=agnostic_mask,
            avg_factor=avg_factor,
            reduction_override=reduction_override)

        with torch.no_grad():
            rot_residual, flip_residual = self._constraint_residuals(
                pred_ori.detach(), pred_rot.detach(), pred_flp.detach(),
                target_ori.detach(), target_rot.detach())
            if agnostic_mask is not None:
                valid = ~agnostic_mask.bool()
                rot_residual = rot_residual[valid]
                flip_residual = flip_residual[valid]
            scale = self.quality_temperature
            rot_quality = torch.exp(-0.5 * (rot_residual / scale).square())
            flip_quality = torch.exp(-0.5 *
                                     (flip_residual / scale).square())
            self._record_quality(rot_quality, flip_quality)

        return loss

