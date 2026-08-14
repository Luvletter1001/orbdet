# Copyright (c) OpenMMLab. All rights reserved.
"""HBox-to-FPN adapter for the group-orbit determinantal objective."""

import math
from typing import Dict, Sequence, Tuple

import torch
from mmdet.structures.bbox import bbox2roi
from mmdet.utils import ConfigType, InstanceList
from torch import Tensor

from mmrotate.models.losses.group_orbit_determinantal_cluster_loss import (
    build_planar_group_orbit)
from mmrotate.registry import MODELS


@MODELS.register_module()
class HBoxFPNGroupOrbitLoss(torch.nn.Module):
    """Extract GT-HBox FPN ROIs and regularize an exact finite-group orbit."""

    _GROUPS = frozenset({'c2', 'c4', 'd1', 'd2'})
    _DIAGNOSTIC_ATTRS = (
        'determinantal', 'spectral_tail', 'fixed_space', 'energy_guard',
        'variance_guard', 'q_gap')

    def __init__(self,
                 roi_extractor: ConfigType,
                 orbit_loss: ConfigType,
                 group: str = 'c2',
                 loss_weight: float = 0.02,
                 min_box_size: float = 2.0) -> None:
        super().__init__()
        if not isinstance(group, str) or group.lower() not in self._GROUPS:
            raise ValueError('group must be a supported nontrivial group')
        if not math.isfinite(loss_weight) or loss_weight <= 0.0:
            raise ValueError('loss_weight must be finite and positive')
        if not math.isfinite(min_box_size) or min_box_size <= 0.0:
            raise ValueError('min_box_size must be finite and positive')

        self.roi_extractor = MODELS.build(roi_extractor)
        self.orbit_loss = MODELS.build(orbit_loss)
        self.group = group.lower()
        self.loss_weight = loss_weight
        self.min_box_size = min_box_size

        self.last_roi_count = torch.tensor(0.0)
        for name in self._DIAGNOSTIC_ATTRS:
            setattr(self, f'last_{name}', torch.tensor(0.0))

    def _reset_diagnostics(self, reference: Tensor) -> None:
        zero = reference.new_zeros(())
        self.last_roi_count = zero
        for name in self._DIAGNOSTIC_ATTRS:
            setattr(self, f'last_{name}', zero)

    def _capture_diagnostics(self, reference: Tensor, roi_count: int) -> None:
        self.last_roi_count = reference.new_tensor(float(roi_count)).detach()
        for name in self._DIAGNOSTIC_ATTRS:
            value = getattr(self.orbit_loss, f'last_{name}')
            setattr(self, f'last_{name}', value.detach())

    def diagnostics(self) -> Dict[str, Tensor]:
        """Return detached scalar diagnostics for logging."""
        values = {'godc_roi_count': self.last_roi_count}
        for name in self._DIAGNOSTIC_ATTRS:
            values[f'godc_{name}'] = getattr(self, f'last_{name}')
        return values

    def _valid_hboxes(self, batch_gt_instances: InstanceList) -> Sequence[Tensor]:
        hbox_list = []
        for gt_instances in batch_gt_instances:
            hboxes = gt_instances.bboxes.convert_to('hbox').tensor
            if hboxes.numel() == 0:
                hbox_list.append(hboxes.reshape(0, 4))
                continue
            widths = hboxes[:, 2] - hboxes[:, 0]
            heights = hboxes[:, 3] - hboxes[:, 1]
            valid = (
                torch.isfinite(hboxes).all(dim=1)
                & (widths >= self.min_box_size)
                & (heights >= self.min_box_size))
            hbox_list.append(hboxes[valid])
        return hbox_list

    def forward(self, features: Tuple[Tensor],
                batch_gt_instances: InstanceList) -> Tensor:
        """Compute the weighted GODC loss on original-view instance ROIs."""
        if not isinstance(features, (tuple, list)) or not features:
            raise ValueError('features must be a non-empty tuple or list')
        if len(features) < self.roi_extractor.num_inputs:
            raise ValueError('insufficient FPN levels for roi_extractor')
        if not batch_gt_instances:
            self._reset_diagnostics(features[0])
            return features[0].sum() * 0.0

        hbox_list = self._valid_hboxes(batch_gt_instances)
        roi_count = sum(len(hboxes) for hboxes in hbox_list)
        if roi_count == 0:
            self._reset_diagnostics(features[0])
            return features[0].sum() * 0.0

        rois = bbox2roi(hbox_list)
        batch_size = len(batch_gt_instances)
        roi_features = self.roi_extractor(
            tuple(feature[:batch_size]
                  for feature in features[:self.roi_extractor.num_inputs]),
            rois)
        orbit = build_planar_group_orbit(roi_features, self.group)
        value = self.orbit_loss(orbit)
        self._capture_diagnostics(value, roi_count)
        return value * self.loss_weight
