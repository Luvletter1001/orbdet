# Copyright (c) OpenMMLab. All rights reserved.
"""Orbdet-v0.2 with an independent symmetry orientation anchor."""

from typing import Union

import torch
from mmdet.structures import SampleList
from torch import Tensor

from mmrotate.models.detectors.h2rbox_v2 import H2RBoxV2Detector
from mmrotate.registry import MODELS


@MODELS.register_module()
class OrbdetV02Detector(H2RBoxV2Detector):
    """Expose detached anchor calibration without changing optimization.

    Prediction is inherited unchanged.  In training, the parent detector and
    head return the official H2RBox-v2 objective; all added values below have
    names without ``loss`` and are used only by the logger.
    """

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        losses = super().loss(batch_inputs, batch_data_samples)
        consistency = self.bbox_head.loss_symmetry_ss
        diagnostic_names = {
            'orbdet_q_rot': 'last_quality_rot',
            'orbdet_q_flip': 'last_quality_flip',
            'orbdet_q_joint': 'last_quality_joint',
            'orbdet_q_high_frac': 'last_quality_high_frac',
        }
        for log_name, attr_name in diagnostic_names.items():
            value = getattr(consistency, attr_name, None)
            if value is not None:
                losses[log_name] = value.detach()

        bbox_loss = losses.get('loss_bbox')
        joint_quality = losses.get('orbdet_q_joint')
        if isinstance(bbox_loss, Tensor) and isinstance(joint_quality, Tensor):
            hbox_fidelity = torch.exp(-bbox_loss.detach().clamp(0.0, 20.0))
            losses['orbdet_hbox_fidelity'] = hbox_fidelity
            losses['orbdet_q_anchored'] = joint_quality * hbox_fidelity
        return losses

