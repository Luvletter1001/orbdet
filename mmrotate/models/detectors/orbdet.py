# Copyright (c) OpenMMLab. All rights reserved.
"""Orbdet detector built on the mature H2RBox dense proposal path."""

from typing import Union

from mmdet.structures import SampleList
from torch import Tensor

from mmrotate.models.detectors.h2rbox import H2RBoxDetector
from mmrotate.registry import MODELS


@MODELS.register_module()
class OrbdetDetector(H2RBoxDetector):
    """H2RBox detector with Orbdet harmonic-quality diagnostics.

    Prediction is intentionally inherited without modification: the harmonic
    quality is a training-time reliability weight and never a second inference
    score multiplier.
    """

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        losses = super().loss(batch_inputs, batch_data_samples)
        consistency_loss = self.bbox_head.loss_bbox_ss
        diagnostic_names = {
            'orbdet_q_mean': 'last_quality_mean',
            'orbdet_q_min': 'last_quality_min',
            'orbdet_q_high_frac': 'last_quality_high_frac',
        }
        for log_name, attr_name in diagnostic_names.items():
            value = getattr(consistency_loss, attr_name, None)
            if value is not None:
                losses[log_name] = value.detach()
        return losses
