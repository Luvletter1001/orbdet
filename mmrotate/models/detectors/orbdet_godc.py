# Copyright (c) OpenMMLab. All rights reserved.
"""Orbdet-v0.2 with an HBox/FPN group-orbit auxiliary objective."""

from typing import Tuple

from mmdet.utils import ConfigType, InstanceList
from torch import Tensor

from mmrotate.models.detectors.orbdet_v0_2 import OrbdetV02Detector
from mmrotate.registry import MODELS


@MODELS.register_module()
class OrbdetGODCDetector(OrbdetV02Detector):
    """Add GODC during training while preserving v0.2 prediction exactly."""

    def __init__(self, godc_auxiliary: ConfigType, **kwargs) -> None:
        super().__init__(**kwargs)
        self.godc_auxiliary = MODELS.build(godc_auxiliary)

    def _add_auxiliary_losses(self, losses: dict, features: Tuple[Tensor],
                              batch_gt_instances: InstanceList) -> dict:
        losses = super()._add_auxiliary_losses(
            losses, features, batch_gt_instances)
        losses['loss_godc'] = self.godc_auxiliary(
            features, batch_gt_instances)
        losses.update(self.godc_auxiliary.diagnostics())
        return losses
