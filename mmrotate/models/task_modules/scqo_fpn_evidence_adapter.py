# Copyright (c) OpenMMLab. All rights reserved.
"""GT-HBox/FPN adapter for read-only SCQO evidence."""

from typing import Dict, List, Sequence, Tuple

import torch
from mmdet.structures.bbox import bbox2roi
from mmdet.utils import ConfigType, InstanceList
from torch import Tensor

from mmrotate.registry import MODELS


@MODELS.register_module()
class SCQOFPNInstanceEvidence(torch.nn.Module):
    """Extract detached SCQO evidence from square GT-HBox FPN RoIs."""

    def __init__(self,
                 roi_extractor: ConfigType,
                 evidence: ConfigType,
                 min_box_size: float = 2.0) -> None:
        super().__init__()
        if min_box_size <= 0:
            raise ValueError('min_box_size must be positive')
        self.roi_extractor = MODELS.build(roi_extractor)
        self.evidence = MODELS.build(evidence)
        self.min_box_size = min_box_size

    def _square_boxes(
            self, batch_gt_instances: InstanceList
    ) -> Tuple[List[Tensor], Tensor, Tensor, Tensor]:
        boxes_per_image = []
        batch_indices = []
        instance_indices = []
        labels = []
        for batch_index, instances in enumerate(batch_gt_instances):
            hboxes = instances.bboxes.convert_to('hbox').tensor
            if hboxes.numel() == 0:
                boxes_per_image.append(hboxes.reshape(0, 4))
                continue
            width = hboxes[:, 2] - hboxes[:, 0]
            height = hboxes[:, 3] - hboxes[:, 1]
            valid = (
                torch.isfinite(hboxes).all(1)
                & (width >= self.min_box_size)
                & (height >= self.min_box_size))
            original_index = valid.nonzero().reshape(-1)
            hboxes = hboxes[valid]
            center = (hboxes[:, :2] + hboxes[:, 2:]) / 2
            side = torch.maximum(
                hboxes[:, 2] - hboxes[:, 0],
                hboxes[:, 3] - hboxes[:, 1])
            square = torch.cat(
                (center - side[:, None] / 2,
                 center + side[:, None] / 2),
                dim=1)
            boxes_per_image.append(square)
            batch_indices.append(
                original_index.new_full(original_index.shape, batch_index))
            instance_indices.append(original_index)
            labels.append(instances.labels[valid])
        reference = batch_gt_instances[0].bboxes.tensor
        empty_long = reference.new_empty((0, ), dtype=torch.long)
        return (
            boxes_per_image,
            torch.cat(batch_indices) if batch_indices else empty_long,
            torch.cat(instance_indices) if instance_indices else empty_long,
            torch.cat(labels) if labels else empty_long)

    def _support(self, rois: Tensor,
                 batch_img_metas: Sequence[dict]) -> Tensor:
        size = self.evidence.roi_size
        grid = (
            torch.arange(size, device=rois.device, dtype=rois.dtype) + 0.5
        ) / size
        x = rois[:, 1, None] + grid[None] * (
            rois[:, 3] - rois[:, 1])[:, None]
        y = rois[:, 2, None] + grid[None] * (
            rois[:, 4] - rois[:, 2])[:, None]
        masks = []
        for row, x_row, y_row in zip(rois, x, y):
            image_index = int(row[0])
            image_height, image_width = batch_img_metas[
                image_index]['img_shape'][:2]
            valid_x = (x_row >= 0) & (x_row < image_width)
            valid_y = (y_row >= 0) & (y_row < image_height)
            masks.append(valid_y[:, None] & valid_x[None, :])
        return torch.stack(masks).unsqueeze(1).to(dtype=rois.dtype)

    def _empty_result(self, reference: Tensor) -> Dict[str, Tensor]:
        empty_features = reference.new_empty(
            (0, self.evidence.channels, self.evidence.roi_size,
             self.evidence.roi_size))
        empty_long = reference.new_empty((0, ), dtype=torch.long)
        result = self.evidence(empty_features)
        result.update(
            batch_index=empty_long,
            instance_index=empty_long.clone(),
            label=empty_long.clone(),
            fpn_level=empty_long.clone(),
            square_hbox=reference.new_empty((0, 4)))
        return result

    def forward(self,
                features: Tuple[Tensor, ...],
                batch_gt_instances: InstanceList,
                batch_img_metas: Sequence[dict]) -> Dict[str, Tensor]:
        """Return detached per-instance evidence and source identities."""
        if not features or len(features) < self.roi_extractor.num_inputs:
            raise ValueError('insufficient FPN features')
        if len(batch_gt_instances) != len(batch_img_metas):
            raise ValueError('instances and image metadata must align')
        if not batch_gt_instances:
            return self._empty_result(features[0])
        boxes, batch_index, instance_index, labels = self._square_boxes(
            batch_gt_instances)
        rois = bbox2roi(boxes)
        if rois.shape[0] == 0:
            return self._empty_result(features[0])
        roi_features = self.roi_extractor(
            tuple(item.detach()
                  for item in features[:self.roi_extractor.num_inputs]),
            rois)
        result = self.evidence(roi_features,
                               self._support(rois, batch_img_metas))
        fpn_level = self.roi_extractor.map_roi_levels(
            rois, self.roi_extractor.num_inputs)
        result.update(
            batch_index=batch_index.detach(),
            instance_index=instance_index.detach(),
            label=labels.detach(),
            fpn_level=fpn_level.detach(),
            square_hbox=rois[:, 1:].detach())
        return result
