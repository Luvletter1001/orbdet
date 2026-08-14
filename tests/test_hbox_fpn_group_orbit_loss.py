import math

import pytest
import torch
from mmengine.structures import InstanceData

from mmrotate.models.losses.hbox_fpn_group_orbit_loss import (
    HBoxFPNGroupOrbitLoss)
from mmrotate.registry import MODELS
from mmrotate.structures.bbox import RotatedBoxes
from mmrotate.utils import register_all_modules


def _adapter_cfg():
    return dict(
        type='HBoxFPNGroupOrbitLoss',
        group='c2',
        loss_weight=0.02,
        min_box_size=2.0,
        roi_extractor=dict(
            type='mmdet.SingleRoIExtractor',
            roi_layer=dict(
                type='RoIAlign', output_size=7, sampling_ratio=2),
            out_channels=4,
            featmap_strides=[1]),
        orbit_loss=dict(
            type='GroupOrbitDeterminantalClusterLoss',
            energy_guard_weight=0.0,
            variance_guard_weight=0.0))


def _instances(boxes):
    instances = InstanceData()
    instances.bboxes = RotatedBoxes(
        torch.tensor(boxes, dtype=torch.float32).reshape(-1, 5))
    instances.labels = torch.zeros(len(boxes), dtype=torch.long)
    return instances


def test_adapter_builds_from_registry_and_backpropagates_through_roi():
    register_all_modules()
    adapter = MODELS.build(_adapter_cfg())
    feature = torch.randn(1, 4, 20, 20, requires_grad=True)

    value = adapter((feature, ), [_instances([[10, 10, 12, 8, 0]])])
    value.backward()

    assert isinstance(adapter, HBoxFPNGroupOrbitLoss)
    assert value.shape == ()
    assert torch.isfinite(value)
    assert float(value) > 0.0
    assert feature.grad is not None
    assert torch.isfinite(feature.grad).all()
    assert float(feature.grad.abs().sum()) > 0.0
    assert float(adapter.last_roi_count) == 1.0
    for diagnostic in adapter.diagnostics().values():
        assert diagnostic.shape == ()
        assert torch.isfinite(diagnostic)
        assert not diagnostic.requires_grad


def test_empty_and_invalid_boxes_return_graph_connected_zero():
    register_all_modules()
    adapter = MODELS.build(_adapter_cfg())
    feature = torch.randn(1, 4, 20, 20, requires_grad=True)
    boxes = [[math.nan, 10, 4, 4, 0], [10, 10, 0, 5, 0]]

    value = adapter((feature, ), [_instances(boxes)])
    value.backward()

    assert float(value) == 0.0
    assert value.requires_grad
    assert feature.grad is not None
    assert torch.equal(feature.grad, torch.zeros_like(feature.grad))
    assert float(adapter.last_roi_count) == 0.0
    assert all(float(item) == 0.0
               for item in adapter.diagnostics().values())


@pytest.mark.parametrize(
    'override, message', [
        (dict(group='c1'), 'nontrivial'),
        (dict(loss_weight=0.0), 'loss_weight'),
        (dict(min_box_size=0.0), 'min_box_size'),
    ])
def test_adapter_rejects_invalid_contract(override, message):
    register_all_modules()
    cfg = _adapter_cfg()
    cfg.update(override)

    with pytest.raises(ValueError, match=message):
        MODELS.build(cfg)
