import math

import pytest
import torch
from mmengine.structures import InstanceData

from mmrotate.models.task_modules.scqo_fpn_evidence_adapter import (
    SCQOFPNInstanceEvidence)
from mmrotate.registry import MODELS
from mmrotate.structures import RotatedBoxes
from mmrotate.utils import register_all_modules


def _instances(boxes, labels):
    value = InstanceData()
    value.bboxes = RotatedBoxes(
        torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 5))
    value.labels = torch.as_tensor(labels, dtype=torch.long)
    return value


def _cfg(featmap_strides=(1, ), finest_scale=None):
    roi_extractor = dict(
        type='mmdet.SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=2),
        out_channels=4,
        featmap_strides=list(featmap_strides))
    if finest_scale is not None:
        roi_extractor['finest_scale'] = finest_scale
    return dict(
        type='SCQOFPNInstanceEvidence',
        min_box_size=2.0,
        roi_extractor=roi_extractor,
        evidence=dict(
            type='SCQOStabilizerEvidence',
            roi_size=14,
            channels=4,
            negative_seed=3407))


def test_adapter_preserves_batch_instance_identity_and_square_boxes():
    register_all_modules()
    module = MODELS.build(_cfg())
    features = (torch.randn(2, 4, 64, 64, requires_grad=True), )
    instances = [
        _instances([
            [10, 10, 8, 4, math.pi / 6],
            [7, 8, 1, 1, 0],
            [30, 30, 6, 10, 0],
        ], [3, 30, 7]),
        _instances([
            [20, 18, 6, 10, 0],
            [10, 10, math.nan, 4, 0],
            [45, 12, 4, 4, 0],
        ], [4, 40, 8]),
    ]
    metas = [dict(img_shape=(64, 64)), dict(img_shape=(64, 64))]

    result = module(features, instances, metas)

    assert torch.equal(result['batch_index'], torch.tensor([0, 0, 1, 1]))
    assert torch.equal(result['instance_index'], torch.tensor([0, 2, 0, 2]))
    assert torch.equal(result['label'], torch.tensor([3, 7, 4, 8]))
    assert torch.equal(result['fpn_level'], torch.tensor([0, 0, 0, 0]))
    expected_squares = torch.tensor([
        [5.535898, 5.535898, 14.464102, 14.464102],
        [25, 25, 35, 35],
        [15, 13, 25, 23],
        [43, 10, 47, 14],
    ])
    assert torch.allclose(result['square_hbox'], expected_squares, atol=1e-5)
    widths = result['square_hbox'][:, 2] - result['square_hbox'][:, 0]
    heights = result['square_hbox'][:, 3] - result['square_hbox'][:, 1]
    assert torch.allclose(widths, heights)
    assert all(not value.requires_grad for value in result.values())
    assert features[0].grad is None


def test_adapter_reports_single_roi_extractor_fpn_levels():
    register_all_modules()
    module = MODELS.build(_cfg((1, 2, 4), finest_scale=8))
    features = (
        torch.randn(1, 4, 96, 96, requires_grad=True),
        torch.randn(1, 4, 48, 48, requires_grad=True),
        torch.randn(1, 4, 24, 24, requires_grad=True),
    )
    instances = [_instances([
        [16, 16, 8, 8, 0],
        [40, 40, 16, 16, 0],
        [68, 68, 32, 32, 0],
    ], [0, 1, 2])]

    result = module(features, instances, [dict(img_shape=(96, 96))])

    rois = torch.cat((
        result['batch_index'].to(result['square_hbox']).unsqueeze(1),
        result['square_hbox']),
                     dim=1)
    expected = module.roi_extractor.map_roi_levels(
        rois, module.roi_extractor.num_inputs)
    assert torch.equal(result['fpn_level'], torch.tensor([0, 1, 2]))
    assert torch.equal(result['fpn_level'], expected)
    assert all(feature.grad is None for feature in features)


def test_edge_box_records_bin_center_support_and_empty_results_are_finite():
    register_all_modules()
    module = MODELS.build(_cfg())
    features = (torch.randn(1, 4, 20, 20), )
    edge = [_instances([[1, 1, 8, 4, 0]], [0])]

    result = module(features, edge, [dict(img_shape=(20, 20))])
    empty = module(
        features,
        [_instances(torch.empty(0, 5), torch.empty(0, dtype=torch.long))],
        [dict(img_shape=(20, 20))])
    empty_batch = module((torch.empty(0, 4, 20, 20), ), [], [])

    assert torch.equal(result['square_hbox'],
                       torch.tensor([[-3.0, -3.0, 5.0, 5.0]]))
    assert torch.allclose(result['support_fraction'], torch.tensor([81 / 196]))
    assert float(result['support_fraction']) < 1.0
    assert not bool(result['valid'])
    assert empty.keys() == result.keys()
    assert empty_batch.keys() == result.keys()
    for empty_result in (empty, empty_batch):
        assert all(value.shape[0] == 0 for value in empty_result.values())
        assert all(
            torch.isfinite(value).all() for value in empty_result.values()
            if value.dtype != torch.bool)
        assert all(not value.requires_grad for value in empty_result.values())


@pytest.mark.parametrize('min_box_size', [0.0, -1.0])
def test_adapter_rejects_nonpositive_minimum_box_size(min_box_size):
    register_all_modules()
    cfg = _cfg()
    cfg['min_box_size'] = min_box_size

    with pytest.raises(ValueError, match='min_box_size'):
        MODELS.build(cfg)


def test_adapter_rejects_insufficient_features_or_unaligned_metadata():
    register_all_modules()
    module = MODELS.build(_cfg((1, 2)))
    instances = [_instances([[8, 8, 4, 4, 0]], [0])]

    with pytest.raises(ValueError, match='insufficient FPN features'):
        module((torch.randn(1, 4, 16, 16), ), instances,
               [dict(img_shape=(16, 16))])
    with pytest.raises(ValueError, match='align'):
        module((torch.randn(1, 4, 16, 16),
                torch.randn(1, 4, 8, 8)), instances, [])
