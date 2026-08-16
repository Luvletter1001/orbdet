import math

import pytest
import torch
from mmengine.structures import InstanceData

from mmrotate.models.task_modules.scqo_fpn_evidence_adapter import (
    SCQOFPNInstanceEvidence)
from mmrotate.registry import MODELS
from mmrotate.structures import RotatedBoxes
from mmrotate.utils import register_all_modules


@MODELS.register_module()
class SCQOContractAuditExtractor(torch.nn.Module):

    def __init__(self, num_inputs=1):
        super().__init__()
        self.num_inputs = num_inputs

    def forward(self, features, rois):
        return features[0].new_zeros((rois.shape[0], 4, 14, 14))

    def map_roi_levels(self, rois, num_levels):
        return rois.new_zeros((rois.shape[0], ), dtype=torch.long)


@MODELS.register_module()
class SCQOHalfLevelAuditExtractor(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.num_inputs = 2
        self.finest_scale = 8

    def map_roi_levels(self, rois, num_levels):
        work_rois = rois.float()
        scale = torch.sqrt((work_rois[:, 3] - work_rois[:, 1]) *
                           (work_rois[:, 4] - work_rois[:, 2]))
        target_levels = torch.floor(
            torch.log2(scale / self.finest_scale + 1e-6))
        return target_levels.clamp(min=0, max=num_levels - 1).long()

    def forward(self, features, rois):
        aligned_rois = rois.type_as(features[0])
        target_levels = self.map_roi_levels(aligned_rois, len(features))
        batch_index = aligned_rois[:, 0].long()
        output = features[0].new_empty((rois.shape[0], 4, 14, 14))
        for level, feature in enumerate(features):
            indices = (target_levels == level).nonzero().reshape(-1)
            if indices.numel():
                output[indices] = feature.index_select(
                    0, batch_index.index_select(0, indices))
        return output


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


def _contract_cfg(num_inputs=1):
    cfg = _cfg()
    cfg['roi_extractor'] = dict(
        type='SCQOContractAuditExtractor', num_inputs=num_inputs)
    return cfg


def _half_level_cfg():
    cfg = _cfg()
    cfg['roi_extractor'] = dict(type='SCQOHalfLevelAuditExtractor')
    return cfg


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


@pytest.mark.parametrize('min_box_size', [True, False])
def test_adapter_rejects_boolean_minimum_box_size(min_box_size):
    register_all_modules()
    cfg = _cfg()
    cfg['min_box_size'] = min_box_size

    with pytest.raises((TypeError, ValueError), match='min_box_size'):
        MODELS.build(cfg)


@pytest.mark.parametrize('min_box_size', ['2', None, 2 + 0j,
                                           torch.tensor(2.0)])
def test_adapter_rejects_non_real_minimum_box_size(min_box_size):
    register_all_modules()
    cfg = _cfg()
    cfg['min_box_size'] = min_box_size

    with pytest.raises(TypeError, match='min_box_size'):
        MODELS.build(cfg)


@pytest.mark.parametrize('min_box_size', [math.nan, math.inf, -math.inf])
def test_adapter_rejects_nonfinite_minimum_box_size(min_box_size):
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


@pytest.mark.parametrize('batch_sizes', [(0, ), (2, ), (1, 0), (1, 2)])
def test_adapter_rejects_consumed_feature_batch_mismatch(batch_sizes):
    register_all_modules()
    module = MODELS.build(_contract_cfg(len(batch_sizes)))
    features = tuple(
        torch.randn(batch_size, 4, 16 // (2**index), 16 // (2**index))
        for index, batch_size in enumerate(batch_sizes))
    instances = [_instances([[8, 8, 4, 4, 0]], [0])]

    with pytest.raises(ValueError, match='batch'):
        module(features, instances, [dict(img_shape=(16, 16))])


@pytest.mark.parametrize('feature', [
    torch.randn(1, 4, 16),
    torch.tensor(1.0),
])
def test_adapter_rejects_feature_without_nchw_shape(feature):
    register_all_modules()
    module = MODELS.build(_contract_cfg())
    instances = [_instances([[8, 8, 4, 4, 0]], [0])]

    with pytest.raises(ValueError, match='four-dimensional'):
        module((feature, ), instances, [dict(img_shape=(16, 16))])


def test_adapter_ignores_unconsumed_extra_feature_batch_size():
    register_all_modules()
    module = MODELS.build(_contract_cfg())
    features = (torch.randn(1, 4, 16, 16), torch.randn(3, 4, 8, 8))
    instances = [_instances([[8, 8, 4, 4, 0]], [0])]

    result = module(features, instances, [dict(img_shape=(16, 16))])

    assert result['batch_index'].shape == (1, )


@pytest.mark.parametrize('meta', [
    pytest.param({}, id='missing'),
    pytest.param(dict(img_shape=None), id='none'),
    pytest.param(dict(img_shape=()), id='empty'),
    pytest.param(dict(img_shape=(20, )), id='missing-width'),
    pytest.param(dict(img_shape=(True, 20)), id='boolean-height'),
    pytest.param(dict(img_shape=(20, False)), id='boolean-width'),
    pytest.param(dict(img_shape=(0, 20)), id='zero-height'),
    pytest.param(dict(img_shape=(20, -1)), id='negative-width'),
    pytest.param(dict(img_shape=(math.nan, 20)), id='nan-height'),
    pytest.param(dict(img_shape=(20, math.inf)), id='infinite-width'),
    pytest.param(dict(img_shape=('20', 20)), id='non-real-height'),
    pytest.param(dict(img_shape=(20, 2 + 0j)), id='non-real-width'),
])
def test_adapter_rejects_invalid_image_shape(meta):
    register_all_modules()
    module = MODELS.build(_contract_cfg())
    instances = [_instances([[8, 8, 4, 4, 0]], [0])]

    with pytest.raises(ValueError, match='img_shape'):
        module((torch.randn(1, 4, 20, 20), ), instances, [meta])


def test_adapter_validates_image_shape_even_without_instances():
    register_all_modules()
    module = MODELS.build(_contract_cfg())
    instances = [_instances(torch.empty(0, 5),
                            torch.empty(0, dtype=torch.long))]

    with pytest.raises(ValueError, match='img_shape'):
        module((torch.randn(1, 4, 20, 20), ), instances,
               [dict(img_shape=(0, 20))])


def test_adapter_accepts_positive_real_image_shape_with_extra_dimensions():
    register_all_modules()
    module = MODELS.build(_contract_cfg())
    instances = [_instances([[8, 8, 4, 4, 0]], [0])]

    result = module((torch.randn(1, 4, 20, 20), ), instances,
                    [dict(img_shape=(20.5, 19.5, 3))])

    assert result['support_fraction'].shape == (1, )


def test_half_rois_align_actual_sampling_fpn_report_and_square_coordinates():
    register_all_modules()
    module = MODELS.build(_half_level_cfg())
    features = (
        torch.ones(1, 4, 14, 14, dtype=torch.float16),
        torch.full((1, 4, 14, 14), 2.0, dtype=torch.float16),
    )
    instances = [_instances([[32, 32, 15.999, 15.999, 0]], [0])]

    result = module(features, instances, [dict(img_shape=(64, 64))])

    assert float(result['energy']) == 4.0
    assert result['fpn_level'].item() == 1
    assert torch.equal(result['square_hbox'],
                       torch.tensor([[24.0, 24.0, 40.0, 40.0]]))
    assert {value.device for value in result.values()} == {features[0].device}
    assert {
        value.dtype
        for value in result.values() if value.is_floating_point()
    } == {torch.float32}


def test_half_empty_result_has_uniform_device_and_floating_dtype():
    register_all_modules()
    module = MODELS.build(_half_level_cfg())
    features = (
        torch.empty(0, 4, 14, 14, dtype=torch.float16),
        torch.empty(0, 4, 14, 14, dtype=torch.float16),
    )

    result = module(features, [], [])

    assert {value.device for value in result.values()} == {features[0].device}
    assert {
        value.dtype
        for value in result.values() if value.is_floating_point()
    } == {torch.float32}
