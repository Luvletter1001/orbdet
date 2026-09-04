import math

import torch

from mmrotate.models.losses.low_rank_orientation_evidence import (
    low_rank_channel_orientation_evidence)


def _axis_delta(prediction, target):
    return torch.abs(torch.remainder(prediction - target + math.pi / 2,
                                     math.pi) - math.pi / 2)


def _linear_texture(angle, channels=(1.0, 2.0, 3.0), size=17):
    coordinates = torch.linspace(-1.0, 1.0, size)
    yy, xx = torch.meshgrid(coordinates, coordinates, indexing='ij')
    normal = angle - math.pi / 2
    plane = xx * math.cos(normal) + yy * math.sin(normal)
    return torch.stack([amplitude * plane for amplitude in channels]).unsqueeze(0)


def test_decodes_tangent_axis_from_rank_one_channel_evidence():
    output = low_rank_channel_orientation_evidence(_linear_texture(0.35))
    assert float(_axis_delta(output['axis_angle'], torch.tensor([0.35]))) < 1e-4
    assert float(output['confidence']) > 0.999
    assert float(output['anisotropy']) > 0.999
    assert bool(output['valid'])


def test_orthogonal_channel_populations_are_marked_ambiguous():
    features = _linear_texture(0.0, channels=(1.0,))
    features = torch.cat([features, _linear_texture(math.pi / 2, channels=(1.0,))], dim=1)
    output = low_rank_channel_orientation_evidence(features, min_anisotropy=0.1)
    assert float(output['anisotropy']) < 1e-5
    assert not bool(output['valid'])


def test_constant_features_are_finite_and_invalid():
    output = low_rank_channel_orientation_evidence(torch.ones(2, 3, 9, 9))
    assert torch.isfinite(output['axis_angle']).all()
    assert torch.isfinite(output['confidence']).all()
    assert torch.equal(output['valid'], torch.zeros(2, dtype=torch.bool))


def test_result_preserves_autograd_connectivity():
    features = torch.randn(2, 4, 9, 9, requires_grad=True)
    output = low_rank_channel_orientation_evidence(features)
    (output['axis_angle'].sum() + output['confidence'].sum() + output['anisotropy'].sum()).backward()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()
