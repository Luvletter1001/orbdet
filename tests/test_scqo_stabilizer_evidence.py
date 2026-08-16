import pytest
import torch

from mmrotate.models.task_modules.scqo_stabilizer_evidence import \
    SCQOStabilizerEvidence


def _cross(size=14):
    image = torch.zeros(1, 1, size, size)
    middle = size // 2
    image[:, :, 2:-2, middle - 1:middle + 1] = 1.0
    image[:, :, middle - 1:middle + 1, 2:-2] = 1.0
    return image


def _bar(size=14):
    image = torch.zeros(1, 1, size, size)
    middle = size // 2
    image[:, :, middle - 1:middle + 1, 2:-2] = 1.0
    return image


def _disk(size=14):
    coordinate = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
    rows, columns = torch.meshgrid(coordinate, coordinate, indexing='ij')
    return ((rows.square() + columns.square()) <= 16).float()[None, None]


def test_c4_evidence_separates_cross_and_bar_with_negative_margin():
    evidence = SCQOStabilizerEvidence(channels=1)

    cross = evidence(_cross())
    bar = evidence(_bar())

    assert float(cross['c4_fixed_space']) < 1e-6
    assert float(bar['c4_fixed_space']) > 1e-3
    assert float(cross['c4_negative_margin']) > 0


def test_patic_and_c4_evidence_reject_controls_and_partial_cross():
    torch.manual_seed(19)
    evidence = SCQOStabilizerEvidence(channels=1)
    random_texture = evidence(torch.randn(1, 1, 14, 14))
    disk = evidence(_disk())
    bar = evidence(_bar())
    full_cross = evidence(_cross())
    occluded_cross = _cross()
    occluded_cross[:, :, :7, :7] = 0
    partial_cross = evidence(occluded_cross)

    assert float(disk['c4_fixed_space']) < 1e-6
    assert float(bar['a2']) > float(disk['a2']) + 0.1
    assert float(bar['a2']) > float(random_texture['a2'])
    assert float(partial_cross['c4_fixed_space']) > float(
        full_cross['c4_fixed_space']) + 1e-4


def test_evidence_stays_float32_and_finite_inside_cpu_autocast():
    evidence = SCQOStabilizerEvidence(channels=1)

    with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
        result = evidence(torch.randn(1, 1, 14, 14))

    assert all(
        torch.isfinite(value).all() for value in result.values()
        if value.dtype != torch.bool)
    assert all(value.dtype == torch.float32 for value in result.values()
               if value.dtype != torch.bool)


def test_zero_features_have_exactly_zero_patic_evidence():
    evidence = SCQOStabilizerEvidence(channels=1)

    result = evidence(torch.zeros(1, 1, 14, 14))

    assert result['a2'].item() == 0.0
    assert result['a4'].item() == 0.0
    assert not bool(result['valid'])


def test_patic_is_invariant_to_orthogonal_channel_mixing():
    generator = torch.Generator().manual_seed(73)
    features = torch.randn(1, 4, 14, 14, generator=generator)
    mixing, _ = torch.linalg.qr(torch.randn(4, 4, generator=generator))
    mixed_features = torch.einsum('ij,njhw->nihw', mixing, features)
    evidence = SCQOStabilizerEvidence(channels=4)

    original = evidence(features)
    mixed = evidence(mixed_features)

    assert torch.allclose(original['a2'], mixed['a2'], rtol=1e-6, atol=1e-7)
    assert torch.allclose(original['a4'], mixed['a4'], rtol=1e-6, atol=1e-7)


def test_large_finite_features_produce_finite_invalid_evidence():
    evidence = SCQOStabilizerEvidence(channels=1)
    features = torch.full((1, 1, 14, 14), 1e20)
    features[:, :, ::2] = -1e20

    result = evidence(features)

    assert all(
        torch.isfinite(value).all() for value in result.values()
        if value.dtype != torch.bool)
    assert not bool(result['valid'])


@pytest.mark.parametrize('kwargs', [
    dict(roi_size=True),
    dict(block_size=True),
    dict(channels=True),
    dict(negative_seed=True),
    dict(min_energy=True),
    dict(min_variance=True),
    dict(min_support_fraction=True),
    dict(roi_size=2, block_size=1),
])
def test_constructor_rejects_boolean_or_degenerate_parameters(kwargs):
    with pytest.raises((TypeError, ValueError)):
        SCQOStabilizerEvidence(**kwargs)


def test_guards_invalidate_constant_features_and_empty_support():
    evidence = SCQOStabilizerEvidence(channels=1)

    constant = evidence(torch.ones(1, 1, 14, 14))
    empty_support = evidence(
        torch.randn(1, 1, 14, 14), valid_support=torch.zeros(1, 1, 14, 14))

    assert not bool(constant['valid'])
    assert float(constant['variance_guard']) > 0
    assert not bool(empty_support['valid'])


def test_low_precision_is_finite_and_nan_is_rejected():
    evidence = SCQOStabilizerEvidence(channels=1)
    result = evidence(torch.randn(1, 1, 14, 14).half())

    assert all(
        torch.isfinite(value).all() for value in result.values()
        if value.dtype != torch.bool)
    with pytest.raises(ValueError, match='finite'):
        evidence(torch.full((1, 1, 14, 14), float('nan')))


@pytest.mark.parametrize('dtype', [torch.bool, torch.int64])
def test_nonfloating_valid_support_is_rejected(dtype):
    evidence = SCQOStabilizerEvidence(channels=1)
    support = torch.ones(1, 1, 14, 14, dtype=dtype)

    with pytest.raises(TypeError, match='floating'):
        evidence(torch.randn(1, 1, 14, 14), valid_support=support)


def test_half_converted_module_keeps_evidence_computation_in_float32():
    evidence = SCQOStabilizerEvidence(channels=1).half()

    result = evidence(torch.randn(1, 1, 14, 14).half())

    assert all(
        torch.isfinite(value).all() for value in result.values()
        if value.dtype != torch.bool)
    assert all(value.dtype == torch.float32 for value in result.values()
               if value.dtype != torch.bool)


def test_evidence_is_detached_and_negative_control_preserves_energy():
    evidence = SCQOStabilizerEvidence(channels=1)
    features = torch.randn(2, 1, 14, 14, requires_grad=True)

    result = evidence(features)

    assert all(not value.requires_grad for value in result.values())
    assert torch.equal(result['negative_energy'], result['energy'])
    assert features.grad is None


def test_registry_builds_evidence_and_empty_batch_returns_vectors():
    from mmrotate.registry import MODELS
    from mmrotate.utils import register_all_modules

    register_all_modules()
    evidence = MODELS.build(dict(type='SCQOStabilizerEvidence', channels=4))
    result = evidence(torch.empty(0, 4, 14, 14))

    assert isinstance(evidence, SCQOStabilizerEvidence)
    assert all(value.shape == (0, ) for value in result.values())
