import pytest
import torch

from mmrotate.models.losses.group_orbit_determinantal_cluster_loss import (
    GroupOrbitDeterminantalClusterLoss, build_planar_group_orbit)


def _loss_without_guards(**kwargs):
    return GroupOrbitDeterminantalClusterLoss(
        energy_guard_weight=0.0,
        variance_guard_weight=0.0,
        **kwargs)


def test_planar_group_orbits_have_exact_elements():
    x = torch.arange(1, 10, dtype=torch.float32).reshape(1, 1, 3, 3)

    c2 = build_planar_group_orbit(x, 'c2')
    assert torch.equal(c2[:, 0], x)
    assert torch.equal(c2[:, 1], torch.rot90(x, 2, (-2, -1)))

    c4 = build_planar_group_orbit(x, 'c4')
    assert c4.shape == (1, 4, 1, 3, 3)
    for index in range(4):
        assert torch.equal(c4[:, index],
                           torch.rot90(x, index, (-2, -1)))

    d1 = build_planar_group_orbit(x, 'd1')
    assert torch.equal(d1[:, 0], x)
    assert torch.equal(d1[:, 1], torch.flip(x, (-1, )))

    d2 = build_planar_group_orbit(x, 'd2')
    expected = (x, torch.rot90(x, 2, (-2, -1)),
                torch.flip(x, (-1, )), torch.flip(x, (-2, )))
    for index, element in enumerate(expected):
        assert torch.equal(d2[:, index], element)


@pytest.mark.parametrize('group', ['c1', 'unknown'])
def test_planar_group_orbit_rejects_trivial_or_unknown_groups(group):
    with pytest.raises(ValueError):
        build_planar_group_orbit(torch.ones(1, 1, 3, 3), group)


def test_c4_requires_square_support():
    with pytest.raises(ValueError, match='square'):
        build_planar_group_orbit(torch.ones(1, 1, 2, 3), 'c4')


def test_invariant_orbit_has_zero_rank_and_fixed_residuals():
    base = torch.tensor([[[[1., 2., 2., 1.], [3., 4., 4., 3.]]]])
    orbit = build_planar_group_orbit(base, 'd1')
    loss_fn = _loss_without_guards(spectral_tail_weight=1.0)
    value = loss_fn(orbit)

    assert float(value) < 1e-6
    assert float(loss_fn.last_determinantal) < 1e-6
    assert float(loss_fn.last_spectral_tail) < 1e-6
    assert float(loss_fn.last_fixed_space) < 1e-6


def test_asymmetric_orbit_costs_more_than_symmetric_orbit():
    symmetric = torch.tensor([[[[1., 2., 2., 1.],
                                [3., 4., 4., 3.]]]])
    asymmetric = torch.tensor([[[[1., 2., 3., 4.],
                                 [2., 5., 7., 11.]]]])
    loss_fn = _loss_without_guards()

    symmetric_loss = loss_fn(build_planar_group_orbit(symmetric, 'd1'))
    asymmetric_loss = loss_fn(build_planar_group_orbit(asymmetric, 'd1'))
    assert float(asymmetric_loss) > float(symmetric_loss) + 1e-4


def test_fixed_space_rejects_scaled_rank_one_orbit():
    base = torch.tensor([[1., 2., 4., 8.]])
    orbit = torch.stack((base, 2.0 * base), dim=1)
    loss_fn = _loss_without_guards(
        determinantal_weight=1.0,
        spectral_tail_weight=0.0,
        fixed_space_weight=1.0)

    loss_fn(orbit)
    assert float(loss_fn.last_determinantal) < 1e-6
    assert float(loss_fn.last_fixed_space) > 1e-3


def test_zero_constant_and_empty_support_trigger_guards():
    loss_fn = GroupOrbitDeterminantalClusterLoss(
        determinantal_weight=0.0,
        fixed_space_weight=0.0,
        spectral_tail_weight=0.0,
        min_energy=0.5,
        min_variance=0.5)
    zero = torch.zeros(1, 2, 1, 2, 2)
    constant = torch.ones(1, 2, 1, 2, 2)

    assert float(loss_fn(zero)) > 0.0
    assert float(loss_fn.last_energy_guard) > 0.0
    assert float(loss_fn(constant)) > 0.0
    assert float(loss_fn.last_variance_guard) > 0.0
    assert float(
        loss_fn(constant, support_mask=torch.zeros(1, 2, 2))) > 0.0


def test_mask_weights_reductions_and_avg_factor_follow_contract():
    torch.manual_seed(3407)
    orbit = torch.randn(2, 2, 3, 4, 4)
    mask = torch.ones(2, 4, 4)
    weight = torch.tensor([1.0, 0.0])
    none_fn = _loss_without_guards(reduction='none')

    unmasked = none_fn(orbit)
    values = none_fn(orbit, support_mask=mask, weight=weight)
    assert values.shape == (2, )
    assert torch.allclose(values[0], unmasked[0])
    assert values[1] == 0

    mean_fn = _loss_without_guards(reduction='mean')
    averaged = mean_fn(
        orbit, support_mask=mask, weight=weight, avg_factor=1.0)
    assert torch.allclose(averaged, values.sum())


def test_empty_batch_has_finite_graph_connected_reductions():
    orbit = torch.randn(0, 2, 3, 4, 4, requires_grad=True)

    none_value = _loss_without_guards(reduction='none')(orbit)
    mean_value = _loss_without_guards(reduction='mean')(orbit)
    sum_value = _loss_without_guards(reduction='sum')(orbit)

    assert none_value.shape == (0, )
    assert mean_value.shape == ()
    assert sum_value.shape == ()
    assert torch.isfinite(mean_value)
    assert torch.isfinite(sum_value)
    assert mean_value.requires_grad


def test_asymmetric_orbit_has_finite_nonzero_gradient():
    torch.manual_seed(2026)
    features = torch.randn(2, 3, 5, 5, requires_grad=True)
    loss_fn = _loss_without_guards()
    loss = loss_fn(build_planar_group_orbit(features, 'd1'))
    loss.backward()

    assert features.grad is not None
    assert torch.isfinite(features.grad).all()
    assert float(features.grad.abs().sum()) > 0.0
    assert not loss_fn.last_q_gap.requires_grad
    assert 0.0 <= float(loss_fn.last_q_gap) <= 1.0


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_low_precision_input_uses_finite_float32_gram(dtype):
    torch.manual_seed(42)
    orbit = torch.randn(2, 2, 3, 4, 4).to(dtype)
    value = _loss_without_guards()(orbit)

    assert value.dtype == torch.float32
    assert torch.isfinite(value)


def test_loss_rejects_invalid_or_nonfinite_inputs():
    loss_fn = _loss_without_guards()

    with pytest.raises(ValueError, match='at least 2'):
        loss_fn(torch.ones(1, 1, 4))
    with pytest.raises(ValueError, match='finite'):
        loss_fn(torch.tensor([[[1.0, float('nan')], [1.0, 2.0]]]))
    with pytest.raises(ValueError, match='non-negative'):
        loss_fn(
            torch.ones(1, 2, 1, 2, 2),
            support_mask=-torch.ones(1, 2, 2))
    with pytest.raises(ValueError, match='positive'):
        loss_fn(torch.ones(1, 2, 4), avg_factor=0.0)


def test_loss_builds_from_mmrotate_registry_and_public_export():
    from mmrotate.models.losses import (
        GroupOrbitDeterminantalClusterLoss as ExportedLoss)
    from mmrotate.registry import MODELS
    from mmrotate.utils import register_all_modules

    register_all_modules()
    module = MODELS.build(dict(type='GroupOrbitDeterminantalClusterLoss'))

    assert isinstance(module, GroupOrbitDeterminantalClusterLoss)
    assert ExportedLoss is GroupOrbitDeterminantalClusterLoss
