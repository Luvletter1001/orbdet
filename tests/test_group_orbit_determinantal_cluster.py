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
