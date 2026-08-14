import pytest
import torch

from mmrotate.models.losses.group_orbit_determinantal_cluster_loss import (
    build_planar_group_orbit)


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
