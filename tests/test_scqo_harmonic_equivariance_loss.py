import math

import pytest
import torch

from mmrotate.registry import MODELS
from mmrotate.utils import register_all_modules


def _rotation(angle, *, dtype=torch.float32):
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return torch.tensor([[cosine, -sine], [sine, cosine]], dtype=dtype)


@pytest.mark.parametrize('order', [2, 4])
def test_induced_actions_match_harmonic_rotation_and_reflection(order):
    from mmrotate.models.losses.scqo_harmonic_equivariance_loss import (
        induced_harmonic_action, )

    theta, phi = 0.31, 0.47
    harmonic = torch.tensor([math.cos(order * theta), math.sin(order * theta)])
    expected_rotation = torch.tensor(
        [math.cos(order * (theta + phi)),
         math.sin(order * (theta + phi))])
    expected_reflection = torch.tensor(
        [math.cos(-order * theta),
         math.sin(-order * theta)])

    rotated = induced_harmonic_action(_rotation(phi), order) @ harmonic
    reflected = induced_harmonic_action(
        torch.tensor([[1.0, 0.0], [0.0, -1.0]]), order) @ harmonic

    assert torch.allclose(rotated, expected_rotation, atol=1e-6)
    assert torch.allclose(reflected, expected_reflection, atol=1e-6)


def test_equivariance_loss_is_zero_for_correct_views_and_differentiable():
    from mmrotate.models.losses.scqo_harmonic_equivariance_loss import (
        SCQOHarmonicEquivarianceLoss, induced_harmonic_action)

    q_ref = torch.tensor([1.0, 0.0], requires_grad=True)
    transforms = torch.stack([_rotation(0.37), _rotation(0.61)])
    actions = induced_harmonic_action(transforms, order=2)
    views = torch.einsum('bij,j->bi', actions, q_ref)
    loss_fn = SCQOHarmonicEquivarianceLoss(order=2)

    correct_loss = loss_fn(q_ref.expand_as(views), views, transforms)
    incorrect_loss = loss_fn(
        q_ref.expand_as(views), q_ref.expand_as(views), transforms)

    assert float(correct_loss) <= 1e-6
    assert float(incorrect_loss) > 0.1
    correct_loss.backward()
    assert q_ref.grad is not None
    assert torch.isfinite(q_ref.grad).all()


def test_zero_carriers_have_finite_unit_loss_without_reduction():
    from mmrotate.models.losses.scqo_harmonic_equivariance_loss import (
        SCQOHarmonicEquivarianceLoss, )

    loss_fn = SCQOHarmonicEquivarianceLoss(order=2, reduction='none')
    zero = torch.zeros(3, 2)
    identity = torch.eye(2).expand(3, -1, -1)

    loss = loss_fn(zero, zero, identity)

    assert torch.isfinite(loss).all()
    assert torch.equal(loss, torch.ones(3))


def test_singleton_weights_preserve_none_reduction_loss_shape():
    from mmrotate.models.losses.scqo_harmonic_equivariance_loss import (
        SCQOHarmonicEquivarianceLoss, )

    loss_fn = SCQOHarmonicEquivarianceLoss(order=2, reduction='none')
    reference = torch.tensor([1.0, 0.0])
    transform = torch.eye(2)
    scalar_loss = loss_fn(
        reference, reference, transform, weight=torch.tensor([2.0]))

    batched_reference = reference.expand(3, -1)
    batched_loss = loss_fn(
        batched_reference,
        batched_reference,
        transform.expand(3, -1, -1),
        weight=torch.tensor([[2.0]]))

    assert scalar_loss.shape == torch.Size([])
    assert batched_loss.shape == torch.Size([3])


def test_normalize_harmonic_handles_extreme_finite_carriers():
    from mmrotate.models.losses.scqo_harmonic_equivariance_loss import (
        normalize_harmonic, )

    normalized = normalize_harmonic(
        torch.tensor([[2e19, 2e19]], dtype=torch.float32))
    expected = torch.full((1, 2), math.sqrt(0.5), dtype=torch.float32)

    assert torch.isfinite(normalized).all()
    assert torch.any(normalized != 0)
    assert torch.allclose(normalized, expected, atol=1e-6)


def test_boolean_hyperparameters_and_weights_are_rejected():
    from mmrotate.models.losses.scqo_harmonic_equivariance_loss import (
        SCQOHarmonicEquivarianceLoss, induced_harmonic_action,
        normalize_harmonic)

    carrier = torch.tensor([1.0, 0.0])
    with pytest.raises(ValueError):
        normalize_harmonic(carrier, eps=True)
    with pytest.raises(ValueError):
        induced_harmonic_action(torch.eye(2), 2, atol=True)
    with pytest.raises(ValueError):
        SCQOHarmonicEquivarianceLoss(eps=True)
    with pytest.raises(ValueError):
        SCQOHarmonicEquivarianceLoss(loss_weight=True)

    loss_fn = SCQOHarmonicEquivarianceLoss()
    for weight in (True, torch.tensor(True)):
        with pytest.raises(ValueError):
            loss_fn(carrier, carrier, torch.eye(2), weight=weight)


def test_float16_inputs_are_promoted_to_float32():
    from mmrotate.models.losses.scqo_harmonic_equivariance_loss import (
        SCQOHarmonicEquivarianceLoss, induced_harmonic_action)

    q_ref = torch.tensor([1.0, 0.0], dtype=torch.float16)
    q_view = torch.tensor([1.0, 0.0], dtype=torch.float16)
    transform = torch.eye(2, dtype=torch.float16)
    action = induced_harmonic_action(transform, 4)
    loss = SCQOHarmonicEquivarianceLoss(order=4)(q_ref, q_view, transform)

    assert action.dtype == torch.float32
    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)


@pytest.mark.parametrize('order', [2, 4])
def test_decode_harmonic_angle_respects_quotient_period(order):
    from mmrotate.models.losses.scqo_harmonic_equivariance_loss import (
        decode_harmonic_angle, )

    angles = torch.tensor([-1.2, -0.2, 0.7])
    carriers = torch.stack(
        [torch.cos(order * angles),
         torch.sin(order * angles)], dim=-1)
    decoded = decode_harmonic_angle(carriers, order)
    period = 2 * math.pi / order
    wrapped_error = torch.remainder(decoded - angles + period / 2,
                                    period) - period / 2

    assert torch.allclose(wrapped_error, torch.zeros_like(angles), atol=1e-6)


def test_registry_builds_loss_and_rejects_non_orthogonal_transform():
    from mmrotate.models.losses.scqo_harmonic_equivariance_loss import (
        induced_harmonic_action, )

    register_all_modules()
    loss = MODELS.build(dict(type='SCQOHarmonicEquivarianceLoss', order=4))

    assert loss.order == 4
    with pytest.raises(ValueError, match='orthogonal'):
        induced_harmonic_action(torch.tensor([[1.0, 1.0], [0.0, 1.0]]), 2)
    with pytest.raises(ValueError, match='orthogonal'):
        loss(
            torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0]),
            torch.tensor([[1.0, 1.0], [0.0, 1.0]]))
