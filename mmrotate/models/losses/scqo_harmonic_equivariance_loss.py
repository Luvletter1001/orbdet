# Copyright (c) OpenMMLab. All rights reserved.
"""Exact O(2) actions and equivariance loss for SCQO harmonics."""

import math
from typing import Optional

import torch
from torch import Tensor

from mmrotate.registry import MODELS


def _validate_order(order):
    """Validate a positive harmonic order."""
    if isinstance(order, bool) or not isinstance(order, int) or order <= 0:
        raise ValueError('order must be a positive integer')
    return order


def _validate_finite_real(value, name: str, *, positive: bool) -> None:
    if isinstance(value, bool) or (isinstance(value, Tensor)
                                   and value.dtype == torch.bool):
        raise ValueError(f'{name} must not be boolean')
    try:
        is_finite = math.isfinite(value)
    except (TypeError, ValueError):
        raise ValueError(f'{name} must be a finite real number') from None
    if not is_finite or (value <= 0 if positive else value < 0):
        bound = 'positive' if positive else 'non-negative'
        raise ValueError(f'{name} must be finite and {bound}')


def _validate_vector(vector: Tensor, name: str) -> None:
    if not isinstance(vector, Tensor):
        raise ValueError(f'{name} must be a Tensor')
    if vector.ndim < 1 or vector.shape[-1] != 2:
        raise ValueError(f'{name} must have shape [..., 2]')
    if not vector.is_floating_point():
        raise ValueError(f'{name} must have a floating-point dtype')
    if not torch.isfinite(vector).all():
        raise ValueError(f'{name} must contain only finite values')


def _promoted_float(vector: Tensor) -> Tensor:
    if vector.dtype in (torch.float16, torch.bfloat16):
        return vector.float()
    return vector


def normalize_harmonic(vector: Tensor, eps: float = 1e-8) -> Tensor:
    """Normalize harmonic carrier vectors while leaving zero vectors zero."""
    _validate_vector(vector, 'vector')
    _validate_finite_real(eps, 'eps', positive=True)
    vector = _promoted_float(vector)
    scale = vector.abs().amax(dim=-1, keepdim=True)
    normalization_scale = torch.where(scale > 1, scale, torch.ones_like(scale))
    scaled = vector / normalization_scale
    denominator = (
        torch.linalg.vector_norm(scaled, dim=-1, keepdim=True) +
        eps / normalization_scale)
    normalized = scaled / denominator
    return torch.where(scale > 0, normalized, torch.zeros_like(normalized))


def induced_harmonic_action(transform: Tensor,
                            order: int,
                            atol: float = 1e-4) -> Tensor:
    """Return the order-``order`` harmonic action induced by an O(2) map."""
    _validate_order(order)
    if not isinstance(transform, Tensor):
        raise ValueError('transform must be a Tensor')
    if transform.ndim < 2 or transform.shape[-2:] != (2, 2):
        raise ValueError('transform must have shape [..., 2, 2]')
    if not transform.is_floating_point():
        raise ValueError('transform must have a floating-point dtype')
    if not torch.isfinite(transform).all():
        raise ValueError('transform must contain only finite values')
    _validate_finite_real(atol, 'atol', positive=False)

    transform = _promoted_float(transform)
    identity = torch.eye(2, dtype=transform.dtype, device=transform.device)
    identity = identity.expand_as(transform)
    gram = transform.transpose(-1, -2) @ transform
    if not torch.allclose(gram, identity, atol=atol, rtol=atol):
        raise ValueError('transform must be orthogonal')

    determinant = torch.linalg.det(transform)
    if not torch.allclose(
            determinant.abs(), torch.ones_like(determinant), atol=atol,
            rtol=atol):
        raise ValueError('transform determinant must have absolute value one')

    phi = torch.atan2(transform[..., 1, 0], transform[..., 0, 0])
    harmonic_phi = order * phi
    cosine, sine = torch.cos(harmonic_phi), torch.sin(harmonic_phi)
    action = torch.stack((
        torch.stack((cosine, -sine), dim=-1),
        torch.stack((sine, cosine), dim=-1),
    ),
                         dim=-2)
    reflection = torch.tensor([[1.0, 0.0], [0.0, -1.0]],
                              dtype=transform.dtype,
                              device=transform.device)
    reflected_action = action @ reflection
    return torch.where((determinant < 0)[..., None, None], reflected_action,
                       action)


def decode_harmonic_angle(vector: Tensor,
                          order: int,
                          eps: float = 1e-8) -> Tensor:
    """Decode an angle modulo the harmonic quotient period ``2π / order``."""
    _validate_order(order)
    vector = normalize_harmonic(vector, eps=eps)
    return torch.atan2(vector[..., 1], vector[..., 0]) / order


@MODELS.register_module()
class SCQOHarmonicEquivarianceLoss(torch.nn.Module):
    """Bounded cosine loss between a harmonic carrier and its O(2) view."""

    def __init__(self,
                 order: int = 2,
                 eps: float = 1e-8,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0):
        super().__init__()
        self.order = _validate_order(order)
        _validate_finite_real(eps, 'eps', positive=True)
        if reduction not in ('none', 'mean', 'sum'):
            raise ValueError('reduction must be one of none, mean, or sum')
        _validate_finite_real(loss_weight, 'loss_weight', positive=False)
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                reference: Tensor,
                view: Tensor,
                transform: Tensor,
                weight: Optional[Tensor] = None,
                reduction_override: Optional[str] = None) -> Tensor:
        """Compute discrepancies; only unweighted samples lie in ``[0, 2]``."""
        _validate_vector(reference, 'reference')
        _validate_vector(view, 'view')
        if reference.shape != view.shape:
            raise ValueError('reference and view must have the same shape')
        action = induced_harmonic_action(transform, self.order)
        if action.shape[:-2] != reference.shape[:-1]:
            raise ValueError(
                'action batch shape must match vector batch shape')
        if reduction_override not in (None, 'none', 'mean', 'sum'):
            raise ValueError('invalid reduction_override')

        dtype = torch.promote_types(
            torch.promote_types(
                _promoted_float(reference).dtype,
                _promoted_float(view).dtype),
            _promoted_float(action).dtype)
        reference_unit = normalize_harmonic(reference, self.eps).to(dtype)
        view_unit = normalize_harmonic(view, self.eps).to(dtype)
        action = _promoted_float(action).to(dtype)
        expected = torch.matmul(action,
                                reference_unit.unsqueeze(-1)).squeeze(-1)
        loss = torch.clamp(1 - (view_unit * expected).sum(dim=-1), 0, 2)

        if weight is not None:
            if not isinstance(weight, Tensor):
                weight = torch.as_tensor(weight, device=loss.device)
            if weight.is_complex() or weight.dtype == torch.bool:
                raise ValueError('weight must have a real numeric dtype')
            weight = weight.to(dtype=dtype, device=loss.device)
            if not torch.isfinite(weight).all() or torch.any(weight < 0):
                raise ValueError('weight must be finite and non-negative')
            if weight.numel() == 1:
                loss = loss * weight.reshape(())
            elif weight.numel() == loss.numel():
                loss = loss * weight.reshape(loss.shape)
            else:
                raise ValueError(
                    'weight must be scalar or one per loss sample')

        reduction = reduction_override or self.reduction
        if reduction == 'none':
            result = loss
        elif reduction == 'sum':
            result = loss.sum()
        else:
            result = loss.sum() if loss.numel() == 0 else loss.mean()
        return result * self.loss_weight
