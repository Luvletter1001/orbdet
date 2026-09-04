# Copyright (c) OpenMMLab. All rights reserved.
"""Gaussian symmetric-space representation of oriented boxes (GDA Gate-A).

Core object: the covariance of the uniform distribution on the rotated
rectangle ``(w, h, theta)``,

    Sigma = R(theta) diag(w^2/12, h^2/12) R(theta)^T .

The space Sym+(2) of such covariances is a Riemannian symmetric space with
KAK/Iwasawa coordinates ``(t, a, psi)``:

    Sigma = e^t * (cosh a * I + sinh a * [[cos psi,  sin psi],
                                          [sin psi, -cos psi]]),

where ``t`` is log-scale, ``a >= 0`` is anisotropy (half log eigenvalue
ratio) and ``psi = 2 * theta`` is the double angle (the u2 harmonic
carrier).  Rotation invariants ``(t, a)`` are separated from the orbit
coordinate ``psi`` by construction.

Group facts (all verified analytically in tests/test_gda_gate_a.py):

* Encoding group V4: ``(w, h, theta)``, ``(h, w, theta + pi/2)`` and
  ``(w, h, theta + pi)`` are the same physical rectangle and give the
  *same* Sigma.  Sigma quotients out the encoding group exactly, which is
  the group-theoretic reason Gaussian/covariance coders (BCGE, GauCho,
  structure-tensor coders) are boundary-continuous.
* O(2) acts by conjugation: rotating the image by ``phi`` maps
  ``Sigma -> R(phi) Sigma R(phi)^T``, i.e. ``psi -> psi + 2 phi`` with
  ``(t, a)`` unchanged; reflecting across the x-axis maps
  ``Sigma -> F Sigma F`` with ``F = diag(1, -1)``, i.e. ``psi -> -psi``.
* HBox supervision stabilizer: the axis-aligned envelope ``(W, H)`` of the
  rectangle satisfies

      W^2 = 12 * Sigma11 + 12 * sqrt(det Sigma) * |sin 2 theta|
      H^2 = 12 * Sigma22 + 12 * sqrt(det Sigma) * |sin 2 theta|

  which depends on Sigma only through its V4 invariants.  Hence HBox
  consistency can supervise the orbit coordinate ``|theta|`` (equivalently
  ``cos 2 theta``) but is exactly blind to one discrete bit,
  ``sign(theta)`` (the chamber bit).  For ``w == h`` the supervision
  stabilizer grows from V4 to D4 (envelope acquires period pi/2).
"""
import torch
from torch import Tensor

_EPS = 1e-12


def _stack_sigma(s11: Tensor, s12: Tensor, s22: Tensor) -> Tensor:
    """Assemble (..., 2, 2) symmetric matrices from entries."""
    row0 = torch.stack([s11, s12], dim=-1)
    row1 = torch.stack([s12, s22], dim=-1)
    return torch.stack([row0, row1], dim=-2)


def wh_theta_to_sigma(w: Tensor, h: Tensor, theta: Tensor) -> Tensor:
    """Rectangle (w along theta, h perpendicular) -> covariance.

    Angles in radians.  Broadcastable, returns (..., 2, 2).
    """
    c, s = torch.cos(theta), torch.sin(theta)
    lam_w = w * w / 12.0
    lam_h = h * h / 12.0
    s11 = lam_w * c * c + lam_h * s * s
    s22 = lam_w * s * s + lam_h * c * c
    s12 = (lam_w - lam_h) * s * c
    return _stack_sigma(s11, s12, s22)


def tapsi_to_sigma(t: Tensor, a: Tensor, psi: Tensor) -> Tensor:
    """KAK coordinates (t, a, psi=2*theta) -> covariance."""
    et = torch.exp(t)
    cha, sha = torch.cosh(a), torch.sinh(a)
    s11 = et * (cha + sha * torch.cos(psi))
    s22 = et * (cha - sha * torch.cos(psi))
    s12 = et * sha * torch.sin(psi)
    return _stack_sigma(s11, s12, s22)


def sigma_to_tapsi(sigma: Tensor,
                   iso_eps: float = 1e-9) -> tuple:
    """Covariance -> KAK coordinates (t, a, psi).

    At the isotropic point (a = 0) the angle coordinate is undefined;
    psi = 0 is returned by convention and never produces NaN.
    """
    s11 = sigma[..., 0, 0]
    s22 = sigma[..., 1, 1]
    s12 = sigma[..., 0, 1]
    det = (s11 * s22 - s12 * s12).clamp_min(_EPS)
    t = 0.5 * torch.log(det)
    diff = s11 - s22
    off = 2.0 * s12
    # +1e-24 keeps the sqrt backward finite at the isotropic point
    gap = torch.sqrt(diff * diff + off * off + 1e-24)  # lambda_+ - lambda_-
    lam_p = 0.5 * (s11 + s22 + gap)
    lam_m = 0.5 * (s11 + s22 - gap)
    a = 0.5 * torch.log(lam_p.clamp_min(_EPS) / lam_m.clamp_min(_EPS))
    psi = torch.atan2(off, diff)
    psi = torch.where(gap > iso_eps, psi, torch.zeros_like(psi))
    return t, a, psi


def sigma_to_wh_theta(sigma: Tensor) -> tuple:
    """le90 decode: covariance -> (w >= h, theta in (-pi/2, pi/2]).

    The decode convention picks the eigenvector of the *larger* eigenvalue
    as the long edge.  For an isotropic Sigma the angle is a gauge choice;
    theta = 0 is returned.
    """
    t, a, psi = sigma_to_tapsi(sigma)
    lam_p = torch.exp(t + a)
    lam_m = torch.exp(t - a)
    w = torch.sqrt(12.0 * lam_p)
    h = torch.sqrt(12.0 * lam_m)
    theta = 0.5 * psi
    return w, h, theta


def rotate_sigma(sigma: Tensor, phi: Tensor) -> Tensor:
    """SO(2) conjugation: Sigma -> R(phi) Sigma R(phi)^T."""
    c, s = torch.cos(phi), torch.sin(phi)
    row0 = torch.stack([c, -s], dim=-1)
    row1 = torch.stack([s, c], dim=-1)
    rot = torch.stack([row0, row1], dim=-2)
    return rot @ sigma @ rot.transpose(-1, -2)


def reflect_sigma(sigma: Tensor) -> Tensor:
    """Reflection across the x-axis: Sigma -> F Sigma F, F = diag(1, -1)."""
    out = sigma.clone()
    out[..., 0, 1] = -out[..., 0, 1]
    out[..., 1, 0] = -out[..., 1, 0]
    return out


def envelope_from_wh_theta(w: Tensor, h: Tensor,
                           theta: Tensor) -> tuple:
    """Axis-aligned envelope (W, H) of the rectangle."""
    cw, sw = torch.cos(theta).abs(), torch.sin(theta).abs()
    return w * cw + h * sw, w * sw + h * cw


def envelope_from_sigma(sigma: Tensor) -> tuple:
    """Axis-aligned envelope (W, H) directly from the covariance.

    Uses  W^2 = 12*Sigma11 + 12*sqrt(det) * |sin 2 theta|  with
    |sin 2 theta| = |2 Sigma12| / (lambda_+ - lambda_-).  The ratio is
    bounded in [0, 1]; at exact isotropy it is 0/0 and is set to 0, which
    is the smooth-limit convention (the envelope of a near-square is
    insensitive to the angle to first order in the eigenvalue gap).
    """
    s11 = sigma[..., 0, 0]
    s22 = sigma[..., 1, 1]
    s12 = sigma[..., 0, 1]
    diff = s11 - s22
    off = 2.0 * s12
    gap = torch.sqrt(diff * diff + off * off + 1e-24)
    det = (s11 * s22 - s12 * s12).clamp_min(_EPS)
    abs_sin2 = (off.abs() / gap.clamp_min(_EPS)).clamp(0.0, 1.0)
    abs_sin2 = torch.where(gap > _EPS, abs_sin2, torch.zeros_like(abs_sin2))
    boost = torch.sqrt(det) * abs_sin2
    w_env = torch.sqrt((12.0 * (s11 + boost)).clamp_min(_EPS))
    h_env = torch.sqrt((12.0 * (s22 + boost)).clamp_min(_EPS))
    return w_env, h_env


def orbit_coordinate(sigma: Tensor) -> Tensor:
    """V4 orbit coordinate cos(2 theta) in [-1, 1] (monotone in |theta|).

    HBox-identifiable: two Sigmas with the same envelope have the same
    orbit coordinate.  Isotropic inputs return 1.0 by convention.
    """
    s11 = sigma[..., 0, 0]
    s22 = sigma[..., 1, 1]
    s12 = sigma[..., 0, 1]
    diff = s11 - s22
    off = 2.0 * s12
    gap = torch.sqrt(diff * diff + off * off)
    coord = diff / gap.clamp_min(_EPS)
    return torch.where(gap > _EPS, coord, torch.ones_like(coord))


def chamber_bit(sigma: Tensor) -> Tensor:
    """The HBox-invisible discrete bit sign(sin 2 theta) = sign(Sigma12)."""
    return (sigma[..., 0, 1] > 0).to(sigma.dtype)


def envelope_consistency_loss(pred_sigma: Tensor, target_wh: Tensor,
                              beta: float = 0.05) -> Tensor:
    """Smooth-L1 between predicted envelope and target (W, H).

    This is the H2RBox-style view-consistency term written in the
    Gaussian representation: it only sees V4 invariants of ``pred_sigma``
    and therefore never penalizes gauge-equivalent angle encodings.
    """
    w_env, h_env = envelope_from_sigma(pred_sigma)
    tw = target_wh[..., 0]
    th = target_wh[..., 1]

    def _sl1(x: Tensor) -> Tensor:
        ax = x.abs()
        return torch.where(ax < beta, 0.5 * ax * ax / beta, ax - 0.5 * beta)

    return _sl1(w_env - tw) + _sl1(h_env - th)
