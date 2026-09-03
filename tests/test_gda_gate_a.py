# Copyright (c) OpenMMLab. All rights reserved.
"""Gate-A analytic tests for the GDA Gaussian symmetric-space core.

Pure-CPU, no dataset, no detector.  Every test pins one mathematical claim
of the design doc:

A1  (t, a, psi) <-> Sigma round-trip (KAK coordinates are faithful)
A2  (w, h, theta) -> Sigma -> decode round-trip modulo the encoding group
A3  the encoding group V4 collapses to a point: (w,h,th), (h,w,th+pi/2),
    (w,h,th+pi) give identical Sigma
A4  SO(2) acts by conjugation and psi -> psi + 2 phi, (t, a) invariant
A5  reflection acts as psi -> -psi
A6  envelope_from_sigma == analytic envelope == brute-force corners
A7  the envelope is V4-invariant (th, -th, th+pi, pi-th give same (W, H))
A8  the envelope separates off-orbit angles (th vs pi/2 - th) when w != h
A9  the envelope is exactly blind to the chamber bit sign(theta)
A10 squares grow the supervision stabilizer to D4 (period pi/2 envelope)
A11 gradients of the envelope loss are finite everywhere incl. a = 0
A12 near isotropy, a 90-degree decode flip costs a tiny Sigma-space
    distance and a tiny envelope change: flips are decode-coordinate
    artifacts, not events the representation-space loss can see
"""
import math

import torch

from mmrotate.models.utils.gda_symmetric_space import (
    chamber_bit, envelope_consistency_loss, envelope_from_sigma,
    envelope_from_wh_theta, orbit_coordinate, reflect_sigma, rotate_sigma,
    sigma_to_tapsi, sigma_to_wh_theta, tapsi_to_sigma, wh_theta_to_sigma)

torch.manual_seed(3407)
PI = math.pi


def _rand_wh_theta(n, dtype=torch.float64, near_square=False):
    if near_square:
        w = torch.rand(n, dtype=dtype) * 50 + 50
        h = w * (1.0 + (torch.rand(n, dtype=dtype) - 0.5) * 0.04)
    else:
        w = torch.rand(n, dtype=dtype) * 150 + 20
        h = torch.rand(n, dtype=dtype) * 150 + 20
    theta = (torch.rand(n, dtype=dtype) - 0.5) * PI
    return w, h, theta


def _wrap_pi(x):
    return (x + PI / 2) % PI - PI / 2


def test_a1_tapsi_roundtrip():
    n = 4096
    t = torch.rand(n, dtype=torch.float64) * 5 + 1.0
    a = torch.rand(n, dtype=torch.float64) * 2 + 1e-3
    psi = (torch.rand(n, dtype=torch.float64) - 0.5) * 2 * PI
    sigma = tapsi_to_sigma(t, a, psi)
    assert torch.all(torch.linalg.eigvalsh(sigma) > 0), 'Sigma not SPD'
    t2, a2, psi2 = sigma_to_tapsi(sigma)
    assert torch.allclose(t, t2, atol=1e-8)
    assert torch.allclose(a, a2, atol=1e-8)
    # psi is a double angle: compare on the circle via exp(i psi)
    assert torch.allclose(torch.cos(psi), torch.cos(psi2), atol=1e-8)
    assert torch.allclose(torch.sin(psi), torch.sin(psi2), atol=1e-8)


def test_a2_wh_theta_decode_roundtrip_mod_encoding_group():
    w, h, theta = _rand_wh_theta(4096)
    sigma = wh_theta_to_sigma(w, h, theta)
    w2, h2, theta2 = sigma_to_wh_theta(sigma)
    assert torch.all(w2 >= h2 - 1e-8), 'decode must give w >= h'
    assert torch.all(theta2 > -PI / 2) and torch.all(theta2 <= PI / 2)
    # re-encoding the decode must reproduce Sigma exactly: the encoding
    # group (swap + pi/2, or +pi) is invisible in representation space
    sigma2 = wh_theta_to_sigma(w2, h2, theta2)
    assert torch.allclose(sigma, sigma2, atol=1e-8)


def test_a3_encoding_group_same_sigma():
    w, h, theta = _rand_wh_theta(1024)
    base = wh_theta_to_sigma(w, h, theta)
    swapped = wh_theta_to_sigma(h, w, theta + PI / 2)
    half_turn = wh_theta_to_sigma(w, h, theta + PI)
    assert torch.allclose(base, swapped, atol=1e-8)
    assert torch.allclose(base, half_turn, atol=1e-8)


def test_a4_rotation_conjugation_action():
    w, h, theta = _rand_wh_theta(1024)
    phi = torch.rand(1024, dtype=torch.float64) * 2 * PI - PI
    sigma = wh_theta_to_sigma(w, h, theta)
    rotated = rotate_sigma(sigma, phi)
    direct = wh_theta_to_sigma(w, h, theta + phi)
    assert torch.allclose(rotated, direct, atol=1e-8)
    t0, a0, psi0 = sigma_to_tapsi(sigma)
    t1, a1, psi1 = sigma_to_tapsi(rotated)
    assert torch.allclose(t0, t1, atol=1e-8)  # invariants untouched
    assert torch.allclose(a0, a1, atol=1e-8)
    assert torch.allclose(torch.cos(psi0 + 2 * phi), torch.cos(psi1),
                          atol=1e-8)
    assert torch.allclose(torch.sin(psi0 + 2 * phi), torch.sin(psi1),
                          atol=1e-8)


def test_a5_reflection_negates_psi():
    w, h, theta = _rand_wh_theta(1024)
    sigma = wh_theta_to_sigma(w, h, theta)
    refl = reflect_sigma(sigma)
    direct = wh_theta_to_sigma(w, h, -theta)
    assert torch.allclose(refl, direct, atol=1e-8)
    _, _, psi0 = sigma_to_tapsi(sigma)
    _, _, psi1 = sigma_to_tapsi(refl)
    assert torch.allclose(torch.cos(psi1), torch.cos(-psi0), atol=1e-8)
    assert torch.allclose(torch.sin(psi1), torch.sin(-psi0), atol=1e-8)


def _bruteforce_envelope(w, h, theta):
    cx = torch.stack([w / 2, w / 2, -w / 2, -w / 2], dim=-1)
    cy = torch.stack([h / 2, -h / 2, h / 2, -h / 2], dim=-1)
    c, s = torch.cos(theta).unsqueeze(-1), torch.sin(theta).unsqueeze(-1)
    x = c * cx - s * cy
    y = s * cx + c * cy
    return (x.max(-1).values - x.min(-1).values,
            y.max(-1).values - y.min(-1).values)


def test_a6_envelope_formula_matches_geometry():
    w, h, theta = _rand_wh_theta(4096)
    sigma = wh_theta_to_sigma(w, h, theta)
    we_s, he_s = envelope_from_sigma(sigma)
    we_a, he_a = envelope_from_wh_theta(w, h, theta)
    we_b, he_b = _bruteforce_envelope(w, h, theta)
    assert torch.allclose(we_s, we_a, atol=1e-6, rtol=1e-9)
    assert torch.allclose(he_s, he_a, atol=1e-6, rtol=1e-9)
    assert torch.allclose(we_s, we_b, atol=1e-6, rtol=1e-9)
    assert torch.allclose(he_s, he_b, atol=1e-6, rtol=1e-9)


def test_a7_envelope_v4_invariance():
    w, h, theta = _rand_wh_theta(4096)
    ref = envelope_from_wh_theta(w, h, theta)
    for shifted in (-theta, theta + PI, PI - theta):
        got = envelope_from_wh_theta(w, h, shifted)
        assert torch.allclose(ref[0], got[0], atol=1e-8)
        assert torch.allclose(ref[1], got[1], atol=1e-8)


def test_a8_envelope_separates_off_orbit_angles():
    # theta vs pi/2 - theta must be distinguishable when w != h:
    # the supervision orbit is {th, -th, pi-th, pi+th}, nothing larger
    w = torch.full((256,), 120.0, dtype=torch.float64)
    h = torch.full((256,), 40.0, dtype=torch.float64)
    theta = (torch.rand(256, dtype=torch.float64) * 0.4 + 0.05) * PI / 2
    w0, _ = envelope_from_wh_theta(w, h, theta)
    w1, _ = envelope_from_wh_theta(w, h, PI / 2 - theta)
    assert torch.all((w0 - w1).abs() > 1.0)


def test_a9_envelope_blind_to_chamber_bit():
    w, h, theta = _rand_wh_theta(1024)
    theta = theta.clamp(0.05, 1.45)  # keep both signs well-defined
    sig_pos = wh_theta_to_sigma(w, h, theta)
    sig_neg = wh_theta_to_sigma(w, h, -theta)
    # the two chambers are different points in representation space ...
    assert not torch.allclose(sig_pos, sig_neg, atol=1e-6)
    assert torch.all(chamber_bit(sig_pos) != chamber_bit(sig_neg))
    # ... share the orbit coordinate ...
    assert torch.allclose(orbit_coordinate(sig_pos),
                          orbit_coordinate(sig_neg), atol=1e-9)
    # ... and produce identical envelopes: HBox cannot see the bit
    env_pos = envelope_from_sigma(sig_pos)
    env_neg = envelope_from_sigma(sig_neg)
    assert torch.allclose(env_pos[0], env_neg[0], atol=1e-8)
    assert torch.allclose(env_pos[1], env_neg[1], atol=1e-8)


def test_a10_square_grows_stabilizer_to_d4():
    s = torch.rand(512, dtype=torch.float64) * 100 + 20
    theta = (torch.rand(512, dtype=torch.float64) - 0.5) * PI
    ref = envelope_from_wh_theta(s, s, theta)
    # period pi/2 and reflection pi/2 - theta appear only at w == h
    q = envelope_from_wh_theta(s, s, theta + PI / 2)
    r = envelope_from_wh_theta(s, s, PI / 2 - theta)
    assert torch.allclose(ref[0], q[0], atol=1e-8)
    assert torch.allclose(ref[1], q[1], atol=1e-8)
    assert torch.allclose(ref[0], r[0], atol=1e-8)
    assert torch.allclose(ref[1], r[1], atol=1e-8)
    # isotropic Sigma: a = 0, decode is a finite gauge choice
    sigma = wh_theta_to_sigma(s, s, theta)
    _, a, _ = sigma_to_tapsi(sigma)
    assert torch.all(a < 1e-9)
    w2, h2, th2 = sigma_to_wh_theta(sigma)
    assert torch.isfinite(w2).all() and torch.isfinite(th2).all()
    assert torch.allclose(w2, h2, atol=1e-8)


def test_a11_envelope_loss_gradients_finite_everywhere():
    n = 2048
    t = torch.rand(n, dtype=torch.float64) * 5 + 1.0
    # include the singular point a = 0 and tiny anisotropies
    a = torch.rand(n, dtype=torch.float64) * 2
    a[: 256] = 0.0
    a[256: 512] = 1e-6
    psi = (torch.rand(n, dtype=torch.float64) - 0.5) * 2 * PI
    t.requires_grad_(True)
    a.requires_grad_(True)
    psi.requires_grad_(True)
    sigma = tapsi_to_sigma(t, a, psi)
    target = torch.rand(n, 2, dtype=torch.float64) * 150 + 30
    loss = envelope_consistency_loss(sigma, target).sum()
    loss.backward()
    for name, g in (('t', t.grad), ('a', a.grad), ('psi', psi.grad)):
        assert g is not None and torch.isfinite(g).all(), name


def test_a12_decode_flip_is_representation_artifact():
    # Near isotropy, two Sigmas only ~3e-2 apart in Frobenius norm decode
    # to le90 angles 90 degrees apart, while their envelopes are the same
    # two numbers (swapped).  The angle flip is a decode-coordinate
    # artifact: nothing in representation space or in any HBox-visible
    # quantity changes.  This is the analytic reason such instances must
    # be quotient-graded instead of angle-graded.
    side, eps_side = 100.0, 6e-4
    w = torch.tensor(side + eps_side, dtype=torch.float64)
    h = torch.tensor(side - eps_side, dtype=torch.float64)
    sigma0 = wh_theta_to_sigma(w, h, torch.tensor(0.0, dtype=torch.float64))
    sigma1 = wh_theta_to_sigma(
        w, h, torch.tensor(PI / 2, dtype=torch.float64))
    _, _, th0 = sigma_to_wh_theta(sigma0)
    _, _, th1 = sigma_to_wh_theta(sigma1)
    jump = abs(_wrap_pi(th1 - th0).item())
    dist = torch.linalg.norm(sigma1 - sigma0).item()
    env0, _ = torch.sort(torch.stack(envelope_from_sigma(sigma0)))
    env1, _ = torch.sort(torch.stack(envelope_from_sigma(sigma1)))
    assert jump > PI / 3, f'expected a ~90deg decode flip, got {jump}'
    assert dist < 0.05, dist
    assert torch.allclose(env0, env1, atol=1e-6)
