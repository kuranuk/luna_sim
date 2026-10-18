from __future__ import annotations

import math

import numpy as np
import pytest

from lunarsim.config.schema import HapkeConfig, ReflectanceConfig
from lunarsim.core.illumination.brdf import (
    hapke,
    hapke_h,
    lambert,
    lunar_lambert,
    reflectance,
    roughness_correction,
)


def _geom(i: float, e: float, g: float) -> tuple[np.ndarray, ...]:
    mu0, mu = np.cos(np.radians(i)), np.cos(np.radians(e))
    si, se = np.sin(np.radians(i)), np.sin(np.radians(e))
    cpsi = (np.cos(np.radians(g)) - mu0 * mu) / (si * se) if si * se > 1e-12 else 1.0
    return (
        np.array(mu0),
        np.array(mu),
        np.array(np.cos(np.radians(g))),
        np.array(np.arccos(np.clip(cpsi, -1, 1))),
    )


def test_h_function_limits() -> None:
    assert float(hapke_h(np.array(0.7), 1e-9)) == pytest.approx(1.0, abs=1e-6)
    # Exact Chandrasekhar H(1) for conservative isotropic scattering = 2.9078.
    assert float(hapke_h(np.array(1.0), 0.999999)) == pytest.approx(2.9078, rel=0.01)


def test_roughness_zero_is_identity() -> None:
    mu0, mu, _, psi = _geom(40, 20, 50)
    a, b, s = roughness_correction(mu0, mu, psi, 0.0)
    assert float(a) == pytest.approx(float(mu0)) and float(b) == pytest.approx(float(mu))
    assert float(s) == 1.0


def test_hapke_smooth_matches_closed_form() -> None:
    p = HapkeConfig(w=0.4, b=0.3, c=0.4, bs0=1.0, hs=0.06, theta_bar_deg=0.0)
    for i, e, g in [(30, 0, 30), (60, 30, 80), (10, 50, 45)]:
        mu0, mu, cg, psi = _geom(i, e, g)
        b, c = p.b, p.c
        ph = (
            0.5 * (1 + c) * (1 - b * b) / (1 - 2 * b * cg + b * b) ** 1.5
            + 0.5 * (1 - c) * (1 - b * b) / (1 + 2 * b * cg + b * b) ** 1.5
        )
        bsh = 1 + p.bs0 / (1 + np.tan(np.radians(g) / 2) / p.hs)
        M = hapke_h(mu0, p.w) * hapke_h(mu, p.w) - 1
        expected = p.w / (4 * math.pi) * mu0 / (mu0 + mu) * (ph * bsh + M)
        assert float(hapke(mu0, mu, cg, psi, p)) == pytest.approx(float(expected), rel=1e-12)


@pytest.mark.parametrize("theta", [0.0, 23.657, 35.0])
def test_hapke_reciprocity(theta: float) -> None:
    p = HapkeConfig(theta_bar_deg=theta)
    for i, e, g in [(30, 60, 50), (70, 20, 80), (10, 45, 40), (80, 5, 82)]:
        a = hapke(*_geom(i, e, g), p) / np.cos(np.radians(i))
        b = hapke(*_geom(e, i, g), p) / np.cos(np.radians(e))
        assert float(a) == pytest.approx(float(b), rel=1e-9)


def test_opposition_surge_and_shadowed() -> None:
    p = HapkeConfig()
    near0 = float(hapke(*_geom(30, 30, 0.0001), p))
    at5 = float(hapke(*_geom(30, 30.0, 5.0), p))
    assert near0 > 1.2 * at5
    flat = HapkeConfig(bs0=0.0)
    assert float(hapke(*_geom(30, 30, 0.0001), flat)) < 1.1 * float(
        hapke(*_geom(30, 30, 5.0), flat)
    )
    assert float(hapke(*_geom(95, 30, 100), p)) == 0.0  # sun below local horizon


def test_default_hapke_reflectance_level() -> None:
    # Sample default chosen for I/F ≈ 0.1 at i = 30°, e = 0° (see HapkeConfig docstring).
    assert math.pi * float(hapke(*_geom(30, 0, 30), HapkeConfig())) == pytest.approx(0.1, abs=0.01)


def test_lunar_lambert() -> None:
    mu0, mu, cg, _ = _geom(40, 0, 40)
    a = 40.0
    L = 1 - 0.019 * a + 2.42e-4 * a**2 - 1.46e-6 * a**3
    expected = 0.12 * (2 * L * mu0 / (mu0 + mu) + (1 - L) * mu0) / math.pi
    assert float(lunar_lambert(mu0, mu, cg, 0.12)) == pytest.approx(float(expected))
    # At zero phase L = 1: I/F = 2 A mu0 / (mu0 + mu).
    m = np.array(0.8)
    assert float(lunar_lambert(m, m, np.array(1.0), 0.1)) * math.pi == pytest.approx(0.1)


def test_reflectance_dispatch_vectors() -> None:
    n = np.array([[0.0, 0.0, 1.0]])
    sun = np.array([[0.0, np.sin(np.radians(30)), np.cos(np.radians(30))]])
    v = np.array([[0.0, 0.0, 1.0]])
    r = reflectance(ReflectanceConfig(model="lambert", lambert_albedo=0.2), n, sun, v)
    assert float(r[0]) == pytest.approx(0.2 * np.cos(np.radians(30)) / math.pi)
    assert float(lambert(np.array(-0.1), np.array(1.0), 0.2)) == 0.0
    r_h = reflectance(ReflectanceConfig(), n, sun, v)
    assert r_h[0] > 0
