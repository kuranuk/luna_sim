"""Regolith reflectance models.

All functions return the **bidirectional reflectance** ``r(i, e, g)`` [1/sr], i.e.
radiance = r × (solar irradiance normal to the beam). ``I/F = π r``.

* :func:`hapke` – Hapke model with the isotropic multiple-scattering approximation
  (IMSA), shadow-hiding (SHOE) and coherent-backscatter (CBOE) opposition effects,
  porosity factor K and macroscopic roughness (Hapke 1984; Hapke 2002, Icarus 157,
  523; Hapke 2008, Icarus 195, 918; Hapke 2012, *Theory of Reflectance and Emittance
  Spectroscopy*, 2nd ed.). Phase function: two-parameter double Henyey-Greenstein
  (b, c) as used by Sato et al. (2014, JGR Planets 119, 1775) for LROC WAC maps.
* :func:`lunar_lambert` – McEwen (1991, Icarus 92, 298) lunar-Lambert with the cubic
  phase polynomial ``L(α) = 1 - 0.019 α + 2.42e-4 α² - 1.46e-6 α³`` (α in deg;
  coefficients as implemented in USGS ISIS ``LunarLambertMcEwen``).
* :func:`lambert` – for tests.

Angles in radians. Inputs may be arrays (broadcast together).
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from lunarsim.config.schema import HapkeConfig, ReflectanceConfig

__all__ = [
    "geometry_from_vectors",
    "hapke",
    "hapke_h",
    "lambert",
    "lunar_lambert",
    "reflectance",
    "roughness_correction",
]

FloatArray = npt.NDArray[np.float64]


def geometry_from_vectors(
    n: FloatArray, to_light: FloatArray, v: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """``(mu0, mu, cos_g, psi)`` from unit normal, to-light and to-viewer vectors.

    ``psi`` is the azimuth between the incidence and emission planes [rad].
    """
    mu0 = np.einsum("...k,...k->...", n, to_light)
    mu = np.einsum("...k,...k->...", n, v)
    cos_g = np.clip(np.einsum("...k,...k->...", to_light, v), -1.0, 1.0)
    si = np.sqrt(np.clip(1.0 - mu0 * mu0, 0.0, None))
    se = np.sqrt(np.clip(1.0 - mu * mu, 0.0, None))
    den = si * se
    with np.errstate(invalid="ignore", divide="ignore"):
        cpsi = np.where(den > 1e-12, (cos_g - mu0 * mu) / np.where(den > 1e-12, den, 1.0), 1.0)
    psi = np.arccos(np.clip(cpsi, -1.0, 1.0))
    return mu0, mu, cos_g, psi


def hapke_h(x: FloatArray, w: float) -> FloatArray:
    """Chandrasekhar H function, Hapke (2002) approximation (Eq. 13)."""
    gamma = math.sqrt(1.0 - w)
    r0 = (1.0 - gamma) / (1.0 + gamma)
    xs = np.maximum(x, 1e-12)
    return np.asarray(
        1.0 / (1.0 - w * xs * (r0 + 0.5 * (1.0 - 2.0 * r0 * xs) * np.log((1.0 + xs) / xs)))
    )


def roughness_correction(
    mu0: FloatArray, mu: FloatArray, psi: FloatArray, theta_bar: float
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Hapke (1984) macroscopic roughness: effective cosines and shadowing factor S.

    Returns ``(mu0_e, mu_e, S)``. ``theta_bar`` is the mean slope angle [rad].
    """
    mu0 = np.clip(mu0, 1e-9, 1.0)
    mu = np.clip(mu, 1e-9, 1.0)
    if theta_bar <= 0.0:
        return mu0, mu, np.ones_like(mu0 * mu)
    tt = math.tan(theta_bar)
    chi = 1.0 / math.sqrt(1.0 + math.pi * tt * tt)
    i = np.arccos(mu0)
    e = np.arccos(mu)
    si, se = np.sin(i), np.sin(e)

    def e1(x: FloatArray) -> FloatArray:
        with np.errstate(divide="ignore", over="ignore"):
            return np.asarray(np.exp(-2.0 / (math.pi * tt * np.maximum(np.tan(x), 1e-12))))

    def e2(x: FloatArray) -> FloatArray:
        with np.errstate(divide="ignore", over="ignore"):
            return np.asarray(
                np.exp(-1.0 / (math.pi * tt * tt * np.maximum(np.tan(x), 1e-12) ** 2))
            )

    def eta(x: FloatArray) -> FloatArray:
        return np.asarray(chi * (np.cos(x) + np.sin(x) * tt * e2(x) / (2.0 - e1(x))))

    e1i, e1e, e2i, e2e = e1(i), e1(e), e2(i), e2(e)
    s2 = np.sin(psi / 2.0) ** 2
    cpsi = np.cos(psi)
    with np.errstate(over="ignore"):
        f = np.exp(-2.0 * np.tan(np.minimum(psi, math.pi - 1e-9) / 2.0))
    eta_i, eta_e = eta(i), eta(e)

    # Case i <= e
    den_a = 2.0 - e1e - (psi / math.pi) * e1i
    mu0e_a = chi * (mu0 + si * tt * (cpsi * e2e + s2 * e2i) / den_a)
    mue_a = chi * (mu + se * tt * (e2e - s2 * e2i) / den_a)
    s_a = mue_a / eta_e * mu0 / eta_i * chi / (1.0 - f + f * chi * mu0 / eta_i)
    # Case e < i
    den_b = 2.0 - e1i - (psi / math.pi) * e1e
    mu0e_b = chi * (mu0 + si * tt * (e2i - s2 * e2e) / den_b)
    mue_b = chi * (mu + se * tt * (cpsi * e2i + s2 * e2e) / den_b)
    s_b = mue_b / eta_e * mu0 / eta_i * chi / (1.0 - f + f * chi * mu / eta_e)

    case_a = i <= e
    return (
        np.asarray(np.where(case_a, mu0e_a, mu0e_b)),
        np.asarray(np.where(case_a, mue_a, mue_b)),
        np.asarray(np.where(case_a, s_a, s_b)),
    )


def hapke(
    mu0: FloatArray, mu: FloatArray, cos_g: FloatArray, psi: FloatArray, p: HapkeConfig
) -> FloatArray:
    """Hapke bidirectional reflectance [1/sr] (0 where not illuminated or not visible)."""
    lit = (mu0 > 0.0) & (mu > 0.0)
    g = np.arccos(np.clip(cos_g, -1.0, 1.0))
    mu0e, mue, S = roughness_correction(mu0, mu, psi, math.radians(p.theta_bar_deg))
    if p.filling_factor is not None:
        y = 1.209 * p.filling_factor ** (2.0 / 3.0)
        K = -math.log(1.0 - y) / y
    else:
        K = 1.0
    b, c = p.b, p.c
    ph = (
        0.5 * (1.0 + c) * (1.0 - b * b) / (1.0 - 2.0 * b * cos_g + b * b) ** 1.5
        + 0.5 * (1.0 - c) * (1.0 - b * b) / (1.0 + 2.0 * b * cos_g + b * b) ** 1.5
    )
    tg2 = np.tan(g / 2.0)
    b_sh = 1.0 + p.bs0 / (1.0 + tg2 / p.hs)
    x = tg2 / p.hc
    with np.errstate(divide="ignore", invalid="ignore"):
        cb_term = np.where(x > 1e-12, (1.0 - np.exp(-x)) / np.where(x > 1e-12, x, 1.0), 1.0)
    b_cb = 1.0 + p.bc0 * 0.5 * (1.0 + cb_term) / (1.0 + x) ** 2
    M = hapke_h(mu0e / K, p.w) * hapke_h(mue / K, p.w) - 1.0
    r = K * p.w / (4.0 * math.pi) * mu0e / (mu0e + mue) * (ph * b_sh + M) * b_cb * S
    return np.asarray(np.where(lit, r, 0.0), dtype=np.float64)


def lunar_lambert(
    mu0: FloatArray,
    mu: FloatArray,
    cos_g: FloatArray,
    albedo: float,
    coeffs: tuple[float, float, float] = (-0.019, 2.42e-4, -1.46e-6),
) -> FloatArray:
    """McEwen lunar-Lambert reflectance [1/sr]: ``I/F = A [2L μ0/(μ0+μ) + (1-L) μ0]``."""
    lit = (mu0 > 0.0) & (mu > 0.0)
    a = np.degrees(np.arccos(np.clip(cos_g, -1.0, 1.0)))
    L = 1.0 + coeffs[0] * a + coeffs[1] * a**2 + coeffs[2] * a**3
    m0 = np.maximum(mu0, 0.0)
    m = np.maximum(mu, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        i_f = albedo * (2.0 * L * m0 / np.maximum(m0 + m, 1e-12) + (1.0 - L) * m0)
    return np.asarray(np.where(lit, np.maximum(i_f, 0.0) / math.pi, 0.0), dtype=np.float64)


def lambert(mu0: FloatArray, mu: FloatArray, albedo: float) -> FloatArray:
    """Lambertian reflectance [1/sr]: ``r = A μ0 / π``."""
    lit = (mu0 > 0.0) & (mu > 0.0)
    return np.asarray(np.where(lit, albedo * mu0 / math.pi, 0.0), dtype=np.float64)


def reflectance(
    cfg: ReflectanceConfig, n: FloatArray, to_light: FloatArray, to_viewer: FloatArray
) -> FloatArray:
    """Bidirectional reflectance [1/sr] for unit vectors (shape ``(..., 3)``)."""
    mu0, mu, cos_g, psi = geometry_from_vectors(n, to_light, to_viewer)
    if cfg.model == "hapke":
        return hapke(mu0, mu, cos_g, psi, cfg.hapke)
    if cfg.model == "lunar_lambert":
        ll = cfg.lunar_lambert
        return lunar_lambert(mu0, mu, cos_g, ll.albedo, (ll.m1, ll.m2, ll.m3))
    return lambert(mu0, mu, cfg.lambert_albedo)
