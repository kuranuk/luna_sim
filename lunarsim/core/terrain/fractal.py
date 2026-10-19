"""Self-affine fractal terrain by spectral (Fourier) synthesis.

A 2-D self-affine surface with Hurst exponent ``H`` has an isotropic power
spectral density ``P(k) ∝ k^-(2H + 2)`` (e.g. Turcotte, *Fractals and Chaos in
Geology and Geophysics*, 2nd ed., 1997). We filter white Gaussian noise
in the Fourier domain with ``sqrt(P(k))`` and then fix the absolute amplitude so
that the RMS slope at the configured baseline matches exactly (see
:class:`lunarsim.config.FractalConfig`).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import scipy.fft as sfft

from lunarsim.config.schema import FractalConfig

__all__ = [
    "fractal_psd",
    "increment_variance",
    "rms_slope_at_baseline",
    "spectral_amplitude",
    "spectral_exponent",
    "synthesize_fractal",
]

FloatArray = npt.NDArray[np.float64]


def spectral_exponent(hurst: float) -> float:
    """Exponent ``beta`` of the 2-D isotropic PSD ``P(k) ∝ k^-beta`` (``beta = 2H + 2``)."""
    return 2.0 * hurst + 2.0


def rms_slope_at_baseline(z: FloatArray, resolution: float, lag_cells: int) -> float:
    """RMS height difference over ``lag_cells`` cells divided by the baseline [m/m].

    Differences along x and y are pooled.
    """
    if lag_cells < 1:
        raise ValueError("lag_cells must be >= 1")
    ny, nx = z.shape
    lag = lag_cells
    sx = sy = 0.0
    # Row blocks keep the temporaries in cache (17x faster than whole-array differences
    # on a 4001² grid); the block size is fixed so the sum order is deterministic.
    for i in range(0, ny, 256):
        blk = z[i : i + 256]
        dx = blk[:, lag:] - blk[:, :-lag]
        sx += float(np.einsum("ij,ij->", dx, dx))
        hi = min(i + 256, ny - lag)
        if hi > i:
            dy = z[i + lag : hi + lag] - z[i:hi]
            sy += float(np.einsum("ij,ij->", dy, dy))
    n = ny * max(nx - lag, 0) + max(ny - lag, 0) * nx
    return float(np.sqrt((sx + sy) / n) / (lag * resolution))


def synthesize_fractal(
    shape: tuple[int, int],
    resolution: float,
    cfg: FractalConfig,
    rng: np.random.Generator,
    workers: int | None = None,
    highpass_k: float | None = None,
) -> FloatArray:
    """Zero-mean fractal elevation field [m] of the given grid ``shape = (ny, nx)``.

    With ``highpass_k`` [cycles/m] (DEM mode) the amplitude is fixed on the full
    spectrum, then wavelengths longer than ``1 / highpass_k`` are removed with the
    complement of a cos² taper (0 below ``k/2``, 1 above ``k``).

    Args:
        shape: Output grid shape ``(ny, nx)``.
        resolution: Cell spacing [m].
        cfg: Fractal parameters.
        rng: Dedicated generator (e.g. stream ``"terrain/fractal"``).
        workers: Threads for the FFT (``None`` -> all cores). Each 1-D transform is
            independent, so the result does not depend on the thread count.
    """
    ny, nx = shape
    if not cfg.enabled or cfg.rms_slope_deg == 0.0:
        return np.zeros(shape, dtype=np.float64)
    nyp = sfft.next_fast_len(int(np.ceil(ny * cfg.pad_factor)), real=True)
    nxp = sfft.next_fast_len(int(np.ceil(nx * cfg.pad_factor)), real=True)
    w = -1 if workers is None else workers

    noise = rng.standard_normal((nyp, nxp))
    spec = sfft.rfft2(noise, workers=w)
    del noise
    ky = sfft.fftfreq(nyp, d=resolution)  # [cycles/m]
    kx = sfft.rfftfreq(nxp, d=resolution)
    k2 = ky[:, None] ** 2 + kx[None, :] ** 2
    if cfg.rolloff_wavelength_m is not None:
        k2 += (1.0 / cfg.rolloff_wavelength_m) ** 2
    beta = spectral_exponent(cfg.hurst)
    with np.errstate(divide="ignore"):
        amp = k2 ** (-beta / 4.0)  # sqrt(P(k)) = k^(-beta/2)
    amp[0, 0] = 0.0  # zero mean; the DC term carries no slope information
    spec *= amp
    del amp
    field = sfft.irfft2(spec, s=(nyp, nxp), workers=w)

    z = np.ascontiguousarray(field[:ny, :nx], dtype=np.float64)
    z -= z.mean()
    lag = max(1, round(cfg.baseline_m / resolution))
    s = rms_slope_at_baseline(z, resolution, lag)
    target = float(np.tan(np.deg2rad(cfg.rms_slope_deg)))
    if highpass_k is not None:
        x = np.clip((np.sqrt(k2) - 0.5 * highpass_k) / (0.5 * highpass_k), 0.0, 1.0)
        spec *= 1.0 - np.cos(0.5 * np.pi * x) ** 2
        field = sfft.irfft2(spec, s=(nyp, nxp), workers=w)
        z = np.ascontiguousarray(field[:ny, :nx], dtype=np.float64)
        z -= z.mean()
    del spec, k2
    z *= target / s
    return z


# ---------------------------------------------------------------------------
# Absolute (ensemble) normalisation, used by the two-layer terrain.
#
# For white noise w (unit variance) on an n_y x n_x grid of spacing r, the field
# z = irfft2(rfft2(w) * A(k)) with A(k) = sqrt(P(k)) / r has variance
# sum_k P(k) Δk_x Δk_y, i.e. P is a two-sided PSD [m^2 / (cycles/m)^2]. The
# expected squared increment at lag L along x is
#     <(z(x + L) - z(x))^2> = sum_k P(k) Δk_x Δk_y · 2 (1 - cos(2π k_x L)).


def fractal_psd(k2: FloatArray, cfg: FractalConfig) -> FloatArray:
    """Unnormalised PSD ``(k² + k0²)^(-beta/2)`` (``k0`` from the roll-off), 0 at k = 0."""
    kk = k2 + ((1.0 / cfg.rolloff_wavelength_m) ** 2 if cfg.rolloff_wavelength_m else 0.0)
    with np.errstate(divide="ignore"):
        p = kk ** (-spectral_exponent(cfg.hurst) / 2.0)
    p[k2 == 0.0] = 0.0
    return np.asarray(p, dtype=np.float64)


def _freq_grid(ny: int, nx: int, res: float) -> tuple[FloatArray, FloatArray]:
    ky = sfft.fftfreq(ny, d=res)
    kx = sfft.rfftfreq(nx, d=res)
    return ky, kx


def increment_variance(
    shape: tuple[int, int],
    res: float,
    psd: FloatArray,
    lag_m: float,
) -> float:
    """Expected squared increment [m²] at ``lag_m`` along x of a field synthesised on
    the rfft grid of ``shape`` with PSD values ``psd`` (rfft layout)."""
    ny, nx = shape
    ky, kx = _freq_grid(ny, nx, res)
    dk2 = 1.0 / (ny * res * nx * res)
    # rfft stores half the plane: weight 2 for columns that have a mirrored partner.
    wcol = np.full(kx.size, 2.0)
    wcol[0] = 1.0
    if nx % 2 == 0:
        wcol[-1] = 1.0
    term = psd * dk2 * 2.0 * (1.0 - np.cos(2.0 * np.pi * kx[None, :] * lag_m))
    del ky
    return float(np.sum(term * wcol[None, :]))


def spectral_amplitude(psd: FloatArray, res: float) -> FloatArray:
    """Fourier filter amplitude ``sqrt(P) / r`` for the rfft synthesis."""
    return np.asarray(np.sqrt(psd) / res, dtype=np.float64)
