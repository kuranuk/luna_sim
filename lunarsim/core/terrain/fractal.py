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

__all__ = ["rms_slope_at_baseline", "spectral_exponent", "synthesize_fractal"]

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
    dx = z[:, lag_cells:] - z[:, :-lag_cells]
    dy = z[lag_cells:, :] - z[:-lag_cells, :]
    ms = (float(np.sum(dx * dx)) + float(np.sum(dy * dy))) / (dx.size + dy.size)
    return float(np.sqrt(ms) / (lag_cells * resolution))


def synthesize_fractal(
    shape: tuple[int, int],
    resolution: float,
    cfg: FractalConfig,
    rng: np.random.Generator,
    workers: int | None = None,
) -> FloatArray:
    """Zero-mean fractal elevation field [m] of the given grid ``shape = (ny, nx)``.

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
    del amp, k2
    field = sfft.irfft2(spec, s=(nyp, nxp), workers=w)
    del spec

    z = np.ascontiguousarray(field[:ny, :nx], dtype=np.float64)
    z -= z.mean()
    lag = max(1, round(cfg.baseline_m / resolution))
    s = rms_slope_at_baseline(z, resolution, lag)
    target = float(np.tan(np.deg2rad(cfg.rms_slope_deg)))
    z *= target / s
    return z
