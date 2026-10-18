"""Photon-counting model of a SPAD flash LiDAR pixel array.

Per pixel and frame (``N`` laser cycles):

1. Mean signal photons per cycle ``n_s`` from the lidar equation for an extended
   Lambertian target (see :class:`lunarsim.config.schema.LidarSpadConfig`).
2. Background count rate ``Φ`` (sunlit surface in the filter band + dark counts).
   Pile-up / dead time: in ``multi_event`` mode the pixel is armed with the
   steady-state probability ``1 / (1 + Φ τ_d)`` (non-paralysable dead time τ_d) and
   counts background at ``Φ / (1 + Φ τ_d)``; in ``first_photon`` mode a cycle can
   register a return at t only if no background photon came first, ``exp(-Φ t)``.
3. Peak search: the histogram is scanned with a window of ``w`` bins (~2 x pulse
   FWHM). The detection threshold ``T`` is the smallest count such that a window of
   pure background exceeds it with probability ≤ ``P_fa`` over the whole gate
   (Poisson statistics, worst case at the start of the gate).
4. A return is detected if the window counts at the return,
   ``k = Binomial(N, p_s) + Poisson(μ_b)``, reach ``T``; ``p_s = armed · (1 - e^{-n_s})``.
5. Range = centroid of the window photons. Signal photons follow the first-photon
   arrival density ``n g(t) exp(-n G(t)) / (1 - e^{-n})`` of a Gaussian pulse
   (σ² = σ_pulse² + σ_jitter²), whose mean shifts early as ``n`` grows (pile-up
   range walk, not corrected); TDC quantisation adds ``Δt²/12`` per photon;
   background photons spread uniformly over the window.
6. Pixels without a detected return produce a false alarm with the probability that
   background alone crosses ``T`` somewhere in the gate; its range is uniform.

Neglected (TODO): optical crosstalk, afterpulsing, multiple returns per pixel,
motion during the frame's pulse train, sun disc inside the field of view.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import stats

from lunarsim.config.schema import LidarNoiseConfig, LidarSpadConfig

__all__ = [
    "C_LIGHT",
    "SpadResult",
    "first_photon_moments",
    "spad_measure",
    "spad_rates",
    "spad_threshold",
]

FloatArray = npt.NDArray[np.float64]
C_LIGHT = 299_792_458.0
PLANCK = 6.62607015e-34


@dataclass(frozen=True)
class SpadResult:
    """Per-pixel SPAD outcome (arrays of length P).

    Attributes:
        detected: True return detected.
        false_alarm: Background-only detection (no or undetected true return).
        range_m: Measured range [m] (nan where neither).
        signal_counts: Signal photons in the peak window.
        window_counts: All photons in the peak window.
        n_signal: Expected signal photons per cycle.
        bg_rate_hz: Background count rate [1/s].
        threshold: Detection threshold [counts].
        sigma_range_m: Predicted range standard deviation [m] (nan if not detected).
        bias_range_m: Predicted pile-up range bias [m] (negative = short).
    """

    detected: npt.NDArray[np.bool_]
    false_alarm: npt.NDArray[np.bool_]
    range_m: FloatArray
    signal_counts: npt.NDArray[np.int64]
    window_counts: npt.NDArray[np.int64]
    n_signal: FloatArray
    bg_rate_hz: FloatArray
    threshold: npt.NDArray[np.int64]
    sigma_range_m: FloatArray
    bias_range_m: FloatArray


def spad_rates(
    r: FloatArray,
    cos_inc: FloatArray,
    rho: FloatArray,
    bg_radiance: FloatArray,
    omega_pix: FloatArray,
    omega_fov: float,
    s: LidarSpadConfig,
) -> tuple[FloatArray, FloatArray]:
    """``(n_signal per cycle, background rate [1/s])`` per pixel.

    ``bg_radiance`` is the in-band surface radiance [W m^-2 sr^-1]; ``r`` may be nan.
    """
    e_ph = PLANCK * C_LIGHT / s.wavelength_m
    a_rx = math.pi * (0.5 * s.rx_aperture_diameter_m) ** 2
    e_pix = s.pulse_energy_j * s.tx_transmission * omega_pix / omega_fov
    with np.errstate(invalid="ignore", divide="ignore"):
        e_rx = e_pix * rho * cos_inc * a_rx * s.rx_transmission / (math.pi * r * r)
    n_s = np.where(np.isfinite(e_rx), e_rx / e_ph * s.photon_detection_efficiency, 0.0)
    phi = bg_radiance * a_rx * omega_pix * s.rx_transmission / e_ph
    phi = phi * s.photon_detection_efficiency + s.dark_count_rate_hz
    return np.asarray(n_s, dtype=np.float64), np.asarray(phi, dtype=np.float64)


def _window_bins(s: LidarSpadConfig) -> int:
    return max(1, math.ceil(2.0 * s.pulse_fwhm_s / s.bin_width_s))


def _n_windows(s: LidarSpadConfig, n: LidarNoiseConfig) -> int:
    gate = 2.0 * n.range_max_m / C_LIGHT
    return max(1, math.ceil(gate / (s.bin_width_s * _window_bins(s))))


def _armed(phi: FloatArray, s: LidarSpadConfig) -> FloatArray:
    """Steady-state probability that a multi-event pixel is armed (non-paralysable)."""
    return np.asarray(1.0 / (1.0 + phi * s.dead_time_s))


def _bg_window_mean(phi: FloatArray, s: LidarSpadConfig) -> FloatArray:
    """Expected background counts in one search window at the start of the gate."""
    w_t = _window_bins(s) * s.bin_width_s
    rate = phi * _armed(phi, s) if s.detection_mode == "multi_event" else phi
    return np.asarray(s.pulses_per_frame * (1.0 - np.exp(-rate * w_t)))


def spad_threshold(phi: FloatArray, s: LidarSpadConfig, n: LidarNoiseConfig) -> FloatArray:
    """Detection threshold [counts] per pixel for false-alarm probability ``P_fa``."""
    mu_b0 = _bg_window_mean(phi, s)
    q = s.false_alarm_probability / _n_windows(s, n)
    k = stats.poisson.isf(q, mu_b0)  # smallest k with P(X > k) <= q
    return np.asarray(np.maximum(k + 1, 1), dtype=np.float64)


def first_photon_moments(n: FloatArray, sigma: float) -> tuple[FloatArray, FloatArray]:
    """Mean shift [s] and std [s] of the first-photon arrival time for a Gaussian pulse
    of std ``sigma`` and ``n`` mean photons per cycle (n -> 0: 0 and sigma)."""
    z = np.linspace(-7.0, 7.0, 281)
    g = stats.norm.pdf(z)
    G = stats.norm.cdf(z)
    dz = z[1] - z[0]

    def integrate(y: FloatArray) -> FloatArray:  # trapezoid rule on the uniform grid
        return np.asarray(dz * (y.sum(axis=1) - 0.5 * (y[:, 0] + y[:, -1])))

    nn = np.asarray(n, dtype=np.float64)[:, None]
    f = g[None, :] * np.exp(-nn * G[None, :])
    f /= integrate(f)[:, None]
    mean = integrate(f * z[None, :])
    var = integrate(f * (z[None, :] - mean[:, None]) ** 2)
    return mean * sigma, np.sqrt(var) * sigma


def spad_measure(
    r_true: FloatArray,
    has_target: npt.NDArray[np.bool_],
    n_s: FloatArray,
    phi: FloatArray,
    s: LidarSpadConfig,
    n: LidarNoiseConfig,
    rng: np.random.Generator,
) -> SpadResult:
    """Simulate detection and ranging for all pixels (fixed-size random draws)."""
    p = r_true.shape[0]
    N = s.pulses_per_frame
    w_t = _window_bins(s) * s.bin_width_s
    sigma_g = math.sqrt((s.pulse_fwhm_s / 2.3548200450309493) ** 2 + s.timing_jitter_s**2)
    T = spad_threshold(phi, s, n)

    in_gate = has_target & (r_true >= n.range_min_m) & (r_true <= n.range_max_m)
    t0 = np.where(in_gate, 2.0 * np.nan_to_num(r_true) / C_LIGHT, 0.0)
    if s.detection_mode == "multi_event":
        surv = _armed(phi, s)  # background is stationary: no time dependence
        mu_bw = _bg_window_mean(phi, s)
    else:
        surv = np.exp(-phi * t0)
        mu_bw = N * surv * (1.0 - np.exp(-phi * w_t))
    p_s = np.where(in_gate, surv * (1.0 - np.exp(-n_s)), 0.0)

    u_sig = rng.binomial(N, np.clip(p_s, 0.0, 1.0))
    u_bg = rng.poisson(mu_bw)
    g_t = rng.standard_normal(p)
    u_fa = rng.random(p)
    u_far = rng.random(p)

    k_tot = u_sig + u_bg
    detected = in_gate & (k_tot >= T)

    shift = np.zeros(p)
    sd_fp = np.full(p, sigma_g)
    sel = np.flatnonzero(in_gate)
    if sel.size:
        shift[sel], sd_fp[sel] = first_photon_moments(n_s[sel], sigma_g)
    var_sig = sd_fp**2 + s.bin_width_s**2 / 12.0
    with np.errstate(invalid="ignore", divide="ignore"):
        var_t = (u_sig * var_sig + u_bg * w_t**2 / 12.0) / np.maximum(k_tot, 1) ** 2
    t_hat = t0 + shift + np.sqrt(var_t) * g_t
    r_meas = np.where(detected, 0.5 * C_LIGHT * t_hat, np.nan)
    sigma_r = np.where(detected, 0.5 * C_LIGHT * np.sqrt(var_t), np.nan)

    mu_b0 = _bg_window_mean(phi, s)
    p_fa = np.minimum(1.0, _n_windows(s, n) * stats.poisson.sf(T - 1, mu_b0))
    false_alarm = ~detected & (u_fa < p_fa)
    r_fa = n.range_min_m + u_far * (n.range_max_m - n.range_min_m)
    r_meas = np.where(false_alarm, r_fa, r_meas)

    return SpadResult(
        detected=detected,
        false_alarm=false_alarm,
        range_m=r_meas,
        signal_counts=np.where(detected, u_sig, 0).astype(np.int64),
        window_counts=np.where(detected | false_alarm, np.where(detected, k_tot, T), 0).astype(
            np.int64
        ),
        n_signal=n_s,
        bg_rate_hz=phi,
        threshold=T.astype(np.int64),
        sigma_range_m=sigma_r,
        bias_range_m=0.5 * C_LIGHT * shift,
    )
