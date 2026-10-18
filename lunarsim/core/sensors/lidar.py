"""LiDAR simulation: scan patterns, ray casting and measurement models.

Sensor frame: FLU (x forward, y left, z up). Points are reported in the sensor
frame *at their own measurement time*, like a raw driver output, so platform motion
during a sweep produces the usual motion distortion (see :class:`SensorMotion`).

Measurement models (``noise.model``):

* ``"spad"`` (default) – photon-counting SPAD flash LiDAR, see :mod:`.spad`:
  solar background from sunlit terrain, pile-up range walk, false alarms.
* ``"gaussian"`` – simple model: Gaussian range noise, logistic detection
  probability vs. received signal, reflectivity intensity.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from lunarsim.config.schema import LidarConfig, LidarNoiseConfig, LidarPatternConfig
from lunarsim.core.frames import Pose, matrix_from_rotvec
from lunarsim.core.illumination.sun import SunState
from lunarsim.core.raycast.heightfield import RayHits
from lunarsim.core.render import Caster, sample_surface
from lunarsim.core.sensors.spad import SpadResult, spad_measure, spad_rates
from lunarsim.core.terrain.generate import Terrain
from lunarsim.core.terrain.labels import Label

__all__ = [
    "LidarScan",
    "ScanPattern",
    "SensorMotion",
    "detection_probability",
    "range_sigma",
    "render_lidar",
    "scan_pattern",
]

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class ScanPattern:
    """Beam directions of one frame.

    Attributes:
        directions: Unit vectors in the sensor frame (FLU), ``(N, 3)``.
        ring: Channel index (0 = lowest beam for spinning, image row for flash), ``(N,)``.
        time: Firing time relative to the frame start [s], ``(N,)``.
        shape: ``(rows, cols)`` of the organised range image (row-major), or None.
        solid_angle: Pixel solid angle [sr] (flash only), ``(N,)`` or None.
    """

    directions: FloatArray
    ring: npt.NDArray[np.int64]
    time: FloatArray
    shape: tuple[int, int] | None
    solid_angle: FloatArray | None = None


def _unit_from_angles(azimuth: FloatArray, elevation: FloatArray) -> FloatArray:
    ce = np.cos(elevation)
    return np.stack([ce * np.cos(azimuth), ce * np.sin(azimuth), np.sin(elevation)], axis=-1)


def scan_pattern(cfg: LidarPatternConfig) -> ScanPattern:
    """Build the beam directions/timing of one frame (see :class:`LidarPatternConfig`)."""
    if cfg.kind == "spinning":
        if cfg.vertical_angles_deg is not None:
            elev = np.sort(np.radians(np.asarray(cfg.vertical_angles_deg, dtype=np.float64)))
        else:
            elev = np.radians(np.linspace(cfg.vertical_min_deg, cfg.vertical_max_deg, cfg.channels))
        n_ch = elev.size
        n_col = max(1, round(360.0 / cfg.horizontal_resolution_deg))
        col = np.arange(n_col)
        az = -math.pi + 2.0 * math.pi * (col + 0.5) / n_col
        # Range-image rows: top (highest elevation) first; ring 0 = lowest beam.
        ring_rows = np.arange(n_ch)[::-1]
        AZ, RING = np.meshgrid(az, ring_rows)
        COL = np.broadcast_to(col, AZ.shape)
        d = _unit_from_angles(AZ.ravel(), elev[RING.ravel()])
        t = (COL.ravel() + 0.5) / n_col / cfg.rotation_hz
        return ScanPattern(d, RING.ravel().astype(np.int64), t, (n_ch, n_col))
    if cfg.kind == "rosette":
        n = cfg.rosette_points
        t = (np.arange(n) + 0.5) / n * cfg.rosette_period_s
        a = math.radians(cfg.rosette_fov_deg) / 4.0
        w1, w2 = 2 * math.pi * cfg.rosette_f1_hz * t, 2 * math.pi * cfg.rosette_f2_hz * t
        dy = a * (np.cos(w1) + np.cos(w2))
        dz = a * (np.sin(w1) - np.sin(w2))
        th = np.hypot(dy, dz)
        psi = np.arctan2(dz, dy)
        d = np.stack([np.cos(th), np.sin(th) * np.cos(psi), np.sin(th) * np.sin(psi)], axis=1)
        return ScanPattern(d, np.zeros(n, dtype=np.int64), t, None)
    # flash: pixel centres on a regular grid in the tangent plane (pinhole-like)
    w, h = cfg.flash_width, cfg.flash_height
    tu = math.tan(math.radians(cfg.flash_hfov_deg) / 2)
    tv = math.tan(math.radians(cfg.flash_vfov_deg) / 2)
    du, dv = 2 * tu / w, 2 * tv / h
    u = -tu + du * (np.arange(w) + 0.5)
    v = -tv + dv * (np.arange(h) + 0.5)
    U, V = np.meshgrid(u, v)
    d = np.stack([np.ones(U.size), -U.ravel(), -V.ravel()], axis=1)
    norm = np.linalg.norm(d, axis=1)
    d /= norm[:, None]
    omega = du * dv / norm**3  # solid angle of a tangent-plane pixel
    ring = np.repeat(np.arange(h), w).astype(np.int64)
    return ScanPattern(d, ring, np.zeros(U.size), (h, w), omega)


@dataclass(frozen=True)
class SensorMotion:
    """Constant velocity of the sensor during a frame (world frame).

    Attributes:
        linear: Velocity of the sensor origin [m/s] in world ENU.
        angular: Angular velocity [rad/s] in world ENU (rotation vector rate).
    """

    linear: FloatArray
    angular: FloatArray


def range_sigma(r: FloatArray, n: LidarNoiseConfig) -> FloatArray:
    """Gaussian model: range noise standard deviation [m] at range ``r`` [m]."""
    return np.asarray(np.sqrt(n.sigma0_m**2 + (n.sigma_per_m * r) ** 2), dtype=np.float64)


def detection_probability(signal: FloatArray, n: LidarNoiseConfig) -> FloatArray:
    """Gaussian model: detection probability for signal ``rho cos(theta) / r²`` [1/m²]."""
    s50 = n.reference_reflectance / n.range_max_m**2
    with np.errstate(divide="ignore", over="ignore"):
        ratio = np.where(signal > 0.0, signal / s50, 0.0)
        p = np.where(ratio > 0.0, 1.0 / (1.0 + ratio ** (-n.detection_steepness)), 0.0)
    return np.asarray(p, dtype=np.float64)


def _subray_offsets(n: int, half_angle: float) -> FloatArray:
    """Deterministic sub-ray offsets (Fibonacci disc) [rad], ``(n, 2)``."""
    if n == 1 or half_angle == 0.0:
        return np.zeros((1, 2))
    k = np.arange(n)
    r = half_angle * np.sqrt((k + 0.5) / n)
    phi = k * math.pi * (3.0 - math.sqrt(5.0))
    return np.stack([r * np.cos(phi), r * np.sin(phi)], axis=1)


@dataclass(frozen=True)
class LidarScan:
    """One LiDAR frame (returned points only, unless noted).

    Attributes:
        points: ``(M, 4)`` float32: x, y, z [m] in the sensor frame at measurement time,
            intensity in [0, 1] (8-bit quantised; SPAD: signal photons per cycle).
        ring: Channel index ``(M,)`` uint16.
        time: Measurement time relative to the frame start [s] ``(M,)`` float32.
        range_m: Measured range [m] ``(M,)``.
        labels: Terrain label of the hit, :attr:`Label.NOISE` for false returns ``(M,)``.
        points_world_gt: Noise-free hit points in world ENU ``(M, 3)`` (nan if false).
        beam_index: Index of each return in the pattern ``(M,)``.
        false_alarm: True for false returns ``(M,)``.
        photon_counts: SPAD photons in the peak window ``(M,)`` (None for gaussian).
        range_image: Organised measured range [m] (nan = no return), or None.
        pose: ``T_world_sensor`` at the frame start.
        motion: Sensor motion used (None = static).
        spad: Full per-pixel SPAD diagnostics (None for gaussian).
    """

    points: npt.NDArray[np.float32]
    ring: npt.NDArray[np.uint16]
    time: npt.NDArray[np.float32]
    range_m: FloatArray
    labels: npt.NDArray[np.uint8]
    points_world_gt: FloatArray
    beam_index: npt.NDArray[np.int64]
    false_alarm: npt.NDArray[np.bool_]
    photon_counts: npt.NDArray[np.int64] | None
    range_image: FloatArray | None
    pose: Pose
    motion: SensorMotion | None
    spad: SpadResult | None = None

    def __len__(self) -> int:
        return int(self.points.shape[0])


@dataclass(frozen=True)
class _Geometry:
    """Ray-cast results per beam."""

    r_true: FloatArray
    any_hit: npt.NDArray[np.bool_]
    hit_fraction: FloatArray
    normals: FloatArray
    labels: npt.NDArray[np.uint8]
    points: FloatArray
    d_center: FloatArray
    p_t: FloatArray
    cos_inc: FloatArray


def _cast_beams(
    layers: Sequence[Terrain],
    caster: Caster,
    cfg: LidarConfig,
    pat: ScanPattern,
    T_world_sensor: Pose,
    motion: SensorMotion | None,
) -> _Geometry:
    nz = cfg.noise
    n_beams = pat.directions.shape[0]
    if motion is None:
        R_t = np.broadcast_to(T_world_sensor.R, (n_beams, 3, 3))
        p_t = np.broadcast_to(T_world_sensor.t, (n_beams, 3))
    else:
        R_t = np.stack(
            [matrix_from_rotvec(motion.angular * tt) @ T_world_sensor.R for tt in pat.time]
        )
        p_t = T_world_sensor.t + pat.time[:, None] * motion.linear

    offs = _subray_offsets(nz.subrays, 0.5 * math.radians(nz.beam_divergence_deg))
    m = offs.shape[0]
    d_s = pat.directions
    helper = np.where(np.abs(d_s[:, 2:3]) < 0.9, [[0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0]])
    e1 = np.cross(d_s, helper)
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    e2 = np.cross(d_s, e1)
    sub = (
        d_s[:, None, :] + offs[None, :, 0:1] * e1[:, None, :] + offs[None, :, 1:2] * e2[:, None, :]
    )
    sub /= np.linalg.norm(sub, axis=2, keepdims=True)
    d_w = np.einsum("nij,nmj->nmi", R_t, sub).reshape(-1, 3)
    o_w = np.repeat(p_t, m, axis=0)
    hits = caster.intersect(o_w, d_w, 0.0, nz.range_max_m * 1.5)
    t_sub = np.where(hits.hit, hits.t, np.nan).reshape(n_beams, m)
    n_hit = np.sum(np.isfinite(t_sub), axis=1)
    any_hit = n_hit > 0
    safe = np.where(any_hit[:, None], t_sub, 0.0)
    with np.errstate(invalid="ignore"):
        if nz.subray_combine == "mean":
            r_true = np.where(any_hit, np.nanmean(safe, axis=1), np.nan)
        else:
            r_true = np.where(
                any_hit, np.nanmin(np.where(np.isnan(safe), np.inf, safe), axis=1), np.nan
            )
    first = np.argmax(np.isfinite(t_sub), axis=1)
    rep = np.arange(n_beams) * m + first
    rep_hits = RayHits(hits.t[rep], hits.qi[rep], hits.qj[rep], hits.layer[rep])
    surf = sample_surface(layers, o_w[rep], d_w[rep], rep_hits)
    normals = np.zeros((n_beams, 3))
    labels = np.full(n_beams, Label.SKY, dtype=np.uint8)
    normals[any_hit] = surf.normals
    labels[any_hit] = surf.labels
    d_center = np.einsum("nij,nj->ni", R_t, d_s)
    cos_inc = np.abs(np.einsum("ni,ni->n", normals, d_center))
    points = p_t + d_center * np.nan_to_num(r_true)[:, None]
    return _Geometry(r_true, any_hit, n_hit / m, normals, labels, points, d_center, p_t, cos_inc)


def _reflectance(n: LidarNoiseConfig, labels: npt.NDArray[np.uint8]) -> FloatArray:
    rho = np.full(labels.shape, n.surface_reflectance)
    if n.rock_reflectance is not None:
        rho[labels == Label.ROCK] = n.rock_reflectance
    return rho


def render_lidar(
    layers: Sequence[Terrain],
    caster: Caster,
    cfg: LidarConfig,
    T_world_sensor: Pose,
    rng: np.random.Generator,
    motion: SensorMotion | None = None,
    sun: SunState | None = None,
    shadow_bias_m: float = 2e-3,
) -> LidarScan:
    """Simulate one LiDAR frame (``sun`` sets the SPAD solar background; None = dark)."""
    pat = scan_pattern(cfg.pattern)
    nz = cfg.noise
    geo = _cast_beams(layers, caster, cfg, pat, T_world_sensor, motion)
    rho = _reflectance(nz, geo.labels)
    n_beams = pat.directions.shape[0]

    spad: SpadResult | None = None
    if nz.model == "spad":
        assert pat.solid_angle is not None
        s = nz.spad
        bg = np.zeros(n_beams)
        if sun is not None and sun.elevation > 0.0:
            mu0 = geo.normals @ sun.direction
            idx = np.flatnonzero(geo.any_hit & (mu0 > 0.0))
            if idx.size:
                o = geo.points[idx] + shadow_bias_m * geo.normals[idx]
                lit = ~caster.occluded(o, np.broadcast_to(sun.direction, o.shape))
                f_band = s.solar_spectral_irradiance_w_m2_nm * s.filter_bandwidth_nm
                f_band /= sun.distance_au**2
                bg[idx[lit]] = rho[idx[lit]] * f_band * mu0[idx[lit]] / math.pi
        r_eff = np.where(geo.any_hit, geo.r_true, np.nan)
        n_s, phi = spad_rates(
            r_eff, geo.cos_inc * geo.hit_fraction, rho, bg, pat.solid_angle,
            float(pat.solid_angle.sum()), s,
        )  # fmt: skip
        spad = spad_measure(np.nan_to_num(r_eff, nan=-1.0), geo.any_hit, n_s, phi, s, nz, rng)
        keep = spad.detected | spad.false_alarm
        r_meas = spad.range_m
        inten = np.clip(spad.signal_counts / s.pulses_per_frame, 0.0, 1.0)
        false_alarm = spad.false_alarm
        counts: npt.NDArray[np.int64] | None = spad.window_counts
    else:
        with np.errstate(invalid="ignore", divide="ignore"):
            signal = np.where(
                geo.any_hit,
                geo.hit_fraction * rho * geo.cos_inc / np.maximum(geo.r_true, 1e-6) ** 2,
                0.0,
            )
        in_range = geo.any_hit & (geo.r_true >= nz.range_min_m) & (geo.r_true <= nz.range_max_m)
        u_det = rng.random(n_beams)
        g_rng = rng.standard_normal(n_beams)
        g_int = rng.standard_normal(n_beams)
        keep = in_range & (u_det < detection_probability(signal, nz))
        r_meas = geo.r_true + g_rng * range_sigma(np.nan_to_num(geo.r_true), nz)
        keep &= (r_meas >= nz.range_min_m) & (r_meas <= nz.range_max_m)
        base = rho * geo.cos_inc if nz.intensity_mode == "reflectivity" else signal
        inten = np.clip(base + nz.intensity_sigma * g_int, 0.0, 1.0)
        false_alarm = np.zeros(n_beams, dtype=bool)
        counts = None

    inten = np.round(inten * 255.0) / 255.0
    idx = np.flatnonzero(keep)
    pts = pat.directions[idx] * r_meas[idx, None]
    gt = np.where(false_alarm[idx, None], np.nan, geo.points[idx])
    labels = np.where(false_alarm[idx], Label.NOISE, geo.labels[idx]).astype(np.uint8)
    range_image = None
    if pat.shape is not None:
        ri = np.full(n_beams, np.nan)
        ri[idx] = r_meas[idx]
        range_image = ri.reshape(pat.shape)
    return LidarScan(
        points=np.concatenate([pts, inten[idx, None]], axis=1).astype(np.float32),
        ring=pat.ring[idx].astype(np.uint16),
        time=pat.time[idx].astype(np.float32),
        range_m=r_meas[idx],
        labels=labels,
        points_world_gt=gt,
        beam_index=idx.astype(np.int64),
        false_alarm=false_alarm[idx],
        photon_counts=None if counts is None else counts[idx],
        range_image=range_image,
        pose=T_world_sensor,
        motion=motion,
        spad=spad,
    )
