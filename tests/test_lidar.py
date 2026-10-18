from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError
from scipy import stats

from lunarsim.config import parse_config
from lunarsim.config.schema import LidarConfig, LidarPatternConfig, LidarSpadConfig
from lunarsim.core.frames import Pose, matrix_from_rotvec
from lunarsim.core.rng import RngFactory
from lunarsim.core.sensors.lidar import SensorMotion, detection_probability, scan_pattern
from lunarsim.core.sensors.spad import first_photon_moments
from lunarsim.core.terrain import Label
from lunarsim.core.world import World
from lunarsim.io.pointcloud import read_pcd, read_ply, write_pcd, write_ply

LOOK_DOWN = Pose.from_euler_zyx(0.0, math.pi / 2, 0.0, (0.0, 0.0, 10.0))


def _world(lidar: dict, sun_el: float = -10.0, size: float = 300.0, z: float = 0.0) -> World:
    return World(
        parse_config(
            {
                "seed": 3,
                "terrain": {
                    "size_x_m": size,
                    "size_y_m": size,
                    "resolution_m": 0.5,
                    "base_elevation_m": z,
                    "fractal": {"enabled": False},
                },
                "sun": {"azimuth_deg": 0.0, "elevation_deg": sun_el},
                "cameras": [],
                "lidars": [{"name": "l", **lidar}],
            }
        )
    )


# ----------------------------------------------------------------------------- patterns


def test_spinning_pattern() -> None:
    p = scan_pattern(LidarPatternConfig(kind="spinning", channels=32))
    assert p.shape == (32, 1800) and p.directions.shape == (32 * 1800, 3)
    np.testing.assert_allclose(np.linalg.norm(p.directions, axis=1), 1.0)
    el = np.degrees(np.arcsin(p.directions[:, 2]))
    assert el.min() == pytest.approx(-25.0) and el.max() == pytest.approx(15.0)
    assert p.ring[0] == 31 and p.ring[-1] == 0  # top row = highest beam
    row0 = p.time[:1800]
    assert np.all(np.diff(row0) > 0) and row0[-1] < 0.1


def test_rosette_pattern() -> None:
    cfg = LidarPatternConfig(kind="rosette", rosette_fov_deg=70.0, rosette_points=20000)
    p = scan_pattern(cfg)
    off = np.degrees(np.arccos(np.clip(p.directions[:, 0], -1, 1)))
    assert off.max() <= 35.0 + 1e-9
    # Non-repetitive: two frames' directions do not coincide.
    p2 = scan_pattern(cfg.model_copy(update={"rosette_period_s": 0.2, "rosette_points": 40000}))
    assert not np.allclose(p2.directions[20000:], p.directions, atol=1e-6)


def test_flash_pattern_solid_angle() -> None:
    cfg = LidarPatternConfig(kind="flash", flash_width=161, flash_height=121)
    p = scan_pattern(cfg)
    centre = p.directions[(121 // 2) * 161 + 161 // 2]
    np.testing.assert_allclose(centre, [1, 0, 0], atol=1e-12)
    assert p.solid_angle is not None
    a, b = math.radians(30.0), math.radians(22.5)
    # Rectangular pyramid with half-angles a, b: 4 atan(tan a tan b / sqrt(1 + tan²a + tan²b)).
    ta, tb = math.tan(a), math.tan(b)
    exact = 4 * math.atan(ta * tb / math.sqrt(1 + ta * ta + tb * tb))
    assert float(p.solid_angle.sum()) == pytest.approx(exact, rel=1e-4)


def test_spad_requires_flash() -> None:
    with pytest.raises(ValidationError):
        LidarConfig(pattern={"kind": "spinning"}, noise={"model": "spad"})


# ----------------------------------------------------------------------------- gaussian


GAUSS = {
    "pattern": {"kind": "flash", "flash_width": 120, "flash_height": 90},
    "noise": {"model": "gaussian", "sigma0_m": 0.03, "sigma_per_m": 0.002, "range_max_m": 200.0},
}


def test_gaussian_range_noise_matches_config() -> None:
    w = _world(GAUSS)
    s = w.render_lidar("l", LOOK_DOWN)
    r_true = np.linalg.norm(s.points_world_gt - LOOK_DOWN.t, axis=1)
    z = (s.range_m - r_true) / np.sqrt(0.03**2 + (0.002 * r_true) ** 2)
    assert len(s) > 9000
    assert abs(z.mean()) < 0.05 and z.std() == pytest.approx(1.0, abs=0.05)
    # Points in the sensor frame lie along the beam directions at the measured range.
    np.testing.assert_allclose(np.linalg.norm(s.points[:, :3], axis=1), s.range_m, rtol=1e-6)


def test_gaussian_detection_probability() -> None:
    cfg = {**GAUSS, "noise": {**GAUSS["noise"], "range_max_m": 12.0, "reference_reflectance": 0.6}}
    w = _world(cfg)
    s = w.render_lidar("l", LOOK_DOWN)
    pat = scan_pattern(w.lidars["l"].pattern)
    ncfg = w.lidars["l"].noise
    cos_i = np.abs(LOOK_DOWN.rotate(pat.directions)[:, 2])
    r = 10.0 / cos_i
    ok = r <= 12.0
    p = detection_probability(ncfg.surface_reflectance * cos_i / r**2, ncfg)
    expected = float(np.sum(p[ok]))
    assert len(s) == pytest.approx(expected, abs=4 * math.sqrt(expected))


# ----------------------------------------------------------------------------- SPAD


SPAD = {
    "pattern": {"kind": "flash", "flash_width": 160, "flash_height": 120},
    "noise": {"model": "spad", "range_max_m": 100.0},
}


def test_first_photon_moments() -> None:
    sig = 4e-10
    m0, s0 = first_photon_moments(np.array([1e-9]), sig)
    assert abs(m0[0]) < 1e-15 and s0[0] == pytest.approx(sig, rel=1e-3)
    n = np.array([0.5, 2.0, 10.0])
    m, s = first_photon_moments(n, sig)
    assert np.all(np.diff(m) < 0) and np.all(np.diff(s) < 0) and m[0] < 0
    # Monte Carlo: first arrival of a Poisson process with Gaussian intensity, given >= 1.
    rng = np.random.default_rng(0)
    for k, nn in enumerate(n):
        cnt = rng.poisson(nn, 200000)
        cnt = cnt[cnt > 0]
        first = np.array([rng.normal(0, sig, c).min() for c in cnt[:20000]])
        assert first.mean() == pytest.approx(m[k], abs=4 * sig / math.sqrt(first.size))
        assert first.std() == pytest.approx(s[k], rel=0.03)


def test_spad_ranging_statistics_dark() -> None:
    w = _world(SPAD, sun_el=-10.0)
    s = w.render_lidar("l", LOOK_DOWN)
    sp = s.spad
    assert sp is not None
    good = ~s.false_alarm
    idx = s.beam_index[good]
    r_true = np.linalg.norm(s.points_world_gt[good] - LOOK_DOWN.t, axis=1)
    z = (s.range_m[good] - r_true - sp.bias_range_m[idx]) / sp.sigma_range_m[idx]
    assert good.sum() > 0.99 * 160 * 120
    assert abs(z.mean()) < 0.05 and z.std() == pytest.approx(1.0, abs=0.05)
    assert np.all(sp.bias_range_m[idx] < 0)  # pile-up makes ranges short


def test_spad_detection_probability_marginal() -> None:
    cfg = {**SPAD, "noise": {**SPAD["noise"], "spad": {"pulse_energy_j": 2e-8}}}
    w = _world(cfg, sun_el=-10.0, z=-40.0)  # ground 50 m below the sensor
    s = w.render_lidar("l", LOOK_DOWN)
    sp = s.spad
    assert sp is not None
    # Poisson approximation of Binomial(N, p_s) + Poisson(mu_b) (p_s << 1 here).
    spc = w.lidars["l"].noise.spad
    n_s = sp.n_signal
    assert n_s.max() < 0.05
    p_s = 1 / (1 + sp.bg_rate_hz * spc.dead_time_s) * (1 - np.exp(-n_s))
    wt = math.ceil(2 * spc.pulse_fwhm_s / spc.bin_width_s) * spc.bin_width_s
    mu = spc.pulses_per_frame * (p_s + (1 - np.exp(-sp.bg_rate_hz * wt)))
    p_det = stats.poisson.sf(sp.threshold - 1, mu)
    expected = float(p_det.sum())
    assert 0.1 * p_det.size < expected < 0.9 * p_det.size  # genuinely marginal
    assert sp.detected.sum() == pytest.approx(expected, abs=5 * math.sqrt(expected))


def test_spad_false_alarm_rate() -> None:
    # Bright sunlit ground beyond the maximum range: every return is a false alarm.
    cfg = {**SPAD, "noise": {**SPAD["noise"], "range_max_m": 8.0}}
    w = _world(cfg, sun_el=60.0)
    rates = []
    for k in range(6):
        s = w.render_lidar("l", LOOK_DOWN, frame_index=k)
        assert s.false_alarm.all()
        assert (s.labels == Label.NOISE).all() and np.isnan(s.points_world_gt).all()
        rates.append(len(s) / (160 * 120))
    p_fa = w.lidars["l"].noise.spad.false_alarm_probability
    n = 6 * 160 * 120
    assert np.mean(rates) <= p_fa + 4 * math.sqrt(p_fa / n)
    assert np.mean(rates) > 0.2 * p_fa


def test_spad_sun_background_shortens_range() -> None:
    cfg = {**SPAD, "noise": {**SPAD["noise"], "spad": {"pulse_energy_j": 5e-8}}}
    pose = Pose.from_euler_zyx(0.0, math.radians(8.0), 0.0, (0.0, 0.0, 2.0))
    det = []
    for el in (-10.0, 10.0, 60.0):
        s = _world(cfg, sun_el=el).render_lidar("l", pose)
        det.append(int((~s.false_alarm).sum()))
    assert det[0] > det[1] > det[2]


def test_spad_reproducible() -> None:
    w = _world(SPAD, sun_el=20.0)
    a = w.render_lidar("l", LOOK_DOWN, frame_index=1)
    b = w.render_lidar("l", LOOK_DOWN, frame_index=1)
    c = w.render_lidar("l", LOOK_DOWN, frame_index=2)
    assert a.points.tobytes() == b.points.tobytes()
    assert a.points.tobytes() != c.points.tobytes()


# ----------------------------------------------------------------------------- motion / io


def test_motion_distortion_consistency() -> None:
    cfg = {
        "pattern": {
            "kind": "spinning",
            "channels": 16,
            "vertical_min_deg": -30,
            "vertical_max_deg": -5,
            "horizontal_resolution_deg": 1.0,
        },
        "noise": {"model": "gaussian", "sigma0_m": 0.0, "sigma_per_m": 0.0, "range_max_m": 200.0},
    }
    w = _world(cfg)
    pose = Pose.from_euler_zyx(0.0, 0.0, 0.0, (0.0, 0.0, 2.0))
    mo = SensorMotion(linear=np.array([5.0, 0.0, 0.0]), angular=np.array([0.0, 0.0, 1.0]))
    s = w.render_lidar("l", pose, motion=mo)
    t = s.time.astype(np.float64)
    R = np.stack([matrix_from_rotvec(mo.angular * tt) @ pose.R for tt in t])
    p_world = np.einsum("nij,nj->ni", R, s.points[:, :3].astype(np.float64))
    p_world += pose.t + t[:, None] * mo.linear
    np.testing.assert_allclose(p_world, s.points_world_gt, atol=1e-4)
    # Interpreting the raw points in the start pose is wrong by up to v * T.
    naive = pose.apply(s.points[:, :3].astype(np.float64))
    err = np.linalg.norm(naive - s.points_world_gt, axis=1)
    assert err.max() > 0.3


def test_pointcloud_io(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    fields = {
        "x": rng.normal(size=50).astype(np.float32),
        "y": rng.normal(size=50).astype(np.float32),
        "z": rng.normal(size=50).astype(np.float32),
        "intensity": rng.random(50).astype(np.float32),
        "ring": rng.integers(0, 32, 50).astype(np.uint16),
        "time": rng.random(50).astype(np.float32),
    }
    write_ply(tmp_path / "a.ply", fields)
    write_pcd(tmp_path / "a.pcd", fields)
    for arr in (read_ply(tmp_path / "a.ply"), read_pcd(tmp_path / "a.pcd")):
        for k, v in fields.items():
            np.testing.assert_array_equal(arr[k], v)
    assert RngFactory(0) is not None and LidarSpadConfig().detection_mode == "multi_event"
