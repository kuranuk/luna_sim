"""Real-DEM base terrain and the IMU."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from lunarsim.config import parse_config
from lunarsim.config.schema import DemConfig, SiteConfig
from lunarsim.core.terrain.dem import DemSampler, load_dem
from lunarsim.core.world import World

R_MOON = 1737.4e3


def _f(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Smooth test topography [m] as a function of latitude/longitude [rad]."""
    return 300.0 * np.sin(40.0 * lat) + 200.0 * np.cos(3.0 * lon) + 50.0 * np.sin(25.0 * lon)


def _polar_geotiff(path: Path, lat0: float, lon0: float, px: float = 20.0, n: int = 400) -> None:
    pytest.importorskip("tifffile")
    from lunarsim.core.terrain.dem import write_geotiff

    rho0 = 2 * R_MOON * math.tan(math.pi / 4 + math.radians(lat0) / 2)
    cx, cy = rho0 * math.sin(math.radians(lon0)), rho0 * math.cos(math.radians(lon0))
    xs = cx + (np.arange(n) - n / 2 + 0.5) * px
    ys = cy - (np.arange(n) - n / 2 + 0.5) * px
    X, Y = np.meshgrid(xs, ys)
    phi = 2 * np.arctan(np.hypot(X, Y) / (2 * R_MOON)) - np.pi / 2
    Z = _f(phi, np.arctan2(X, Y)).astype(np.float32)
    Z[0, 0] = -32768.0  # a nodata pixel far from the site
    write_geotiff(path, Z, xs[0] - px / 2, ys[0] + px / 2, px, nodata=-32768.0)


def _expected(lat0: float, lon0: float, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    la, lo = math.radians(lat0), math.radians(lon0)
    E = np.array([-math.sin(lo), math.cos(lo), 0.0])
    N = np.array([-math.sin(la) * math.cos(lo), -math.sin(la) * math.sin(lo), math.cos(la)])
    U = np.array([math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)])
    P = R_MOON * U + x[:, None] * E + y[:, None] * N
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    return _f(np.arcsin(P[:, 2]), np.arctan2(P[:, 1], P[:, 0])) - _f(np.array(la), np.array(lo))


def test_polar_stereographic_geotiff(tmp_path: Path) -> None:
    lat0, lon0 = -89.4, 30.0
    _polar_geotiff(tmp_path / "ps.tif", lat0, lon0)
    cfg = DemConfig(enabled=True, path=str(tmp_path / "ps.tif"), projection="polar_stereographic")
    r = load_dem(cfg)
    assert r.nodata == -32768.0 and r.dy < 0
    s = DemSampler(cfg, SiteConfig(latitude_deg=lat0, longitude_deg=lon0))
    assert s.pixel_size_m == pytest.approx(20.0, rel=1e-3)
    x = np.array([0.0, 1500.0, 0.0, -2000.0, 2500.0])
    y = np.array([0.0, 0.0, 1500.0, -1200.0, 2500.0])
    np.testing.assert_allclose(s.heights(x, y), _expected(lat0, lon0, x, y), atol=0.01)
    with pytest.raises(ValueError, match="beyond"):
        s.heights(np.array([10_000.0]), np.array([0.0]))


def test_equirectangular_npy_and_local(tmp_path: Path) -> None:
    lon = np.arange(-180.0, 180.0, 0.5) + 0.25
    lat = np.arange(90.0, -90.0, -0.5) - 0.25  # north-up rows
    LON, LAT = np.meshgrid(np.radians(lon), np.radians(lat))
    np.save(tmp_path / "eq.npy", (_f(LAT, LON) / 1000.0).astype(np.float32))  # km
    cfg = DemConfig(
        enabled=True,
        path=str(tmp_path / "eq.npy"),
        projection="equirectangular",
        pixel_size=0.5,
        origin_x=float(lon[0]),
        origin_y=float(lat[0]),
        height_scale=1000.0,
    )
    s = DemSampler(cfg, SiteConfig(latitude_deg=20.0, longitude_deg=179.9))  # across ±180°
    x = np.array([0.0, 30e3, -30e3, 0.0])
    y = np.array([0.0, 0.0, 10e3, -40e3])
    np.testing.assert_allclose(s.heights(x, y), _expected(20.0, 179.9, x, y), atol=2.0)
    z = np.add.outer(np.arange(50.0), np.arange(60.0) * 2.0)  # z = row + 2 col
    np.save(tmp_path / "loc.npy", z)
    loc = DemConfig(
        enabled=True,
        path=str(tmp_path / "loc.npy"),
        pixel_size=1.0,
        origin_x=0.0,
        origin_y=0.0,
        rows_north_up=False,
        center_x_m=30.0,
        center_y_m=20.0,
    )
    sl = DemSampler(loc, SiteConfig())
    assert sl.heights(np.array([3.0]), np.array([-2.0]))[0] == pytest.approx(2 * 3.0 - 2.0)


def _dem_world(tmp_path: Path, near: bool, curvature: bool = False) -> World:
    lat0, lon0 = -89.4, 30.0
    _polar_geotiff(tmp_path / "ps.tif", lat0, lon0)
    terrain: dict[str, object] = {
        "size_x_m": 1000.0,
        "size_y_m": 1000.0,
        "resolution_m": 2.0,
        "fractal": {"rms_slope_deg": 4.0, "baseline_m": 2.0},
        "dem": {
            "enabled": True,
            "path": str(tmp_path / "ps.tif"),
            "projection": "polar_stereographic",
        },
        "curvature": curvature,
    }
    if near:
        terrain["near"] = {
            "enabled": True,
            "size_m": 40.0,
            "resolution_m": 0.25,
            "tile_cells": 128,
            "overlap_cells": 16,
        }
    cfg = {
        "seed": 3,
        "site": {"latitude_deg": lat0, "longitude_deg": lon0},
        "terrain": terrain,
        "cameras": [],
        "lidars": [],
    }
    return World(parse_config(cfg))


def test_world_with_dem_single_layer(tmp_path: Path) -> None:
    w = _dem_world(tmp_path, near=False)
    hm = w.heightmap
    X, Y = np.meshgrid(hm.x, hm.y)
    dem = _expected(-89.4, 30.0, X.ravel(), Y.ravel()).reshape(X.shape)
    resid = (hm.z - dem)[:500, :500]
    # The DEM carries the long wavelengths; the fractal adds only short-wavelength detail.
    assert np.std(dem) > 3.0 and np.std(resid) < 0.1 * np.std(dem)
    lowpass = resid.reshape(25, 20, 25, 20).mean(axis=(1, 3))  # 40 m blocks (> 2 DEM px)
    assert np.std(lowpass) < 0.3 * np.std(resid)


def test_world_with_dem_two_layer_consistent(tmp_path: Path) -> None:
    w = _dem_world(tmp_path, near=True, curvature=True)
    far, near = w.terrain_far, w.terrain
    assert far is not None
    # Near patch = far base (spline) + DEM/curvature + fine detail: at far nodes inside the
    # patch the large-scale difference is only the fine detail (small).
    xs = far.heightmap.x[(far.heightmap.x > -15) & (far.heightmap.x < 15)]
    ys = far.heightmap.y[(far.heightmap.y > -15) & (far.heightmap.y < 15)]
    X, Y = np.meshgrid(xs, ys)
    zf = far.heightmap.height_at_mesh(X, Y)
    zn = near.heightmap.height_at_mesh(X, Y)
    assert np.abs(zn - zf).mean() < 0.3
    z_corner = far.heightmap.z[0, 0]
    assert np.isfinite(z_corner)


def _imu_world(noise: bool = True, rate: float = 100.0, **imu: object) -> World:
    zero = {
        k: 0.0
        for k in (
            "gyro_noise_density",
            "gyro_random_walk",
            "gyro_turn_on_bias",
            "accel_noise_density",
            "accel_random_walk",
            "accel_turn_on_bias",
        )
    }
    cfg = {
        "terrain": {
            "size_x_m": 40,
            "size_y_m": 40,
            "resolution_m": 0.1,
            "fractal": {"enabled": False},
        },
        "site": {"latitude_deg": 30.0},
        "cameras": [],
        "lidars": [],
        "imus": [
            {
                "name": "imu",
                "rate_hz": rate,
                "bandwidth_hz": None,
                "mount": {"x_m": 0.3, "z_m": 0.2},
                **({} if noise else {"noise": zero}),
                **imu,
            }
        ],
    }
    return World(parse_config(cfg))


def test_imu_truth_driving_and_turning() -> None:
    w = _imu_world(noise=False)
    w.set_rover_command(0.2, 0.0)
    w.step(1.0)
    s = w.imu_samples("imu", since_s=0.05)
    omega = 2.6617e-6 * np.array([0.0, math.cos(math.radians(30)), math.sin(math.radians(30))])
    np.testing.assert_allclose(s.accel, np.tile([0.0, 0.0, 1.6243], (len(s), 1)), atol=1e-9)
    np.testing.assert_allclose(s.gyro, np.tile(omega, (len(s), 1)), atol=1e-12)
    w.set_rover_command(0.2, 0.1)
    w.step(2.0)
    s = w.imu_samples("imu", since_s=1.2)
    # Lever arm r = 0.3 m ahead of the turning centre: a = v ω (left) + ω² r (backwards)
    a_exp = np.array([-(0.1**2) * 0.3, 0.2 * 0.1 * 0.98, 1.6243])  # slip_flat = 0.02
    np.testing.assert_allclose(s.accel.mean(axis=0), a_exp, atol=2e-4)
    assert s.gyro[:, 2].mean() == pytest.approx(0.1 + omega[2], abs=1e-9)
    assert s.t[0] > 1.2 and np.allclose(np.diff(s.t), 0.01)


def test_imu_noise_statistics() -> None:
    w = _imu_world(noise=True, rate=200.0)
    w.step(60.0)
    s = w.imu_samples("imu")
    assert len(s) == 12001
    n = w.config.imus[0].noise
    white_g = (s.gyro - s.gyro_true - s.gyro_bias).std(axis=0)
    white_a = (s.accel - s.accel_true - s.accel_bias).std(axis=0)
    np.testing.assert_allclose(white_g, n.gyro_noise_density * math.sqrt(200.0), rtol=0.05)
    np.testing.assert_allclose(white_a, n.accel_noise_density * math.sqrt(200.0), rtol=0.05)
    # Bias random walk: increments have std rw * sqrt(dt).
    inc = np.diff(s.accel_bias, axis=0).std(axis=0)
    np.testing.assert_allclose(inc, n.accel_random_walk * math.sqrt(0.005), rtol=0.05)
    assert np.abs(s.gyro_bias[0]).max() > 0.0  # turn-on bias drawn


def test_imu_independent_of_step_partition_and_reset() -> None:
    a = _imu_world()
    a.set_rover_command(0.1, 0.05)
    a.step(3.0)
    b = _imu_world()
    b.set_rover_command(0.1, 0.05)
    for _ in range(30):
        b.step(0.1)
    sa, sb = a.imu_samples("imu"), b.imu_samples("imu")
    np.testing.assert_array_equal(sa.t, sb.t)
    np.testing.assert_allclose(sa.gyro, sb.gyro, atol=1e-12)
    np.testing.assert_allclose(sa.accel, sb.accel, atol=1e-9)
    a.reset_dynamic()
    a.set_rover_command(0.1, 0.05)
    a.step(3.0)
    np.testing.assert_array_equal(a.imu_samples("imu").gyro, sa.gyro)


def test_imu_low_pass_filter() -> None:
    raw = _imu_world(noise=False)
    filt = _imu_world(noise=False, bandwidth_hz=5.0)
    for w in (raw, filt):
        w.set_rover_command(0.2, 0.0)
        w.step(0.5)
        w.set_rover_command(0.0, 0.0)  # sudden stop: deceleration spike
        w.step(0.5)
    t = raw.imu_samples("imu").t
    a_raw = raw.imu_samples("imu").accel[:, 0]
    a_f = filt.imu_samples("imu").accel[:, 0]
    assert np.abs(a_f).max() < 0.5 * np.abs(a_raw).max()
    # Same velocity change after the start transient (unit DC gain): Δv = Σ a dt.
    before = t < 0.45
    assert a_f[before].sum() * 0.01 == pytest.approx(a_raw[before].sum() * 0.01, rel=1e-3)
    assert a_raw[before].sum() * 0.01 == pytest.approx(0.2 * 0.98, rel=1e-6)
