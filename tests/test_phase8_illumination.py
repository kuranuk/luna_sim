"""Penumbra, earthshine, curvature horizon, horizon maps / PSR, SPICE ephemeris."""

from __future__ import annotations

import math
import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from lunarsim.config import parse_config
from lunarsim.core.illumination.horizon import (
    HorizonMap,
    disc_fraction_above,
    horizon_map,
    illumination_map,
)
from lunarsim.core.illumination.spice import angular_separation_deg, sun_spice
from lunarsim.core.illumination.sun import (
    EARTH_MEAN_RADIUS_M,
    earthshine_irradiance,
    sun_astronomical,
    sun_manual,
)
from lunarsim.core.illumination.visibility import (
    DiscSource,
    SkyLighting,
    disc_visibility,
    sun_source,
)
from lunarsim.core.raycast import HAS_NUMBA, HeightfieldCaster
from lunarsim.core.terrain import Heightmap
from lunarsim.core.world import World

BACKENDS = ["numpy"] + (["numba"] if HAS_NUMBA else [])


def _wall(res: float = 0.1, height: float = 10.0) -> Heightmap:
    """Plane z = 0 for y < 0 and a plateau z = height for y >= 0 (an east-west wall)."""
    x = np.arange(-10.0, 10.0 + 1e-9, res)
    y = np.arange(-80.0, 10.0 + 1e-9, res)
    z = np.where(y[:, None] >= 0.0, height, 0.0) * np.ones((1, x.size))
    return Heightmap(np.ascontiguousarray(z), res, float(x[0]), float(y[0]))


def test_disc_fraction_formula() -> None:
    assert disc_fraction_above(0.0) == pytest.approx(0.5)
    assert disc_fraction_above(1.0) == pytest.approx(1.0)
    assert disc_fraction_above(-1.0) == pytest.approx(0.0)
    # Monte-Carlo check of the segment area.
    rng = np.random.default_rng(0)
    p = rng.uniform(-1, 1, (200_000, 2))
    p = p[np.hypot(p[:, 0], p[:, 1]) < 1]
    for t in (-0.6, 0.3):
        assert disc_fraction_above(t) == pytest.approx(np.mean(p[:, 1] > -t), abs=0.005)


@pytest.mark.parametrize("backend", BACKENDS)
def test_penumbra_profile_matches_disc_geometry(backend: str) -> None:
    """Shadow of a straight wall: visible fraction = disc area above the wall edge."""
    H, el = 10.0, math.radians(10.0)
    hm = _wall(height=H)
    caster = HeightfieldCaster(hm, backend)
    sun = sun_manual(0.0, math.degrees(el))  # due north, behind the wall
    src = DiscSource(sun.direction, math.radians(0.2666), limb_u=0.0)
    d = np.linspace(53.0, 60.0, 57)  # distance south of the wall top edge
    pts = np.c_[np.zeros_like(d), -d, np.zeros_like(d)]
    nrm = np.tile([0.0, 0.0, 1.0], (d.size, 1))
    got = disc_visibility(caster, pts, nrm, src, 1e-3, samples=128)
    edge = H / np.hypot(d, H)  # sin of the wall-top elevation; top edge at y = 0
    theta = np.arcsin(edge)
    expected = disc_fraction_above((el - theta) / src.radius)
    assert np.max(np.abs(got - expected)) < 0.06
    assert got[0] == 0.0 and got[-1] == 1.0  # umbra near the wall, fully lit beyond
    hard = disc_visibility(caster, pts, nrm, src, 1e-3, samples=0)
    assert set(np.unique(hard)) <= {0.0, 1.0}
    # The hard shadow boundary sits where the disc centre crosses the edge.
    d_hard = H / math.tan(el)
    assert np.all(hard[d < d_hard - 0.15] == 0.0) and np.all(hard[d > d_hard + 0.15] == 1.0)
    assert ((got > 0) & (got < 1)).sum() >= 15  # ~2.9 m wide penumbra at 0.125 m spacing


def test_penumbra_sunset_on_flat_ground() -> None:
    hm = Heightmap.centered(np.zeros((41, 41)), 0.5)
    caster = HeightfieldCaster(hm, "numpy")
    pts = np.array([[0.0, 0.0, 0.0]])
    nrm = np.array([[0.0, 0.0, 1.0]])
    for el_deg, expect in ((0.5, 1.0), (0.0, 0.5), (-0.5, 0.0)):
        sky = SkyLighting(sun_manual(90.0, el_deg), penumbra_samples=64, limb_darkening_u=0.0)
        assert sky.sun_visibility(caster, pts, nrm, 1e-3)[0] == pytest.approx(expect, abs=0.03)


def test_limb_darkening_weights_and_radius() -> None:
    s = sun_manual(10.0, 20.0, distance_au=1.0)
    src = sun_source(s)
    assert math.degrees(src.radius) == pytest.approx(0.2665, abs=0.001)
    dirs, w, rim = src.samples(32)
    assert w.sum() == pytest.approx(1.0) and w[0] > w[-1]  # centre brighter than limb
    ang = np.degrees(np.arccos(np.clip(dirs @ s.direction, -1, 1)))
    assert ang.max() <= 0.2666 and np.degrees(np.arccos(rim @ s.direction)).min() > 0.266


def test_curvature_local_horizon() -> None:
    """Sun just below the origin's horizon still lights points far towards the Sun."""
    R = 1737.4e3
    hm = Heightmap.centered(np.zeros((11, 11)), 1.0)
    caster = HeightfieldCaster(hm, "numpy")
    sun = sun_manual(90.0, -0.01)  # due east
    x = np.array([0.0, 200.0, 400.0, 1000.0, -1000.0])
    pts = np.c_[x, np.zeros(5), -(x**2) / (2 * R)]
    up = np.c_[x / R, np.zeros(5), np.ones(5)]
    nrm = up / np.linalg.norm(up, axis=1, keepdims=True)
    flat = SkyLighting(sun).sun_visibility(caster, pts, nrm, 1e-3)
    curved = SkyLighting(sun, curvature_radius_m=R).sun_visibility(caster, pts, nrm, 1e-3)
    assert flat.tolist() == [0, 0, 0, 0, 0]
    # local elevation = -0.01° + x/R rad: positive beyond x = 0.01° * R ≈ 303 m
    assert curved.tolist() == [0, 0, 1, 1, 0]


def test_curvature_in_terrain() -> None:
    base = {
        "seed": 1,
        "terrain": {
            "size_x_m": 400,
            "size_y_m": 400,
            "resolution_m": 2.0,
            "fractal": {"enabled": False},
        },
        "cameras": [],
        "lidars": [],
    }
    flat = World(parse_config(base))
    base["terrain"]["curvature"] = True  # type: ignore[index]
    curved = World(parse_config(base))
    d = curved.heightmap.z - flat.heightmap.z
    X, Y = np.meshgrid(curved.heightmap.x, curved.heightmap.y)
    np.testing.assert_allclose(d, -(X**2 + Y**2) / (2 * 1737.4e3), atol=1e-12)
    assert d.min() == pytest.approx(-(2 * 200.0**2) / (2 * 1737.4e3))


def test_earthshine_phase_and_magnitude() -> None:
    d = 3.844e8
    full = sun_manual(
        0.0, -30.0, earth_azimuth_deg=180.0, earth_elevation_deg=30.0, earth_distance_m=d
    )
    e_full = earthshine_irradiance(full, 0.367)
    assert e_full == pytest.approx(1361.0 * 0.367 * (EARTH_MEAN_RADIUS_M / d) ** 2)
    assert 0.1 < e_full < 0.2  # ≈ 1e-4 of sunlight
    quarter = sun_manual(
        90.0, 0.0, earth_azimuth_deg=0.0, earth_elevation_deg=0.0, earth_distance_m=d
    )
    assert earthshine_irradiance(quarter) == pytest.approx(e_full / math.pi)
    new = sun_manual(0.0, 30.0, earth_azimuth_deg=0.0, earth_elevation_deg=30.0)
    assert earthshine_irradiance(new) == pytest.approx(0.0, abs=1e-12)
    assert earthshine_irradiance(sun_manual(0.0, 30.0)) == 0.0


def test_earthshine_lights_shadowed_scene() -> None:
    cfg = {
        "seed": 4,
        "terrain": {"size_x_m": 30, "size_y_m": 30, "resolution_m": 0.1},
        "sun": {
            "azimuth_deg": 0.0,
            "elevation_deg": -5.0,
            "earthshine": True,
            "earth_azimuth_deg": 180.0,
            "earth_elevation_deg": 40.0,
        },
        "cameras": [
            {
                "name": "c",
                "intrinsics": {"width": 48, "height": 32},
                "sensor": {"auto_exposure": True},
            }
        ],
        "lidars": [],
    }
    w = World(parse_config(cfg))
    assert w.earthshine_w_m2 > 0.1
    f = w.render_camera("c")
    assert f.buffers.sun_fraction.max() == 0.0  # night: no direct sunlight
    assert f.buffers.radiance.max() > 0.0 and f.image.dn.max() > 0
    cfg["sun"]["earthshine"] = False  # type: ignore[index]
    dark = World(parse_config(cfg)).render_camera("c")
    assert dark.buffers.radiance.max() == 0.0
    st = w.state()["earth"]
    assert isinstance(st, dict) and st["elevation_deg"] == pytest.approx(40.0)


def test_penumbra_in_world_render() -> None:
    cfg = {
        "seed": 9,
        "terrain": {
            "size_x_m": 30,
            "size_y_m": 30,
            "resolution_m": 0.1,
            "rocks": {"enabled": True, "k": 0.1, "d_min_m": 0.4, "d_max_m": 2.0},
        },
        "sun": {"azimuth_deg": 80.0, "elevation_deg": 3.0},
        "render": {"penumbra_samples": 16},
        "cameras": [{"name": "c", "intrinsics": {"width": 96, "height": 64}}],
        "lidars": [],
    }
    f = World(parse_config(cfg)).render_camera("c").buffers.sun_fraction
    partial = (f > 0) & (f < 1)
    assert partial.any() and (f == 1).any() and (f == 0).any()


@pytest.mark.parametrize("backend", BACKENDS)
def test_horizon_map_wall(backend: str) -> None:
    hm = _wall(res=0.5, height=10.0)
    hor = horizon_map(hm, n_azimuth=8, stride=10, growth=0.002, backend=backend)
    j = int(np.argmin(np.abs(hor.x)))
    for i in (5, 10, 15):  # nodes south of the wall
        y = hor.y[i]
        assert y <= -5
        # Looking north (bin 0): the wall top edge (z = 10 at y = 0) - bilinear ramp
        # between y = -0.5 and 0 means the edge is seen at the first plateau node.
        assert hor.elevation[i, j, 0] == pytest.approx(math.atan2(10.0, -y), abs=2e-3)
        assert hor.elevation[i, j, 4] == pytest.approx(0.0, abs=1e-6)  # south: flat


def test_horizon_backends_agree_and_curvature() -> None:
    rng = np.random.default_rng(3)
    hm = Heightmap.centered(rng.normal(size=(60, 70)), 1.0)
    a = horizon_map(hm, 16, stride=3, backend="numpy")
    if HAS_NUMBA:
        b = horizon_map(hm, 16, stride=3, backend="numba")
        np.testing.assert_allclose(a.elevation, b.elevation, atol=1e-6)
    flat = Heightmap.centered(np.zeros((50, 50)), 10.0)
    c = horizon_map(flat, 8, curvature_radius_m=1737.4e3, backend="numpy")
    assert np.all(c.elevation <= 0.0)
    assert c.elevation.max() == pytest.approx(-10.0 / (2 * 1737.4e3), rel=1e-3)


def test_illumination_map_pit_is_psr() -> None:
    z = np.zeros((41, 41))
    z[15:26, 15:26] = -20.0  # deep square pit 10 m wide
    hm = Heightmap.centered(z, 1.0)
    hor = horizon_map(hm, 32, backend="numpy")
    suns = [sun_manual(az, 2.0) for az in range(0, 360, 10)]
    m = illumination_map(hor, suns)
    assert isinstance(hor, HorizonMap)
    assert m.psr[20, 20] and not m.psr[2, 2]
    assert m.mean_sun_fraction[2, 2] == pytest.approx(1.0)
    assert np.isnan(m.earth_visible_fraction).all()


def test_world_long_term_illumination() -> None:
    cfg = {
        "seed": 5,
        "site": {"latitude_deg": -89.6, "longitude_deg": 0.0},
        "terrain": {
            "size_x_m": 200,
            "size_y_m": 200,
            "resolution_m": 1.0,
            "craters": {"enabled": True, "sfd": {"d_min_m": 10, "d_max_m": 60}},
        },
        "cameras": [],
        "lidars": [],
    }
    w = World(parse_config(cfg))
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    m = w.long_term_illumination(t0, t0 + timedelta(days=30), timedelta(hours=12), stride=4)
    assert m.n_samples == 61
    assert 0.0 < m.mean_sun_fraction.mean() < 1.0
    assert np.nanmax(m.earth_visible_fraction) > 0.0


def test_analytic_sun_against_reference_values() -> None:
    # Regression of the analytic model (SPICE comparison: docs/references.md).
    s = sun_astronomical(-89.5, 0.0, datetime(2026, 12, 20, tzinfo=timezone.utc))
    assert -3.0 < s.elevation_deg < 3.0
    assert s.earth_direction is not None


def test_spice_mode_requires_kernels(tmp_path: object) -> None:
    with pytest.raises(ValueError):
        parse_config({"sun": {"mode": "astronomical", "ephemeris": "spice"}})
    w = World(
        parse_config(
            {
                "terrain": {"size_x_m": 20, "size_y_m": 20},
                "sun": {"mode": "astronomical", "spice_kernels": ["/nonexistent/de440s.bsp"]},
                "cameras": [],
                "lidars": [],
            }
        )
    )
    assert w.ephemeris == "analytic"  # auto falls back without kernels


SPICE_KERNELS = [k for k in os.environ.get("LUNARSIM_SPICE_KERNELS", "").split(os.pathsep) if k]


@pytest.mark.skipif(not SPICE_KERNELS, reason="set LUNARSIM_SPICE_KERNELS (LSK, SPK, PCK)")
def test_analytic_matches_spice() -> None:
    pytest.importorskip("spiceypy")
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    worst_sun = worst_earth = 0.0
    for h in range(0, 365 * 24, 97):
        t = t0 + timedelta(hours=h)
        a = sun_astronomical(-89.5, 20.0, t)
        b = sun_spice(-89.5, 20.0, t, SPICE_KERNELS)
        s, e = angular_separation_deg(a, b)
        worst_sun, worst_earth = max(worst_sun, s), max(worst_earth, e)
    assert worst_sun < 0.1 and worst_earth < 0.2
