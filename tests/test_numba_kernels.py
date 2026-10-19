"""numba fast paths agree with the numpy reference implementations."""

from __future__ import annotations

import numpy as np
import pytest

from lunarsim.config import parse_config
from lunarsim.config.schema import ReflectanceConfig
from lunarsim.core.raycast import HAS_NUMBA

pytestmark = pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")

TERRAIN = {
    "size_x_m": 60, "size_y_m": 60, "resolution_m": 0.1,
    "craters": {"enabled": True, "sfd": {"d_min_m": 0.4, "d_max_m": 20.0}},
    "rocks": {"enabled": True, "k": 0.1, "d_min_m": 0.3, "d_max_m": 2.0,
              "ejecta": {"enabled": True, "min_crater_diameter_m": 5.0}},
}  # fmt: skip


def test_crater_and_rock_stamping(monkeypatch: pytest.MonkeyPatch) -> None:
    import lunarsim.core.terrain.craters as craters
    import lunarsim.core.terrain.rocks as rocks
    from lunarsim.core.rng import RngFactory
    from lunarsim.core.terrain import build_terrain

    cfg = parse_config({"seed": 3, "terrain": TERRAIN}).terrain
    fast = build_terrain(cfg, RngFactory(3))
    monkeypatch.setattr(craters, "USE_NUMBA", False)
    monkeypatch.setattr(rocks, "USE_NUMBA", False)
    ref = build_terrain(cfg, RngFactory(3))
    np.testing.assert_allclose(fast.heightmap.z, ref.heightmap.z, rtol=0, atol=1e-10)
    np.testing.assert_array_equal(fast.labels, ref.labels)


def test_crater_planes_two_layer(monkeypatch: pytest.MonkeyPatch) -> None:
    import lunarsim.core.terrain.craters as craters
    from lunarsim.core.world import World

    t = {**TERRAIN, "size_x_m": 200, "size_y_m": 200, "resolution_m": 1.0,
         "near": {"enabled": True, "size_m": 30, "resolution_m": 0.1, "tile_cells": 128,
                  "overlap_cells": 16}}  # fmt: skip
    t["craters"] = {"enabled": True, "sfd": {"d_min_m": 0.4, "d_max_m": 40.0}}
    cfg = parse_config({"seed": 8, "terrain": t, "cameras": [], "lidars": []})
    fast = World(cfg)
    monkeypatch.setattr(craters, "USE_NUMBA", False)
    ref = World(cfg)
    for a, b in ((fast.terrain, ref.terrain), (fast.terrain_far, ref.terrain_far)):
        assert a is not None and b is not None
        np.testing.assert_allclose(a.heightmap.z, b.heightmap.z, rtol=0, atol=1e-9)
        np.testing.assert_array_equal(a.labels, b.labels)


def test_hapke_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    import lunarsim.core.illumination.brdf as brdf

    rng = np.random.default_rng(0)

    def unit(n: int) -> np.ndarray:
        v = rng.normal(size=(n, 3))
        v[:, 2] = np.abs(v[:, 2]) - 0.2  # include grazing / back-facing cases
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    n, lt, v = unit(50_000), unit(50_000), unit(50_000)
    for hp in ({}, {"theta_bar_deg": 0.0}, {"bc0": 0.5, "filling_factor": 0.4}):
        cfg = ReflectanceConfig.model_validate({"model": "hapke", "hapke": hp})
        fast = brdf.reflectance(cfg, n, lt, v)
        monkeypatch.setattr(brdf, "USE_NUMBA", False)
        ref = brdf.reflectance(cfg, n, lt, v)
        monkeypatch.setattr(brdf, "USE_NUMBA", True)
        np.testing.assert_allclose(fast, ref, rtol=1e-11, atol=1e-15)
        assert (ref > 0).sum() > 1000


def test_render_and_lidar_fast_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    import lunarsim.core.illumination.brdf as brdf
    import lunarsim.core.render as render
    from lunarsim.core.raycast.layered import LayeredCaster
    from lunarsim.core.world import World

    t = {**TERRAIN, "size_x_m": 200, "size_y_m": 200, "resolution_m": 1.0,
         "near": {"enabled": True, "size_m": 30, "resolution_m": 0.1, "tile_cells": 128,
                  "overlap_cells": 16}}  # fmt: skip
    t["craters"] = {"enabled": True, "sfd": {"d_min_m": 0.4, "d_max_m": 40.0}}
    cfg = parse_config({
        "seed": 2, "terrain": t, "sun": {"azimuth_deg": 70, "elevation_deg": 4},
        "cameras": [{"name": "c", "intrinsics": {"width": 96, "height": 64}}],
        "lidars": [{"name": "l", "pattern": {"kind": "flash", "flash_width": 40,
                                             "flash_height": 30}}],
    })  # fmt: skip
    w = World(cfg)
    fast_cam, fast_lid = w.render_camera("c", frame_index=3), w.render_lidar("l", frame_index=3)
    monkeypatch.setattr(brdf, "USE_NUMBA", False)
    monkeypatch.setattr(render, "USE_NUMBA", False)
    monkeypatch.setattr(LayeredCaster, "fused", False)
    ref_cam, ref_lid = w.render_camera("c", frame_index=3), w.render_lidar("l", frame_index=3)
    np.testing.assert_array_equal(fast_cam.buffers.depth, ref_cam.buffers.depth)
    np.testing.assert_array_equal(fast_cam.buffers.labels, ref_cam.buffers.labels)
    np.testing.assert_allclose(fast_cam.buffers.normals, ref_cam.buffers.normals, atol=1e-14)
    np.testing.assert_allclose(fast_cam.buffers.radiance, ref_cam.buffers.radiance, rtol=1e-10)
    np.testing.assert_array_equal(fast_lid.points, ref_lid.points)


def test_mipmap_and_height_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    import lunarsim.core.raycast.heightfield as hf
    import lunarsim.core.terrain.heightmap as hmod
    from lunarsim.core.terrain import Heightmap

    rng = np.random.default_rng(5)
    for shape in ((37, 53), (64, 64), (2, 9)):
        z = rng.normal(size=shape)
        fast = hf.MaxMipmap.build(z)
        monkeypatch.setattr(hf, "HAS_NUMBA", False)
        ref = hf.MaxMipmap.build(z)
        monkeypatch.setattr(hf, "HAS_NUMBA", True)
        np.testing.assert_array_equal(fast.data, ref.data)
        np.testing.assert_array_equal(fast.offset, ref.offset)
    hm = Heightmap.centered(rng.normal(size=(80, 90)), 0.25)
    x, y = rng.uniform(-14, 14, 500), rng.uniform(-12, 12, 500)
    fast_h = hm.height_at_mesh(x, y)
    monkeypatch.setattr(hmod, "height_mesh_numba", None)
    np.testing.assert_array_equal(fast_h, hm.height_at_mesh(x, y))
