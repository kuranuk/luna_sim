from __future__ import annotations

import numpy as np
import pytest

from lunarsim.core.illumination.shadows import terrain_shadow_mask
from lunarsim.core.illumination.sun import sun_manual
from lunarsim.core.raycast import HAS_NUMBA, HeightfieldCaster, MaxMipmap
from lunarsim.core.raycast.layered import LayeredCaster
from lunarsim.core.terrain import Heightmap, Label, Terrain
from lunarsim.core.terrain.craters import CraterCatalog
from lunarsim.core.terrain.rocks import RockCatalog

BACKENDS = ["numpy"] + (["numba"] if HAS_NUMBA else [])


def _rough(seed: int = 0, shape: tuple[int, int] = (121, 161), res: float = 0.25) -> Heightmap:
    rng = np.random.default_rng(seed)
    return Heightmap.centered(rng.normal(size=shape) * 0.3, res)


def _rays(n: int, seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    o = np.c_[rng.uniform(-18, 18, n), rng.uniform(-14, 14, n), np.full(n, 4.0)]
    d = np.c_[rng.normal(size=(n, 2)), -rng.uniform(0.02, 1.0, n)]
    return o, d / np.linalg.norm(d, axis=1, keepdims=True)


def test_maxmip_bounds() -> None:
    z = np.random.default_rng(0).normal(size=(37, 53))
    mip = MaxMipmap.build(z)
    assert mip.data[mip.offset[-1]] == z.max()
    lvl0 = mip.data[: mip.offset[1]].reshape(mip.ny[0], mip.nx[0])
    assert lvl0[3, 4] == z[3:5, 4:6].max()


@pytest.mark.parametrize("backend", BACKENDS)
def test_hits_on_surface_and_first_hit(backend: str) -> None:
    hm = _rough()
    c = HeightfieldCaster(hm, backend)
    o, d = _rays(3000)
    h = c.intersect(o, d)
    p = o + d * np.where(h.hit, h.t, 0.0)[:, None]
    np.testing.assert_allclose(p[h.hit, 2], hm.height_at_mesh(p[h.hit, 0], p[h.hit, 1]), atol=1e-9)
    # Brute force: march finely; the first sign change must not occur before the hit.
    ts = np.linspace(0.0, 60.0, 6001)
    for k in range(0, 3000, 97):
        q = o[k] + ts[:, None] * d[k]
        inside = hm.contains(q[:, 0], q[:, 1])
        below = inside & (q[:, 2] < hm.height_at_mesh(q[:, 0], q[:, 1]) - 1e-6)
        first = ts[np.argmax(below)] if below.any() else np.inf
        if h.hit[k]:
            assert h.t[k] <= first + 1e-9
            assert first - h.t[k] < 0.02 or not np.isfinite(first)
        else:
            assert not np.isfinite(first)


@pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
def test_numba_numpy_agree() -> None:
    hm = _rough(3)
    o, d = _rays(5000, 4)
    a = HeightfieldCaster(hm, "numba").intersect(o, d)
    b = HeightfieldCaster(hm, "numpy").intersect(o, d)
    np.testing.assert_array_equal(a.hit, b.hit)
    np.testing.assert_allclose(a.t[a.hit], b.t[b.hit], rtol=0, atol=1e-9)
    np.testing.assert_array_equal(a.qi, b.qi)


@pytest.mark.parametrize("backend", BACKENDS)
def test_t_range_and_vertical_rays(backend: str) -> None:
    hm = Heightmap.centered(np.zeros((41, 41)), 0.5)
    c = HeightfieldCaster(hm, backend)
    o = np.array([[0.1, 0.2, 10.0], [0.1, 0.2, 10.0], [0.0, 0.0, -1.0], [50.0, 0.0, 10.0]])
    d = np.array([[0, 0, -1.0], [0, 0, -1.0], [0, 0, 1.0], [0, 0, -1.0]])
    h = c.intersect(o, d, t_max=np.array([20.0, 5.0, 20.0, 20.0]))
    assert h.t[0] == pytest.approx(10.0)
    assert not h.hit[1] and not h.hit[3]  # too short / off-grid
    # Triangles are two-sided: a ray from below hits the surface (counts as occluded).
    assert h.t[2] == pytest.approx(1.0)


def _box_terrain(res: float = 0.05, height: float = 0.5) -> Terrain:
    hm = Heightmap.centered(np.zeros((161, 161)), res)  # +-4 m
    inside = (np.abs(hm.x)[None, :] <= 0.5) & (np.abs(hm.y)[:, None] <= 0.5)
    hm.z[inside] = height
    lab = np.zeros(hm.shape, dtype=np.uint8)
    return Terrain(hm, lab, CraterCatalog.empty(), RockCatalog.empty())


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("el", [15.0, 30.0, 50.0])
def test_box_shadow_length(backend: str, el: float) -> None:
    """Shadow of a 0.5 m box with the sun in the East: length = H / tan(el)."""
    t = _box_terrain()
    hm = t.heightmap
    caster = HeightfieldCaster(hm, backend)
    sun = sun_manual(90.0, el)
    sh = terrain_shadow_mask(t, caster, sun)
    row = sh[np.argmin(np.abs(hm.y)), :]
    west = (hm.x < -0.5 - 1e-9) & row
    tip = hm.x[west].min()
    expected = -0.5 - 0.5 / np.tan(np.radians(el))
    assert tip == pytest.approx(expected, abs=2 * hm.resolution)
    assert not row[hm.x > 0.5 + hm.resolution].any()  # sunward side lit


def test_layered_caster() -> None:
    far = Heightmap.centered(np.zeros((201, 201)), 1.0)  # +-100 m, z = 0
    near = Heightmap.centered(np.full((101, 101), 0.5), 0.1)  # +-5 m, z = 0.5
    c = LayeredCaster(HeightfieldCaster(near), HeightfieldCaster(far))
    o = np.array([[0.0, 0.0, 2.0], [50.0, 0.0, 2.0], [-20.0, 0.0, 2.0]])
    d = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0], [0.0, 0.0, -1.0]])
    h = c.intersect(o, d)
    np.testing.assert_allclose(h.t, [1.5, 2.0, 2.0])
    np.testing.assert_array_equal(h.layer, [0, 1, 1])
    # Ray starting inside the near box, leaving it and hitting the far layer.
    dd = np.array([[1.0, 0.0, -0.05]])
    dd /= np.linalg.norm(dd)
    h2 = c.intersect(np.array([[0.0, 0.0, 1.0]]), dd)
    assert h2.layer[0] == 1 and h2.t[0] == pytest.approx(np.hypot(20.0, 1.0), rel=1e-9)
    assert Label.SKY == 255
