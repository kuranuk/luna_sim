"""Gridding of LOLA shots into a local DEM (scripts/fetch_lola_dem.py).

No network: synthetic points.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load():
    path = ROOT / "scripts" / "fetch_lola_dem.py"
    spec = importlib.util.spec_from_file_location("fetch_lola_dem", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_enu_basis_is_orthonormal_and_up_is_radial() -> None:
    mod = _load()
    site, rot = mod.enu_basis(-89.5, 0.0)
    assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-12)
    assert np.allclose(rot[2], site / np.linalg.norm(site), atol=1e-12)
    assert rot[1][2] > 0  # north has a +z (Moon north pole) component


def test_grid_dem_recovers_a_plane_from_scattered_shots() -> None:
    mod = _load()
    rng = np.random.default_rng(3)
    xy = rng.uniform(-400.0, 400.0, size=(20000, 2))
    grade = np.tan(np.radians(8.0))
    z = -grade * xy[:, 0] + 5.0  # 8 deg down to the east, site 5 m above the reference
    ax, grid, stats = mod.grid_dem(np.column_stack([xy, z]), half_m=200.0, res_m=10.0)
    assert grid.shape == (41, 41) and ax[0] == -200.0 and ax[-1] == 200.0
    xx, _ = np.meshgrid(ax, ax)
    err = np.abs(grid - (-grade * xx + 5.0))
    assert err[4:-4, 4:-4].max() < 0.05  # the smoothing reflects at the border
    assert abs(stats["slope_at_site_deg"] - 8.0) < 0.1
    assert abs(stats["downhill_azimuth_at_site_deg"] - 90.0) < 1.0


def test_geotiff_mode_converts_polar_stereographic_pixels(tmp_path: Path) -> None:
    """A sphere-following raster (h = 100 m everywhere) becomes the curved cap in ENU."""
    tifffile = pytest.importorskip("tifffile")
    mod = _load()
    lat, lon = -89.453, 222.88
    cx, cy = mod.stereo_xy(lat, lon, -1)
    assert cx < 0 and cy < 0  # lon 222.88 E: third quadrant of the south polar map
    n, res = 400, 5.0
    x0, y0 = round(cx / res) * res - n * res / 2, round(cy / res) * res + n * res / 2
    path = tmp_path / "flat.tif"
    tags = [
        (33550, "d", 3, (res, res, 0.0), False),
        (33922, "d", 6, (0.0, 0.0, 0.0, x0, y0, 0.0), False),
        (34736, "d", 4, (-90.0, 0.0, mod.MOON_RADIUS_M, mod.MOON_RADIUS_M), False),
    ]
    tifffile.imwrite(path, np.full((n, n), 100.0, dtype=np.float32), extratags=tags)
    enu, scale = mod.geotiff_points(path, lat, lon, 300.0)
    assert scale == "5 m/pix"
    r = np.hypot(enu[:, 0], enu[:, 1])
    expect = 100.0 - r**2 / (2 * (mod.MOON_RADIUS_M + 100.0))  # sagitta of the sphere
    assert np.abs(enu[:, 2] - expect).max() < 1e-3
    assert r.min() < 5.0 and enu[:, 0].max() > 300.0 and enu[:, 1].min() < -300.0


def test_shipped_dem_matches_its_metadata() -> None:
    meta = json.loads((ROOT / "web/assets/dem/lola_site01.json").read_text())
    z = np.fromfile(ROOT / "web/assets/dem" / meta["data"], dtype="<f4")
    assert meta["format"] == "lunarsim-dem/1"
    assert meta["site"] == {"lat": -89.453, "lon": 222.88}
    assert z.size == meta["n"] ** 2 and np.isfinite(z).all()
    z = z.reshape(meta["n"], meta["n"])
    c = meta["n"] // 2
    assert z[c, c] == 0.0  # heights are relative to the site
    assert meta["x0"] == -0.5 * (meta["n"] - 1) * meta["res"]
    assert "LOLA" in meta["source"] and "Barker" in meta["source"]
    # the ridge top is about 1.96 km above the reference sphere (Barker et al. 2021, Site01)
    assert 1900.0 < meta["height_of_site_above_reference_sphere_m"] < 2000.0
