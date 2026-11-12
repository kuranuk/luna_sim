"""Gridding of LOLA shots into a local DEM (scripts/fetch_lola_dem.py).

No network: synthetic points.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

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


def test_shipped_dem_matches_its_metadata() -> None:
    meta = json.loads((ROOT / "web/assets/dem/lola_89.5S_0E.json").read_text())
    z = np.fromfile(ROOT / "web/assets/dem" / meta["data"], dtype="<f4")
    assert meta["format"] == "lunarsim-dem/1"
    assert z.size == meta["n"] ** 2 and np.isfinite(z).all()
    z = z.reshape(meta["n"], meta["n"])
    c = meta["n"] // 2
    assert abs(z[c, c]) < 1.5  # heights are relative to the site
    assert meta["x0"] == -0.5 * (meta["n"] - 1) * meta["res"]
    assert "LOLA" in meta["source"] and meta["stats"]["shots"] > 1000
