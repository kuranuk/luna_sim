"""Run the browser simulator's JavaScript unit tests (web/tests) with Node.js if available."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not installed")
def test_web_sim_node_tests() -> None:
    files = sorted(str(p) for p in (ROOT / "web" / "tests").glob("*.test.mjs"))
    assert files
    r = subprocess.run(["node", "--test", *files], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]


def test_web_vendor_matches_viewer() -> None:
    """web/vendor holds the same Three.js files (and SRI hashes) as the Python viewer."""
    a = ROOT / "web" / "vendor"
    b = ROOT / "lunarsim" / "viewer" / "static" / "vendor"
    for rel in ("three.module.js", "addons/controls/OrbitControls.js"):
        assert (a / rel).read_bytes() == (b / rel).read_bytes(), rel
    html = (ROOT / "web" / "index.html").read_text()
    viewer = (ROOT / "lunarsim" / "viewer" / "static" / "index.html").read_text()
    for h in ("sha384-61S/Nu32S3E5", "sha384-qlO/ZugKPxAQUAvT"):
        assert h in html and h in viewer


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not installed")
@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_web_lidar_raycast_matches_python(tmp_path: Path, backend: str) -> None:
    """The browser simulator's LiDAR rays hit the same terrain points as the Python caster."""
    import json

    import numpy as np

    from lunarsim.core.raycast.heightfield import HAS_NUMBA, HeightfieldCaster
    from lunarsim.core.terrain.heightmap import Heightmap

    if backend == "numba" and not HAS_NUMBA:
        pytest.skip("numba not installed")
    tool = ROOT / "web" / "tools" / "dump_rays.mjs"
    r = subprocess.run(
        ["node", str(tool), str(tmp_path)], capture_output=True, text=True, timeout=600
    )
    assert r.returncode == 0, r.stderr
    m = json.loads((tmp_path / "meta.json").read_text())
    z = np.fromfile(tmp_path / "z.f64").reshape(m["n"], m["n"])
    rays = np.fromfile(tmp_path / "rays.f64").reshape(-1, 7)
    hm = Heightmap(z, m["res"], m["x0"], m["y0"])
    t = (
        HeightfieldCaster(hm, backend=backend)
        .intersect(rays[:, :3], rays[:, 3:6], t_max=m["tMax"])
        .t
    )
    js = rays[:, 6]
    np.testing.assert_array_equal(np.isfinite(t), np.isfinite(js))
    hit = np.isfinite(js)
    assert hit.sum() > 0.3 * len(js)
    np.testing.assert_allclose(t[hit], js[hit], rtol=0, atol=1e-9)
