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
