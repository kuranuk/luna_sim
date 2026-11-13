"""Run the browser simulator's JavaScript unit tests (web/tests) with Node.js if available."""

from __future__ import annotations

import json
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


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not installed")
def test_web_terrain_model_matches_python_and_references() -> None:
    """Browser terrain model: crater profiles / populations, rocks, fractal vs Python and theory."""
    import json
    import math

    import numpy as np

    from lunarsim.config.schema import CraterSFDConfig, CraterShapeConfig
    from lunarsim.core.terrain.craters import _profile_for, expected_count, sfd_cdf
    from lunarsim.core.terrain.rocks import _number_table, cumulative_area_fraction

    tool = ROOT / "web" / "tools" / "dump_terrain_model.mjs"
    r = subprocess.run(["node", str(tool)], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr
    d = json.loads(r.stdout)

    # crater radial profiles (in crater radii) for fresh .. fully degraded craters
    shape = CraterShapeConfig()
    assert d["shape"]["depthFresh"] == shape.depth_ratio_fresh
    assert d["shape"]["rimFresh"] == shape.rim_ratio_fresh
    for g, row in d["profiles"].items():
        rho, prof = _profile_for(shape, float(g))
        np.testing.assert_allclose(row, np.interp(d["rhos"], rho, prof), atol=1e-3)

    # crater population: Poisson count around the equilibrium density, truncated power law D^-2
    cp = d["craterParams"]
    sfd = CraterSFDConfig(
        mode="equilibrium", equilibrium_fraction=cp["fraction"], d_min_m=cp["dMin"],
        d_max_m=cp["dMax"],
    )  # fmt: skip
    expected = expected_count(sfd, d["craterArea"])
    counts = np.array(d["craterCounts"])
    assert abs(counts.mean() - expected) < 4 * math.sqrt(expected / len(counts))
    for q, dq in d["craterDiameterQuantiles"]:
        assert float(sfd_cdf(dq, sfd)) == pytest.approx(q, abs=2e-3)
    assert d["craterMeanG"] == pytest.approx(0.5, abs=5e-3)  # Beta(1, 1)

    # stamped craters on a fine grid: depth / diameter, rim height, rim-crest radius, ejecta decay
    fresh, old = d["stamped"]
    assert fresh["depthOverD"] == pytest.approx(0.15, abs=2e-3)  # Stopar et al. (2017)
    assert fresh["rimOverD"] == pytest.approx(0.036, abs=1e-3)  # Pike (1977)
    assert fresh["crestRadiusOverR"] == pytest.approx(1.0, abs=0.01)
    tail = 4.0**-3  # (r/R)^-3 tapered to 0 at 4 R (McGetchin et al. 1973)
    expect = [(q**-3 - tail) / (1 - tail) for q in (1.5, 2, 3)]
    np.testing.assert_allclose(fresh["ejecta"], expect, atol=5e-3)
    assert old["depthOverD"] == pytest.approx(0.05, abs=2e-3)  # Basilevsky et al. (2018)
    assert old["rimOverD"] < fresh["rimOverD"] / 3

    # rocks: Golombek & Rapp (1997) cumulative area fraction F_k(D) = k exp(-q D)
    rp = d["rockParams"]
    _, cum = _number_table(rp["k"], rp["dMin"], rp["dMax"])
    n_exp = cum[-1] * d["rockArea"]
    rc = np.array(d["rockCounts"])
    assert abs(rc.mean() - n_exp) < 4 * math.sqrt(n_exp / len(rc))

    def f_above(dd: float) -> float:
        k = rp["k"]
        return float(cumulative_area_fraction(dd, k) - cumulative_area_fraction(rp["dMax"], k))

    assert d["rockCircleAreaFraction"] == pytest.approx(f_above(rp["dMin"]), rel=0.02)
    for dd, frac in d["rockAreaFractionAbove"][:3]:
        assert frac == pytest.approx(f_above(dd), rel=0.12)
    st = d["rockStamp"]
    assert st["area"] == pytest.approx(st["expectedArea"], rel=0.01)  # ellipse a x b
    assert st["top"] == pytest.approx(0.6, abs=1e-3)  # height ratio 0.5 x 1.2 m

    # fractal: RMS slope at the baseline is exact; structure function ~ L^H
    fr = d["fractal"]
    assert fr["measuredSlopeDeg"] == pytest.approx(fr["rmsSlopeDeg"], abs=1e-9)
    assert 0.7 < fr["measuredHurst"] < fr["hurst"] + 0.05  # estimator is biased low near H = 1
    assert abs(fr["mean"]) < 1e-9
    assert d["tilt"]["measuredDeg"] == pytest.approx(d["tilt"]["slopeDeg"], abs=1e-9)
    assert abs(d["tilt"]["across"]) < 1e-9


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not installed")
def test_browser_presets_match_python_scenarios() -> None:
    """Presets of the same name describe the same place, sun and surface models in both versions.

    Grid size / resolution and the smallest crater and rock differ on purpose (they follow the
    grid); everything physical is compared.
    """
    from lunarsim.config.load import load_config

    js = "import('./web/js/sim/scenarios.js').then((m) => console.log(JSON.stringify(m.SCENARIOS)))"
    r = subprocess.run(["node", "-e", js], capture_output=True, text=True, timeout=60, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    presets = json.loads(r.stdout)
    assert set(presets) == {"south_pole_lowsun", "mare_plain", "crater_field"}
    for name, web in presets.items():
        cfg = load_config(ROOT / "configs" / f"{name}.yaml")
        t, wt = cfg.terrain, web["terrain"]
        assert cfg.seed == web["seed"], name
        assert (cfg.site.latitude_deg, cfg.site.longitude_deg) == (
            web["site"]["lat"],
            web["site"]["lon"],
        ), name
        assert cfg.sun.mode == web["sun"]["mode"], name
        if cfg.sun.mode == "astronomical":
            assert cfg.sun.utc == web["sun"]["utc"], name
        else:
            assert cfg.sun.azimuth_deg == web["sun"]["azDeg"], name
            assert cfg.sun.elevation_deg == web["sun"]["elDeg"], name
        f, wf = t.fractal, wt["fractal"]
        assert (f.enabled, f.hurst, f.rms_slope_deg, f.baseline_m, f.rolloff_wavelength_m) == (
            wf["enabled"],
            wf["hurst"],
            wf["rmsSlopeDeg"],
            wf["baseline"],
            wf["rolloffWavelength"],
        ), name
        assert t.tilt.slope_deg == wt["tilt"]["slopeDeg"] == 0.0, name
        wd = wt.get("dem") or {"enabled": False}
        assert t.dem.enabled == wd["enabled"], name
        if t.dem.enabled:
            assert t.dem.detail_below_m == wd["detailBelow"], name
        c, wc = t.craters, wt["craters"]
        assert c.enabled == wc["enabled"], name
        assert c.sfd.equilibrium_fraction == wc["equilibriumFraction"], name
        cap = t.dem.detail_below_m if t.dem.enabled else None
        if cap is None:  # with a DEM both versions cap the craters at detail_below
            assert c.sfd.d_max_m == wc["dMax"], name
        rk, wr = t.rocks, wt["rocks"]
        assert (rk.enabled, rk.k, rk.d_max_m) == (wr["enabled"], wr["k"], wr["dMax"]), name
        assert rk.ejecta.enabled == wr["ejecta"], name
        if rk.ejecta.enabled:
            assert (rk.ejecta.k, rk.ejecta.min_crater_diameter_m) == (
                wr["ejectaK"],
                wr["ejectaMinCraterD"],
            ), name
        rv, wv = cfg.rover, web["rover"]
        assert (rv.wheelbase_m, rv.track_m, rv.origin_height_m, rv.max_speed_m_s) == (
            wv["wheelbase"],
            wv["track"],
            wv["originHeight"],
            wv["maxSpeed"],
        ), name
