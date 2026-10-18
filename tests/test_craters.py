from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from lunarsim.config import TerrainConfig
from lunarsim.config.schema import CraterConfig, CraterSFDConfig, CraterShapeConfig
from lunarsim.core.rng import RngFactory
from lunarsim.core.terrain import Heightmap, Label, build_terrain
from lunarsim.core.terrain.craters import (
    CraterCatalog,
    expected_count,
    generate_craters,
    radial_profile,
    sample_diameters,
    sfd_cdf,
    stamp_craters,
)


@pytest.mark.parametrize("mode,b", [("equilibrium", 2.0), ("power_law", 3.0), ("power_law", 1.5)])
def test_sfd_ks(mode: str, b: float) -> None:
    sfd = CraterSFDConfig(mode=mode, b=b, d_min_m=1.0, d_max_m=200.0)
    d = sample_diameters(20000, sfd, RngFactory(0).generator("s"))
    assert d.min() >= 1.0 and d.max() <= 200.0
    res = stats.kstest(d, lambda x: sfd_cdf(x, sfd))
    assert res.pvalue > 0.01
    # Cumulative slope check: log N(>D) vs log D over the well-sampled range.
    ds = np.geomspace(1.0, 4.0, 8)
    n_gt = np.array([(d > x).sum() for x in ds])
    slope = np.polyfit(np.log(ds), np.log(n_gt), 1)[0]
    expected = np.polyfit(np.log(ds), np.log(ds**-b - 200.0**-b), 1)[0]
    assert slope == pytest.approx(expected, abs=0.05)


def test_ks_rejects_wrong_exponent() -> None:
    sfd = CraterSFDConfig(mode="power_law", b=3.0, d_min_m=1.0, d_max_m=200.0)
    wrong = CraterSFDConfig(mode="power_law", b=2.5, d_min_m=1.0, d_max_m=200.0)
    d = sample_diameters(20000, sfd, RngFactory(0).generator("s"))
    assert stats.kstest(d, lambda x: sfd_cdf(x, wrong)).pvalue < 1e-6


def test_equilibrium_coefficient_and_count() -> None:
    sfd = CraterSFDConfig(mode="equilibrium", equilibrium_fraction=0.02, d_min_m=2.0, d_max_m=50.0)
    assert sfd.coefficient == pytest.approx(0.0308) and sfd.exponent == 2.0
    cfg = CraterConfig(enabled=True, sfd=sfd, shape=CraterShapeConfig(ejecta_extent_radii=2.0))
    extent = (-100.0, 100.0, -100.0, 100.0)
    margin = 0.5 * 50.0 * 2.0
    lam = expected_count(sfd, (200.0 + 2 * margin) ** 2)
    counts = [len(generate_craters(cfg, extent, RngFactory(s))) for s in range(40)]
    # Poisson: mean within 4 standard errors.
    assert abs(np.mean(counts) - lam) < 4.0 * np.sqrt(lam / len(counts))


def test_profile_continuity_and_values() -> None:
    shape = CraterShapeConfig()
    assert radial_profile(0.0, 1.5, 0.3, shape) == pytest.approx(-1.5)
    assert radial_profile(1.0 - 1e-12, 1.5, 0.3, shape) == pytest.approx(0.3)
    assert radial_profile(1.0, 1.5, 0.3, shape) == pytest.approx(0.3)
    assert radial_profile(shape.ejecta_extent_radii, 1.5, 0.3, shape) == 0.0
    r = np.linspace(1.0, shape.ejecta_extent_radii, 50)
    assert np.all(np.diff(radial_profile(r, 1.5, 0.3, shape)) < 0)


def _single(d: float, g: float, x: float = 0.0, y: float = 0.0) -> CraterCatalog:
    shape = CraterShapeConfig()
    dd = shape.depth_ratio_fresh + g * (shape.depth_ratio_degraded - shape.depth_ratio_fresh)
    depth = dd * d
    rim = depth * shape.rim_ratio_fresh / shape.depth_ratio_fresh
    a = np.array
    return CraterCatalog(a([x]), a([y]), a([d]), a([depth]), a([rim]), a([g]), a([-1]))


@pytest.mark.parametrize("g", [0.0, 0.5, 1.0])
def test_single_crater_depth(g: float) -> None:
    hm = Heightmap.centered(np.zeros((601, 601)), 0.1)  # +-30 m; ejecta reach 40 m
    cat = _single(20.0, g)
    stamp_craters(hm, cat, CraterShapeConfig())
    # Rim crest to floor = d exactly (up to discretisation), for any degradation.
    assert hm.z.max() - hm.z.min() == pytest.approx(cat.depth[0], rel=0.02)
    floor = -(cat.depth[0] - cat.rim_height[0])  # nominal; smoothing lowers the rim
    tol = 0.02 if g == 0.0 else 0.25
    assert hm.height_at(0.0, 0.0) == pytest.approx(floor, rel=tol)
    if g == 0.0:
        assert hm.z.max() == pytest.approx(cat.rim_height[0], rel=0.02)
    # Beyond the ejecta reach the terrain is untouched.
    assert hm.z[0, 0] == 0.0


def test_border_crater_shape_matches_interior() -> None:
    """A crater centred off-grid must not be distorted (regression)."""
    shape = CraterShapeConfig()
    hm = Heightmap.centered(np.zeros((201, 201)), 0.1)  # extent +-10 m
    stamp_craters(hm, _single(3.0, 0.8, x=10.5, y=0.0), shape)
    ref = Heightmap.centered(np.zeros((201, 201)), 0.1)
    stamp_craters(ref, _single(3.0, 0.8, x=0.0, y=0.0), shape)
    # Column at x = 10 m (0.5 m from the centre) vs the same offset in the reference.
    np.testing.assert_allclose(hm.z[:, -1], ref.z[:, 105], atol=1e-9)


def test_crater_follows_tilted_surface() -> None:
    hm0 = Heightmap.centered(np.zeros((301, 301)), 0.1)
    tilt = 0.1 * hm0.x[None, :] + 0.05 * hm0.y[:, None]
    hm = Heightmap.centered(tilt.copy(), 0.1)
    stamp_craters(hm, _single(10.0, 0.0), CraterShapeConfig())
    flat = Heightmap.centered(np.zeros((301, 301)), 0.1)
    stamp_craters(flat, _single(10.0, 0.0), CraterShapeConfig())
    np.testing.assert_allclose(hm.z - tilt, flat.z, atol=1e-9)


def test_overprinting_erases_older_interior() -> None:
    shape = CraterShapeConfig()
    hm = Heightmap.centered(np.zeros((301, 301)), 0.1)
    old = _single(3.0, 0.0, x=0.5, y=0.0)
    new = _single(20.0, 0.0)
    both = CraterCatalog(
        *(
            np.concatenate([getattr(old, f), getattr(new, f)])
            for f in ("x", "y", "diameter", "depth", "rim_height", "degradation", "cluster_id")
        )
    )
    stamp_craters(hm, both, shape)
    only_new = Heightmap.centered(np.zeros((301, 301)), 0.1)
    stamp_craters(only_new, new, shape)
    inner = np.hypot(hm.x[None, :], hm.y[:, None]) < 0.85 * 10.0
    np.testing.assert_allclose(hm.z[inner], only_new.z[inner], atol=1e-9)


def test_catalog_order_and_degradation_mapping() -> None:
    cfg = CraterConfig(enabled=True, sfd=CraterSFDConfig(d_min_m=1.0, d_max_m=20.0))
    cat = generate_craters(cfg, (-50.0, 50.0, -50.0, 50.0), RngFactory(2))
    assert np.all(np.diff(cat.degradation) <= 0)  # oldest (most degraded) first
    dd = cat.depth / cat.diameter
    assert dd.max() <= 0.15 + 1e-12 and dd.min() >= 0.05 - 1e-12
    np.testing.assert_allclose(cat.rim_height / cat.depth, 0.036 / 0.15)
    rec = cat.to_records()
    assert rec[0]["id"] == 0 and set(rec[0]) >= {"x_m", "diameter_m", "depth_m", "degradation"}


def test_clustering() -> None:
    base = dict(enabled=True, sfd=CraterSFDConfig(d_min_m=1.0, d_max_m=5.0))
    ext = (-200.0, 200.0, -200.0, 200.0)
    uni = generate_craters(CraterConfig(**base), ext, RngFactory(3))
    clu = generate_craters(
        CraterConfig(**base, clustering={"fraction": 0.5, "mean_cluster_size": 20, "sigma_m": 3}),
        ext,
        RngFactory(3),
    )
    assert (clu.cluster_id >= 0).mean() == pytest.approx(0.5, abs=0.01)
    # Cluster members share the parent's degradation (same event).
    cid = clu.cluster_id[clu.cluster_id >= 0][0]
    assert np.ptp(clu.degradation[clu.cluster_id == cid]) < 0.01
    assert (uni.cluster_id == -1).all()


def test_terrain_labels_and_reproducibility() -> None:
    cfg = TerrainConfig(
        size_x_m=60,
        size_y_m=60,
        resolution_m=0.2,
        craters={"enabled": True, "sfd": {"d_min_m": 1.0, "d_max_m": 20.0}},
    )
    a = build_terrain(cfg, RngFactory(5))
    b = build_terrain(cfg, RngFactory(5), workers=1)
    assert a.heightmap.z.tobytes() == b.heightmap.z.tobytes()
    assert a.labels.tobytes() == b.labels.tobytes()
    assert (a.labels == Label.CRATER_INTERIOR).any() and (a.labels == Label.CRATER_RIM).any()


def test_degraded_profile_table_matches_2d_gaussian() -> None:
    from scipy.ndimage import gaussian_filter

    from lunarsim.core.terrain.craters import _shape_params, degraded_profile_table

    shape = CraterShapeConfig()
    rho_t, table = degraded_profile_table(shape)
    g = 1.0
    depth, rim = _shape_params(np.array([2.0]), np.array([g]), shape)  # R = 1
    h = 0.01
    ax = np.arange(-450, 451) * h
    rho2 = np.hypot(ax[None, :], ax[:, None])
    f2 = gaussian_filter(
        radial_profile(rho2, depth[0] - rim[0], rim[0], shape),
        shape.smoothing_radii_degraded * g / h,
        mode="constant",
        truncate=6.0,
    )
    span = f2[rho2 <= 1.2].max() - f2[rho2 < 1.0].min()
    f2 *= depth[0] / span
    centre_row = f2[450, 450:]
    np.testing.assert_allclose(np.interp(ax[450:], rho_t, table[-1]), centre_row, atol=2e-3)
    for row, d in zip(
        table,
        _shape_params(np.full(len(table), 2.0), np.linspace(0, 1, len(table)), shape)[0],
        strict=True,
    ):
        assert row[rho_t <= 1.2].max() - row[rho_t < 1.0].min() == pytest.approx(d, rel=1e-9)
