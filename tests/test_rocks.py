from __future__ import annotations

import numpy as np
import pytest

from lunarsim.config import TerrainConfig
from lunarsim.config.schema import RockConfig
from lunarsim.core.rng import RngFactory
from lunarsim.core.terrain import Heightmap, Label, build_terrain
from lunarsim.core.terrain.craters import CraterCatalog
from lunarsim.core.terrain.rocks import (
    RockCatalog,
    cumulative_area_fraction,
    generate_rocks,
    q_of_k,
    sample_rock_diameters,
    stamp_rocks,
)


def test_q_of_k() -> None:
    assert q_of_k(0.1) == pytest.approx(1.79 + 1.52)


@pytest.mark.parametrize("k", [0.05, 0.15, 0.3])
def test_cumulative_area_fraction_matches_model(k: float) -> None:
    area = 2.0e5
    d_min, d_max = 0.2, 10.0
    d = sample_rock_diameters(k, d_min, d_max, area, RngFactory(0).generator("r"))
    for x in (0.3, 0.6, 1.0):
        observed = np.sum(0.25 * np.pi * d[d >= x] ** 2) / area
        expected = cumulative_area_fraction(x, k) - cumulative_area_fraction(d_max, k)
        assert observed == pytest.approx(float(expected), rel=0.1)


def test_half_ellipsoid_stamp() -> None:
    hm = Heightmap.centered(np.zeros((101, 101)), 0.02)
    a = np.array
    rocks = RockCatalog(a([0.0]), a([0.0]), a([1.0]), a([0.4]), a([0.5]), a([0.0]), a([-1]))
    labels = np.zeros(hm.shape, dtype=np.uint8)
    stamp_rocks(hm, rocks, labels)
    assert hm.z.max() == pytest.approx(0.4)
    assert hm.height_at(0.45, 0.0) > 0 and hm.height_at(0.0, 0.3) == 0.0  # minor axis 0.25 m
    footprint = (labels == Label.ROCK).sum() * 0.02**2
    assert footprint == pytest.approx(np.pi * 0.5 * 0.25, rel=0.05)


def test_ejecta_rocks_around_fresh_crater_only() -> None:
    cfg = RockConfig(
        enabled=True,
        k=0.01,
        d_min_m=0.2,
        d_max_m=3.0,
        ejecta={"enabled": True, "k": 0.2, "min_crater_diameter_m": 30.0},
    )
    a = np.array
    craters = CraterCatalog(
        a([0.0, 60.0]),
        a([0.0, 0.0]),
        a([50.0, 50.0]),
        a([7.0, 2.5]),
        a([1.8, 0.6]),
        a([0.0, 1.0]),
        a([-1, -1]),
    )
    rocks = generate_rocks(cfg, (-100.0, 100.0, -50.0, 50.0), RngFactory(1), craters)
    src = rocks.source_crater
    assert (src == 0).sum() > 0 and (src == 1).sum() == 0  # fully degraded -> none
    r = np.hypot(rocks.x[src == 0], rocks.y[src == 0])
    assert r.min() >= 0.8 * 25.0 - 1e-9 and r.max() <= 2.5 * 25.0 + 1e-9
    assert rocks.diameter[src == 0].max() <= 0.29 * 50.0**0.66


def test_rocks_independent_of_crater_settings() -> None:
    base = dict(
        size_x_m=60,
        size_y_m=60,
        resolution_m=0.2,
        rocks={"enabled": True, "k": 0.1, "d_min_m": 0.4},
    )
    t1 = build_terrain(TerrainConfig(**base), RngFactory(9))
    t2 = build_terrain(TerrainConfig(**base, craters={"enabled": True}), RngFactory(9))
    np.testing.assert_array_equal(t1.rocks.x, t2.rocks.x)
    np.testing.assert_array_equal(t1.rocks.diameter, t2.rocks.diameter)
    assert (t1.labels == Label.ROCK).any()
