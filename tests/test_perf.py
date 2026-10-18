"""Performance benchmarks (run with LUNARSIM_PERF=1)."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from lunarsim.core.world import World

ROOT = Path(__file__).parents[1]


@pytest.mark.perf
def test_terrain_1km_quarter_metre() -> None:
    t0 = time.perf_counter()
    w = World.from_yaml(ROOT / "configs" / "bench_1km.yaml")
    dt = time.perf_counter() - t0
    print(f"\n1 km^2 @ 0.25 m ({w.heightmap.shape}): {dt:.2f} s")
    assert dt < 60.0
