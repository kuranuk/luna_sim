"""Terrain assembly: base elevation + fractal + regional tilt + craters + rocks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from lunarsim.config.schema import TerrainConfig
from lunarsim.core.rng import RngFactory
from lunarsim.core.terrain.craters import CraterCatalog, generate_craters, stamp_craters
from lunarsim.core.terrain.fractal import synthesize_fractal
from lunarsim.core.terrain.heightmap import Heightmap
from lunarsim.core.terrain.labels import Label
from lunarsim.core.terrain.rocks import RockCatalog, generate_rocks, stamp_rocks

__all__ = ["Terrain", "build_terrain"]

STREAM_FRACTAL = "terrain/fractal"


@dataclass(frozen=True)
class Terrain:
    """Generated terrain and its ground truth.

    Attributes:
        heightmap: Final elevation grid.
        labels: Per-cell :class:`Label` (uint8, same shape as the heightmap).
        craters: Crater catalog (oldest first).
        rocks: Rock catalog.
    """

    heightmap: Heightmap
    labels: npt.NDArray[np.uint8]
    craters: CraterCatalog
    rocks: RockCatalog


def build_terrain(cfg: TerrainConfig, rngs: RngFactory, workers: int | None = None) -> Terrain:
    """Generate the terrain deterministically from ``cfg`` and ``rngs``.

    Order: fractal -> tilt -> craters (oldest first) -> rocks. Every stochastic stage
    uses its own named stream, so e.g. changing the crater settings leaves the
    background rocks unchanged.
    """
    ny, nx = cfg.shape
    z = synthesize_fractal(
        (ny, nx), cfg.resolution_m, cfg.fractal, rngs.generator(STREAM_FRACTAL), workers=workers
    )
    hm = Heightmap.centered(z, cfg.resolution_m)
    z = hm.z  # contiguous float64 array owned by hm
    z += cfg.base_elevation_m
    if cfg.tilt.slope_deg > 0.0:
        az = np.deg2rad(cfg.tilt.downhill_azimuth_deg)
        gx, gy = np.sin(az), np.cos(az)  # downhill unit vector (East, North)
        grade = np.tan(np.deg2rad(cfg.tilt.slope_deg))
        z -= grade * (hm.x[None, :] * gx + hm.y[:, None] * gy)

    labels = np.full((ny, nx), Label.PLAIN, dtype=np.uint8)
    craters = generate_craters(cfg.craters, hm.extent, rngs)
    stamp_craters(hm, craters, cfg.craters.shape, labels, cfg.craters.label_min_diameter_m)
    rocks = generate_rocks(cfg.rocks, hm.extent, rngs, craters)
    stamp_rocks(hm, rocks, labels)
    return Terrain(heightmap=hm, labels=labels, craters=craters, rocks=rocks)
