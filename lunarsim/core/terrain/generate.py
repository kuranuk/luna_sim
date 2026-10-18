"""Terrain assembly: base elevation + fractal + regional tilt (+ craters/rocks later)."""

from __future__ import annotations

import numpy as np

from lunarsim.config.schema import TerrainConfig
from lunarsim.core.rng import RngFactory
from lunarsim.core.terrain.fractal import synthesize_fractal
from lunarsim.core.terrain.heightmap import Heightmap

__all__ = ["build_terrain"]

STREAM_FRACTAL = "terrain/fractal"


def build_terrain(cfg: TerrainConfig, rngs: RngFactory, workers: int | None = None) -> Heightmap:
    """Generate the terrain heightmap deterministically from ``cfg`` and ``rngs``."""
    ny, nx = cfg.shape
    z = synthesize_fractal(
        (ny, nx), cfg.resolution_m, cfg.fractal, rngs.generator(STREAM_FRACTAL), workers=workers
    )
    hm = Heightmap.centered(z, cfg.resolution_m)
    z = hm.z  # contiguous float64 view owned by hm
    z += cfg.base_elevation_m
    if cfg.tilt.slope_deg > 0.0:
        az = np.deg2rad(cfg.tilt.downhill_azimuth_deg)
        gx, gy = np.sin(az), np.cos(az)  # downhill unit vector (East, North)
        grade = np.tan(np.deg2rad(cfg.tilt.slope_deg))
        z -= grade * (hm.x[None, :] * gx + hm.y[:, None] * gy)
    return hm
