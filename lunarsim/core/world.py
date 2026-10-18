"""World: the single container of simulation state.

A :class:`World` is fully determined by its validated :class:`WorldConfig`
(which includes the seed). Later phases add sun, rover, sensors and time.
"""

from __future__ import annotations

from pathlib import Path

from lunarsim.config.load import load_config
from lunarsim.config.schema import WorldConfig, config_hash
from lunarsim.core.rng import RngFactory
from lunarsim.core.terrain import Heightmap, build_terrain

__all__ = ["World"]


class World:
    """Terrain (+ future: sun, rover, sensors) built from config and seed."""

    def __init__(self, config: WorldConfig, workers: int | None = None) -> None:
        self.config = config
        self.rngs = RngFactory(config.seed)
        self.heightmap: Heightmap = build_terrain(config.terrain, self.rngs, workers=workers)

    @classmethod
    def from_yaml(cls, path: str | Path, seed: int | None = None) -> World:
        """Build a world from a scenario YAML file (``seed`` overrides the file)."""
        return cls(load_config(path, seed=seed))

    @property
    def seed(self) -> int:
        """Root seed."""
        return self.config.seed

    @property
    def config_hash(self) -> str:
        """SHA-256 of the canonical config (identifies the world reproducibly)."""
        return config_hash(self.config)

    def metadata(self) -> dict[str, object]:
        """JSON-serialisable summary."""
        hm = self.heightmap
        xmin, xmax, ymin, ymax = hm.extent
        return {
            "name": self.config.name,
            "seed": self.seed,
            "config_hash": self.config_hash,
            "grid_shape": list(hm.shape),
            "resolution_m": hm.resolution,
            "extent_m": {"x_min": xmin, "x_max": xmax, "y_min": ymin, "y_max": ymax},
            "elevation_m": {"min": float(hm.z.min()), "max": float(hm.z.max())},
        }
