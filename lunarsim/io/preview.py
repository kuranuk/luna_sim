"""Quick-look visualisations (NOT physically based rendering; see illumination/ for that)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from lunarsim.core.frames import azel_to_enu
from lunarsim.core.terrain.heightmap import Heightmap

__all__ = ["hillshade", "normalize"]


def hillshade(
    hm: Heightmap, azimuth_deg: float = 315.0, elevation_deg: float = 30.0
) -> npt.NDArray[np.float64]:
    """Lambertian shaded relief in [0, 1] (no cast shadows), for previews only."""
    s = azel_to_enu(np.deg2rad(azimuth_deg), np.deg2rad(elevation_deg))
    shade = hm.normals() @ s
    return np.asarray(np.clip(shade, 0.0, 1.0), dtype=np.float64)


def normalize(
    values: npt.ArrayLike, vmin: float | None = None, vmax: float | None = None
) -> npt.NDArray[np.float64]:
    """Linearly map values to [0, 1]."""
    v = np.asarray(values, dtype=np.float64)
    lo = float(v.min()) if vmin is None else vmin
    hi = float(v.max()) if vmax is None else vmax
    if hi <= lo:
        return np.zeros_like(v)
    return np.asarray(np.clip((v - lo) / (hi - lo), 0.0, 1.0), dtype=np.float64)
