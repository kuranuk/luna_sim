"""Cast shadows on terrain grids (point sun, or the solar disc via :class:`SkyLighting`)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from lunarsim.core.illumination.sun import SunState
from lunarsim.core.illumination.visibility import SkyLighting
from lunarsim.core.render import Caster
from lunarsim.core.terrain.generate import Terrain
from lunarsim.core.terrain.labels import Label

__all__ = ["labels_with_shadow", "terrain_shadow_mask", "terrain_sun_fraction"]


def terrain_sun_fraction(
    terrain: Terrain,
    caster: Caster,
    sun: SunState,
    bias_m: float = 2e-3,
    chunk: int = 1_000_000,
    sky: SkyLighting | None = None,
) -> npt.NDArray[np.float64]:
    """Visible fraction of the solar disc at every grid node (0/1 for a point sun)."""
    hm = terrain.heightmap
    sk = sky if sky is not None else SkyLighting(sun)
    n = hm.normals().reshape(-1, 3)
    X, Y = np.meshgrid(hm.x, hm.y)
    p = np.stack([X.ravel(), Y.ravel(), hm.z.ravel()], axis=1)
    out = np.empty(p.shape[0])
    for s in range(0, p.shape[0], chunk):
        out[s : s + chunk] = sk.sun_visibility(caster, p[s : s + chunk], n[s : s + chunk], bias_m)
    return out.reshape(hm.shape)


def terrain_shadow_mask(
    terrain: Terrain,
    caster: Caster,
    sun: SunState,
    bias_m: float = 2e-3,
    chunk: int = 1_000_000,
    sky: SkyLighting | None = None,
) -> npt.NDArray[np.bool_]:
    """True for grid nodes with less than half the solar disc visible (point sun: unlit)."""
    return terrain_sun_fraction(terrain, caster, sun, bias_m, chunk, sky) < 0.5


def labels_with_shadow(
    labels: npt.NDArray[np.uint8], shadow: npt.NDArray[np.bool_]
) -> npt.NDArray[np.uint8]:
    """Copy of ``labels`` with :attr:`Label.SHADOW` where ``shadow`` (sky kept)."""
    out = labels.copy()
    out[shadow & (labels != Label.SKY)] = Label.SHADOW
    return out
