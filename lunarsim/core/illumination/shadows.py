"""Cast shadows on terrain grids (sun as a point source; penumbra in Phase 8)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from lunarsim.core.illumination.sun import SunState
from lunarsim.core.render import Caster
from lunarsim.core.terrain.generate import Terrain
from lunarsim.core.terrain.labels import Label

__all__ = ["labels_with_shadow", "terrain_shadow_mask"]


def terrain_shadow_mask(
    terrain: Terrain, caster: Caster, sun: SunState, bias_m: float = 2e-3, chunk: int = 1_000_000
) -> npt.NDArray[np.bool_]:
    """True for grid nodes without direct sunlight (self-facing away or cast shadow)."""
    hm = terrain.heightmap
    n = hm.normals().reshape(-1, 3)
    X, Y = np.meshgrid(hm.x, hm.y)
    p = np.stack([X.ravel(), Y.ravel(), hm.z.ravel()], axis=1)
    facing = (n @ sun.direction > 0.0) & (sun.elevation > 0.0)
    shadow = ~facing
    idx = np.flatnonzero(facing)
    for s in range(0, idx.size, chunk):
        ii = idx[s : s + chunk]
        o = p[ii] + bias_m * n[ii]
        d = np.broadcast_to(sun.direction, o.shape)
        shadow[ii] = caster.occluded(o, d)
    return shadow.reshape(hm.shape)


def labels_with_shadow(
    labels: npt.NDArray[np.uint8], shadow: npt.NDArray[np.bool_]
) -> npt.NDArray[np.uint8]:
    """Copy of ``labels`` with :attr:`Label.SHADOW` where ``shadow`` (sky kept)."""
    out = labels.copy()
    out[shadow & (labels != Label.SKY)] = Label.SHADOW
    return out
