"""Two-layer ray casting: a fine near patch embedded in a coarse far heightmap.

Along each ray the near layer owns the parameter interval where the ray's ground
track lies inside the near patch's xy box; the far layer owns the rest. The far
layer's (coarse) surface inside the near box is therefore never hit. A small step
can exist at the patch border where the two surfaces differ by the fine detail.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from lunarsim.core.raycast.heightfield import HeightfieldCaster, RayHits

__all__ = ["LayeredCaster"]

FloatArray = npt.NDArray[np.float64]


def _box_interval(
    o: FloatArray, d: FloatArray, box: tuple[float, float, float, float]
) -> tuple[FloatArray, FloatArray]:
    """Parameter interval ``[ta, tb]`` where the xy track is inside ``box`` (ta > tb: never)."""
    ta = np.full(o.shape[0], -np.inf)
    tb = np.full(o.shape[0], np.inf)
    for axis, lo, hi in ((0, box[0], box[1]), (1, box[2], box[3])):
        p, dd = o[:, axis], d[:, axis]
        par = np.abs(dd) < 1e-15
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (lo - p) / dd
            t2 = (hi - p) / dd
        tmn = np.where(par, np.where((p >= lo) & (p <= hi), -np.inf, np.inf), np.minimum(t1, t2))
        tmx = np.where(par, np.where((p >= lo) & (p <= hi), np.inf, -np.inf), np.maximum(t1, t2))
        ta = np.maximum(ta, tmn)
        tb = np.minimum(tb, tmx)
    return ta, tb


class LayeredCaster:
    """Nearest-hit ray casting over ``[near, far]`` layers (layer index 0 / 1)."""

    def __init__(self, near: HeightfieldCaster, far: HeightfieldCaster) -> None:
        self.near = near
        self.far = far

    @property
    def backend(self) -> str:
        """Backend of the underlying casters."""
        return self.near.backend

    def intersect(
        self,
        origins: npt.ArrayLike,
        directions: npt.ArrayLike,
        t_min: npt.ArrayLike = 0.0,
        t_max: npt.ArrayLike = np.inf,
    ) -> RayHits:
        """Nearest hit (see :meth:`HeightfieldCaster.intersect`)."""
        o = np.asarray(origins, dtype=np.float64).reshape(-1, 3)
        d = np.asarray(directions, dtype=np.float64).reshape(-1, 3)
        n = o.shape[0]
        t0 = np.broadcast_to(np.asarray(t_min, dtype=np.float64), (n,)).copy()
        t1 = np.broadcast_to(np.asarray(t_max, dtype=np.float64), (n,)).copy()
        ta, tb = _box_interval(o, d, self.near.bounds_xy)
        inside = ta <= tb

        t = np.full(n, np.inf)
        qi = np.full(n, -1, dtype=np.int64)
        qj = np.full(n, -1, dtype=np.int64)
        layer = np.full(n, -1, dtype=np.int64)

        def take(hits: RayHits, idx: npt.NDArray[np.intp], lay: int) -> None:
            better = hits.t < t[idx]
            sel = idx[better]
            t[sel] = hits.t[better]
            qi[sel] = hits.qi[better]
            qj[sel] = hits.qj[better]
            layer[sel] = lay

        # Far layer before entering the near box (or everywhere if never inside).
        far_hi = np.where(inside, np.minimum(ta, t1), t1)
        idx = np.flatnonzero(far_hi > t0)
        if idx.size:
            take(self.far.intersect(o[idx], d[idx], t0[idx], far_hi[idx]), idx, 1)
        # Near layer inside the box.
        n_lo, n_hi = np.maximum(ta, t0), np.minimum(tb, t1)
        idx = np.flatnonzero(inside & (n_lo <= n_hi) & ~np.isfinite(t))
        if idx.size:
            take(self.near.intersect(o[idx], d[idx], n_lo[idx], n_hi[idx]), idx, 0)
        # Far layer after leaving the near box.
        f_lo = np.maximum(tb, t0)
        idx = np.flatnonzero(inside & (f_lo < t1) & ~np.isfinite(t))
        if idx.size:
            take(self.far.intersect(o[idx], d[idx], f_lo[idx], t1[idx]), idx, 1)
        return RayHits(t=t, qi=qi, qj=qj, layer=layer)

    def occluded(
        self,
        origins: npt.ArrayLike,
        directions: npt.ArrayLike,
        t_max: npt.ArrayLike = np.inf,
        t_min: npt.ArrayLike = 0.0,
    ) -> npt.NDArray[np.bool_]:
        """True where something blocks the ray before ``t_max``."""
        return self.intersect(origins, directions, t_min, t_max).hit
