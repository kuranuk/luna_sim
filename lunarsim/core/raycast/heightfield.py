"""Ray casting against a heightmap (no external ray-tracing library).

Surface model: each grid quad between four neighbouring samples is split into two
triangles along the (i, j)-(i+1, j+1) diagonal, so the surface is the same
piecewise-planar mesh a viewer or exporter would build from the heightmap.

Acceleration: a *maximum mipmap* (quad-tree of max heights, Tevs et al. 2008,
"Maximum mipmaps for fast, accurate, and scalable dynamic height field rendering").
A ray descends into a node only if its height over the node's footprint can reach
the node's maximum; otherwise it skips to the node's exit and climbs one level.

Two interchangeable implementations of the same traversal:
* numba (``@njit(parallel=True)``) when numba is installed — fast path;
* vectorised numpy fallback — same algorithm, all rays advanced in lock step.
Results agree to floating-point round-off (tested), not necessarily bit-for-bit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from lunarsim.core.terrain.heightmap import Heightmap

__all__ = ["HAS_NUMBA", "HeightfieldCaster", "MaxMipmap", "RayHits"]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

try:  # optional accelerator
    import numba

    HAS_NUMBA = True
except ImportError:  # pragma: no cover - exercised on systems without numba
    numba = None
    HAS_NUMBA = False

_EPS_STEP = 1e-7  # [m] advance past a cell boundary
_EPS_BARY = 1e-9  # barycentric tolerance (closes cracks along triangle edges)
_EPS_T = 1e-6  # [m] tolerance on the segment where a leaf hit is accepted
_EPS_Z = 1e-6  # [m] slack on height bounds (flat terrain has a zero-thickness z slab)
# Horizontal direction components below this are treated as exactly zero: a ray running
# (almost) along a grid line would otherwise compute ~0 cell-exit distances forever.
_EPS_DIR = 1e-12


@dataclass(frozen=True)
class RayHits:
    """Result of a ray cast.

    Attributes:
        t: Distance along the (unit) ray direction to the hit [m]; ``inf`` = miss.
        qi, qj: Row / column index of the hit quad (-1 for a miss).
        layer: Index of the terrain layer that was hit (-1 for a miss).
    """

    t: FloatArray
    qi: IntArray
    qj: IntArray
    layer: IntArray

    @property
    def hit(self) -> npt.NDArray[np.bool_]:
        """True where the ray hit the terrain."""
        return np.asarray(np.isfinite(self.t))


@dataclass(frozen=True)
class MaxMipmap:
    """Flattened max-height pyramid over the grid quads.

    Level 0 has shape ``(ny - 1, nx - 1)`` (max of each quad's four corners); level
    ``l + 1`` takes the max over 2x2 blocks of level ``l`` (ceil sizes).
    """

    data: FloatArray
    offset: IntArray
    nx: IntArray
    ny: IntArray

    @property
    def levels(self) -> int:
        """Number of levels."""
        return int(self.offset.size)

    @classmethod
    def build(cls, z: FloatArray) -> MaxMipmap:
        """Build the pyramid for heights ``z`` of shape ``(ny, nx)``."""
        lvl = np.maximum(np.maximum(z[:-1, :-1], z[:-1, 1:]), np.maximum(z[1:, :-1], z[1:, 1:]))
        arrays = [lvl]
        while lvl.shape[0] > 1 or lvl.shape[1] > 1:
            h, w = lvl.shape
            ph, pw = h + (h % 2), w + (w % 2)
            pad = np.full((ph, pw), -np.inf)
            pad[:h, :w] = lvl
            lvl = np.maximum(
                np.maximum(pad[0::2, 0::2], pad[0::2, 1::2]),
                np.maximum(pad[1::2, 0::2], pad[1::2, 1::2]),
            )
            arrays.append(lvl)
        sizes = np.array([a.size for a in arrays], dtype=np.int64)
        offset = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64)
        return cls(
            data=np.concatenate([a.ravel() for a in arrays]),
            offset=offset,
            nx=np.array([a.shape[1] for a in arrays], dtype=np.int64),
            ny=np.array([a.shape[0] for a in arrays], dtype=np.int64),
        )


# ----------------------------------------------------------------------------- scalar


def _tri_hit(
    ax: float, ay: float, az: float,
    bx: float, by: float, bz: float,
    cx: float, cy: float, cz: float,
    ox: float, oy: float, oz: float,
    dx: float, dy: float, dz: float,
) -> float:  # fmt: skip
    """Möller-Trumbore ray/triangle distance (inf if no hit)."""
    e1x, e1y, e1z = bx - ax, by - ay, bz - az
    e2x, e2y, e2z = cx - ax, cy - ay, cz - az
    px = dy * e2z - dz * e2y
    py = dz * e2x - dx * e2z
    pz = dx * e2y - dy * e2x
    det = e1x * px + e1y * py + e1z * pz
    if abs(det) < 1e-18:
        return math.inf
    inv = 1.0 / det
    sx, sy, sz = ox - ax, oy - ay, oz - az
    u = (sx * px + sy * py + sz * pz) * inv
    if u < -_EPS_BARY or u > 1.0 + _EPS_BARY:
        return math.inf
    qx = sy * e1z - sz * e1y
    qy = sz * e1x - sx * e1z
    qz = sx * e1y - sy * e1x
    v = (dx * qx + dy * qy + dz * qz) * inv
    if v < -_EPS_BARY or u + v > 1.0 + _EPS_BARY:
        return math.inf
    return (e2x * qx + e2y * qy + e2z * qz) * inv


def _trace_one(
    z: FloatArray,
    pyr: FloatArray,
    off: IntArray,
    lnx: IntArray,
    lny: IntArray,
    x0: float,
    y0: float,
    res: float,
    zmin: float,
    zmax: float,
    ox: float,
    oy: float,
    oz: float,
    dx: float,
    dy: float,
    dz: float,
    t0: float,
    t1: float,
) -> tuple[float, int, int]:
    """Nearest hit of one ray within ``[t0, t1]`` -> ``(t, qi, qj)``."""
    nqy = z.shape[0] - 1
    nqx = z.shape[1] - 1
    nlev = off.shape[0]
    u0 = (ox - x0) / res
    v0 = (oy - y0) / res
    du = dx / res if abs(dx) > _EPS_DIR else 0.0
    dv = dy / res if abs(dy) > _EPS_DIR else 0.0
    tlo = t0
    thi = t1
    # Slab clipping against the grid's bounding box (u, v in quad units, z in m).
    for p, d, lo, hi in (
        (u0, du, 0.0, float(nqx)),
        (v0, dv, 0.0, float(nqy)),
        (oz, dz, zmin - _EPS_Z, zmax + _EPS_Z),
    ):
        if abs(d) < 1e-15:
            if p < lo or p > hi:
                return math.inf, -1, -1
        else:
            ta = (lo - p) / d
            tb = (hi - p) / d
            if ta > tb:
                ta, tb = tb, ta
            tlo = max(tlo, ta)
            thi = min(thi, tb)
    if tlo > thi:
        return math.inf, -1, -1

    t = tlo
    level = nlev - 1
    while t <= thi:
        s = 1 << level
        u = u0 + du * t
        v = v0 + dv * t
        j = min(max(int(math.floor(u / s)), 0), lnx[level] - 1)  # noqa: RUF046 (numba)
        i = min(max(int(math.floor(v / s)), 0), lny[level] - 1)  # noqa: RUF046 (numba)
        tx = ((j + 1) * s - u0) / du if du > 0.0 else ((j * s - u0) / du if du < 0.0 else math.inf)
        ty = ((i + 1) * s - v0) / dv if dv > 0.0 else ((i * s - v0) / dv if dv < 0.0 else math.inf)
        tc = min(tx, ty, thi)
        zlow = min(oz + dz * t, oz + dz * tc)
        if zlow > pyr[off[level] + i * lnx[level] + j] + _EPS_Z:
            t = max(t, tc) + _EPS_STEP
            if level < nlev - 1:
                level += 1
            continue
        if level > 0:
            level -= 1
            continue
        xa = x0 + j * res
        xb = xa + res
        ya = y0 + i * res
        yb = ya + res
        z00 = z[i, j]
        z01 = z[i, j + 1]
        z10 = z[i + 1, j]
        z11 = z[i + 1, j + 1]
        h1 = _tri_hit(xa, ya, z00, xb, ya, z01, xb, yb, z11, ox, oy, oz, dx, dy, dz)
        h2 = _tri_hit(xa, ya, z00, xb, yb, z11, xa, yb, z10, ox, oy, oz, dx, dy, dz)
        # Validate each triangle separately: the plane of one triangle can give a hit
        # behind the segment that must not mask a valid hit on the other.
        lo = max(t - _EPS_T, t0)
        hi = tc + _EPS_T
        th = math.inf
        if lo <= h1 <= hi:
            th = h1
        if lo <= h2 <= hi and h2 < th:
            th = h2
        if th < math.inf:
            return th, i, j
        t = max(t, tc) + _EPS_STEP
    return math.inf, -1, -1


if HAS_NUMBA:
    # Re-bind the scalar kernels to their compiled versions; numba resolves globals
    # at compile time, so _trace_one then calls the compiled _tri_hit.
    _tri_hit = numba.njit(cache=True)(_tri_hit)
    _trace_one = numba.njit(cache=True)(_trace_one)

    @numba.njit(cache=True, parallel=True)  # type: ignore[untyped-decorator]
    def _trace_many_nb(  # pragma: no cover - compiled
        z: FloatArray,
        pyr: FloatArray,
        off: IntArray,
        lnx: IntArray,
        lny: IntArray,
        x0: float,
        y0: float,
        res: float,
        zmin: float,
        zmax: float,
        o: FloatArray,
        d: FloatArray,
        t0: FloatArray,
        t1: FloatArray,
        t_out: FloatArray,
        i_out: IntArray,
        j_out: IntArray,
    ) -> None:
        for k in numba.prange(o.shape[0]):
            t, i, j = _trace_one(
                z, pyr, off, lnx, lny, x0, y0, res, zmin, zmax,
                o[k, 0], o[k, 1], o[k, 2], d[k, 0], d[k, 1], d[k, 2], t0[k], t1[k],
            )  # fmt: skip
            t_out[k] = t
            i_out[k] = i
            j_out[k] = j


# ----------------------------------------------------------------------------- numpy


def _tri_hit_np(
    a: tuple[FloatArray, FloatArray, FloatArray],
    b: tuple[FloatArray, FloatArray, FloatArray],
    c: tuple[FloatArray, FloatArray, FloatArray],
    o: FloatArray,
    d: FloatArray,
) -> FloatArray:
    """Vectorised Möller-Trumbore (rows of ``o``, ``d`` against triangles a, b, c)."""
    e1 = np.stack([b[0] - a[0], b[1] - a[1], b[2] - a[2]], axis=1)
    e2 = np.stack([c[0] - a[0], c[1] - a[1], c[2] - a[2]], axis=1)
    p = np.cross(d, e2)
    det = np.einsum("ij,ij->i", e1, p)
    ok = np.abs(det) >= 1e-18
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    s = o - np.stack(a, axis=1)
    u = np.einsum("ij,ij->i", s, p) * inv
    q = np.cross(s, e1)
    v = np.einsum("ij,ij->i", d, q) * inv
    t = np.einsum("ij,ij->i", e2, q) * inv
    ok &= (
        (u >= -_EPS_BARY) & (u <= 1.0 + _EPS_BARY) & (v >= -_EPS_BARY) & (u + v <= 1.0 + _EPS_BARY)
    )
    return np.where(ok, t, np.inf)


def _trace_many_np(
    z: FloatArray,
    mip: MaxMipmap,
    x0: float,
    y0: float,
    res: float,
    o: FloatArray,
    d: FloatArray,
    t0: FloatArray,
    t1: FloatArray,
) -> tuple[FloatArray, IntArray, IntArray]:
    n = o.shape[0]
    nqy, nqx = z.shape[0] - 1, z.shape[1] - 1
    zmin, zmax = float(z.min()), float(z.max())
    t_out = np.full(n, np.inf)
    i_out = np.full(n, -1, dtype=np.int64)
    j_out = np.full(n, -1, dtype=np.int64)
    u0 = (o[:, 0] - x0) / res
    v0 = (o[:, 1] - y0) / res
    du = np.where(np.abs(d[:, 0]) > _EPS_DIR, d[:, 0] / res, 0.0)
    dv = np.where(np.abs(d[:, 1]) > _EPS_DIR, d[:, 1] / res, 0.0)
    tlo = t0.astype(np.float64).copy()
    thi = t1.astype(np.float64).copy()
    valid = np.ones(n, dtype=bool)
    for p, dd, lo, hi in (
        (u0, du, 0.0, nqx),
        (v0, dv, 0.0, nqy),
        (o[:, 2], d[:, 2], zmin - _EPS_Z, zmax + _EPS_Z),
    ):
        par = np.abs(dd) < 1e-15
        valid &= ~(par & ((p < lo) | (p > hi)))
        with np.errstate(divide="ignore", invalid="ignore"):
            ta = (lo - p) / dd
            tb = (hi - p) / dd
        tmn = np.where(par, -np.inf, np.minimum(ta, tb))
        tmx = np.where(par, np.inf, np.maximum(ta, tb))
        tlo = np.maximum(tlo, tmn)
        thi = np.minimum(thi, tmx)
    active = np.flatnonzero(valid & (tlo <= thi))
    t = tlo[active]
    level = np.full(active.size, mip.levels - 1, dtype=np.int64)
    while active.size:
        s = (1 << level).astype(np.float64)
        ua, va, dua, dva = u0[active], v0[active], du[active], dv[active]
        u = ua + dua * t
        v = va + dva * t
        lnx, lny = mip.nx[level], mip.ny[level]
        j = np.clip(np.floor(u / s).astype(np.int64), 0, lnx - 1)
        i = np.clip(np.floor(v / s).astype(np.int64), 0, lny - 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            tx = np.where(
                dua > 0, ((j + 1) * s - ua) / dua, np.where(dua < 0, (j * s - ua) / dua, np.inf)
            )
            ty = np.where(
                dva > 0, ((i + 1) * s - va) / dva, np.where(dva < 0, (i * s - va) / dva, np.inf)
            )
        tc = np.minimum(np.minimum(tx, ty), thi[active])
        oz, dz = o[active, 2], d[active, 2]
        zlow = np.minimum(oz + dz * t, oz + dz * tc)
        hmax = mip.data[mip.offset[level] + i * lnx + j]
        skip = zlow > hmax + _EPS_Z
        descend = ~skip & (level > 0)
        leaf = ~skip & (level == 0)
        new_t = t.copy()
        new_level = level.copy()
        new_t[skip] = np.maximum(t[skip], tc[skip]) + _EPS_STEP
        new_level[skip] = np.minimum(level[skip] + 1, mip.levels - 1)
        new_level[descend] -= 1
        finished = np.zeros(active.size, dtype=bool)
        if leaf.any():
            li, lj = i[leaf], j[leaf]
            ids = active[leaf]
            xa = x0 + lj * res
            xb = xa + res
            ya = y0 + li * res
            yb = ya + res
            z00, z01 = z[li, lj], z[li, lj + 1]
            z10, z11 = z[li + 1, lj], z[li + 1, lj + 1]
            ol, dl = o[ids], d[ids]
            h1 = _tri_hit_np((xa, ya, z00), (xb, ya, z01), (xb, yb, z11), ol, dl)
            h2 = _tri_hit_np((xa, ya, z00), (xb, yb, z11), (xa, yb, z10), ol, dl)
            tl, tcl = t[leaf], tc[leaf]
            lo = np.maximum(tl - _EPS_T, t0[ids])
            hi = tcl + _EPS_T
            h1 = np.where((h1 >= lo) & (h1 <= hi), h1, np.inf)
            h2 = np.where((h2 >= lo) & (h2 <= hi), h2, np.inf)
            th = np.minimum(h1, h2)
            got = np.isfinite(th)
            t_out[ids[got]] = th[got]
            i_out[ids[got]] = li[got]
            j_out[ids[got]] = lj[got]
            fin_leaf = np.zeros(leaf.sum(), dtype=bool)
            fin_leaf[got] = True
            finished[np.flatnonzero(leaf)[fin_leaf]] = True
            adv = np.flatnonzero(leaf)[~got]
            new_t[adv] = np.maximum(t[adv], tc[adv]) + _EPS_STEP
        finished |= new_t > thi[active]
        keep = ~finished
        active, t, level = active[keep], new_t[keep], new_level[keep]
    return t_out, i_out, j_out


# ----------------------------------------------------------------------------- caster


class HeightfieldCaster:
    """Ray caster for one heightmap.

    Args:
        heightmap: Terrain grid.
        backend: ``"auto"`` (numba if available), ``"numba"`` or ``"numpy"``.
    """

    def __init__(self, heightmap: Heightmap, backend: str = "auto") -> None:
        if backend not in ("auto", "numba", "numpy"):
            raise ValueError(f"unknown backend {backend!r}")
        if backend == "numba" and not HAS_NUMBA:
            raise RuntimeError("numba backend requested but numba is not installed")
        self.heightmap = heightmap
        self.backend = "numba" if (backend == "auto" and HAS_NUMBA) else backend
        if self.backend == "auto":
            self.backend = "numpy"
        self.mip = MaxMipmap.build(heightmap.z)
        self._zmin = float(heightmap.z.min())
        self._zmax = float(heightmap.z.max())

    @property
    def bounds_xy(self) -> tuple[float, float, float, float]:
        """``(x_min, x_max, y_min, y_max)`` covered by the mesh [m]."""
        return self.heightmap.extent

    def intersect(
        self,
        origins: npt.ArrayLike,
        directions: npt.ArrayLike,
        t_min: npt.ArrayLike = 0.0,
        t_max: npt.ArrayLike = np.inf,
    ) -> RayHits:
        """Nearest intersection of each ray with the terrain.

        Args:
            origins: Ray origins in world ENU, shape ``(N, 3)`` [m].
            directions: Unit directions, shape ``(N, 3)``.
            t_min, t_max: Valid distance interval per ray (scalar or ``(N,)``) [m].
        """
        o = np.ascontiguousarray(np.asarray(origins, dtype=np.float64).reshape(-1, 3))
        d = np.ascontiguousarray(np.asarray(directions, dtype=np.float64).reshape(-1, 3))
        n = o.shape[0]
        t0 = np.ascontiguousarray(np.broadcast_to(np.asarray(t_min, dtype=np.float64), (n,)))
        t1 = np.ascontiguousarray(np.broadcast_to(np.asarray(t_max, dtype=np.float64), (n,)))
        hm = self.heightmap
        t: FloatArray
        qi: IntArray
        qj: IntArray
        if self.backend == "numba":
            t = np.empty(n)
            qi = np.empty(n, dtype=np.int64)
            qj = np.empty(n, dtype=np.int64)
            if n:
                _trace_many_nb(
                    hm.z, self.mip.data, self.mip.offset, self.mip.nx, self.mip.ny,
                    hm.x0, hm.y0, hm.resolution, self._zmin, self._zmax, o, d, t0, t1, t, qi, qj,
                )  # fmt: skip
        else:
            t, qi, qj = _trace_many_np(hm.z, self.mip, hm.x0, hm.y0, hm.resolution, o, d, t0, t1)
        layer = np.where(np.isfinite(t), 0, -1).astype(np.int64)
        return RayHits(t=t, qi=qi, qj=qj, layer=layer)

    def occluded(
        self,
        origins: npt.ArrayLike,
        directions: npt.ArrayLike,
        t_max: npt.ArrayLike = np.inf,
        t_min: npt.ArrayLike = 0.0,
    ) -> npt.NDArray[np.bool_]:
        """True where something blocks the ray before ``t_max`` (shadow rays)."""
        return self.intersect(origins, directions, t_min, t_max).hit
