"""Horizon maps of a heightmap and long-term illumination / PSR maps (SPEC 4.2).

Horizon map: for every (strided) grid node and ``n_azimuth`` azimuths (from North,
clockwise), the elevation angle [rad] of the local horizon, found by marching outward
over the bilinearly interpolated heightmap at geometrically growing distances
``d_0 = res, d_{k+1} = d_k (1 + growth)`` until the grid edge or ``max_distance_m``.
With ``curvature_radius_m`` the drop ``d² / (2 R)`` of a spherical Moon is applied
(leave it None if the heightmap already includes curvature, ``terrain.curvature``).
Terrain beyond the heightmap is unknown and treated as not blocking.

Illumination over time: for each time sample with Sun azimuth/elevation (same for all
nodes in the tangent-plane world) the visible fraction of the solar disc above the
horizon is the area of a disc segment: with ``t = (el_sun - h(az)) / α``,
``f = 1 - (acos t - t sqrt(1 - t²)) / π`` for |t| < 1 (horizon locally straight).
Nodes with ``f = 0`` at every sample are permanently shadowed over the span (PSR).
One draconic year (346.6 d) covers the full ±1.54° range of solar declinations seen
from the Moon; the Earth visibility uses the Earth centre direction.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import numpy.typing as npt

from lunarsim.core.illumination.sun import AU_M, SunState
from lunarsim.core.terrain.heightmap import Heightmap

try:  # optional accelerator
    import numba

    HAS_NUMBA = True
except ImportError:  # pragma: no cover
    numba = None
    HAS_NUMBA = False

__all__ = [
    "HorizonMap",
    "IlluminationMap",
    "disc_fraction_above",
    "horizon_map",
    "illumination_map",
    "sun_track",
]

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class HorizonMap:
    """Horizon elevation per node and azimuth.

    Attributes:
        elevation: Horizon elevation [rad], shape ``(ny, nx, n_azimuth)`` (float32).
        x: Node x coordinates [m], ``(nx,)``.
        y: Node y coordinates [m], ``(ny,)``.
        z: Node elevations [m], ``(ny, nx)``.
    """

    elevation: npt.NDArray[np.float32]
    x: FloatArray
    y: FloatArray
    z: FloatArray

    @property
    def n_azimuth(self) -> int:
        """Number of azimuth bins (bin k centred on azimuth 2πk/n)."""
        return int(self.elevation.shape[2])

    def at(self, azimuth: float) -> FloatArray:
        """Horizon elevation [rad] of every node towards ``azimuth`` [rad] (linear in azimuth)."""
        n = self.n_azimuth
        u = (azimuth % (2.0 * math.pi)) / (2.0 * math.pi) * n
        k0 = math.floor(u) % n
        f = u - math.floor(u)
        e = self.elevation
        return np.asarray((1.0 - f) * e[:, :, k0] + f * e[:, :, (k0 + 1) % n], dtype=np.float64)


def _distances(res: float, max_d: float, growth: float) -> FloatArray:
    d = [res]
    while d[-1] * (1.0 + growth) <= max_d:
        d.append(max(d[-1] * (1.0 + growth), d[-1] + res))
    return np.asarray(d, dtype=np.float64)


def _horizon_numpy(
    z: FloatArray,
    res: float,
    rows: npt.NDArray[np.int64],
    cols: npt.NDArray[np.int64],
    saz: FloatArray,
    caz: FloatArray,
    dist: FloatArray,
    inv2r: float,
) -> npt.NDArray[np.float32]:
    ny, nx = z.shape
    zmax = float(z.max())
    R, C = np.meshgrid(rows, cols, indexing="ij")
    R, C = R.ravel().astype(np.float64), C.ravel().astype(np.float64)
    h0 = z[R.astype(np.int64), C.astype(np.int64)]
    out = np.empty((R.size, saz.size), dtype=np.float64)
    for a in range(saz.size):
        best = np.full(R.size, -1e300)  # max slope; atan once at the end
        alive = np.ones(R.size, dtype=bool)
        for d in dist:
            u = C + d * saz[a] / res
            v = R + d * caz[a] / res
            alive &= (u >= 0.0) & (u <= nx - 1) & (v >= 0.0) & (v <= ny - 1)
            alive &= zmax - h0 > best * d  # nothing farther can rise above
            if not alive.any():
                break
            idx = np.flatnonzero(alive)
            uu, vv = u[idx], v[idx]
            j0 = np.minimum(np.floor(uu).astype(np.int64), nx - 2)
            i0 = np.minimum(np.floor(vv).astype(np.int64), ny - 2)
            fu, fv = uu - j0, vv - i0
            h = (
                (1 - fu) * (1 - fv) * z[i0, j0]
                + fu * (1 - fv) * z[i0, j0 + 1]
                + (1 - fu) * fv * z[i0 + 1, j0]
                + fu * fv * z[i0 + 1, j0 + 1]
            )
            sl = (h - h0[idx]) / d - d * inv2r
            best[idx] = np.maximum(best[idx], sl)
        out[:, a] = np.where(best > -1e299, np.arctan(best), -0.5 * math.pi)
    return out.reshape(rows.size, cols.size, saz.size).astype(np.float32)


def _horizon_kernel(
    z: FloatArray,
    res: float,
    rows: npt.NDArray[np.int64],
    cols: npt.NDArray[np.int64],
    saz: FloatArray,
    caz: FloatArray,
    dist: FloatArray,
    inv2r: float,
    out: npt.NDArray[np.float32],
) -> None:  # pragma: no cover - compiled by numba
    ny, nx = z.shape
    zmax = z.max()
    for ii in numba.prange(rows.size):
        r = float(rows[ii])
        for jj in range(cols.size):
            c = float(cols[jj])
            h0 = z[rows[ii], cols[jj]]
            for a in range(saz.size):
                best = -1e300  # max slope (tan of the elevation): atan only once at the end
                for d in dist:
                    u = c + d * saz[a] / res
                    v = r + d * caz[a] / res
                    if u < 0.0 or u > nx - 1 or v < 0.0 or v > ny - 1:
                        break
                    if zmax - h0 <= best * d:  # nothing farther can rise above
                        break
                    j0 = min(math.floor(u), nx - 2)
                    i0 = min(math.floor(v), ny - 2)
                    fu = u - j0
                    fv = v - i0
                    h = (
                        (1 - fu) * (1 - fv) * z[i0, j0]
                        + fu * (1 - fv) * z[i0, j0 + 1]
                        + (1 - fu) * fv * z[i0 + 1, j0]
                        + fu * fv * z[i0 + 1, j0 + 1]
                    )
                    sl = (h - h0) / d - d * inv2r
                    if sl > best:
                        best = sl
                out[ii, jj, a] = math.atan(best) if best > -1e299 else -0.5 * math.pi


_kernel = numba.njit(parallel=True, cache=False)(_horizon_kernel) if HAS_NUMBA else None


def horizon_map(
    hm: Heightmap,
    n_azimuth: int = 64,
    stride: int = 1,
    max_distance_m: float | None = None,
    growth: float = 0.02,
    curvature_radius_m: float | None = None,
    backend: str = "auto",
) -> HorizonMap:
    """Horizon elevation [rad] at every ``stride``-th node for ``n_azimuth`` azimuths."""
    if n_azimuth < 4 or stride < 1:
        raise ValueError("n_azimuth must be >= 4 and stride >= 1")
    z = np.ascontiguousarray(hm.z, dtype=np.float64)
    ny, nx = z.shape
    res = hm.resolution
    xmin, xmax, ymin, ymax = hm.extent
    max_d = max_distance_m if max_distance_m is not None else math.hypot(xmax - xmin, ymax - ymin)
    dist = _distances(res, max_d, growth)
    az = 2.0 * math.pi * np.arange(n_azimuth) / n_azimuth
    saz, caz = np.sin(az), np.cos(az)  # azimuth from North: (East, North) = (sin, cos)
    rows = np.arange(0, ny, stride, dtype=np.int64)
    cols = np.arange(0, nx, stride, dtype=np.int64)
    inv2r = 0.0 if curvature_radius_m is None else 0.5 / curvature_radius_m
    use_numba = HAS_NUMBA if backend == "auto" else backend == "numba"
    if use_numba:
        if _kernel is None:
            raise RuntimeError("numba backend requested but numba is not installed")
        out: npt.NDArray[np.float32] = np.empty((rows.size, cols.size, n_azimuth), np.float32)
        _kernel(z, res, rows, cols, saz, caz, dist, inv2r, out)
    else:
        out = _horizon_numpy(z, res, rows, cols, saz, caz, dist, inv2r)
    return HorizonMap(out, hm.x[cols], hm.y[rows], z[np.ix_(rows, cols)])


def disc_fraction_above(t: npt.ArrayLike) -> FloatArray:
    """Area fraction of a unit disc above a line at signed offset ``t`` (radii) below its centre."""
    tt = np.clip(np.asarray(t, dtype=np.float64), -1.0, 1.0)
    return np.asarray(1.0 - (np.arccos(tt) - tt * np.sqrt(1.0 - tt * tt)) / math.pi)


@dataclass(frozen=True)
class IlluminationMap:
    """Long-term illumination statistics on a horizon-map grid.

    Attributes:
        x, y: Node coordinates [m].
        mean_sun_fraction: Time-averaged visible fraction of the solar disc.
        max_sun_fraction: Maximum visible fraction over the span.
        max_sun_elevation_above_horizon_deg: max over time of (el_sun - horizon) [deg].
        earth_visible_fraction: Fraction of samples with the Earth centre above the horizon
            (nan without Earth geometry).
        n_samples: Number of time samples.
    """

    x: FloatArray
    y: FloatArray
    mean_sun_fraction: FloatArray
    max_sun_fraction: FloatArray
    max_sun_elevation_above_horizon_deg: FloatArray
    earth_visible_fraction: FloatArray
    n_samples: int

    @property
    def psr(self) -> npt.NDArray[np.bool_]:
        """Permanently shadowed nodes (no part of the solar disc ever visible)."""
        return np.asarray(self.max_sun_fraction <= 0.0)


def sun_track(
    sun_at: Callable[[datetime], SunState], start: datetime, end: datetime, step: timedelta
) -> list[SunState]:
    """Sun states from ``start`` to ``end`` (inclusive) every ``step``."""
    out, t = [], start
    while t <= end:
        out.append(sun_at(t))
        t += step
    return out


def _illum_kernel(
    elev: npt.NDArray[np.float32],
    sun_u: FloatArray,
    sun_el: FloatArray,
    sun_alpha: FloatArray,
    earth_u: FloatArray,
    earth_el: FloatArray,
    mean: FloatArray,
    mx: FloatArray,
    mel: FloatArray,
    earth: FloatArray,
) -> None:  # pragma: no cover - compiled by numba
    ny, nx, n = elev.shape
    for i in numba.prange(ny):
        for j in range(nx):
            acc = 0.0
            best = 0.0
            bel = -0.5 * math.pi
            ecount = 0.0
            for k in range(sun_u.size):
                u = sun_u[k]
                k0 = math.floor(u) % n
                f = u - math.floor(u)
                h = (1.0 - f) * elev[i, j, k0] + f * elev[i, j, (k0 + 1) % n]
                dl = sun_el[k] - h
                t = dl / sun_alpha[k]
                if t >= 1.0:
                    fr = 1.0
                elif t <= -1.0:
                    fr = 0.0
                else:
                    fr = 1.0 - (math.acos(t) - t * math.sqrt(1.0 - t * t)) / math.pi
                acc += fr
                best = max(best, fr)
                bel = max(bel, dl)
                if earth_u.size:
                    ue = earth_u[k]
                    e0 = math.floor(ue) % n
                    fe = ue - math.floor(ue)
                    he = (1.0 - fe) * elev[i, j, e0] + fe * elev[i, j, (e0 + 1) % n]
                    if earth_el[k] > he:
                        ecount += 1.0
            mean[i, j] = acc
            mx[i, j] = best
            mel[i, j] = bel
            earth[i, j] = ecount


_illum = numba.njit(parallel=True, cache=False)(_illum_kernel) if HAS_NUMBA else None


def illumination_map(
    hor: HorizonMap,
    suns: Sequence[SunState],
    sun_radius_m: float = 6.957e8,
    backend: str = "auto",
) -> IlluminationMap:
    """Accumulate sun/earth visibility over the given sun states."""
    shape = hor.z.shape
    use_numba = HAS_NUMBA if backend == "auto" else backend == "numba"
    if use_numba and _illum is not None and len(suns):
        n = hor.n_azimuth
        two_pi = 2.0 * math.pi
        su = np.array([(s.azimuth % two_pi) / two_pi * n for s in suns])
        sel = np.array([s.elevation for s in suns])
        sal = np.array([math.asin(sun_radius_m / (s.distance_au * AU_M)) for s in suns])
        has_e = all(s.earth_direction is not None for s in suns)
        if has_e:
            ed = np.array([s.earth_direction for s in suns])
            eu = (np.arctan2(ed[:, 0], ed[:, 1]) % two_pi) / two_pi * n
            eel = np.arcsin(np.clip(ed[:, 2], -1.0, 1.0))
        else:
            eu = eel = np.zeros(0)
        out = [np.zeros(shape) for _ in range(4)]
        _illum(hor.elevation, su, sel, sal, eu, eel, *out)
        k = len(suns)
        return IlluminationMap(
            hor.x, hor.y, out[0] / k, out[1], np.degrees(out[2]),
            out[3] / k if has_e else np.full(shape, np.nan), k,
        )  # fmt: skip
    mean = np.zeros(shape)
    mx = np.zeros(shape)
    mel = np.full(shape, -90.0)
    earth = np.zeros(shape)
    has_earth = True
    for s in suns:
        h = hor.at(s.azimuth)
        alpha = math.asin(sun_radius_m / (s.distance_au * AU_M))
        f = disc_fraction_above((s.elevation - h) / alpha)
        mean += f
        np.maximum(mx, f, out=mx)
        np.maximum(mel, np.degrees(s.elevation - h), out=mel)
        if s.earth_direction is None:
            has_earth = False
        elif has_earth:
            e = s.earth_direction
            eaz = math.atan2(e[0], e[1])
            eel = math.asin(max(-1.0, min(1.0, float(e[2]))))
            earth += eel > hor.at(eaz)
    n = max(len(suns), 1)
    return IlluminationMap(
        hor.x,
        hor.y,
        mean / n,
        mx,
        mel,
        earth / n if has_earth else np.full(shape, np.nan),
        len(suns),
    )
