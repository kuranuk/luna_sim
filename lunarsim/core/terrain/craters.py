"""Stochastic crater population: SFD sampling, placement, shape, degradation, stamping.

Pipeline (see :func:`generate_craters` / :func:`stamp_craters`):

1. Number of craters ~ Poisson(A * N(>D_min)) over the terrain area enlarged by the
   ejecta reach of the largest crater (craters centred just outside still matter).
2. Diameters by inverse-transform sampling of the truncated power law.
3. Centres: homogeneous Poisson process, optionally a clustered fraction
   (Neyman-Scott, secondary-crater-like clusters).
4. Degradation state ``g ~ Beta(a, b)``; ``g`` doubles as relative age, so craters
   are stamped oldest (largest g) first and younger craters overprint older ones.
5. Each crater replaces the terrain inside its rim by a fresh bowl referenced to a
   plane fitted on the surrounding ring and adds rim/ejecta outside (see
   :func:`radial_profile`).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import numpy.typing as npt
from scipy.special import i0e

from lunarsim.config.schema import CraterConfig, CraterSFDConfig, CraterShapeConfig
from lunarsim.core.rng import RngFactory
from lunarsim.core.terrain._stamp_numba import stamp_craters_numba
from lunarsim.core.terrain.heightmap import Heightmap
from lunarsim.core.terrain.labels import Label

__all__ = [
    "CraterCatalog",
    "degraded_profile_table",
    "expected_count",
    "generate_craters",
    "radial_profile",
    "sample_diameters",
    "sfd_cdf",
    "stamp_craters",
]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

STREAM_COUNT = "terrain/craters/count"
STREAM_SIZE = "terrain/craters/size"
STREAM_POSITION = "terrain/craters/position"
STREAM_CLUSTER = "terrain/craters/cluster"
STREAM_DEGRADATION = "terrain/craters/degradation"

# Interior replacement: weight 1 for r <= 0.9 R, falling linearly to 0 at 1.1 R.
_REPLACE_IN, _REPLACE_OUT = 0.9, 1.1
# Ring used to fit the local pre-impact reference plane [crater radii].
_RING_IN, _RING_OUT = 1.1, 1.6
# Label bands [crater radii].
_RIM_IN, _RIM_OUT = 0.85, 1.15
# Stamp with the numba kernel when available (set False to force the numpy loop).
USE_NUMBA = True


@dataclass(frozen=True)
class CraterCatalog:
    """Ground truth of the crater population (as formed, before overprinting).

    Arrays are ordered by age, oldest first (= stamping order).

    Attributes:
        x, y: Centre in world ENU [m].
        diameter: Rim-crest diameter D [m].
        depth: Rim crest to floor depth d [m]. Exact in the stamped terrain (before
            overprinting by younger craters/rocks), also for smoothed degraded craters.
        rim_height: Nominal rim crest height above the local pre-impact surface h [m].
            For degraded craters the smoothing lowers the actual rim below this value
            (and deepens the floor correspondingly, keeping d).
        degradation: g in [0, 1] (0 fresh, 1 most degraded).
        cluster_id: -1 for background craters, else the cluster index.
        uid: Optional stable string ids (two-layer terrain); default: age rank.
    """

    x: FloatArray
    y: FloatArray
    diameter: FloatArray
    depth: FloatArray
    rim_height: FloatArray
    degradation: FloatArray
    cluster_id: IntArray
    uid: npt.NDArray[np.str_] | None = None

    def __len__(self) -> int:
        return int(self.x.size)

    @classmethod
    def empty(cls) -> CraterCatalog:
        """Catalog with no craters."""
        z = np.zeros(0)
        return cls(z, z, z, z, z, z, np.zeros(0, dtype=np.int64))

    def subset(self, idx: npt.ArrayLike) -> CraterCatalog:
        """Catalog restricted to (and ordered by) ``idx``."""
        ix = np.asarray(idx, dtype=np.intp)
        return CraterCatalog(
            self.x[ix], self.y[ix], self.diameter[ix], self.depth[ix], self.rim_height[ix],
            self.degradation[ix], self.cluster_id[ix], None if self.uid is None else self.uid[ix],
        )  # fmt: skip

    @classmethod
    def concat(cls, parts: list[CraterCatalog]) -> CraterCatalog:
        """Concatenate catalogs (uids required on all parts or none)."""
        parts = [p for p in parts if len(p)]
        if not parts:
            return cls.empty()
        fields = ("x", "y", "diameter", "depth", "rim_height", "degradation", "cluster_id")
        arrays = [np.concatenate([getattr(p, f) for p in parts]) for f in fields]
        uids = None
        if all(p.uid is not None for p in parts):
            uids = np.concatenate([p.uid for p in parts if p.uid is not None])
        x, y, d, dep, rim, g, cid = arrays
        return cls(x, y, d, dep, rim, g, cid, uids)

    def to_records(self) -> list[dict[str, float | int | str]]:
        """List of JSON-serialisable dicts (``id`` = age rank, 0 = oldest)."""
        return [
            {
                "id": i if self.uid is None else str(self.uid[i]),
                "x_m": float(self.x[i]),
                "y_m": float(self.y[i]),
                "diameter_m": float(self.diameter[i]),
                "depth_m": float(self.depth[i]),
                "rim_height_m": float(self.rim_height[i]),
                "depth_to_diameter": float(self.depth[i] / self.diameter[i]),
                "degradation": float(self.degradation[i]),
                "cluster_id": int(self.cluster_id[i]),
            }
            for i in range(len(self))
        ]


# --------------------------------------------------------------------------- SFD


def expected_count(sfd: CraterSFDConfig, area_m2: float) -> float:
    """Expected number of craters with ``D_min <= D <= D_max`` on ``area_m2``."""
    c, b = sfd.coefficient, sfd.exponent
    return float(area_m2 * c * (sfd.d_min_m**-b - sfd.d_max_m**-b))


def sfd_cdf(d: npt.ArrayLike, sfd: CraterSFDConfig) -> FloatArray:
    """CDF of the truncated power-law diameter distribution, ``P(D' <= d)``."""
    b = sfd.exponent
    lo, hi = sfd.d_min_m**-b, sfd.d_max_m**-b
    da = np.clip(np.asarray(d, dtype=np.float64), sfd.d_min_m, sfd.d_max_m)
    return np.asarray((lo - da**-b) / (lo - hi), dtype=np.float64)


def sample_diameters(n: int, sfd: CraterSFDConfig, rng: np.random.Generator) -> FloatArray:
    """Inverse-transform sampling of ``n`` diameters [m] from the truncated power law."""
    b = sfd.exponent
    lo, hi = sfd.d_min_m**-b, sfd.d_max_m**-b
    u = rng.random(n)
    return np.asarray((lo - u * (lo - hi)) ** (-1.0 / b), dtype=np.float64)


# ------------------------------------------------------------------------- shape


def radial_profile(
    rho: npt.ArrayLike, floor_depth: float, rim_height: float, shape: CraterShapeConfig
) -> FloatArray:
    """Elevation change [m] of a fresh crater vs normalised radius ``rho = r / R``.

    * ``rho < 1``: parabolic bowl from ``-floor_depth`` (centre) to ``+rim_height`` (rim).
    * ``1 <= rho < rho_max``: rim/ejecta ``h (rho^-n - rho_max^-n) / (1 - rho_max^-n)``,
      i.e. the McGetchin-type ``rho^-n`` decay tapered to reach 0 at ``rho_max``.
    """
    r = np.asarray(rho, dtype=np.float64)
    n = shape.ejecta_exponent
    rmax = shape.ejecta_extent_radii
    inner = -floor_depth + (floor_depth + rim_height) * r * r
    tail = rmax**-n
    with np.errstate(divide="ignore"):
        outer = rim_height * (np.maximum(r, 1.0) ** -n - tail) / (1.0 - tail)
    out = np.where(r < 1.0, inner, np.where(r < rmax, outer, 0.0))
    return np.asarray(out, dtype=np.float64)


def _shape_params(
    diameter: FloatArray, g: FloatArray, shape: CraterShapeConfig
) -> tuple[FloatArray, FloatArray]:
    """Depth d and rim height h [m] interpolated linearly in g (fresh -> degraded)."""
    dd = shape.depth_ratio_fresh + g * (shape.depth_ratio_degraded - shape.depth_ratio_fresh)
    depth = dd * diameter
    # Rim height shrinks in proportion to depth (keeps h/d of fresh craters).
    rim = depth * (shape.rim_ratio_fresh / shape.depth_ratio_fresh)
    return depth, rim


_TABLE_NG = 33  # degradation bins
_TABLE_NR = 513  # radial samples over [0, rho_max]


@lru_cache(maxsize=16)
def degraded_profile_table(shape: CraterShapeConfig) -> tuple[FloatArray, FloatArray]:
    """Radial profiles of craters vs degradation, in units of the crater radius R.

    Returns ``(rho, table)`` with ``table[i]`` the elevation change / R at ``rho`` for
    ``g = i / (_TABLE_NG - 1)``. Degradation smooths the fresh profile with an
    isotropic 2-D Gaussian of ``sigma = smoothing_radii_degraded * g`` (in R). For a
    radially symmetric f the 2-D convolution is the 1-D integral

        F(rho) = ∫ f(r) exp(-(rho^2 + r^2) / 2s^2) I0(rho r / s^2) r dr / s^2,

    evaluated with the trapezoid rule (``i0e`` for numerical stability). The result
    is then rescaled so that rim crest to floor equals the target depth d(g), and set
    to 0 beyond ``rho_max``. Independent of the grid resolution.
    """
    rmax = shape.ejecta_extent_radii
    rho: FloatArray = np.linspace(0.0, rmax, _TABLE_NR, dtype=np.float64)
    w = np.full(_TABLE_NR, rho[1] - rho[0])
    w[0] *= 0.5
    w[-1] *= 0.5
    g_grid: FloatArray = np.linspace(0.0, 1.0, _TABLE_NG, dtype=np.float64)
    depth_r, rim_r = _shape_params(np.full(_TABLE_NG, 2.0), g_grid, shape)  # D = 2 R
    table: FloatArray = np.empty((_TABLE_NG, _TABLE_NR))
    inside = rho < 1.0
    near = rho <= 1.2
    for i, g in enumerate(g_grid):
        f = radial_profile(rho, depth_r[i] - rim_r[i], rim_r[i], shape)
        s = shape.smoothing_radii_degraded * g
        if s > 0.0:
            a = rho[:, None] * rho[None, :] / (s * s)
            k = np.exp(-((rho[:, None] - rho[None, :]) ** 2) / (2 * s * s)) * i0e(a) / (s * s)
            f = (k * (rho * w)[None, :]) @ f
            span = float(f[near].max() - f[inside].min())
            f *= depth_r[i] / span
        f[rho >= rmax] = 0.0
        table[i] = f
    rho.setflags(write=False)
    table.setflags(write=False)
    return rho, table


def _profile_for(shape: CraterShapeConfig, g: float) -> tuple[FloatArray, FloatArray]:
    """Profile / R for degradation g (linear interpolation between table rows)."""
    rho, table = degraded_profile_table(shape)
    t = min(max(g, 0.0), 1.0) * (_TABLE_NG - 1)
    i = min(int(t), _TABLE_NG - 2)
    f = t - i
    return rho, (1.0 - f) * table[i] + f * table[i + 1]


# ----------------------------------------------------------------------- sampling


def generate_craters(
    cfg: CraterConfig, extent: tuple[float, float, float, float], rngs: RngFactory
) -> CraterCatalog:
    """Sample the crater population for a terrain with the given cell-centre ``extent``."""
    if not cfg.enabled:
        return CraterCatalog.empty()
    sfd = cfg.sfd
    margin = 0.5 * sfd.d_max_m * cfg.shape.ejecta_extent_radii
    xmin, xmax = extent[0] - margin, extent[1] + margin
    ymin, ymax = extent[2] - margin, extent[3] + margin
    area = (xmax - xmin) * (ymax - ymin)
    n = int(rngs.generator(STREAM_COUNT).poisson(expected_count(sfd, area)))
    if n == 0:
        return CraterCatalog.empty()

    diameter = sample_diameters(n, sfd, rngs.generator(STREAM_SIZE))
    pos_rng = rngs.generator(STREAM_POSITION)
    x = pos_rng.uniform(xmin, xmax, n)
    y = pos_rng.uniform(ymin, ymax, n)
    g = rngs.generator(STREAM_DEGRADATION).beta(cfg.degradation.beta_a, cfg.degradation.beta_b, n)
    cluster_id: IntArray = np.full(n, -1, dtype=np.int64)

    cl = cfg.clustering
    n_clustered = round(cl.fraction * n)
    if n_clustered > 0:
        crng = rngs.generator(STREAM_CLUSTER)
        n_parents = max(1, round(n_clustered / cl.mean_cluster_size))
        px = crng.uniform(xmin, xmax, n_parents)
        py = crng.uniform(ymin, ymax, n_parents)
        pg = crng.beta(cfg.degradation.beta_a, cfg.degradation.beta_b, n_parents)
        members = crng.choice(n, size=n_clustered, replace=False)
        parent = crng.integers(0, n_parents, n_clustered)
        x[members] = np.clip(px[parent] + crng.normal(0.0, cl.sigma_m, n_clustered), xmin, xmax)
        y[members] = np.clip(py[parent] + crng.normal(0.0, cl.sigma_m, n_clustered), ymin, ymax)
        # Same impact event -> same age; tiny jitter keeps the ordering strict.
        g[members] = np.clip(pg[parent] + crng.normal(0.0, 1e-3, n_clustered), 0.0, 1.0)
        cluster_id[members] = parent

    order = np.argsort(-g, kind="stable")  # oldest (most degraded) first
    x, y, diameter, g, cluster_id = x[order], y[order], diameter[order], g[order], cluster_id[order]
    depth, rim = _shape_params(diameter, g, cfg.shape)
    return CraterCatalog(x, y, diameter, depth, rim, g, cluster_id)


# ----------------------------------------------------------------------- stamping


def _fit_plane(xs: FloatArray, ys: FloatArray, zs: FloatArray) -> tuple[float, float, float] | None:
    """Least-squares plane ``z = a + b x + c y`` (None if under-determined)."""
    if zs.size < 3:
        return None
    A = np.stack([np.ones_like(xs), xs, ys], axis=1)
    coef, _, rank, _ = np.linalg.lstsq(A, zs, rcond=None)
    if rank < 3:
        return None
    return float(coef[0]), float(coef[1]), float(coef[2])


def stamp_craters(
    hm: Heightmap,
    catalog: CraterCatalog,
    shape: CraterShapeConfig,
    labels: npt.NDArray[np.uint8] | None = None,
    label_min_diameter_m: float | None = None,
    planes: FloatArray | None = None,
    out_planes: FloatArray | None = None,
) -> None:
    """Stamp craters into ``hm.z`` in place, in catalog (age) order.

    If ``labels`` is given, crater interior/rim labels are written for craters with
    ``D >= label_min_diameter_m`` (default 4 cells); younger craters overwrite older.

    ``planes`` (N, 3), if given, supplies the pre-impact reference plane
    ``z = a + b dx + c dy`` (relative to the crater centre) for rows that are finite;
    other rows are fitted on the current terrain. ``out_planes`` (N, 3) receives the
    planes actually used.
    """
    z = hm.z
    ny, nx = z.shape
    res = hm.resolution
    lab_min = 4.0 * res if label_min_diameter_m is None else label_min_diameter_m
    rmax_rel = shape.ejecta_extent_radii
    if USE_NUMBA and stamp_craters_numba is not None and len(catalog):
        n = len(catalog)
        p_in = np.full((n, 3), np.nan) if planes is None else np.ascontiguousarray(planes, float)
        p_out = np.empty((n, 3))
        lab = labels if labels is not None else np.zeros((1, 1), dtype=np.uint8)
        _, table = degraded_profile_table(shape)
        stamp_craters_numba(
            z, hm.x0, hm.y0, res,
            np.ascontiguousarray(catalog.x, float), np.ascontiguousarray(catalog.y, float),
            0.5 * np.asarray(catalog.diameter, float),
            np.ascontiguousarray(catalog.degradation, float),
            np.ascontiguousarray(table), rmax_rel, p_in, p_out, lab, labels is not None,
            lab_min, _RING_IN, _RING_OUT, _REPLACE_IN, _REPLACE_OUT, _RIM_IN, _RIM_OUT,
            int(Label.CRATER_INTERIOR), int(Label.CRATER_RIM),
        )  # fmt: skip
        if out_planes is not None:
            out_planes[...] = p_out
        return
    for k in range(len(catalog)):
        cx, cy = float(catalog.x[k]), float(catalog.y[k])
        R = 0.5 * float(catalog.diameter[k])
        reach = rmax_rel * R
        j0 = max(0, int(np.floor((cx - reach - hm.x0) / res)))
        j1 = min(nx, int(np.ceil((cx + reach - hm.x0) / res)) + 1)
        i0 = max(0, int(np.floor((cy - reach - hm.y0) / res)))
        i1 = min(ny, int(np.ceil((cy + reach - hm.y0) / res)) + 1)
        if j0 >= j1 or i0 >= i1:
            continue
        dx = (hm.x0 + res * np.arange(j0, j1))[None, :] - cx
        dy = (hm.y0 + res * np.arange(i0, i1))[:, None] - cy
        rho = np.hypot(dx, dy) / R
        patch = z[i0:i1, j0:j1]
        prof_rho, prof = _profile_for(shape, float(catalog.degradation[k]))
        delta = R * np.interp(rho, prof_rho, prof, right=0.0)

        ring = (rho >= _RING_IN) & (rho <= _RING_OUT)
        dxb = np.broadcast_to(dx, rho.shape)
        dyb = np.broadcast_to(dy, rho.shape)
        # Fit a plane only when most of the ring lies on the grid; a partial ring
        # (crater at the border) can make the fit extrapolate wildly.
        if planes is not None and np.isfinite(planes[k]).all():
            a, b, c = (float(v) for v in planes[k])
        else:
            n_ring_full = np.pi * (_RING_OUT**2 - _RING_IN**2) * (R / res) ** 2
            n_ring = int(ring.sum())
            plane = (
                _fit_plane(dxb[ring], dyb[ring], patch[ring])
                if n_ring >= 0.5 * n_ring_full
                else None
            )
            if plane is None:
                level = float(patch[ring].mean()) if n_ring > 0 else float(patch.mean())
                a, b, c = level, 0.0, 0.0
            else:
                a, b, c = plane
        if out_planes is not None:
            out_planes[k] = (a, b, c)
        ref = a + b * dx + c * dy
        w = np.clip((_REPLACE_OUT - rho) / (_REPLACE_OUT - _REPLACE_IN), 0.0, 1.0)
        patch += delta + w * (ref - patch)

        if labels is not None and 2.0 * R >= lab_min:
            lp = labels[i0:i1, j0:j1]
            lp[rho < _RIM_IN] = Label.CRATER_INTERIOR
            lp[(rho >= _RIM_IN) & (rho < _RIM_OUT)] = Label.CRATER_RIM
