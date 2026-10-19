"""Rocks: Golombek-Rapp exponential size model, Poisson placement, heightmap stamping.

The cumulative fractional area covered by rocks of diameter >= D is

    F_k(D) = k exp(-q D),   q(k) = 1.79 + 0.152 / k        (D in m)

(Golombek & Rapp 1997). The number of rocks per unit area per unit diameter follows
from ``-dF/dD`` divided by the footprint area of one rock (pi D^2 / 4)::

    n(D) = k q exp(-q D) / (pi D^2 / 4)

Rocks are added to the heightmap as half-ellipsoids (decision Q2 in docs/PLAN.md).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from lunarsim.config.schema import RockConfig
from lunarsim.core.rng import RngFactory
from lunarsim.core.terrain._stamp_numba import stamp_rocks_numba
from lunarsim.core.terrain.craters import CraterCatalog
from lunarsim.core.terrain.heightmap import Heightmap
from lunarsim.core.terrain.labels import Label

# Stamp with the numba kernel when available (set False to force the numpy loop).
USE_NUMBA = True

__all__ = [
    "RockCatalog",
    "background_rocks",
    "cumulative_area_fraction",
    "ejecta_rocks",
    "generate_rocks",
    "q_of_k",
    "sample_rock_diameters",
    "stamp_rocks",
]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

STREAM_BACKGROUND = "terrain/rocks/background"
STREAM_EJECTA = "terrain/rocks/ejecta"

_N_TABLE = 4096  # diameter grid for numerical inverse-CDF sampling


@dataclass(frozen=True)
class RockCatalog:
    """Ground truth of rocks.

    Attributes:
        x, y: Centre in world ENU [m].
        diameter: Major footprint diameter [m].
        height: Height above the local surface [m].
        axis_ratio: Minor / major footprint axis.
        orientation: Major-axis azimuth from East, counter-clockwise [rad].
        source_crater: Index into the crater catalog, -1 for background rocks.
    """

    x: FloatArray
    y: FloatArray
    diameter: FloatArray
    height: FloatArray
    axis_ratio: FloatArray
    orientation: FloatArray
    source_crater: IntArray

    def __len__(self) -> int:
        return int(self.x.size)

    @classmethod
    def empty(cls) -> RockCatalog:
        """Catalog with no rocks."""
        z = np.zeros(0)
        return cls(z, z, z, z, z, z, np.zeros(0, dtype=np.int64))

    @classmethod
    def concat(cls, parts: list[RockCatalog]) -> RockCatalog:
        """Concatenate catalogs."""
        parts = [p for p in parts if len(p)]
        if not parts:
            return cls.empty()
        return cls(
            *(
                np.concatenate([getattr(p, f) for p in parts])
                for f in (
                    "x",
                    "y",
                    "diameter",
                    "height",
                    "axis_ratio",
                    "orientation",
                    "source_crater",
                )
            )
        )

    def subset(self, sel: npt.ArrayLike) -> RockCatalog:
        """Catalog restricted to a boolean mask or index array."""
        m = np.asarray(sel)
        return RockCatalog(
            self.x[m],
            self.y[m],
            self.diameter[m],
            self.height[m],
            self.axis_ratio[m],
            self.orientation[m],
            self.source_crater[m],
        )

    def with_source(self, source_crater: IntArray) -> RockCatalog:
        """Copy with replaced ``source_crater`` indices."""
        return RockCatalog(
            self.x,
            self.y,
            self.diameter,
            self.height,
            self.axis_ratio,
            self.orientation,
            np.asarray(source_crater, dtype=np.int64),
        )

    def to_records(self) -> list[dict[str, float | int]]:
        """List of JSON-serialisable dicts."""
        return [
            {
                "id": i,
                "x_m": float(self.x[i]),
                "y_m": float(self.y[i]),
                "diameter_m": float(self.diameter[i]),
                "height_m": float(self.height[i]),
                "axis_ratio": float(self.axis_ratio[i]),
                "orientation_rad": float(self.orientation[i]),
                "source_crater": int(self.source_crater[i]),
            }
            for i in range(len(self))
        ]


def q_of_k(k: float) -> float:
    """Decay constant ``q(k) = 1.79 + 0.152 / k`` [1/m] (Golombek & Rapp 1997)."""
    return 1.79 + 0.152 / k


def cumulative_area_fraction(d: npt.ArrayLike, k: float) -> FloatArray:
    """``F_k(D) = k exp(-q(k) D)``: area fraction covered by rocks >= D [-]."""
    return np.asarray(k * np.exp(-q_of_k(k) * np.asarray(d, dtype=np.float64)))


def _number_table(k: float, d_min: float, d_max: float) -> tuple[FloatArray, FloatArray]:
    """Diameter grid and cumulative number density N(d_min..D) [1/m^2]."""
    d: FloatArray = np.geomspace(d_min, d_max, _N_TABLE, dtype=np.float64)
    q = q_of_k(k)
    n = k * q * np.exp(-q * d) / (0.25 * np.pi * d * d)
    cum: FloatArray = np.concatenate([[0.0], np.cumsum(0.5 * (n[1:] + n[:-1]) * np.diff(d))])
    return d, cum


def sample_rock_diameters(
    k: float, d_min: float, d_max: float, area_m2: float, rng: np.random.Generator
) -> FloatArray:
    """Poisson number of rocks on ``area_m2`` and their diameters [m] (inverse CDF)."""
    if d_max <= d_min:
        return np.zeros(0)
    d, cum = _number_table(k, d_min, d_max)
    n = int(rng.poisson(cum[-1] * area_m2))
    u = rng.random(n) * cum[-1]
    return np.asarray(np.interp(u, cum, d), dtype=np.float64)


def _shapes(
    diam: FloatArray, cfg: RockConfig, rng: np.random.Generator
) -> tuple[FloatArray, FloatArray, FloatArray]:
    n = diam.size
    axis = rng.uniform(cfg.axis_ratio_min, 1.0, n)
    orient = rng.uniform(0.0, np.pi, n)
    return cfg.height_ratio * diam, axis, orient


def ejecta_rocks(
    cfg: RockConfig,
    crater_x: float,
    crater_y: float,
    crater_d: float,
    degradation: float,
    crater_index: int,
    rng: np.random.Generator,
) -> RockCatalog:
    """Blocks around one crater (rim/proximal ejecta annulus), see EjectaRockConfig."""
    ej = cfg.ejecta
    k_e = ej.k * (1.0 - degradation)
    if k_e <= 1e-6 or crater_d < ej.min_crater_diameter_m:
        return RockCatalog.empty()
    d_max = min(cfg.d_max_m, ej.max_size_coeff * crater_d**ej.max_size_exp)
    r_in, r_out = ej.inner_radii * 0.5 * crater_d, ej.outer_radii * 0.5 * crater_d
    de = sample_rock_diameters(k_e, cfg.d_min_m, d_max, np.pi * (r_out**2 - r_in**2), rng)
    if de.size == 0:
        return RockCatalog.empty()
    r = np.sqrt(rng.uniform(r_in**2, r_out**2, de.size))
    th = rng.uniform(0.0, 2.0 * np.pi, de.size)
    he, ae, oe = _shapes(de, cfg, rng)
    return RockCatalog(
        crater_x + r * np.cos(th),
        crater_y + r * np.sin(th),
        de,
        he,
        ae,
        oe,
        np.full(de.size, crater_index, dtype=np.int64),
    )


def background_rocks(
    cfg: RockConfig,
    box: tuple[float, float, float, float],
    rng: np.random.Generator,
) -> RockCatalog:
    """Background rocks with centres uniform in ``box = (xmin, xmax, ymin, ymax)``."""
    xmin, xmax, ymin, ymax = box
    diam = sample_rock_diameters(
        cfg.k, cfg.d_min_m, cfg.d_max_m, (xmax - xmin) * (ymax - ymin), rng
    )
    x = rng.uniform(xmin, xmax, diam.size)
    y = rng.uniform(ymin, ymax, diam.size)
    h, axis, orient = _shapes(diam, cfg, rng)
    return RockCatalog(x, y, diam, h, axis, orient, np.full(diam.size, -1, dtype=np.int64))


def generate_rocks(
    cfg: RockConfig,
    extent: tuple[float, float, float, float],
    rngs: RngFactory,
    craters: CraterCatalog | None = None,
) -> RockCatalog:
    """Sample background rocks and (optionally) blocks around fresh craters."""
    if not cfg.enabled:
        return RockCatalog.empty()
    m = 0.5 * cfg.d_max_m
    xmin, xmax, ymin, ymax = extent[0] - m, extent[1] + m, extent[2] - m, extent[3] + m
    rng = rngs.generator(STREAM_BACKGROUND)
    diam = sample_rock_diameters(
        cfg.k, cfg.d_min_m, cfg.d_max_m, (xmax - xmin) * (ymax - ymin), rng
    )
    x = rng.uniform(xmin, xmax, diam.size)
    y = rng.uniform(ymin, ymax, diam.size)
    h, axis, orient = _shapes(diam, cfg, rng)
    parts = [RockCatalog(x, y, diam, h, axis, orient, np.full(diam.size, -1, dtype=np.int64))]

    ej = cfg.ejecta
    if ej.enabled and craters is not None and len(craters):
        erng = rngs.generator(STREAM_EJECTA)
        for ci in np.flatnonzero(craters.diameter >= ej.min_crater_diameter_m):
            freshness = 1.0 - float(craters.degradation[ci])
            k_e = ej.k * freshness
            if k_e <= 1e-6:
                continue
            dc = float(craters.diameter[ci])
            d_max = min(cfg.d_max_m, ej.max_size_coeff * dc**ej.max_size_exp)
            r_in, r_out = ej.inner_radii * 0.5 * dc, ej.outer_radii * 0.5 * dc
            area = np.pi * (r_out**2 - r_in**2)
            de = sample_rock_diameters(k_e, cfg.d_min_m, d_max, area, erng)
            if de.size == 0:
                continue
            # Uniform in the annulus.
            r = np.sqrt(erng.uniform(r_in**2, r_out**2, de.size))
            th = erng.uniform(0.0, 2.0 * np.pi, de.size)
            he, ae, oe = _shapes(de, cfg, erng)
            parts.append(
                RockCatalog(
                    craters.x[ci] + r * np.cos(th),
                    craters.y[ci] + r * np.sin(th),
                    de,
                    he,
                    ae,
                    oe,
                    np.full(de.size, ci, dtype=np.int64),
                )
            )
    return RockCatalog.concat(parts)


def stamp_rocks(
    hm: Heightmap, rocks: RockCatalog, labels: npt.NDArray[np.uint8] | None = None
) -> None:
    """Add half-ellipsoid rocks to ``hm.z`` in place (and label their footprints)."""
    z = hm.z
    ny, nx = z.shape
    res = hm.resolution
    if USE_NUMBA and stamp_rocks_numba is not None and len(rocks):
        lab = labels if labels is not None else np.zeros((1, 1), dtype=np.uint8)
        stamp_rocks_numba(
            z, hm.x0, hm.y0, res,
            np.ascontiguousarray(rocks.x, float), np.ascontiguousarray(rocks.y, float),
            np.ascontiguousarray(rocks.diameter, float),
            np.ascontiguousarray(rocks.axis_ratio, float),
            np.ascontiguousarray(rocks.orientation, float),
            np.ascontiguousarray(rocks.height, float), lab, labels is not None, int(Label.ROCK),
        )  # fmt: skip
        return
    for k in range(len(rocks)):
        cx, cy = float(rocks.x[k]), float(rocks.y[k])
        a = 0.5 * float(rocks.diameter[k])
        b = a * float(rocks.axis_ratio[k])
        j0 = max(0, int(np.floor((cx - a - hm.x0) / res)))
        j1 = min(nx, int(np.ceil((cx + a - hm.x0) / res)) + 1)
        i0 = max(0, int(np.floor((cy - a - hm.y0) / res)))
        i1 = min(ny, int(np.ceil((cy + a - hm.y0) / res)) + 1)
        if j0 >= j1 or i0 >= i1:
            continue
        dx = (hm.x0 + res * np.arange(j0, j1))[None, :] - cx
        dy = (hm.y0 + res * np.arange(i0, i1))[:, None] - cy
        co, si = np.cos(rocks.orientation[k]), np.sin(rocks.orientation[k])
        u = (dx * co + dy * si) / a
        v = (-dx * si + dy * co) / b
        s = 1.0 - u * u - v * v
        inside = s > 0.0
        if not inside.any():
            continue
        z[i0:i1, j0:j1] += float(rocks.height[k]) * np.sqrt(np.where(inside, s, 0.0))
        if labels is not None:
            labels[i0:i1, j0:j1][inside] = Label.ROCK
