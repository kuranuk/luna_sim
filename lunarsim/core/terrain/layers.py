"""Two-layer terrain: coarse far heightmap + fine near patch (decision Q3).

Far layer (``terrain.resolution_m`` over ``terrain.size_*``):
    fractal limited to the far band (cos² taper to zero between 0.5 and 1 x the
    coarse Nyquist frequency k_c) + tilt + *large* craters (D >= D_split = 2 coarse
    cells), generated over the whole terrain.

Near patch (``terrain.near``):
    cubic-spline upsampling of the far layer *before craters*
    + fine fractal detail carrying the complementary band (1 - taper² below k_c,
      full spectrum above, up to the fine Nyquist frequency)
    + large craters intersecting the patch and *small* craters (D < D_split)
    + rocks, all stamped at the fine resolution in age order.

Fine detail, small craters and background rocks are generated per tile of a
global tile lattice (``near.tile_cells`` fine cells, anchored at the world origin)
from tile-indexed random streams, so any patch position yields the same terrain
where patches overlap. Neighbouring detail tiles overlap by ``2 * overlap_cells``
and are blended with cos/sin weights (w_a² + w_b² = 1 preserves the variance of
independent fields).

Amplitude: the PSD constant C is set analytically so that the *expected* RMS slope
at ``fractal.baseline_m`` of far band + detail band equals the target. The far
realisation is then rescaled so that its increment variance at the nearest coarse
lag equals its ensemble expectation, removing most of the low-frequency
realisation scatter (the fine detail has many modes and little scatter).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import scipy.fft as sfft
from scipy.ndimage import map_coordinates

from lunarsim.config.schema import TerrainConfig
from lunarsim.core.rng import RngFactory
from lunarsim.core.terrain.craters import (
    CraterCatalog,
    _shape_params,
    sample_diameters,
    stamp_craters,
)
from lunarsim.core.terrain.dem import BaseSurface
from lunarsim.core.terrain.fractal import (
    fractal_psd,
    increment_variance,
    rms_slope_at_baseline,
    spectral_amplitude,
)
from lunarsim.core.terrain.generate import Terrain
from lunarsim.core.terrain.heightmap import Heightmap
from lunarsim.core.terrain.labels import Label
from lunarsim.core.terrain.rocks import RockCatalog, background_rocks, ejecta_rocks, stamp_rocks

__all__ = ["TwoLayerTerrain"]

FloatArray = npt.NDArray[np.float64]


def _taper2(k: FloatArray, k_c: float) -> FloatArray:
    """Squared far-band weight: 1 below k_c/2, cos² ramp to 0 at k_c."""
    x = np.clip((k - 0.5 * k_c) / (0.5 * k_c), 0.0, 1.0)
    return np.asarray(np.cos(0.5 * np.pi * x) ** 2, dtype=np.float64)


def _tile_weight(m_total: int, overlap: int) -> FloatArray:
    """1-D blend weights of a tile of ``m_total = T + 2 O`` cells."""
    w = np.ones(m_total)
    ramp = 0.5 * np.pi * (np.arange(2 * overlap) + 0.5) / (2 * overlap)
    w[: 2 * overlap] = np.sin(ramp)
    w[m_total - 2 * overlap :] = np.cos(ramp)
    return w


def _tile_range(lo: float, hi: float, margin: float, size: float) -> range:
    """Indices of tiles of edge ``size`` overlapping ``[lo - margin, hi + margin]``."""
    return range(math.floor((lo - margin) / size), math.floor((hi + margin) / size) + 1)


def _overlaps(
    x: FloatArray, y: FloatArray, radius: FloatArray, ext: tuple[float, float, float, float]
) -> npt.NDArray[np.bool_]:
    """True where the disc (x, y, radius) intersects the box ``ext``."""
    inside_x = (x + radius >= ext[0]) & (x - radius <= ext[1])
    return np.asarray(inside_x & (y + radius >= ext[2]) & (y - radius <= ext[3]))


@dataclass
class TwoLayerTerrain:
    """Builder/holder of the far layer and a (re-centrable) near patch."""

    cfg: TerrainConfig
    rngs: RngFactory
    workers: int | None = None
    base: BaseSurface | None = None
    far: Terrain = field(init=False)
    near: Terrain = field(init=False)
    psd_constant: float = field(init=False)
    _far_base: Heightmap = field(init=False, repr=False)
    _large: CraterCatalog = field(init=False, repr=False)
    _large_planes: FloatArray = field(init=False, repr=False)
    _detail_cache: dict[tuple[int, int], FloatArray] = field(
        init=False, repr=False, default_factory=dict
    )
    _psd_cache: dict[str, FloatArray | None] = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self) -> None:
        if not self.cfg.near.enabled:
            raise ValueError("TwoLayerTerrain requires terrain.near.enabled = true")
        self._build_far()
        nr = self.cfg.near
        self.near = self.build_near(nr.center_x_m, nr.center_y_m)

    # ------------------------------------------------------------------ geometry
    @property
    def d_split(self) -> float:
        """Craters at least this large are stamped on the far layer [m]."""
        return 2.0 * self.cfg.resolution_m

    @property
    def k_c(self) -> float:
        """Coarse Nyquist frequency [cycles/m]."""
        return 0.5 / self.cfg.resolution_m

    def _tile_shape(self) -> tuple[int, int]:
        nr = self.cfg.near
        m = nr.tile_cells + 2 * nr.overlap_cells
        return m, m

    # ------------------------------------------------------------------ spectra
    def _detail_psd_unit(self) -> FloatArray:
        if self._psd_cache.get("unit") is None:
            self._psd_cache["unit"] = self._detail_psd_unit_compute()
        return self._psd_cache["unit"]

    def _detail_psd_unit_compute(self) -> FloatArray:
        m, _ = self._tile_shape()
        r = self.cfg.near.resolution_m
        ky = sfft.fftfreq(m, d=r)
        kx = sfft.rfftfreq(m, d=r)
        k2 = ky[:, None] ** 2 + kx[None, :] ** 2
        return fractal_psd(k2, self.cfg.fractal) * (1.0 - _taper2(np.sqrt(k2), self.k_c))

    def _highpass(self, k2: FloatArray) -> FloatArray | float:
        """DEM mode: suppress wavelengths the DEM already carries (1 - taper² at its Nyquist)."""
        hk = self.base.highpass_k if self.base is not None else None
        if hk is None:
            return 1.0
        return 1.0 - _taper2(np.sqrt(k2), hk)

    def _detail_psd_synth(self) -> FloatArray:
        """Detail PSD used for synthesis (constant per terrain; cached)."""
        if self._psd_cache.get("synth") is None:
            self._psd_cache["synth"] = self._detail_psd_synth_compute()
        return self._psd_cache["synth"]

    def _detail_psd_synth_compute(self) -> FloatArray:
        m, _ = self._tile_shape()
        r = self.cfg.near.resolution_m
        ky = sfft.fftfreq(m, d=r)
        kx = sfft.rfftfreq(m, d=r)
        k2 = ky[:, None] ** 2 + kx[None, :] ** 2
        return self._detail_psd_unit() * self._highpass(k2)

    # ------------------------------------------------------------------ far
    def _build_far(self) -> None:
        cfg = self.cfg
        fc = cfg.fractal
        ny, nx = cfg.shape
        r = cfg.resolution_m
        w = -1 if self.workers is None else self.workers
        z = np.zeros((ny, nx))
        self.psd_constant = 0.0
        if fc.enabled and fc.rms_slope_deg > 0.0:
            nyp = sfft.next_fast_len(int(np.ceil(ny * fc.pad_factor)), real=True)
            nxp = sfft.next_fast_len(int(np.ceil(nx * fc.pad_factor)), real=True)
            ky = sfft.fftfreq(nyp, d=r)
            kx = sfft.rfftfreq(nxp, d=r)
            k2 = ky[:, None] ** 2 + kx[None, :] ** 2
            psd_far = fractal_psd(k2, fc) * _taper2(np.sqrt(k2), self.k_c)
            hp = self._highpass(k2)
            del k2
            L = fc.baseline_m
            target = (math.tan(math.radians(fc.rms_slope_deg)) * L) ** 2
            s_far = increment_variance((nyp, nxp), r, psd_far, L)
            s_det = increment_variance(
                self._tile_shape(), cfg.near.resolution_m, self._detail_psd_unit(), L
            )
            C = target / (s_far + s_det)  # amplitude of the full (un-high-passed) spectrum
            self.psd_constant = C
            psd_far = psd_far * hp
            noise = self.rngs.generator("terrain/fractal/far").standard_normal((nyp, nxp))
            spec = sfft.rfft2(noise, workers=w)
            del noise
            spec *= spectral_amplitude(C * psd_far, r)
            field_ = sfft.irfft2(spec, s=(nyp, nxp), workers=w)
            del spec
            z = np.ascontiguousarray(field_[:ny, :nx])
            z -= z.mean()
            lag = max(1, round(L / r))
            expected = increment_variance((nyp, nxp), r, C * psd_far, lag * r)
            realised = (rms_slope_at_baseline(z, r, lag) * lag * r) ** 2
            if realised > 0.0 and expected > 0.0:
                z *= math.sqrt(expected / realised)
        hm = Heightmap.centered(z, r)
        hm.z[...] += cfg.base_elevation_m
        if cfg.tilt.slope_deg > 0.0:
            az = math.radians(cfg.tilt.downhill_azimuth_deg)
            grade = math.tan(math.radians(cfg.tilt.slope_deg))
            hm.z[...] -= grade * (hm.x[None, :] * math.sin(az) + hm.y[:, None] * math.cos(az))
        self._far_base = Heightmap(hm.z.copy(), hm.resolution, hm.x0, hm.y0)
        if self.base is not None:  # DEM + curvature, evaluated per layer (not interpolated)
            hm.z[...] += self.base.grid(hm.x, hm.y)

        self._large = self._large_craters(hm.extent)
        labels = np.full(hm.shape, Label.PLAIN, dtype=np.uint8)
        # Reference planes of large craters are fitted once on the far layer and reused
        # by every near patch, so near stamping does not depend on the patch extent.
        self._large_planes = np.full((len(self._large), 3), np.nan)
        stamp_craters(
            hm, self._large, cfg.craters.shape, labels, cfg.craters.label_min_diameter_m,
            out_planes=self._large_planes,
        )  # fmt: skip
        self.far = Terrain(hm, labels, self._large, RockCatalog.empty())

    def _large_craters(self, extent: tuple[float, float, float, float]) -> CraterCatalog:
        cc = self.cfg.craters
        if not cc.enabled:
            return CraterCatalog.empty()
        sfd = cc.sfd
        lo = max(sfd.d_min_m, self.d_split)
        if lo >= sfd.d_max_m:
            return CraterCatalog.empty()
        sub = sfd.model_copy(update={"d_min_m": lo})
        margin = 0.5 * sfd.d_max_m * cc.shape.ejecta_extent_radii
        xmin, xmax = extent[0] - margin, extent[1] + margin
        ymin, ymax = extent[2] - margin, extent[3] + margin
        area = (xmax - xmin) * (ymax - ymin)
        c, b = sfd.coefficient, sfd.exponent
        lam = area * c * (lo**-b - sfd.d_max_m**-b)
        rng = self.rngs.generator("terrain/craters/large")
        n = int(rng.poisson(lam))
        d = sample_diameters(n, sub, rng)
        x = rng.uniform(xmin, xmax, n)
        y = rng.uniform(ymin, ymax, n)
        g = rng.beta(cc.degradation.beta_a, cc.degradation.beta_b, n)
        order = np.argsort(-g, kind="stable")
        x, y, d, g = x[order], y[order], d[order], g[order]
        depth, rim = _shape_params(d, g, cc.shape)
        uid = np.array([f"L{i}" for i in range(n)], dtype=np.str_)
        return CraterCatalog(x, y, d, depth, rim, g, np.full(n, -1, dtype=np.int64), uid)

    # ------------------------------------------------------------------ tiles
    def _detail_tile(self, a: int, b: int) -> FloatArray:
        """Weighted fine-detail field of tile (a, b) (cached)."""
        key = (a, b)
        if key in self._detail_cache:
            return self._detail_cache[key]
        nr = self.cfg.near
        m, _ = self._tile_shape()
        if self.psd_constant > 0.0:
            rng = self.rngs.generator(f"terrain/detail/tile/{a}/{b}")
            spec = sfft.rfft2(rng.standard_normal((m, m)))
            if self._psd_cache.get("amp") is None:
                self._psd_cache["amp"] = spectral_amplitude(
                    self.psd_constant * self._detail_psd_synth(), nr.resolution_m
                )
            amp = self._psd_cache["amp"]
            assert amp is not None
            spec *= amp
            f = sfft.irfft2(spec, s=(m, m))
            w1 = _tile_weight(m, nr.overlap_cells)
            f *= w1[:, None] * w1[None, :]
        else:
            f = np.zeros((m, m))
        if len(self._detail_cache) > 256:
            self._detail_cache.clear()
        self._detail_cache[key] = f
        return np.asarray(f, dtype=np.float64)

    def _small_craters_tile(self, a: int, b: int) -> CraterCatalog:
        cc = self.cfg.craters
        sfd = cc.sfd
        hi = min(sfd.d_max_m, self.d_split)
        if not cc.enabled or sfd.d_min_m >= hi:
            return CraterCatalog.empty()
        nr = self.cfg.near
        size = nr.tile_cells * nr.resolution_m
        x0, y0 = b * size, a * size  # tile core (a: rows/y, b: columns/x)
        sub = sfd.model_copy(update={"d_max_m": hi})
        c, bb = sfd.coefficient, sfd.exponent
        rng = self.rngs.generator(f"terrain/craters/small/tile/{a}/{b}")
        n = int(rng.poisson(size * size * c * (sfd.d_min_m**-bb - hi**-bb)))
        d = sample_diameters(n, sub, rng)
        x = rng.uniform(x0, x0 + size, n)
        y = rng.uniform(y0, y0 + size, n)
        g = rng.beta(cc.degradation.beta_a, cc.degradation.beta_b, n)
        depth, rim = _shape_params(d, g, cc.shape)
        uid = np.array([f"S{a}_{b}_{i}" for i in range(n)], dtype=np.str_)
        return CraterCatalog(x, y, d, depth, rim, g, np.full(n, -1, dtype=np.int64), uid)

    # ------------------------------------------------------------------ near
    def build_near(self, center_x: float, center_y: float) -> Terrain:
        """Generate the near patch centred (snapped to the fine lattice) at ``(x, y)`` [m]."""
        cfg, nr = self.cfg, self.cfg.near
        r = nr.resolution_m
        n_out = nr.shape[0]
        # Padding so that small craters near the border see the same surroundings for
        # their reference-plane fit as in any other patch (2 x (ejecta reach + ring)).
        shape_cfg = cfg.craters.shape
        pad_m = 2.0 * 0.5 * self.d_split * (shape_cfg.ejecta_extent_radii + 1.6)
        pad = math.ceil(pad_m / r) if cfg.craters.enabled else 0
        n = n_out + 2 * pad
        gx0 = round((center_x - 0.5 * nr.size_m) / r) - pad
        gy0 = round((center_y - 0.5 * nr.size_m) / r) - pad
        x0, y0 = gx0 * r, gy0 * r
        fb = self._far_base
        xmin, xmax, ymin, ymax = fb.extent
        if x0 < xmin or y0 < ymin or x0 + (n - 1) * r > xmax or y0 + (n - 1) * r > ymax:
            raise ValueError("near patch must lie inside the far terrain")

        # 1) upsampled far base
        xs = x0 + r * np.arange(n)
        ys = y0 + r * np.arange(n)
        uu = (xs - fb.x0) / fb.resolution
        vv = (ys - fb.y0) / fb.resolution
        V, U = np.meshgrid(vv, uu, indexing="ij")
        z = map_coordinates(fb.z, [V, U], order=3, mode="nearest")
        del U, V
        if self.base is not None:
            z += self.base.grid(np.asarray(xs, np.float64), np.asarray(ys, np.float64))

        # 2) fine detail from tiles
        tc, ov = nr.tile_cells, nr.overlap_cells
        m = tc + 2 * ov
        for a in range(math.floor((gy0 - ov) / tc) - 1, math.floor((gy0 + n + ov) / tc) + 2):
            ty0 = a * tc - ov
            i0, i1 = max(gy0, ty0), min(gy0 + n, ty0 + m)
            if i0 >= i1:
                continue
            for b in range(math.floor((gx0 - ov) / tc) - 1, math.floor((gx0 + n + ov) / tc) + 2):
                tx0 = b * tc - ov
                j0, j1 = max(gx0, tx0), min(gx0 + n, tx0 + m)
                if j0 >= j1:
                    continue
                tile = self._detail_tile(a, b)
                z[i0 - gy0 : i1 - gy0, j0 - gx0 : j1 - gx0] += tile[
                    i0 - ty0 : i1 - ty0, j0 - tx0 : j1 - tx0
                ]
        hm = Heightmap(np.ascontiguousarray(z), r, x0, y0)
        labels: npt.NDArray[np.uint8] = np.full(hm.shape, Label.PLAIN, dtype=np.uint8)

        # 3) craters intersecting the patch: large (global) + small (per tile), age order
        shape = cfg.craters.shape
        ext = hm.extent
        parts: list[CraterCatalog] = []
        plane_parts: list[FloatArray] = []
        if len(self._large):
            large = self._large
            reach = 0.5 * large.diameter * shape.ejecta_extent_radii
            sel = np.flatnonzero(_overlaps(large.x, large.y, reach, ext))
            parts.append(large.subset(sel))
            plane_parts.append(self._large_planes[sel])
        size = tc * r
        small_reach = 0.5 * self.d_split * shape.ejecta_extent_radii
        for a in _tile_range(ext[2], ext[3], small_reach, size):
            for b in _tile_range(ext[0], ext[1], small_reach, size):
                sc = self._small_craters_tile(a, b)
                parts.append(sc)
                plane_parts.append(np.full((len(sc), 3), np.nan))
        cat = CraterCatalog.concat(parts)
        planes = np.concatenate(plane_parts) if plane_parts else np.zeros((0, 3))
        if len(cat):
            order = np.argsort(-cat.degradation, kind="stable")
            cat, planes = cat.subset(order), planes[order]
        stamp_craters(hm, cat, shape, labels, cfg.craters.label_min_diameter_m, planes=planes)

        # 4) rocks: background per tile + ejecta per crater
        rc = cfg.rocks
        rock_parts: list[RockCatalog] = []
        if rc.enabled:
            m_r = 0.5 * rc.d_max_m
            for a in _tile_range(ext[2], ext[3], m_r, size):
                for b in _tile_range(ext[0], ext[1], m_r, size):
                    rng = self.rngs.generator(f"terrain/rocks/background/tile/{a}/{b}")
                    box = (b * size, (b + 1) * size, a * size, (a + 1) * size)
                    rock_parts.append(background_rocks(rc, box, rng))
            if rc.ejecta.enabled and len(cat):
                uids = cat.uid if cat.uid is not None else np.arange(len(cat)).astype(np.str_)
                for ci in np.flatnonzero(cat.diameter >= rc.ejecta.min_crater_diameter_m):
                    rng = self.rngs.generator(f"terrain/rocks/ejecta/{uids[ci]}")
                    rock_parts.append(
                        ejecta_rocks(
                            rc,
                            float(cat.x[ci]),
                            float(cat.y[ci]),
                            float(cat.diameter[ci]),
                            float(cat.degradation[ci]),
                            int(ci),
                            rng,
                        )
                    )
        rocks = RockCatalog.concat(rock_parts)
        if len(rocks):
            rocks = rocks.subset(_overlaps(rocks.x, rocks.y, 0.5 * rocks.diameter, ext))
        stamp_rocks(hm, rocks, labels)
        if pad:
            core = slice(pad, pad + n_out)
            hm = Heightmap(
                np.ascontiguousarray(hm.z[core, core]), r, hm.x0 + pad * r, hm.y0 + pad * r
            )
            labels = np.ascontiguousarray(labels[core, core], dtype=np.uint8)
            cat, rocks = self._crop_catalogs(hm, cat, rocks)
        return Terrain(hm, labels, cat, rocks)

    @staticmethod
    def _crop_catalogs(
        hm: Heightmap, cat: CraterCatalog, rocks: RockCatalog
    ) -> tuple[CraterCatalog, RockCatalog]:
        """Keep only features that touch the (cropped) patch; re-index rock sources."""
        ext = hm.extent
        if len(cat):
            idx = np.flatnonzero(_overlaps(cat.x, cat.y, 0.5 * cat.diameter, ext))
            remap = np.full(len(cat), -1, dtype=np.int64)
            remap[idx] = np.arange(idx.size)
            cat = cat.subset(idx)
            if len(rocks):
                src = rocks.source_crater
                rocks = rocks.with_source(np.where(src >= 0, remap[np.maximum(src, 0)], -1))
        if len(rocks):
            rocks = rocks.subset(_overlaps(rocks.x, rocks.y, 0.5 * rocks.diameter, ext))
        return cat, rocks

    def recenter(self, center_x: float, center_y: float) -> Terrain:
        """Regenerate the near patch at a new centre and return it."""
        self.near = self.build_near(center_x, center_y)
        return self.near
