"""Real DEM input (GeoTIFF via optional ``tifffile``, or ``.npy``) and the base surface.

:class:`BaseSurface` is the deterministic large-scale surface added to the synthetic
terrain on every layer, evaluated analytically at each layer's own grid nodes:

* the DEM resampled at the world points (``terrain.dem``), relative to its value at the
  world origin, and
* the lunar curvature drop ``-(x² + y²) / (2 R)`` of the tangent-plane world
  (``terrain.curvature``).

World points map onto the Moon through the tangent plane at the site: the point
``R·up + x·E + y·N`` is projected radially onto the sphere, giving latitude/longitude,
then the DEM's map projection gives map coordinates and pixel indices (cubic spline).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
from scipy.ndimage import distance_transform_edt, map_coordinates

from lunarsim.config.schema import DemConfig, SiteConfig, TerrainConfig

__all__ = ["BaseSurface", "DemRaster", "DemSampler", "load_dem", "write_geotiff"]

FloatArray = npt.NDArray[np.float64]

# GeoTIFF tags (GeoTIFF 1.0 spec / GDAL).
TAG_PIXEL_SCALE = 33550
TAG_TIEPOINT = 33922
TAG_GDAL_NODATA = 42113


@dataclass(frozen=True)
class DemRaster:
    """DEM pixels and their map geometry.

    Attributes:
        data: Raw pixel values ``(rows, cols)`` (possibly a read-only memmap).
        x0: Map x of the centre of pixel [0, 0].
        y0: Map y of the centre of pixel [0, 0].
        dx: Map x step per column (> 0).
        dy: Map y step per row (< 0 for north-up rasters).
        nodata: Pixel value marking missing data (or None).
    """

    data: npt.NDArray[np.generic]
    x0: float
    y0: float
    dx: float
    dy: float
    nodata: float | None


def _read_tiff(path: Path) -> tuple[npt.NDArray[np.generic], dict[int, object]]:
    try:
        import tifffile
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError("GeoTIFF DEMs need the optional 'tifffile' (pip install .[dem])") from exc
    with tifffile.TiffFile(path) as tf:
        page = tf.pages[0]
        tags = {t.code: t.value for t in page.tags.values()}  # type: ignore[union-attr]
    data: npt.NDArray[np.generic]
    try:
        data = np.asarray(tifffile.memmap(path))  # uncompressed, contiguous: no full read
    except ValueError:
        data = np.asarray(tifffile.imread(path))
    return data, tags


def load_dem(cfg: DemConfig) -> DemRaster:
    """Open the DEM of ``cfg`` (GeoTIFF georeferencing from tags, ``.npy`` from ``cfg``)."""
    path = Path(cfg.path)
    if path.suffix.lower() == ".npy":
        data = np.load(path, mmap_mode="r", allow_pickle=False)
        if cfg.pixel_size is None or cfg.origin_x is None or cfg.origin_y is None:
            raise ValueError(".npy DEMs need terrain.dem.pixel_size, origin_x and origin_y")
        dy = -cfg.pixel_size if cfg.rows_north_up else cfg.pixel_size
        return DemRaster(data, cfg.origin_x, cfg.origin_y, cfg.pixel_size, dy, cfg.nodata)
    data, tags = _read_tiff(path)
    if data.ndim != 2:
        raise ValueError(f"{path}: expected a single-band DEM, got shape {data.shape}")
    scale = tags.get(TAG_PIXEL_SCALE)
    tie = tags.get(TAG_TIEPOINT)
    if scale is None or tie is None:
        raise ValueError(f"{path}: missing ModelPixelScale / ModelTiepoint GeoTIFF tags")
    sx, sy = float(scale[0]), float(scale[1])  # type: ignore[index]
    i, j, _, mx, my, _ = (float(v) for v in tie[:6])  # type: ignore[index]
    # Tiepoint raster (i, j) refers to pixel corners (PixelIsArea); centre of [0, 0]:
    x0 = mx + (0.5 - i) * sx
    y0 = my - (0.5 - j) * sy
    nodata = cfg.nodata
    nd_tag = tags.get(TAG_GDAL_NODATA)
    if nodata is None and nd_tag is not None:
        nodata = float(str(nd_tag).strip("\x00 "))
    return DemRaster(data, x0, y0, sx, -sy, nodata)


def write_geotiff(
    path: str | Path,
    data: npt.ArrayLike,
    x0_corner: float,
    y0_corner: float,
    pixel_size: float,
    nodata: float | None = None,
) -> None:
    """Write a north-up single-band GeoTIFF with pixel scale/tiepoint tags (tests, tools).

    ``(x0_corner, y0_corner)`` is the map position of the upper-left corner of pixel [0, 0].
    """
    import tifffile

    extratags: list[tuple[int, str, int, object, bool]] = [
        (TAG_PIXEL_SCALE, "d", 3, (pixel_size, pixel_size, 0.0), False),
        (TAG_TIEPOINT, "d", 6, (0.0, 0.0, 0.0, x0_corner, y0_corner, 0.0), False),
    ]
    if nodata is not None:
        extratags.append((TAG_GDAL_NODATA, "s", 0, f"{nodata:g}", False))
    tifffile.imwrite(path, np.asarray(data), extratags=extratags)


class DemSampler:
    """Heights of a DEM at world ENU points (tangent plane at the site)."""

    def __init__(self, cfg: DemConfig, site: SiteConfig, raster: DemRaster | None = None) -> None:
        self.cfg = cfg
        self.raster = raster if raster is not None else load_dem(cfg)
        la, lo = math.radians(site.latitude_deg), math.radians(site.longitude_deg)
        self._east = np.array([-math.sin(lo), math.cos(lo), 0.0])
        self._north = np.array(
            [-math.sin(la) * math.cos(lo), -math.sin(la) * math.sin(lo), math.cos(la)]
        )
        self._up = np.array(
            [math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)]
        )
        self._site_lat = site.latitude_deg
        self._site_lon = site.longitude_deg
        if cfg.projection == "polar_stereographic" and abs(site.latitude_deg) < 1e-9:
            raise ValueError("polar_stereographic DEMs need a non-equatorial site latitude")
        self.h_origin = 0.0
        self.h_origin = float(self._sample(np.zeros(1), np.zeros(1))[0])

    @property
    def pixel_size_m(self) -> float:
        """Approximate DEM pixel size on the ground at the site [m] (coarser axis)."""
        r = self.raster
        R = self.cfg.radius_m
        if self.cfg.projection == "local":
            return max(abs(r.dx), abs(r.dy))
        if self.cfg.projection == "polar_stereographic":
            k = 2.0 / (1.0 + abs(math.sin(math.radians(self._site_lat))))  # scale factor
            return max(abs(r.dx), abs(r.dy)) / k
        dlat = R * math.radians(abs(r.dy))
        dlon = R * math.cos(math.radians(self._site_lat)) * math.radians(abs(r.dx))
        return max(dlat, dlon)

    @property
    def nyquist_k(self) -> float:
        """DEM Nyquist frequency on the ground [cycles/m]."""
        return 0.5 / self.pixel_size_m

    def _map_xy(self, x: FloatArray, y: FloatArray) -> tuple[FloatArray, FloatArray]:
        cfg = self.cfg
        if cfg.projection == "local":
            return cfg.center_x_m + x, cfg.center_y_m + y
        R = cfg.radius_m
        p = (
            R * self._up[None, :]
            + x.reshape(-1, 1) * self._east[None, :]
            + y.reshape(-1, 1) * self._north[None, :]
        )
        p /= np.linalg.norm(p, axis=1, keepdims=True)
        lat = np.arcsin(np.clip(p[:, 2], -1.0, 1.0))
        lon = np.arctan2(p[:, 1], p[:, 0])
        if cfg.projection == "equirectangular":
            # Unwrap around the site longitude (continuous across ±180°), then shift by 360°
            # so the site lies inside the raster's longitude range.
            lon_deg = self._site_lon + np.mod(np.degrees(lon) - self._site_lon + 180.0, 360.0)
            lon_deg -= 180.0
            r = self.raster
            lo_min = min(r.x0, r.x0 + r.dx * (r.data.shape[1] - 1))
            lon_deg += 360.0 * math.floor((lo_min + 360.0 - self._site_lon) / 360.0)
            return lon_deg.reshape(x.shape), np.degrees(lat).reshape(x.shape)
        if self._site_lat < 0.0:  # south polar stereographic
            rho = 2.0 * R * np.tan(0.25 * np.pi + 0.5 * lat)
            mx, my = rho * np.sin(lon), rho * np.cos(lon)
        else:
            rho = 2.0 * R * np.tan(0.25 * np.pi - 0.5 * lat)
            mx, my = rho * np.sin(lon), -rho * np.cos(lon)
        return mx.reshape(x.shape), my.reshape(x.shape)

    def _sample(self, x: FloatArray, y: FloatArray) -> FloatArray:
        r = self.raster
        mx, my = self._map_xy(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))
        col = (mx - r.x0) / r.dx
        row = (my - r.y0) / r.dy
        rows, cols = r.data.shape
        pad = 16  # cubic-spline prefilter edge effects decay as 0.268^pad (window-independent)
        wrap = self.cfg.projection == "equirectangular" and abs(abs(r.dx) * cols - 360.0) < 1e-6
        r0 = max(0, math.floor(float(row.min())) - pad)
        r1 = min(rows, math.ceil(float(row.max())) + pad + 1)
        if row.min() < -0.5 or row.max() > rows - 0.5:
            raise ValueError("terrain extends beyond the DEM")
        if wrap:  # global raster: columns wrap around in longitude
            c0 = math.floor(float(col.min())) - pad
            c1 = math.ceil(float(col.max())) + pad + 1
            win = np.array(r.data[r0:r1][:, np.arange(c0, c1) % cols], dtype=np.float64)
        else:
            if col.min() < -0.5 or col.max() > cols - 0.5:
                raise ValueError("terrain extends beyond the DEM")
            c0 = max(0, math.floor(float(col.min())) - pad)
            c1 = min(cols, math.ceil(float(col.max())) + pad + 1)
            win = np.array(r.data[r0:r1, c0:c1], dtype=np.float64)
        bad = ~np.isfinite(win)
        if r.nodata is not None:
            bad |= win == r.nodata
        if bad.all():
            raise ValueError("DEM window contains no valid data")
        if bad.any():  # fill missing pixels with the nearest valid value
            idx = distance_transform_edt(bad, return_distances=False, return_indices=True)
            win = win[tuple(idx)]
        win = win * self.cfg.height_scale + self.cfg.height_offset_m
        h = map_coordinates(win, [row - r0, col - c0], order=3, mode="nearest")
        return np.asarray(h, dtype=np.float64).reshape(np.shape(x)) - self.h_origin

    def heights(self, x: npt.ArrayLike, y: npt.ArrayLike) -> FloatArray:
        """DEM height [m] at world points relative to the DEM height at the origin."""
        xa, ya = np.broadcast_arrays(np.asarray(x, np.float64), np.asarray(y, np.float64))
        return self._sample(xa, ya)


@dataclass
class BaseSurface:
    """DEM + curvature evaluated at world points (``None`` parts are skipped)."""

    dem: DemSampler | None
    curvature_radius_m: float | None

    @classmethod
    def from_config(cls, terrain: TerrainConfig, site: SiteConfig) -> BaseSurface | None:
        """Base surface for a terrain config, or None if neither DEM nor curvature is on."""
        dem = DemSampler(terrain.dem, site) if terrain.dem.enabled else None
        curv = terrain.moon_radius_m if terrain.curvature else None
        if dem is None and curv is None:
            return None
        return cls(dem, curv)

    @property
    def highpass_k(self) -> float | None:
        """Frequency [cycles/m] below which the fractal is suppressed (DEM Nyquist)."""
        return self.dem.nyquist_k if self.dem is not None else None

    def grid(self, x: FloatArray, y: FloatArray) -> FloatArray:
        """Base height on the grid ``x`` (columns) × ``y`` (rows) [m], shape ``(ny, nx)``."""
        X, Y = np.meshgrid(x, y)
        z = np.zeros(X.shape)
        if self.dem is not None:
            z += self.dem.heights(X, Y)
        if self.curvature_radius_m is not None:
            z -= (X * X + Y * Y) / (2.0 * self.curvature_radius_m)
        return z
