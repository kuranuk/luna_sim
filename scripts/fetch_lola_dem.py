"""Build a local base DEM for the browser simulator from LOLA laser shots (LRO).

Source: USGS Astrogeology analysis-ready data on AWS Open Data, bucket ``astrogeo-ard``,
``moon/lro/lola/LolaRDR_<lat band>_<lon band>.copc.laz`` (LOLA RDR shots as cloud-optimised point
clouds, body-fixed Cartesian metres, IAU 2015 Moon sphere R = 1737.4 km; NASA / GSFC / LOLA team
data, public domain). Only the points around the site are downloaded (HTTP range requests).

The shots within ``half`` metres of the site are projected into the site's local ENU frame (tangent
plane at the site, curvature included), interpolated on a regular grid (Delaunay, linear), smoothed
with a small Gaussian (orbit-to-orbit offsets of LOLA tracks are about 1 m) and written as

    web/assets/dem/<name>.f32   float32, row-major, row 0 = south, heights [m] relative to the site
    web/assets/dem/<name>.json  grid geometry, site, source, statistics

Needs the optional packages ``laspy[lazrs]`` and ``requests`` (not part of the runtime):

    pip install "laspy[lazrs]" requests
    python scripts/fetch_lola_dem.py --lat -89.5 --lon 0 --name lola_89.5S_0E

Alternative source (``--geotiff``): a finished polar-stereographic LOLA DEM, e.g. the 5 m/pix
south-pole site DEMs of NASA GSFC PGDA (Barker et al. 2021, Planet. Space Sci. 203, 105119;
https://pgda.gsfc.nasa.gov/products/78, ``<site>_final_adj_5mpp_surf.tif``). Download the file by
hand; its pixels around the site are converted to the same local ENU grid (needs ``tifffile``):

    python scripts/fetch_lola_dem.py --geotiff Site01_final_adj_5mpp_surf.tif \
        --lat -89.453 --lon 222.88 --res 5 --name lola_site01
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
from pathlib import Path

import numpy as np
import numpy.typing as npt
from scipy.interpolate import RegularGridInterpolator, griddata
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
MOON_RADIUS_M = 1737400.0
BUCKET = "https://astrogeo-ard.s3-us-west-2.amazonaws.com/moon/lro/lola"
FloatArray = npt.NDArray[np.float64]


def enu_basis(lat_deg: float, lon_deg: float) -> tuple[FloatArray, FloatArray]:
    """Site position [m] (on the reference sphere) and the rows E, N, U in Moon-fixed axes."""
    la, lo = math.radians(lat_deg), math.radians(lon_deg)
    sl, cl, so, co = math.sin(la), math.cos(la), math.sin(lo), math.cos(lo)
    site = MOON_RADIUS_M * np.array([cl * co, cl * so, sl])
    basis = np.array([[-so, co, 0.0], [-sl * co, -sl * so, cl], [cl * co, cl * so, sl]])
    return site, basis


def tile_names(lat_deg: float, lon_deg: float, half_m: float) -> list[str]:
    """LOLA tiles (15 deg x 15 deg) touched by the square window around the site."""
    site, basis = enu_basis(lat_deg, lon_deg)
    names: set[str] = set()
    for e in np.linspace(-half_m, half_m, 9):
        for n in np.linspace(-half_m, half_m, 9):
            p = site + e * basis[0] + n * basis[1]
            lat = math.degrees(math.asin(p[2] / np.linalg.norm(p)))
            lon = math.degrees(math.atan2(p[1], p[0])) % 360.0
            la0 = min(math.floor(lat / 15.0) * 15, 75)
            lo0 = min(math.floor(lon / 15.0) * 15, 345)
            names.add(f"LolaRDR_{la0}N{la0 + 15}N_{lo0}E{lo0 + 15}E")
    return sorted(names)


def fetch_points(lat_deg: float, lon_deg: float, half_m: float) -> tuple[FloatArray, list[str]]:
    """LOLA shots near the site in the local ENU frame, shape (N, 3) [m]."""
    import laspy  # optional dependency
    from laspy.copc import Bounds

    site, basis = enu_basis(lat_deg, lon_deg)
    # COPC queries are 2-D in Moon-fixed x, y: a generous box (the window plus the height range
    # along the local vertical), filtered exactly in ENU below
    margin = 1.5 * half_m + 3000.0
    lo, hi = site[:2] - margin, site[:2] + margin
    parts, used = [], []
    for name in tile_names(lat_deg, lon_deg, half_m):
        with laspy.CopcReader.open(f"{BUCKET}/{name}.copc.laz") as reader:
            q = reader.query(Bounds(mins=lo, maxs=hi))
            xyz = np.stack([np.asarray(q.x), np.asarray(q.y), np.asarray(q.z)], axis=1)
        if len(xyz):
            parts.append(xyz)
            used.append(name)
    if not parts:
        raise SystemExit("no LOLA shots found around the site")
    enu = (np.concatenate(parts) - site) @ basis.T
    keep = (np.abs(enu[:, 0]) <= half_m * 1.3) & (np.abs(enu[:, 1]) <= half_m * 1.3)
    return enu[keep], used


def grid_dem(
    enu: FloatArray, half_m: float, res_m: float, smooth_cells: float = 1.0
) -> tuple[FloatArray, FloatArray, dict[str, float]]:
    """Regular grid (rows south -> north) from scattered shots; returns (axis, z, statistics)."""
    n = round(2 * half_m / res_m) + 1
    ax = -half_m + res_m * np.arange(n)
    X, Y = np.meshgrid(ax, ax)
    z = griddata((enu[:, 0], enu[:, 1]), enu[:, 2], (X, Y), method="linear")
    if np.isnan(z).any():  # outside the convex hull of the shots: nearest shot
        zn = griddata((enu[:, 0], enu[:, 1]), enu[:, 2], (X, Y), method="nearest")
        z = np.where(np.isnan(z), zn, z)
    if smooth_cells > 0:
        z = gaussian_filter(z, smooth_cells)
    f = RegularGridInterpolator((ax, ax), z, bounds_error=False)
    m = (np.abs(enu[:, 0]) < half_m) & (np.abs(enu[:, 1]) < half_m)
    d = enu[m, 2] - f(np.stack([enu[m, 1], enu[m, 0]], axis=1))
    gy, gx = np.gradient(z, res_m)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    c = n // 2
    stats = {
        "shots": int(m.sum()),
        "mean_shot_spacing_m": float(2 * half_m / math.sqrt(max(int(m.sum()), 1))),
        "rms_residual_m": float(np.sqrt(np.nanmean(d**2))),
        "slope_at_site_deg": float(slope[c, c]),
        "slope_median_deg": float(np.median(slope)),
        "slope_max_deg": float(slope.max()),
        "downhill_azimuth_at_site_deg": float(math.degrees(math.atan2(-gx[c, c], -gy[c, c])) % 360),
    }
    return ax, z, stats


def stereo_xy(lat_deg: float, lon_deg: float, pole: int) -> tuple[float, float]:
    """Polar stereographic x, y [m] (true scale at the pole, lon 0 along +y south / -y north)."""
    la, lo = math.radians(lat_deg), math.radians(lon_deg)
    rho = 2.0 * MOON_RADIUS_M * math.tan(math.pi / 4.0 - pole * la / 2.0)
    return rho * math.sin(lo), -pole * rho * math.cos(lo)


def geotiff_points(
    path: Path, lat_deg: float, lon_deg: float, half_m: float
) -> tuple[FloatArray, str]:
    """Pixels of a polar-stereographic height GeoTIFF near the site as ENU points [m]."""
    import tifffile  # optional dependency

    with tifffile.TiffFile(path) as tf:
        page = tf.pages[0]
        tags = {t.code: t.value for t in page.tags.values()}
        z = np.asarray(page.asarray(), dtype=np.float64)
    sx, sy = tags[33550][:2]  # ModelPixelScale
    tie = tags[33922]  # ModelTiepoint: raster (i, j, k) -> model (x, y, z)
    doubles = tags.get(34736, ())
    if MOON_RADIUS_M not in doubles or (-90.0 not in doubles and 90.0 not in doubles):
        raise ValueError("expected a polar stereographic GeoTIFF on the 1737.4 km sphere")
    pole = -1 if -90.0 in doubles else 1
    # GDAL_METADATA node_offset=1 / PixelIsArea: the tiepoint is the outer corner of pixel (0, 0)
    x0, y0 = tie[3] - tie[0] * sx, tie[4] + tie[1] * sy
    cx, cy = stereo_xy(lat_deg, lon_deg, pole)
    m = 1.5 * half_m + 100.0
    j0, j1 = int((cx - m - x0) / sx), int((cx + m - x0) / sx) + 1
    i0, i1 = int((y0 - cy - m) / sy), int((y0 - cy + m) / sy) + 1
    if j0 < 0 or i0 < 0 or j1 > z.shape[1] or i1 > z.shape[0]:
        raise ValueError(f"the window around the site leaves the raster ({path.name})")
    x = x0 + (np.arange(j0, j1) + 0.5) * sx
    y = y0 - (np.arange(i0, i1) + 0.5) * sy
    xx, yy = np.meshgrid(x, y)
    h = z[i0:i1, j0:j1]
    ok = np.isfinite(h)
    xx, yy, h = xx[ok], yy[ok], h[ok]
    rho = np.hypot(xx, yy)
    lat = -pole * (2.0 * np.arctan(rho / (2.0 * MOON_RADIUS_M)) - math.pi / 2.0)
    lon = np.arctan2(xx, -pole * yy)
    r = MOON_RADIUS_M + h
    xyz = np.stack([r * np.cos(lat) * np.cos(lon), r * np.cos(lat) * np.sin(lon), r * np.sin(lat)])
    site, basis = enu_basis(lat_deg, lon_deg)
    enu = (basis @ (xyz - site[:, None])).T
    keep = (np.abs(enu[:, 0]) <= 1.3 * half_m) & (np.abs(enu[:, 1]) <= 1.3 * half_m)
    return enu[keep], f"{sx:g} m/pix"


def write_dem(out_dir: Path, name: str, lat: float, lon: float, ax: FloatArray, z: FloatArray,
              stats: dict[str, float], tiles: list[str], source: str | None = None,
              processing: str | None = None) -> None:  # fmt: skip
    c = len(ax) // 2
    z0 = float(z[c, c])
    out_dir.mkdir(parents=True, exist_ok=True)
    (z - z0).astype("<f4").tofile(out_dir / f"{name}.f32")
    meta = {
        "format": "lunarsim-dem/1",
        "site": {"lat": lat, "lon": lon},
        "n": len(ax), "res": float(ax[1] - ax[0]), "x0": float(ax[0]), "y0": float(ax[0]),
        "data": f"{name}.f32",
        "height_of_site_above_reference_sphere_m": z0,
        "reference_radius_m": MOON_RADIUS_M,
        "frame": "local ENU at the site; rows south to north, columns west to east; heights "
                 "relative to the site (curvature of the Moon included)",
        "source": source
        or "LRO LOLA RDR laser shots (NASA GSFC, LOLA team), USGS Astrogeology "
        "analysis-ready COPC tiles on AWS Open Data: s3://astrogeo-ard/moon/lro/lola/",
        "tiles": tiles,
        "processing": processing
        or "Delaunay linear interpolation of the shots + Gaussian smoothing (1 cell)",
        "created": datetime.date.today().isoformat(),
        "stats": stats,
    }  # fmt: skip
    (out_dir / f"{name}.json").write_text(json.dumps(meta, indent=1) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--lat", type=float, required=True, help="site latitude [deg]")
    ap.add_argument("--lon", type=float, required=True, help="site longitude [deg east]")
    ap.add_argument("--half", type=float, default=640.0, help="half size of the DEM [m]")
    ap.add_argument("--res", type=float, default=10.0, help="grid spacing [m]")
    ap.add_argument("--name", required=True, help="output base name")
    ap.add_argument("--out", type=Path, default=ROOT / "web" / "assets" / "dem")
    ap.add_argument("--geotiff", type=Path, help="polar stereographic DEM instead of the shots")
    ap.add_argument("--source", help="source text stored in the file (with --geotiff)")
    args = ap.parse_args(argv)
    if args.geotiff:
        enu, scale = geotiff_points(args.geotiff, args.lat, args.lon, args.half)
        ax, z, stats = grid_dem(enu, args.half, args.res, smooth_cells=0.0)
        stats["source_pixels"] = stats.pop("shots")
        del stats["mean_shot_spacing_m"]
        source = args.source or f"polar stereographic LOLA DEM {args.geotiff.name}"
        processing = f"{scale} pixels -> local ENU -> Delaunay linear interpolation, no smoothing"
        write_dem(args.out, args.name, args.lat, args.lon, ax, z, stats, [args.geotiff.name],
                  source, processing)  # fmt: skip
    else:
        enu, tiles = fetch_points(args.lat, args.lon, args.half)
        ax, z, stats = grid_dem(enu, args.half, args.res)
        write_dem(args.out, args.name, args.lat, args.lon, ax, z, stats, tiles)
    n = len(ax)
    print(f"wrote {args.out / args.name}.json/.f32: {n} x {n} at {args.res} m")
    print(json.dumps(stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
