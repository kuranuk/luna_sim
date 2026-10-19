"""Phase 8 demo: optional features. Outputs go to out/phase8/.

1. penumbra       - hard vs soft shadow edge at low sun, visible solar-disc fraction
2. earthshine     - night scene lit only by the Earth (sun below the horizon)
3. curvature      - sunset terminator on a 12 km flat plain with/without curvature
4. dem            - real LOLA DEM (Tycho) + fractal detail + craters, oblique view
5. psr            - one-draconic-year illumination / PSR / Earth visibility maps
6. imu            - 200 Hz IMU along a waypoint drive (plot)
7. spice          - analytic ephemeris vs SPICE over a year (needs LUNARSIM_SPICE_KERNELS)

Usage: python scripts/demo_phase8.py [--only NAME ...] [--dem PATH/moon_dem.bin]
The DEM step uses the 1024x512 LOLA global DEM of the sibling ``moonsim`` repository
(NASA/USGS, public domain); it is skipped if the file is not found.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from lunarsim.config import load_config, parse_config
from lunarsim.config.schema import ImuConfig, WorldConfig
from lunarsim.core.frames import Pose
from lunarsim.core.illumination.shadows import terrain_sun_fraction
from lunarsim.core.sensors.camera import camera_pose_from_body
from lunarsim.core.world import World
from lunarsim.io.preview import hillshade
from lunarsim.io.raster import write_png8

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "phase8"


def tone(dn: np.ndarray, pct: float = 99.5) -> np.ndarray:
    v = dn.astype(np.float64)
    hi = max(float(np.percentile(v, pct)), 1e-12)
    return np.asarray(np.clip(v / hi, 0.0, 1.0) ** (1 / 2.2))


def save(name: str, img: np.ndarray) -> Path:
    path = OUT / name
    write_png8(path, img, flip_north_up=False)
    return path


def line_plot(
    path: Path,
    t: np.ndarray,
    series: Sequence[tuple[str, np.ndarray, tuple[int, int, int]]],
    title: str,
    ylabel: str,
    size: tuple[int, int] = (900, 300),
) -> None:
    """Minimal PIL line plot (no plotting dependency)."""
    w, h = size
    img = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = 70, 30, w - 20, h - 35
    ys = np.concatenate([s[1] for s in series])
    lo, hi = float(np.nanmin(ys)), float(np.nanmax(ys))
    if hi - lo < 1e-12:
        hi, lo = hi + 1.0, lo - 1.0
    pad = 0.05 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    d.rectangle([x0, y0, x1, y1], outline="black")
    for k in range(5):
        v = lo + (hi - lo) * k / 4
        yy = y1 - (y1 - y0) * k / 4
        d.line([x0 - 4, yy, x0, yy], fill="black")
        d.text((5, yy - 6), f"{v:.3g}", fill="black")
    for k in range(6):
        v = t[0] + (t[-1] - t[0]) * k / 5
        xx = x0 + (x1 - x0) * k / 5
        d.line([xx, y1, xx, y1 + 4], fill="black")
        d.text((xx - 12, y1 + 8), f"{v:.3g}", fill="black")
    d.text((x0, 8), f"{title}   [{ylabel}]", fill="black")
    for i, (label, y, col) in enumerate(series):
        px = x0 + (x1 - x0) * (t - t[0]) / max(t[-1] - t[0], 1e-12)
        py = y1 - (y1 - y0) * (y - lo) / (hi - lo)
        d.line(list(zip(px.tolist(), py.tolist(), strict=True)), fill=col, width=1)
        d.text((x1 - 170, y0 + 6 + 14 * i), label, fill=col)
    img.save(path)


def no_lights(w: World) -> World:
    for name in w.lights:
        w.set_light(name, False)
    return w


# ---------------------------------------------------------------------------
def demo_penumbra(report: dict[str, object]) -> None:
    base = load_config(ROOT / "configs" / "south_pole_lowsun.yaml")
    cfg = base.model_copy(
        update={
            "sun": base.sun.model_copy(
                update={"mode": "manual", "azimuth_deg": 60.0, "elevation_deg": 1.5}
            )
        }
    )
    hard = no_lights(World(cfg))
    soft = no_lights(
        World(
            cfg.model_copy(
                update={"render": cfg.render.model_copy(update={"penumbra_samples": 32})}
            )
        )
    )
    t0 = time.perf_counter()
    fh = hard.render_camera("navcam")
    th = time.perf_counter() - t0
    t0 = time.perf_counter()
    fs = soft.render_camera("navcam", exposure_s=fh.image.exposure_s)
    ts = time.perf_counter() - t0
    a, b = tone(fh.image.dn), tone(fs.image.dn)
    a /= max(a.max(), 1e-9)
    b /= max(b.max(), 1e-9)
    save("penumbra_hard.png", a)
    save("penumbra_soft.png", b)
    save("penumbra_sun_fraction.png", fs.buffers.sun_fraction)
    # Zoom on the region with the most partially lit pixels.
    sf = fs.buffers.sun_fraction
    part = ((sf > 0) & (sf < 1)).astype(float)
    k = 80
    cs = np.cumsum(np.cumsum(np.pad(part, ((1, 0), (1, 0))), 0), 1)
    win = cs[k:, k:] - cs[:-k, k:] - cs[k:, :-k] + cs[:-k, :-k]
    win[: sf.shape[0] // 2] = -1  # near field (lower half): wide, resolved penumbrae
    i, j = np.unravel_index(np.argmax(win), win.shape)
    zoom = np.hstack(
        [
            a[i : i + k, j : j + k],
            np.ones((k, 2)),
            b[i : i + k, j : j + k],
            np.ones((k, 2)),
            sf[i : i + k, j : j + k],
        ]
    )
    big = np.kron(zoom, np.ones((4, 4)))
    save("penumbra_zoom_hard_vs_soft.png", big)
    report["penumbra"] = {
        "render_s_hard": round(th, 3),
        "render_s_penumbra32": round(ts, 3),
        "partial_pixels": int(part.sum()),
    }


def demo_earthshine(report: dict[str, object]) -> None:
    base = load_config(ROOT / "configs" / "south_pole_lowsun.yaml")
    sun = base.sun.model_copy(
        update={
            "mode": "manual",
            "azimuth_deg": 200.0,
            "elevation_deg": -3.0,
            "earthshine": True,
            "earth_azimuth_deg": 0.0,
            "earth_elevation_deg": 6.0,
        }
    )
    cam = base.cameras[0]
    cam = cam.model_copy(
        update={
            "sensor": cam.sensor.model_copy(
                update={"auto_exposure": True, "ae_max_exposure_s": 30.0}
            )
        }
    )
    cfg = base.model_copy(update={"sun": sun, "cameras": [cam]})
    w = no_lights(World(cfg))
    w.rover.reset(0.0, 0.0, math.radians(90.0))  # face north, towards the Earth
    f = w.render_camera("navcam")
    save("earthshine_navcam.png", tone(f.image.dn))
    off = no_lights(
        World(cfg.model_copy(update={"sun": sun.model_copy(update={"earthshine": False})}))
    )
    off.rover.reset(0.0, 0.0, math.radians(90.0))
    fo = off.render_camera("navcam", exposure_s=f.image.exposure_s)
    report["earthshine"] = {
        "earthshine_w_m2": round(w.earthshine_w_m2, 4),
        "auto_exposure_s": round(f.image.exposure_s, 3),
        "mean_dn_with": float(f.image.dn.mean()),
        "mean_dn_without": float(fo.image.dn.mean()),
    }


def demo_curvature(report: dict[str, object]) -> None:
    out: dict[str, object] = {}
    imgs = []
    for curv, rough in ((False, True), (True, True), (True, False)):
        cfg = parse_config(
            {
                "seed": 11,
                "terrain": {
                    "size_x_m": 12000,
                    "size_y_m": 3000,
                    "resolution_m": 10.0,
                    "fractal": {
                        "enabled": rough,
                        "rms_slope_deg": 1.0,
                        "baseline_m": 10.0,
                        "rolloff_wavelength_m": 2000.0,
                    },
                    "curvature": curv,
                },
                "sun": {"azimuth_deg": 90.0, "elevation_deg": -0.08},
                "cameras": [],
                "lidars": [],
            }
        )
        w = World(cfg)
        sky = w.sky
        frac = terrain_sun_fraction(w.terrain, w.caster, w.sun, sky=sky)
        key = ("curved" if curv else "flat") + ("_rough" if rough else "_smooth")
        out[key + "_lit_fraction"] = round(float(frac.mean()), 4)
        if rough:
            imgs.append(np.flipud(0.25 + 0.75 * frac))
        else:  # smooth sphere: terminator where the local elevation of the sun crosses 0
            lit_x = w.heightmap.x[frac.mean(axis=0) > 0.5]
            out["smooth_terminator_x_m"] = float(lit_x.min()) if lit_x.size else None
            out["expected_x_m"] = round(math.radians(0.08) * cfg.terrain.moon_radius_m, 1)
    sep = np.ones((6, imgs[0].shape[1]))
    save("curvature_flat_vs_curved.png", np.vstack([imgs[0], sep, imgs[1]]))
    report["curvature"] = out


def demo_dem(report: dict[str, object], dem_bin: Path) -> None:
    if not dem_bin.is_file():
        print(f"  DEM {dem_bin} not found: skipped")
        report["dem"] = "skipped (no DEM file)"
        return
    meta = json.loads(dem_bin.with_suffix(".json").read_text())
    z = np.fromfile(dem_bin, dtype="<i2").reshape(meta["height"], meta["width"])
    npy = OUT / "lola_global.npy"
    np.save(npy, z)
    px = 360.0 / meta["width"]
    cfg = parse_config(
        {
            "seed": 43,
            "site": {"latitude_deg": -43.3, "longitude_deg": -11.2},  # Tycho
            "terrain": {
                "size_x_m": 160_000,
                "size_y_m": 160_000,
                "resolution_m": 250.0,
                "fractal": {"hurst": 0.9, "rms_slope_deg": 6.0, "baseline_m": 250.0},
                "craters": {"enabled": True, "sfd": {"d_min_m": 1000.0, "d_max_m": 15000.0}},
                "dem": {
                    "enabled": True,
                    "path": str(npy),
                    "projection": "equirectangular",
                    "height_scale": meta["scale"],
                    "pixel_size": px,
                    "origin_x": -180.0 + 0.5 * px,
                    "origin_y": 90.0 - 0.5 * px,
                },
                "curvature": True,
            },
            "sun": {"azimuth_deg": 100.0, "elevation_deg": 12.0},
            "cameras": [
                {
                    "name": "orbit",
                    "intrinsics": {"width": 960, "height": 540, "hfov_deg": 50.0},
                    "sensor": {"auto_exposure": True},
                }
            ],
            "lidars": [],
        }
    )
    t0 = time.perf_counter()
    w = World(cfg)
    tb = time.perf_counter() - t0
    hm = w.heightmap
    save("dem_tycho_hillshade.png", np.flipud(hillshade(hm, 100.0, 20.0)))
    zc = w.height_at(0.0, -60_000.0)
    pose: Pose = camera_pose_from_body(
        [0.0, -75_000.0, zc + 14_000.0], 0.0, math.radians(18.0), math.radians(90.0)
    )
    f = w.render_camera("orbit", pose)
    save("dem_tycho_oblique.png", tone(f.image.dn))
    report["dem"] = {
        "source": meta.get("name"),
        "world_build_s": round(tb, 2),
        "relief_m": float(hm.z.max() - hm.z.min()),
        "dem_pixel_m": round(w.base_surface.dem.pixel_size_m, 1),  # type: ignore[union-attr]
        "craters_added": len(w.terrain.craters),
    }


def demo_psr(report: dict[str, object]) -> None:
    w = World.from_yaml(ROOT / "configs" / "south_pole_lowsun.yaml")
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    tic = time.perf_counter()
    m = w.long_term_illumination(t0, t0 + timedelta(days=346.62), timedelta(hours=2), stride=4)
    dt = time.perf_counter() - tic
    far = w.terrain_far
    assert far is not None
    z = far.heightmap.z[::4, ::4]
    from lunarsim.core.terrain import Heightmap

    shade = hillshade(Heightmap(np.ascontiguousarray(z), 4.0, 0.0, 0.0))
    rgb = np.stack([shade] * 3, axis=-1) * 0.8
    rgb[m.psr] = [0.15, 0.35, 1.0]
    save("psr_overlay.png", np.flipud(rgb))
    save("psr_mean_sun_fraction.png", np.flipud(np.clip(m.mean_sun_fraction / 0.6, 0, 1)))
    save("psr_earth_visible_fraction.png", np.flipud(m.earth_visible_fraction))
    report["psr"] = {
        "grid": list(m.psr.shape),
        "spacing_m": 4.0,
        "samples": m.n_samples,
        "compute_s": round(dt, 1),
        "psr_area_fraction": round(float(m.psr.mean()), 4),
        "mean_illumination": round(float(m.mean_sun_fraction.mean()), 4),
        "best_illumination": round(float(m.mean_sun_fraction.max()), 4),
        "earth_visible_mean": round(float(np.nanmean(m.earth_visible_fraction)), 4),
    }


def demo_imu(report: dict[str, object]) -> None:
    base = load_config(ROOT / "configs" / "south_pole_lowsun.yaml")
    cfg: WorldConfig = base.model_copy(update={"imus": [ImuConfig(name="imu", rate_hz=200.0)]})
    w = World(cfg)
    w.set_waypoints([(4.0, 0.0), (4.0, 3.0), (0.0, 3.0)], 0.2)
    t0 = time.perf_counter()
    w.step(40.0)
    dt = time.perf_counter() - t0
    s = w.imu_samples("imu")
    line_plot(
        OUT / "imu_gyro.png",
        s.t,
        [
            ("gyro z (meas)", s.gyro[:, 2], (200, 0, 0)),
            ("gyro z (true)", s.gyro_true[:, 2], (0, 0, 0)),
        ],
        "IMU 200 Hz: yaw rate along the waypoint drive",
        "rad/s",
    )
    line_plot(
        OUT / "imu_accel.png",
        s.t,
        [
            ("accel x", s.accel[:, 0], (200, 0, 0)),
            ("accel y", s.accel[:, 1], (0, 140, 0)),
            ("accel z - g", s.accel[:, 2] - 1.6243, (0, 0, 200)),
        ],
        "IMU 200 Hz: specific force (5 Hz 2nd-order low-pass truth + noise)",
        "m/s^2",
    )
    report["imu"] = {
        "samples": len(s),
        "sim_s": 40.0,
        "wall_s": round(dt, 2),
        "gyro_z_rms_error": float(np.sqrt(np.mean((s.gyro[:, 2] - s.gyro_true[:, 2]) ** 2))),
        "accel_z_mean": float(s.accel[:, 2].mean()),
    }


def demo_spice(report: dict[str, object]) -> None:
    kernels = [k for k in os.environ.get("LUNARSIM_SPICE_KERNELS", "").split(os.pathsep) if k]
    if not kernels:
        print("  LUNARSIM_SPICE_KERNELS not set: skipped")
        report["spice"] = "skipped (no kernels)"
        return
    from lunarsim.core.illumination.spice import angular_separation_deg, sun_spice
    from lunarsim.core.illumination.sun import sun_astronomical

    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    t, es, ee, el_a, el_s = [], [], [], [], []
    for h in range(0, 365 * 24, 6):
        tt = t0 + timedelta(hours=h)
        a = sun_astronomical(-89.5, 0.0, tt)
        b = sun_spice(-89.5, 0.0, tt, kernels)
        s, e = angular_separation_deg(a, b)
        t.append(h / 24.0)
        es.append(s)
        ee.append(e)
        el_a.append(a.elevation_deg)
        el_s.append(b.elevation_deg)
    ta = np.array(t)
    line_plot(
        OUT / "spice_vs_analytic_error.png",
        ta,
        [
            ("Sun direction error", np.array(es), (200, 120, 0)),
            ("Earth direction error", np.array(ee), (0, 0, 200)),
        ],
        "analytic vs SPICE, site 89.5S 0E, 2026",
        "deg",
    )
    line_plot(
        OUT / "spice_sun_elevation.png",
        ta,
        [("SPICE", np.array(el_s), (0, 0, 0)), ("analytic", np.array(el_a), (200, 0, 0))],
        "Sun elevation at 89.5S 0E, 2026",
        "deg",
    )
    report["spice"] = {
        "kernels": [Path(k).name for k in kernels],
        "sun_max_deg": round(max(es), 4),
        "sun_rms_deg": round(float(np.sqrt(np.mean(np.square(es)))), 4),
        "earth_max_deg": round(max(ee), 4),
        "earth_rms_deg": round(float(np.sqrt(np.mean(np.square(ee)))), 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--dem", default=str(ROOT.parent / "moonsim" / "data" / "moon_dem.bin"))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {}
    steps = {
        "penumbra": lambda: demo_penumbra(report),
        "earthshine": lambda: demo_earthshine(report),
        "curvature": lambda: demo_curvature(report),
        "dem": lambda: demo_dem(report, Path(args.dem)),
        "psr": lambda: demo_psr(report),
        "imu": lambda: demo_imu(report),
        "spice": lambda: demo_spice(report),
    }
    for name, fn in steps.items():
        if args.only and name not in args.only:
            continue
        print(f"[{name}]")
        t0 = time.perf_counter()
        fn()
        print(f"  done in {time.perf_counter() - t0:.1f} s: {report.get(name)}")
    prev = OUT / "report.json"
    old = json.loads(prev.read_text()) if prev.exists() and args.only else {}
    prev.write_text(json.dumps(old | report, indent=2, default=str))
    print(f"report: {prev}")


if __name__ == "__main__":
    main()
