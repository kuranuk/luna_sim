"""Phase 4 demo: SPAD flash LiDAR (and a spinning LiDAR) in the south-polar scene.

Outputs in out/phase4/<scenario>/: PLY/PCD point clouds, range/intensity/photon images,
bird's-eye views, and a sun-elevation sweep of the SPAD detection range.

Usage: python scripts/demo_phase4.py [--config configs/south_pole_lowsun.yaml]
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from lunarsim.core.frames import Pose
from lunarsim.core.sensors.lidar import LidarScan
from lunarsim.core.world import World
from lunarsim.io.pointcloud import write_pcd, write_ply
from lunarsim.io.preview import normalize
from lunarsim.io.raster import write_png8

ROOT = Path(__file__).resolve().parents[1]


def bev(scan: LidarScan, half: float = 40.0, px: int = 600) -> np.ndarray:
    """Bird's-eye view in the sensor frame (x forward = up), RGB: real=intensity, false=red."""
    img = np.zeros((px, px, 3))
    p = scan.points[:, :3]
    j = ((p[:, 1] * -1 + half) / (2 * half) * (px - 1)).astype(int)
    i = ((half - p[:, 0]) / (2 * half) * (px - 1)).astype(int)
    ok = (i >= 0) & (i < px) & (j >= 0) & (j < px)
    val = 0.25 + 0.75 * normalize(p[:, 2], -3.0, 3.0)
    img[i[ok], j[ok]] = np.stack([val[ok]] * 3, axis=1)
    fa = ok & scan.false_alarm
    img[i[fa], j[fa]] = [1.0, 0.1, 0.1]
    return img


def save_scan(out: Path, tag: str, scan: LidarScan, max_range: float) -> dict[str, object]:
    fields = {
        "x": scan.points[:, 0],
        "y": scan.points[:, 1],
        "z": scan.points[:, 2],
        "intensity": scan.points[:, 3],
        "ring": scan.ring,
        "time": scan.time,
        "label": scan.labels,
    }
    write_ply(out / f"{tag}.ply", fields)
    write_pcd(out / f"{tag}.pcd", fields)
    write_png8(out / f"{tag}_bev.png", bev(scan), flip_north_up=False)
    if scan.range_image is not None:
        ri = scan.range_image
        write_png8(
            out / f"{tag}_range.png",
            np.where(np.isfinite(ri), 1.0 - normalize(np.nan_to_num(ri), 0, max_range), 0.0),
            flip_north_up=False,
        )
    if scan.spad is not None and scan.range_image is not None:
        shp = scan.range_image.shape
        cnt = scan.spad.signal_counts.reshape(shp)
        write_png8(out / f"{tag}_photons.png", normalize(np.log1p(cnt)), flip_north_up=False)
        bg = scan.spad.bg_rate_hz.reshape(shp)
        write_png8(out / f"{tag}_background.png", normalize(np.log10(bg)), flip_north_up=False)
    real = ~scan.false_alarm
    return {
        "points": len(scan),
        "false_alarms": int(scan.false_alarm.sum()),
        "max_range_m": float(np.max(scan.range_m[real])) if real.any() else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "south_pole_lowsun.yaml"))
    ap.add_argument("--out", default=str(ROOT / "out" / "phase4"))
    args = ap.parse_args()
    world = World.from_yaml(args.config)
    out = Path(args.out) / world.config.name
    out.mkdir(parents=True, exist_ok=True)
    sun = world.sun
    z0 = world.height_at(0.0, 0.0)
    stats: dict[str, object] = {
        "sun_elevation_deg": sun.elevation_deg,
        "sun_azimuth_deg": sun.azimuth_deg,
    }

    def pose(view_az: float, pitch_deg: float = 8.0) -> Pose:
        return Pose.from_euler_zyx(
            0.0, math.radians(pitch_deg), math.radians(90.0 - view_az), (0.0, 0.0, z0 + 1.5)
        )

    for name in world.lidars:
        mr = world.lidars[name].noise.range_max_m
        for tag, az in {
            "down_sun": sun.azimuth_deg + 180,
            "cross_sun": sun.azimuth_deg + 90,
        }.items():
            world.render_lidar(name, pose(az))
            t = time.perf_counter()
            scan = world.render_lidar(name, pose(az))
            dt = time.perf_counter() - t
            stats[f"{name}/{tag}"] = save_scan(out, f"{name}_{tag}", scan, mr) | {
                "time_s": round(dt, 3)
            }

    # SPAD detection vs sun elevation (same view, sunlit terrain in front).
    sweep = {}
    for el in (-5.0, 2.0, 10.0, 30.0, 60.0):
        world.set_sun_manual(sun.azimuth_deg, el)
        scan = world.render_lidar("spad_flash", pose(sun.azimuth_deg + 180))
        real = ~scan.false_alarm
        sweep[f"{el}"] = {
            "returns": int(real.sum()),
            "false_alarms": int(scan.false_alarm.sum()),
            "p95_range_m": float(np.percentile(scan.range_m[real], 95)) if real.any() else None,
        }
        if el in (-5.0, 60.0):
            save_scan(out, f"spad_flash_sun{el:+05.1f}", scan, 100.0)
    world.set_sun_utc(world.config.sun.utc)
    stats["spad_sweep_by_sun_elevation_deg"] = sweep
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
