"""Phase 5 demo: rover drive with terrain contact, stereo navcam and headlights.

Outputs in out/phase5/<scenario>/: trajectory over the near patch (PNG), per-step
state log (CSV), stereo pairs along the drive, and a headlight on/off comparison
looking into a shadowed area.

Usage: python scripts/demo_phase5.py [--config configs/south_pole_lowsun_stereo.yaml]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np

from lunarsim.core.world import CameraFrame, World
from lunarsim.io.preview import hillshade, normalize
from lunarsim.io.raster import write_png8

ROOT = Path(__file__).resolve().parents[1]


def save_pair(out: Path, tag: str, left: CameraFrame, right: CameraFrame, bits: int) -> None:
    pair = np.concatenate([left.image.dn, right.image.dn], axis=1) / (2**bits - 1)
    write_png8(out / f"{tag}_stereo.png", pair, flip_north_up=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "south_pole_lowsun_stereo.yaml"))
    ap.add_argument("--out", default=str(ROOT / "out" / "phase5"))
    ap.add_argument("--duration", type=float, default=120.0)
    args = ap.parse_args()
    world = World.from_yaml(args.config)
    out = Path(args.out) / world.config.name
    out.mkdir(parents=True, exist_ok=True)
    cam = world.config.cameras[0]
    bits = cam.sensor.bits
    hm0 = world.heightmap  # patch at start, for the trajectory map

    # Drive: straight, then a gentle left arc (skid-steer), sampled every 0.5 s.
    dt = 0.5
    log = []
    frames = 0
    t0 = time.perf_counter()
    n_steps = int(args.duration / dt)
    for k in range(n_steps):
        omega = 0.0 if k < n_steps // 3 else math.radians(3.0)
        world.set_rover_command(0.2, omega)
        st = world.step(dt)
        c = world.rover.contact
        log.append(
            {
                "t_s": round(world.time_s, 3),
                "x_m": world.rover.x,
                "y_m": world.rover.y,
                "z_m": float(c.pose.t[2]),
                "yaw_deg": math.degrees(world.rover.yaw),
                "roll_deg": math.degrees(c.roll),
                "pitch_deg": math.degrees(c.pitch),
                "slope_deg": st.slope_deg,
                "slip": st.slip,
                "lateral_v_m_s": st.lateral_velocity,
                "warnings": ";".join(st.warnings),
            }
        )
        if k % (n_steps // 4) == 0:
            left, right = world.render_stereo(cam.name, frame_index=frames)
            save_pair(out, f"t{world.time_s:05.1f}", left, right, bits)
            frames += 1
    t_drive = time.perf_counter() - t0
    with (out / "rover_log.csv").open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(log[0].keys()))
        wr.writeheader()
        wr.writerows(log)

    # Trajectory over the initial near patch (North up).
    img = np.repeat(hillshade(hm0, world.sun.azimuth_deg, 20.0)[..., None], 3, axis=2)
    for row in log:
        j = round((row["x_m"] - hm0.x0) / hm0.resolution)
        i = round((row["y_m"] - hm0.y0) / hm0.resolution)
        if 0 <= i < hm0.shape[0] and 0 <= j < hm0.shape[1]:
            img[max(0, i - 3) : i + 4, max(0, j - 3) : j + 4] = [1.0, 0.2, 0.1]
    write_png8(out / "trajectory.png", img)

    # Headlights: look into the darkest direction (largest shadow fraction) at the end.
    best = None
    for az in range(0, 360, 30):
        world.rover.reset(world.rover.x, world.rover.y, math.radians(az))
        b = world.render_camera(cam.name, frame_index=99).buffers
        vis = np.isfinite(b.depth[240:])
        frac = float(b.shadow[240:][vis].mean()) if vis.any() else 0.0
        if best is None or frac > best[0]:
            best = (frac, az)
    assert best is not None
    world.rover.reset(world.rover.x, world.rover.y, math.radians(best[1]))
    fixed_exposure = 0.02
    res = {}
    for on in (False, True):
        for name in world.lights:
            world.set_light(name, on)
        f = world.render_camera(cam.name, frame_index=100, exposure_s=fixed_exposure)
        write_png8(
            out / f"headlights_{'on' if on else 'off'}.png",
            f.image.dn / (2**bits - 1),
            flip_north_up=False,
        )
        res["on" if on else "off"] = float(f.image.dn.mean())
    rad = f.buffers.radiance
    write_png8(
        out / "headlights_on_radiance_log.png",
        normalize(np.log10(np.maximum(rad, 1e-4))),
        flip_north_up=False,
    )

    stats = {
        "drive_duration_s": args.duration,
        "wall_time_s": round(t_drive, 2),
        "distance_m": world.rover.odometer_m,
        "final_pose": log[-1],
        "max_slope_deg": max(r["slope_deg"] for r in log),
        "max_abs_roll_deg": max(abs(r["roll_deg"]) for r in log),
        "max_abs_pitch_deg": max(abs(r["pitch_deg"]) for r in log),
        "slip_range": [min(r["slip"] for r in log), max(r["slip"] for r in log)],
        "warnings_seen": sorted({w for r in log for w in r["warnings"].split(";") if w}),
        "near_patch_extent_m": list(world.heightmap.extent),
        "headlight_view_azimuth_from_east_deg": best[1],
        "headlight_view_shadow_fraction": best[0],
        "mean_dn_lights_off_on": res,
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
