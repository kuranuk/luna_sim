"""Phase 3 demo: south-polar low-sun scene rendered by the navigation camera.

Outputs in out/phase3/<scenario>/: camera images (8-bit preview + 16-bit DN), depth,
labels (with shadow), shadow masks, a sun-elevation sweep and the far-layer shadow map.

Usage: python scripts/demo_phase3.py [--config configs/south_pole_lowsun.yaml]
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from lunarsim.core.illumination.shadows import labels_with_shadow, terrain_shadow_mask
from lunarsim.core.sensors.camera import camera_pose_from_body
from lunarsim.core.world import CameraFrame, World
from lunarsim.io.groundtruth import write_label_png
from lunarsim.io.preview import hillshade, normalize
from lunarsim.io.raster import save_npy, write_png8, write_uint16_png

ROOT = Path(__file__).resolve().parents[1]


def save_frame(out: Path, tag: str, f: CameraFrame, bits: int) -> None:
    b = f.buffers
    write_uint16_png(out / f"{tag}_dn16.png", f.image.dn)
    write_png8(out / f"{tag}.png", f.image.dn / (2**bits - 1), flip_north_up=False)
    finite = np.isfinite(b.depth)
    d = np.where(finite, b.depth, 0.0)
    write_png8(
        out / f"{tag}_depth.png",
        np.where(finite, 1.0 - normalize(np.log1p(d)), 0.0),
        flip_north_up=False,
    )
    write_label_png(
        out / f"{tag}_labels.png",
        labels_with_shadow(b.labels, b.shadow),
        color=True,
        flip_north_up=False,
    )
    save_npy(out / f"{tag}_depth.npy", b.depth.astype(np.float32))
    save_npy(out / f"{tag}_radiance.npy", b.radiance.astype(np.float32))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "south_pole_lowsun.yaml"))
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default=str(ROOT / "out" / "phase3"))
    args = ap.parse_args()

    t0 = time.perf_counter()
    world = World.from_yaml(args.config, seed=args.seed)
    t_world = time.perf_counter() - t0
    for light in world.lights:  # sunlight only in this demo (lights: demo_phase5)
        world.set_light(light, False)
    out = Path(args.out) / world.config.name
    out.mkdir(parents=True, exist_ok=True)
    cam_name = world.config.cameras[0].name
    bits = world.config.cameras[0].sensor.bits
    sun = world.sun
    stats: dict[str, object] = {"world": world.metadata(), "world_build_s": round(t_world, 2)}

    z0 = world.height_at(0.0, 0.0)
    views = {
        "down_sun": sun.azimuth_deg + 180.0,  # sun behind the camera
        "cross_sun": sun.azimuth_deg + 90.0,
        "toward_sun": sun.azimuth_deg + 20.0,
    }
    timings = {}
    for tag, view_az in views.items():
        pose = camera_pose_from_body(
            [0.0, 0.0, z0 + 1.5], pitch=math.radians(8.0), yaw=math.radians(90.0 - view_az)
        )
        world.render_camera(cam_name, pose)  # warm-up (JIT compile on first call)
        t = time.perf_counter()
        f = world.render_camera(cam_name, pose, frame_index=0)
        timings[tag] = round(time.perf_counter() - t, 3)
        save_frame(out, tag, f, bits)
        vis = np.isfinite(f.buffers.depth)
        stats[tag] = {
            "view_azimuth_deg": view_az % 360.0,
            "render_s": timings[tag],
            "exposure_s": f.image.exposure_s,
            "shadow_fraction_of_ground": float(f.buffers.shadow[vis].mean()),
            "saturated_fraction": float(f.image.saturated.mean()),
            "far_layer_pixel_fraction": float((f.buffers.layer == 1).mean()),
        }

    # Sun elevation sweep (manual mode, same azimuth), fixed exposure for comparability.
    pose = camera_pose_from_body(
        [0.0, 0.0, z0 + 1.5], pitch=math.radians(8.0), yaw=math.radians(90.0 - views["cross_sun"])
    )
    sweep = {}
    for el in (1.0, 3.0, 10.0, 30.0):
        world.set_sun_manual(sun.azimuth_deg, el)
        f = world.render_camera(cam_name, pose, exposure_s=2.0e-4)
        save_frame(out, f"sweep_el{el:04.1f}", f, bits)
        vis = np.isfinite(f.buffers.depth)
        sweep[f"{el}"] = float(f.buffers.shadow[vis].mean())
    stats["sweep_shadow_fraction_by_elevation_deg"] = sweep
    world.set_sun_utc(world.config.sun.utc)

    # Far-layer cast-shadow map (labels "shadow") and hillshade, North up.
    far = world.terrain_far
    if far is not None:
        t = time.perf_counter()
        sh = terrain_shadow_mask(far, world.caster.far, world.sun)  # type: ignore[union-attr]
        stats["far_shadow_map_s"] = round(time.perf_counter() - t, 2)
        stats["far_shadow_fraction"] = float(sh.mean())
        write_png8(
            out / "far_hillshade.png",
            hillshade(far.heightmap, world.sun.azimuth_deg, max(world.sun.elevation_deg, 5)),
        )
        write_png8(out / "far_shadow_map.png", (~sh).astype(float))
        write_label_png(
            out / "far_labels_shadow.png", labels_with_shadow(far.labels, sh), color=True
        )
    near_sh = terrain_shadow_mask(world.terrain, world.caster, world.sun)
    write_png8(out / "near_shadow_map.png", (~near_sh).astype(float))
    stats["near_shadow_fraction"] = float(near_sh.mean())
    (out / "stats.json").write_text(json.dumps(stats, indent=2, default=str))
    print(json.dumps(stats, indent=2, default=str))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
