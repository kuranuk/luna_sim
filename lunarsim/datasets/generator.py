"""Generate SLAM / VO datasets: the same trajectory under several illuminations.

Layout (``<output_dir>/``)::

    dataset.yaml, scenario.yaml        job and fully resolved scenario (with seed)
    terrain/                           far/near heightmaps (.npy), craters/rocks (.json)
    <variation>/
        times.txt                      frame times [s]
        cameras/<cam>/left|right/NNNNNN.png   16-bit digital numbers
        cameras/<cam>/depth/NNNNNN.npy        float32 Z [m] (left camera, inf = sky)
        cameras/<cam>/labels/NNNNNN.png       uint8 semantic labels (left camera)
        cameras/<cam>/shadow/NNNNNN.png       uint8 0/255 shadow mask
        lidars/<lidar>/NNNNNN.bin             float32 x y z intensity (sensor frame, KITTI)
        poses/<cam>_kitti.txt                 left camera, KITTI odometry format
        poses/<sensor>_tum.txt, body_tum.txt  TUM format (world ENU)
        calib/<cam>_kitti.txt, calib.yaml     intrinsics and extrinsics
        rover_state.csv, metadata.json

The terrain and the rover motion do not depend on the sun, so all variations share
identical poses (tested); only the sensor data differ.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from lunarsim.config.load import load_config
from lunarsim.core.frames import R_BODY_FROM_CAMERA, Pose
from lunarsim.core.world import World
from lunarsim.datasets.schema import DatasetConfig, SunVariation, TrajectoryConfig
from lunarsim.io.bundle import lidar_fields
from lunarsim.io.groundtruth import write_label_png
from lunarsim.io.pointcloud import write_pcd, write_ply
from lunarsim.io.raster import save_npy, write_uint16_png
from lunarsim.io.trajectory import kitti_calib_text, write_kitti_poses, write_tum
from lunarsim.version import __version__

__all__ = ["generate_dataset"]


def _apply_variation(world: World, v: SunVariation) -> None:
    if v.utc is not None:
        world.set_sun_utc(v.utc)
    elif v.azimuth_deg is not None and v.elevation_deg is not None:
        world.set_sun_manual(v.azimuth_deg, v.elevation_deg)
    if v.lights_on is not None:
        for name in world.lights:
            world.set_light(name, v.lights_on)


def _command_at(traj: TrajectoryConfig, t: float) -> tuple[float, float, float] | None:
    if traj.commands is None:
        return None
    seg = None
    for c in sorted(traj.commands, key=lambda c: c.t_s):
        if c.t_s <= t + 1e-9:
            seg = c
    return (0.0, 0.0, 0.0) if seg is None else (seg.v_m_s, seg.omega_rad_s, seg.steer_rad)


def _calibration(world: World, cfg: DatasetConfig) -> dict[str, Any]:
    out: dict[str, Any] = {
        "frames": {
            "world": "local ENU (x East, y North, z Up)",
            "body": "rover FLU (x forward, y left, z up)",
            "camera": "OpenCV optical (x right, y down, z forward)",
            "lidar": "FLU, points at their own measurement time",
        },
        "cameras": {},
        "lidars": {},
    }
    for name in cfg.cameras:
        cam_cfg, model = world.cameras[name]
        T = world.mount_pose(cam_cfg.mount) @ Pose(R_BODY_FROM_CAMERA, np.zeros(3))
        out["cameras"][name] = {
            "width": model.width,
            "height": model.height,
            "K": model.K.tolist(),
            "distortion_k1_k2_p1_p2_k3": list(model.dist),
            "stereo_baseline_m": cam_cfg.stereo_baseline_m,
            "T_body_camera": T.matrix().tolist(),
            "bits": cam_cfg.sensor.bits,
        }
    for name in cfg.lidars:
        lc = world.lidars[name]
        out["lidars"][name] = {
            "pattern": lc.pattern.model_dump(mode="json"),
            "noise_model": lc.noise.model,
            "range_max_m": lc.noise.range_max_m,
            "T_body_lidar": world.mount_pose(lc.mount).matrix().tolist(),
        }
    return out


def _write_terrain(world: World, root: Path) -> None:
    t = root / "terrain"
    t.mkdir(parents=True, exist_ok=True)
    hm = world.terrain.heightmap
    save_npy(t / "near_heightmap.npy", hm.z)
    geo = {"near": {"resolution_m": hm.resolution, "x0_m": hm.x0, "y0_m": hm.y0}}
    (t / "near_craters.json").write_text(json.dumps(world.terrain.craters.to_records()))
    (t / "near_rocks.json").write_text(json.dumps(world.terrain.rocks.to_records()))
    far = world.terrain_far
    if far is not None:
        fh = far.heightmap
        save_npy(t / "far_heightmap.npy", fh.z)
        geo["far"] = {"resolution_m": fh.resolution, "x0_m": fh.x0, "y0_m": fh.y0}
        (t / "far_craters.json").write_text(json.dumps(far.craters.to_records()))
    (t / "geometry.json").write_text(json.dumps(geo, indent=2))


def _run(world: World, cfg: DatasetConfig, v: SunVariation, out: Path) -> dict[str, Any]:
    world.reset_dynamic()
    _apply_variation(world, v)
    traj = cfg.trajectory
    if traj.waypoints is not None:
        world.set_waypoints(list(traj.waypoints), traj.speed_m_s)
    n_frames = math.floor(cfg.duration_s * cfg.frame_rate_hz + 1e-9) + 1
    dt = 1.0 / cfg.frame_rate_hz
    times: list[float] = []
    body: list[Pose] = []
    cam_poses: dict[str, list[Pose]] = {c: [] for c in cfg.cameras}
    lidar_poses: dict[str, list[Pose]] = {name: [] for name in cfg.lidars}
    rows: list[dict[str, Any]] = []
    for k in range(n_frames):
        times.append(world.time_s)
        body.append(world.rover.pose)
        st = world.state()["rover"]
        assert isinstance(st, dict)
        rows.append({"frame": k, "t_s": world.time_s, **{
            f: st[f] for f in ("roll_deg", "pitch_deg", "yaw_deg", "slope_deg", "slip",
                               "odometer_m")
        }, "x_m": st["position_m"][0], "y_m": st["position_m"][1], "z_m": st["position_m"][2],
            "warnings": ";".join(st["warnings"])})  # fmt: skip
        for name in cfg.cameras:
            cam_cfg, _ = world.cameras[name]
            d = out / "cameras" / name
            left = world.render_camera(name, None, k)
            cam_poses[name].append(left.pose)
            (d / "left").mkdir(parents=True, exist_ok=True)
            write_uint16_png(d / "left" / f"{k:06d}.png", left.image.dn)
            if cfg.stereo and cam_cfg.stereo_baseline_m is not None:
                right = world.render_camera(name, None, k, left.image.exposure_s, "right")
                (d / "right").mkdir(parents=True, exist_ok=True)
                write_uint16_png(d / "right" / f"{k:06d}.png", right.image.dn)
            b = left.buffers
            for buf in cfg.camera_buffers:
                (d / buf).mkdir(parents=True, exist_ok=True)
                if buf == "depth":
                    save_npy(d / "depth" / f"{k:06d}.npy", b.depth.astype(np.float32))
                elif buf == "labels":
                    write_label_png(d / "labels" / f"{k:06d}.png", b.labels, flip_north_up=False)
                else:
                    mask = (b.shadow.astype(np.uint8) * 255).astype(np.uint8)
                    write_label_png(d / "shadow" / f"{k:06d}.png", mask, flip_north_up=False)
        for name in cfg.lidars:
            scan = world.render_lidar(name, None, k)
            lidar_poses[name].append(scan.pose)
            d = out / "lidars" / name
            d.mkdir(parents=True, exist_ok=True)
            for fmt in cfg.lidar_formats:
                if fmt == "bin":
                    scan.points.astype("<f4").tofile(d / f"{k:06d}.bin")
                elif fmt == "ply":
                    write_ply(d / f"{k:06d}.ply", lidar_fields(scan))
                elif fmt == "pcd":
                    write_pcd(d / f"{k:06d}.pcd", lidar_fields(scan))
                else:
                    np.savez(d / f"{k:06d}.npz", points=scan.points, ring=scan.ring,
                             time=scan.time, labels=scan.labels,
                             points_world_gt=scan.points_world_gt)  # fmt: skip
        if k < n_frames - 1:
            cmd = _command_at(traj, world.time_s)
            if cmd is not None:
                world.set_rover_command(*cmd)
            world.step(dt)

    (out / "times.txt").write_text("".join(f"{t:.6f}\n" for t in times))
    p = out / "poses"
    p.mkdir(parents=True, exist_ok=True)
    write_tum(p / "body_tum.txt", times, body)
    for name, poses in cam_poses.items():
        write_kitti_poses(p / f"{name}_kitti.txt", poses)
        write_tum(p / f"{name}_tum.txt", times, poses)
    for name, poses in lidar_poses.items():
        write_tum(p / f"{name}_tum.txt", times, poses)
    with (out / "rover_state.csv").open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    meta = {
        "variation": v.model_dump(mode="json"),
        "frames": n_frames,
        "frame_rate_hz": cfg.frame_rate_hz,
        "sun": world.state()["sun"],
        "lights": dict(world.light_on),
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    return meta


def generate_dataset(
    cfg: DatasetConfig, scenario_path: str | Path, output_dir: str | Path | None = None
) -> Path:
    """Generate all variations of a dataset; returns the output directory."""
    root = Path(output_dir or cfg.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    scenario = load_config(scenario_path, seed=cfg.seed)
    world = World(scenario)
    (root / "dataset.yaml").write_text(yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False))
    (root / "scenario.yaml").write_text(
        yaml.safe_dump(scenario.model_dump(mode="json"), sort_keys=False)
    )
    _write_terrain(world, root)
    calib = _calibration(world, cfg)
    lidar0 = cfg.lidars[0] if cfg.lidars else None
    summary: dict[str, Any] = {
        "lunarsim_version": __version__,
        "seed": scenario.seed,
        "config_hash": world.config_hash,
        "variations": {},
    }
    for v in cfg.variations:
        out = root / v.name
        summary["variations"][v.name] = _run(world, cfg, v, out)
        c = out / "calib"
        c.mkdir(parents=True, exist_ok=True)
        (c / "calib.yaml").write_text(yaml.safe_dump(calib, sort_keys=False))
        for name in cfg.cameras:
            cam_cfg, model = world.cameras[name]
            T_cam_lidar = None
            if lidar0 is not None:
                T_body_cam = world.mount_pose(cam_cfg.mount) @ Pose(R_BODY_FROM_CAMERA, np.zeros(3))
                T_cam_lidar = T_body_cam.inverse() @ world.mount_pose(world.lidars[lidar0].mount)
            (c / f"{name}_kitti.txt").write_text(
                kitti_calib_text(model.K, cam_cfg.stereo_baseline_m, T_cam_lidar)
            )
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    return root
