"""Phase 6 demo: run the server (CLI) and drive the rover through the Python client.

Starts ``lunarsim serve`` on 127.0.0.1 with a token, creates the south-polar world,
measures REST latencies, checks that the event loop stays responsive during a
render, then drives along waypoints over the WebSocket while receiving camera and
LiDAR frames. Outputs go to out/phase6/<scenario>/.

Usage: python scripts/demo_phase6.py [--config configs/south_pole_lowsun_stereo.yaml]
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from lunarsim.client import LunarSimClient
from lunarsim.io.pointcloud import write_ply
from lunarsim.io.raster import write_png8

ROOT = Path(__file__).resolve().parents[1]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def timed(fn: Any, *a: Any, **k: Any) -> tuple[Any, float]:
    t = time.perf_counter()
    out = fn(*a, **k)
    return out, time.perf_counter() - t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "south_pole_lowsun_stereo.yaml"))
    ap.add_argument("--out", default=str(ROOT / "out" / "phase6"))
    args = ap.parse_args()
    out = Path(args.out) / Path(args.config).stem
    out.mkdir(parents=True, exist_ok=True)

    port, token = free_port(), secrets.token_hex(16)
    env = dict(os.environ, LUNARSIM_TOKEN=token)
    server = subprocess.Popen(
        [sys.executable, "-m", "lunarsim.cli", "serve", "--port", str(port)],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    stats: dict[str, object] = {}
    try:
        c = LunarSimClient(f"http://127.0.0.1:{port}", token=token)
        for _ in range(200):
            try:
                c.health()
                break
            except OSError:
                time.sleep(0.05)
        stats["version"] = c.version()
        res, dt = timed(c.create_world, args.config)
        wid = res["world_id"]
        stats["create_world_s"] = round(dt, 2)
        meta = res["metadata"]
        cam = "navcam"
        lid = "spad_flash"

        lat = {}
        _, lat["state"] = timed(c.state, wid)
        _, lat["heightmap_npy"] = timed(c.heightmap, wid)
        _, lat["step_1s"] = timed(c.step, wid, 1.0)
        frame, lat["camera_npz"] = timed(c.camera, wid, cam, 0, "left", "npz", ["depth", "labels"])
        _, lat["camera_npz_cached"] = timed(c.camera, wid, cam, 0, "left", "npz", ["depth"])
        scan, lat["lidar_npz"] = timed(c.lidar, wid, lid)
        stats["latency_s"] = {k: round(v, 3) for k, v in lat.items()}

        # Event loop responsiveness: /health while a fresh render is in progress.
        health_times: list[float] = []
        th = threading.Thread(target=lambda: c.camera(wid, cam, 7, "right", "npz"))
        th.start()
        while th.is_alive():
            _, h = timed(c.health)
            health_times.append(h)
            time.sleep(0.02)
        th.join()
        stats["health_during_render_s"] = {
            "n": len(health_times),
            "max": round(max(health_times), 4),
            "median": round(float(np.median(health_times)), 4),
        }

        # Drive along waypoints over the WebSocket, receiving sensor frames.
        x0, y0 = res["metadata"]["rover"]["x_m"], res["metadata"]["rover"]["y_m"]
        pts = [(x0 + 6.0, y0), (x0 + 6.0, y0 + 6.0), (x0, y0 + 6.0)]
        c.waypoints(wid, pts, speed_m_s=0.2)
        received = {"camera": 0, "lidar": 0}
        frames_saved = 0
        t0 = time.perf_counter()
        with c.stream(wid) as s:
            s.send(type="subscribe", cameras=[{"name": cam}], lidars=[lid], every_n_steps=10)
            assert s.receive()["type"] == "subscribed"
            for k in range(1, 121):
                state = s.step(0.5)  # state message of this step
                if k % 10:
                    continue
                for _ in range(2):  # camera then lidar frame of this step
                    m = s.receive()
                    received[m["type"]] += 1
                    if m["type"] == "camera":
                        write_png8(
                            out / f"ws_cam_{m['frame_index']:03d}.png",
                            m["data"]["image"] / 4095.0,
                            flip_north_up=False,
                        )
                        frames_saved += 1
                    else:
                        d = m["data"]["points"]
                        write_ply(
                            out / f"ws_lidar_{m['frame_index']:03d}.ply",
                            {"x": d[:, 0], "y": d[:, 1], "z": d[:, 2], "intensity": d[:, 3]},
                        )
        stats["ws_drive"] = {
            "steps": 120,
            "dt_s": 0.5,
            "wall_s": round(time.perf_counter() - t0, 2),
            "frames_received": received,
            "final_ground_track_m": state["rover"]["ground_track_m"],
            "waypoints_remaining": state["rover"]["waypoints_remaining"],
            "odometer_m": round(state["rover"]["odometer_m"], 2),
        }
        write_png8(out / "rest_camera_left.png", frame["image"] / 4095.0, flip_north_up=False)
        stats["rest_lidar_points"] = int(scan["points"].shape[0])
        stats["world"] = {k: meta[k] for k in ("name", "seed", "config_hash", "grid_shape")}
        c.delete_world(wid)
    finally:
        server.terminate()
        server.wait(timeout=10)
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
