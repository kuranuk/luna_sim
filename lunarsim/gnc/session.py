"""Protocol handling for one browser connection, independent of the WebSocket library."""

from __future__ import annotations

import json
import math
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import IO, Any

import numpy as np
import numpy.typing as npt

from .base import Gnc
from .protocol import PROTOCOL_VERSION, decode
from .types import CameraImage, Command, ImuBatch, LidarScan, Observation, SimInfo

Blobs = dict[str, npt.NDArray[Any]]


def observation_from_message(header: dict[str, Any], blobs: Blobs) -> Observation:
    """Build an :class:`Observation` from a decoded ``obs`` message (ground truth is dropped)."""
    imu = ImuBatch(
        t=blobs.get("imu/t", np.zeros(0)),
        gyro=blobs.get("imu/gyro", np.zeros((0, 3))),
        accel=blobs.get("imu/accel", np.zeros((0, 3))),
    )
    scans = []
    for s in header.get("lidar", []):
        p = f"lidar/{s['name']}/{s['frame']}/"
        scans.append(
            LidarScan(
                name=s["name"],
                frame=int(s["frame"]),
                t=float(s["t"]),
                points=blobs[p + "points"],
                labels=blobs[p + "labels"],
                time=blobs[p + "time"],
                ring=blobs[p + "ring"],
                beam=blobs.get(p + "beam", np.zeros(0, dtype=np.uint32)),
                pattern=dict(s.get("pattern", {})),
            )
        )
    cams = [
        CameraImage(
            name=c["name"], frame=int(c["frame"]), t=float(c["t"]),
            image=blobs[f"camera/{c['name']}/{c['frame']}/image"], bits=int(c["bits"]),
            exposure=float(c["exposure"]), fx=float(c["fx"]), fy=float(c["fy"]),
            cx=float(c["cx"]), cy=float(c["cy"]), saturated=int(c.get("saturated", 0)),
        )
        for c in header.get("cameras", [])
    ]  # fmt: skip
    w = header.get("wheel", {})
    return Observation(
        t=float(header["t"]),
        dt=float(header["dt"]),
        imu=imu,
        lidar=scans,
        wheel_v=float(w.get("v", 0.0)),
        wheel_omega=float(w.get("omega", 0.0)),
        waypoints=[(float(a), float(b)) for a, b in header.get("waypoints", [])],
        cameras=cams,
    )


def command_message(cmd: Command, seq: int) -> dict[str, Any]:
    v, w = float(cmd.v), float(cmd.omega)
    if not (math.isfinite(v) and math.isfinite(w)):
        raise ValueError(f"non-finite command v={v} omega={w}")
    msg: dict[str, Any] = {"type": "cmd", "seq": seq, "v": v, "omega": w, "status": cmd.status}
    if cmd.estimate is not None:
        msg["estimate"] = cmd.estimate.to_dict()
    return msg


class RunLogger:
    """Writes one run (between two ``hello`` messages) to ``<log_dir>/<run>/``:
    ``info.json``, ``truth_tum.txt`` (10 Hz), ``estimate_tum.txt`` (per control period),
    ``commands.csv``. TUM lines: ``t x y z qx qy qz qw`` (world ENU, body frame)."""

    def __init__(self, root: Path, info: dict[str, Any]) -> None:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        k = 0
        while (root / f"run_{stamp}_{k:02d}").exists():
            k += 1
        self.dir = root / f"run_{stamp}_{k:02d}"
        self.dir.mkdir(parents=True)
        (self.dir / "info.json").write_text(json.dumps(info, indent=1))
        self.files: dict[str, IO[str]] = {
            n: open(self.dir / n, "w", buffering=1)
            for n in ("truth_tum.txt", "estimate_tum.txt", "commands.csv")
        }
        self.files["commands.csv"].write("t,v,omega,status\n")

    def record(self, header: dict[str, Any], blobs: Blobs, cmd: Command) -> None:
        for t, x, y, z, qw, qx, qy, qz in blobs.get("truth/poses", np.zeros((0, 8))):
            self.files["truth_tum.txt"].write(
                f"{t:.6f} {x:.6f} {y:.6f} {z:.6f} {qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}\n"
            )
        t = float(header["t"])
        if cmd.estimate is not None:
            e = cmd.estimate
            qw, qx, qy, qz = e.quaternion()
            self.files["estimate_tum.txt"].write(
                f"{t:.6f} {e.x:.6f} {e.y:.6f} {e.z:.6f} {qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}\n"
            )
        status = cmd.status.replace(",", ";").replace("\n", " ")
        self.files["commands.csv"].write(f"{t:.6f},{cmd.v:.6f},{cmd.omega:.6f},{status}\n")

    def close(self) -> None:
        for f in self.files.values():
            f.close()


class GncSession:
    """State of one connection: feeds messages to a :class:`Gnc` and returns the replies."""

    def __init__(
        self, gnc: Gnc, log_dir: Path | None = None, log: Callable[[str], None] = print
    ) -> None:
        self.gnc, self.log_dir, self.log = gnc, log_dir, log
        self.logger: RunLogger | None = None
        self.ready = False
        self.steps = 0

    def handle_text(self, text: str) -> dict[str, Any] | None:
        msg = json.loads(text)
        if msg.get("type") != "hello":
            return None
        if int(msg.get("protocol", 0)) != PROTOCOL_VERSION:
            return {
                "type": "error",
                "message": f"protocol {msg.get('protocol')} not supported "
                f"(server speaks {PROTOCOL_VERSION})",
            }
        info = SimInfo.from_dict(msg["info"])
        self.close()
        try:
            self.gnc.reset(info)
        except Exception:
            return self._error()
        if self.log_dir is not None:
            self.logger = RunLogger(self.log_dir, msg["info"])
            self.log(f"logging to {self.logger.dir}")
        self.ready, self.steps = True, 0
        p = info.initial_pose
        self.log(
            f"reset: start ({p.x:.2f}, {p.y:.2f}) yaw {math.degrees(p.yaw):.1f} deg, "
            f"control period {info.control_period} s"
        )
        return {"type": "ready", "name": getattr(self.gnc, "name", type(self.gnc).__name__)}

    def handle_binary(self, data: bytes) -> dict[str, Any]:
        header, blobs = decode(data)
        if header.get("type") != "obs" or not self.ready:
            return {"type": "error", "message": "unexpected binary message (send hello first)"}
        try:
            cmd = self.gnc.step(observation_from_message(header, blobs))
            reply = command_message(cmd, int(header["seq"]))
        except Exception:
            return self._error()
        if self.logger is not None:
            self.logger.record(header, blobs, cmd)
        self.steps += 1
        return reply

    def _error(self) -> dict[str, Any]:
        tb = traceback.format_exc()
        self.log(tb)
        return {"type": "error", "message": tb.strip().splitlines()[-1], "traceback": tb}

    def close(self) -> None:
        if self.logger is not None:
            self.logger.close()
            self.logger = None
