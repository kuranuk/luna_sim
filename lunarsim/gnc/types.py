"""Data types passed between the simulator and a guidance / navigation / control (GNC) algorithm.

Frames and units follow the simulator (CLAUDE.md): world = local ENU [m], body = FLU, yaw [rad]
from East counter-clockwise, quaternions (w, x, y, z). Sensor data are in each sensor's own frame;
the sensor mounts (body FLU xyz [m] + roll/pitch/yaw [deg]) are in :class:`SimInfo`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class Pose:
    """Planar-friendly pose: position [m] in world ENU and yaw [rad] (East = 0, CCW).

    ``roll`` / ``pitch`` [rad] are optional (ZYX Euler, pitch > 0 = nose down).
    """

    x: float
    y: float
    z: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "yaw": self.yaw,
            "roll": self.roll,
            "pitch": self.pitch,
        }

    def quaternion(self) -> tuple[float, float, float, float]:
        """(w, x, y, z) of R = Rz(yaw) Ry(pitch) Rx(roll), w >= 0."""
        cr, sr = math.cos(self.roll / 2), math.sin(self.roll / 2)
        cp, sp = math.cos(self.pitch / 2), math.sin(self.pitch / 2)
        cy, sy = math.cos(self.yaw / 2), math.sin(self.yaw / 2)
        q = (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        )
        return q if q[0] >= 0 else (-q[0], -q[1], -q[2], -q[3])


@dataclass
class Mount:
    """Sensor mount on the rover body (FLU): position [m] and roll/pitch/yaw [deg]."""

    xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rpy_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> Mount:
        d = d or {}
        return cls(
            (float(d.get("x", 0)), float(d.get("y", 0)), float(d.get("z", 0))),
            (float(d.get("rollDeg", 0)), float(d.get("pitchDeg", 0)), float(d.get("yawDeg", 0))),
        )


@dataclass
class SimInfo:
    """Session information sent once on connect (and again after a reset / new terrain).

    ``initial_pose`` is the known starting pose (navigation starts from it); ``raw`` holds the
    full scenario dictionary from the browser (camelCase keys) for anything not exposed here.
    """

    control_period: float
    initial_pose: Pose
    imu_rate_hz: float | None
    imu_mount: Mount | None
    lidar_mounts: dict[str, Mount]
    camera_mounts: dict[str, Mount]
    cameras_available: bool
    max_speed: float
    max_yaw_rate: float
    waypoints: list[tuple[float, float]]
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, info: dict[str, Any]) -> SimInfo:
        sc = info.get("scenario", {})
        p = info["initialPose"]
        imu = sc.get("imu")
        rover = sc.get("rover", {})
        return cls(
            control_period=float(info["controlPeriod"]),
            initial_pose=Pose(
                p["x"], p["y"], p["z"], p["yaw"], p.get("roll", 0.0), p.get("pitch", 0.0)
            ),
            imu_rate_hz=float(imu["rateHz"]) if imu else None,
            imu_mount=Mount.from_dict(imu.get("mount")) if imu else None,
            camera_mounts={
                cc["name"]: Mount.from_dict(cc.get("mount")) for cc in sc.get("cameras", [])
            },
            cameras_available=bool(info.get("camerasAvailable", False)),
            lidar_mounts={
                lc["name"]: Mount.from_dict(lc.get("mount")) for lc in sc.get("lidars", [])
            },
            max_speed=float(rover.get("maxSpeed", 0.2)),
            max_yaw_rate=math.radians(float(rover.get("maxYawRateDeg", 20.0))),
            waypoints=[(float(a), float(b)) for a, b in info.get("waypoints", [])],
            raw=info,
        )


@dataclass
class ImuBatch:
    """IMU samples since the previous observation: t (N,) [s], gyro (N, 3) [rad/s],
    accel (N, 3) [m/s²] (specific force), all in the IMU frame."""

    t: npt.NDArray[Any]
    gyro: npt.NDArray[Any]
    accel: npt.NDArray[Any]

    def __len__(self) -> int:
        return int(self.t.shape[0])


@dataclass
class LidarScan:
    """One LiDAR frame, in the sensor frame (FLU) at the scan time ``t`` [s].

    points (N, 4) float32 = x, y, z [m], intensity; labels (N,) uint8 (254 = false alarm / noise);
    time (N,) float32 = per-point offset from ``t`` [s]; ring (N,) uint16 = channel / row.
    """

    name: str
    frame: int
    t: float
    points: npt.NDArray[Any]
    labels: npt.NDArray[Any]
    time: npt.NDArray[Any]
    ring: npt.NDArray[Any]


@dataclass
class CameraImage:
    """One camera frame at time ``t`` [s]: linear digital numbers, top row first.

    image (H, W) uint16 in [0, 2^bits - 1] (Hapke surface + sun + headlights, shot / read noise,
    auto exposure); pinhole intrinsics fx, fy, cx, cy [px] (OpenCV, pixel-centre origin, no
    distortion); exposure [s]; the camera body is mounted as ``SimInfo.camera_mounts[name]``
    (optical frame: x right, y down, z forward = body +x).
    """

    name: str
    frame: int
    t: float
    image: npt.NDArray[Any]
    bits: int
    exposure: float
    fx: float
    fy: float
    cx: float
    cy: float
    saturated: int = 0

    @property
    def K(self) -> npt.NDArray[np.float64]:
        """3x3 camera matrix."""
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])


@dataclass
class Observation:
    """Everything the rover measured during the last control period (no ground truth).

    ``cameras`` holds the frames taken during the period (empty when the browser has no WebGL2
    and in ``web/tools/headless.mjs``). ``wheel_v`` [m/s] / ``wheel_omega`` [rad/s] are
    wheel-odometry rates (the applied command, i.e. without slip); ``waypoints`` are the goals
    set in the browser UI (world ENU [m]).
    """

    t: float
    dt: float
    imu: ImuBatch
    lidar: list[LidarScan]
    wheel_v: float
    wheel_omega: float
    waypoints: list[tuple[float, float]]
    cameras: list[CameraImage] = field(default_factory=list)


@dataclass
class Command:
    """Rover command for the next control period: forward speed v [m/s], yaw rate omega [rad/s].

    ``estimate`` (optional) is the algorithm's current pose estimate, shown against the truth in
    the browser and logged. ``status`` is an optional short text shown in the browser HUD.
    """

    v: float
    omega: float
    estimate: Pose | None = None
    status: str = ""
