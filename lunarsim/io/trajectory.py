"""Trajectory and calibration writers in common SLAM / VO evaluation formats.

* KITTI odometry poses: one line per frame, the 3x4 row-major matrix ``T_c0_ck``
  (camera pose at frame k expressed in the camera frame of frame 0).
* TUM RGB-D: ``timestamp tx ty tz qx qy qz qw`` (pose of the sensor in the world).
* KITTI ``calib.txt``: ``P0`` / ``P1`` (left / right projection, rectified pair) and
  ``Tr`` (3x4, LiDAR -> left camera).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt

from lunarsim.core.frames import Pose

__all__ = ["kitti_calib_text", "write_kitti_poses", "write_tum"]


def _row(m: npt.NDArray[np.float64]) -> str:
    return " ".join(f"{v:.12e}" for v in m.ravel())


def write_kitti_poses(path: str | Path, poses_world: Sequence[Pose]) -> None:
    """KITTI odometry pose file relative to the first pose."""
    if not poses_world:
        Path(path).write_text("")
        return
    inv0 = poses_world[0].inverse()
    lines = [_row((inv0 @ T).matrix()[:3, :]) for T in poses_world]
    Path(path).write_text("\n".join(lines) + "\n")


def write_tum(path: str | Path, times: Sequence[float], poses_world: Sequence[Pose]) -> None:
    """TUM trajectory file (quaternion written as qx qy qz qw)."""
    lines = []
    for t, T in zip(times, poses_world, strict=True):
        w, x, y, z = T.quat()
        tx, ty, tz = T.t
        lines.append(f"{t:.6f} {tx:.9f} {ty:.9f} {tz:.9f} {x:.12f} {y:.12f} {z:.12f} {w:.12f}")
    Path(path).write_text("\n".join(lines) + ("\n" if lines else ""))


def kitti_calib_text(
    K: npt.NDArray[np.float64], baseline_m: float | None, T_cam_lidar: Pose | None
) -> str:
    """KITTI-style calib.txt content (rectified, undistorted pinhole assumed)."""
    P0 = np.hstack([K, np.zeros((3, 1))])
    out = [f"P0: {_row(P0)}"]
    if baseline_m is not None:
        P1 = P0.copy()
        P1[0, 3] = -K[0, 0] * baseline_m  # right camera: x shifted by +baseline
        out.append(f"P1: {_row(P1)}")
    if T_cam_lidar is not None:
        out.append(f"Tr: {_row(T_cam_lidar.matrix()[:3, :])}")
    return "\n".join(out) + "\n"
