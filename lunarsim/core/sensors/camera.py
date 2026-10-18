"""Pinhole camera with Brown-Conrady distortion (OpenCV conventions).

Camera frame: OpenCV optical (x right, y down, z forward). Pixel (u, v) with the
origin at the *centre* of the top-left pixel. Distortion acts on normalised image
coordinates ``(x, y) = (X/Z, Y/Z)``::

    r² = x² + y²
    x_d = x (1 + k1 r² + k2 r⁴ + k3 r⁶) + 2 p1 x y + p2 (r² + 2 x²)
    y_d = y (1 + k1 r² + k2 r⁴ + k3 r⁶) + p1 (r² + 2 y²) + 2 p2 x y
    u = fx x_d + cx,  v = fy y_d + cy
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from lunarsim.config.schema import CameraIntrinsicsConfig
from lunarsim.core.frames import R_BODY_FROM_CAMERA, Pose, matrix_from_euler_zyx

__all__ = ["CameraModel", "camera_pose_from_body"]

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class CameraModel:
    """Intrinsics and distortion of a pinhole camera."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    dist: tuple[float, float, float, float, float]  # k1, k2, p1, p2, k3

    @classmethod
    def from_config(cls, c: CameraIntrinsicsConfig) -> CameraModel:
        """Build from config; ``fx`` defaults to the value giving ``hfov_deg``."""
        fx = c.fx if c.fx is not None else 0.5 * c.width / math.tan(math.radians(c.hfov_deg) / 2)
        fy = c.fy if c.fy is not None else fx
        cx = c.cx if c.cx is not None else 0.5 * (c.width - 1)
        cy = c.cy if c.cy is not None else 0.5 * (c.height - 1)
        return cls(c.width, c.height, fx, fy, cx, cy, (c.k1, c.k2, c.p1, c.p2, c.k3))

    @property
    def K(self) -> FloatArray:
        """3x3 camera matrix."""
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])

    def distort(self, x: FloatArray, y: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Apply Brown-Conrady distortion to normalised coordinates."""
        k1, k2, p1, p2, k3 = self.dist
        r2 = x * x + y * y
        rad = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))
        xd = x * rad + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        yd = y * rad + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
        return xd, yd

    def undistort(
        self, xd: FloatArray, yd: FloatArray, iterations: int = 30
    ) -> tuple[FloatArray, FloatArray]:
        """Invert :meth:`distort` by Newton iteration (exact for zero distortion)."""
        if not any(self.dist):
            return xd.copy(), yd.copy()
        k1, k2, p1, p2, k3 = self.dist
        x, y = xd.copy(), yd.copy()
        for _ in range(iterations):
            r2 = x * x + y * y
            rad = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))
            drad = k1 + r2 * (2.0 * k2 + 3.0 * k3 * r2)  # d rad / d r2
            fx_ = x * rad + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x) - xd
            fy_ = y * rad + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y - yd
            a = rad + 2.0 * x * x * drad + 2.0 * p1 * y + 6.0 * p2 * x
            b = 2.0 * x * y * drad + 2.0 * p1 * x + 2.0 * p2 * y
            c = 2.0 * x * y * drad + 2.0 * p1 * x + 2.0 * p2 * y
            d = rad + 2.0 * y * y * drad + 6.0 * p1 * y + 2.0 * p2 * x
            det = a * d - b * c
            x = x - (d * fx_ - b * fy_) / det
            y = y - (-c * fx_ + a * fy_) / det
        return x, y

    def project(self, points_cam: npt.ArrayLike) -> tuple[FloatArray, npt.NDArray[np.bool_]]:
        """Pixels ``(N, 2)`` of camera-frame points ``(N, 3)`` and a validity mask.

        Valid = in front of the camera (Z > 0) and inside the image.
        """
        p = np.asarray(points_cam, dtype=np.float64).reshape(-1, 3)
        z = p[:, 2]
        front = z > 1e-9
        zs = np.where(front, z, 1.0)
        xd, yd = self.distort(p[:, 0] / zs, p[:, 1] / zs)
        u = self.fx * xd + self.cx
        v = self.fy * yd + self.cy
        inside = (u >= -0.5) & (u <= self.width - 0.5) & (v >= -0.5) & (v <= self.height - 0.5)
        return np.stack([u, v], axis=1), front & inside

    def unproject(self, pixels: npt.ArrayLike) -> FloatArray:
        """Unit ray directions in the camera frame for pixels ``(N, 2)``."""
        px = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
        xd = (px[:, 0] - self.cx) / self.fx
        yd = (px[:, 1] - self.cy) / self.fy
        x, y = self.undistort(xd, yd)
        d = np.stack([x, y, np.ones_like(x)], axis=1)
        return np.asarray(d / np.linalg.norm(d, axis=1, keepdims=True))

    def pixel_rays(self) -> FloatArray:
        """Unit rays (camera frame) for all pixel centres, shape ``(H*W, 3)`` row-major."""
        u, v = np.meshgrid(np.arange(self.width, dtype=np.float64), np.arange(self.height))
        return self.unproject(np.stack([u.ravel(), v.ravel()], axis=1))


def camera_pose_from_body(
    position: npt.ArrayLike, roll: float = 0.0, pitch: float = 0.0, yaw: float = 0.0
) -> Pose:
    """``T_world_camera`` (optical frame) for a FLU camera body at ``position`` [m].

    ``yaw`` is measured from East (world +x) counter-clockwise, ``pitch`` positive
    nose-down (FLU, right-handed about +y), ``roll`` about the forward axis [rad].
    """
    R_world_body = matrix_from_euler_zyx(roll, pitch, yaw)
    return Pose(R_world_body @ R_BODY_FROM_CAMERA, np.asarray(position, dtype=np.float64))
