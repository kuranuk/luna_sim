"""Coordinate frames and rigid transforms. All angles in radians, lengths in metres.

Frames (all right-handed):

* **World** – local ENU: x = East, y = North, z = Up. Origin at the terrain grid centre.
* **Body** – rover body FLU: x = Forward, y = Left, z = Up.
* **Camera** – OpenCV optical frame: x = Right, y = Down, z = Forward (optical axis).
* **LiDAR** – FLU like the body frame.

Naming convention: ``R_a_b`` / ``T_a_b`` maps coordinates expressed in frame *b*
into frame *a* (``p_a = R_a_b @ p_b``).

Directions given as (azimuth, elevation) use **azimuth measured from North
towards East (clockwise seen from above)** and elevation above the local horizon.

Quaternions are stored as ``(w, x, y, z)`` with unit norm.
Euler angles follow the aerospace ZYX convention: ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

__all__ = [
    "R_BODY_FROM_CAMERA",
    "R_CAMERA_FROM_BODY",
    "Pose",
    "azel_to_enu",
    "enu_to_azel",
    "euler_zyx_from_matrix",
    "matrix_from_euler_zyx",
    "matrix_from_quat",
    "matrix_from_rotvec",
    "quat_from_matrix",
    "rot_x",
    "rot_y",
    "rot_z",
    "rotvec_from_matrix",
]

FloatArray = npt.NDArray[np.float64]


def rot_x(angle: float) -> FloatArray:
    """Active rotation about +x by ``angle`` [rad]."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(angle: float) -> FloatArray:
    """Active rotation about +y by ``angle`` [rad]."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(angle: float) -> FloatArray:
    """Active rotation about +z by ``angle`` [rad]."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def matrix_from_euler_zyx(roll: float, pitch: float, yaw: float) -> FloatArray:
    """Rotation matrix ``Rz(yaw) @ Ry(pitch) @ Rx(roll)`` (angles in rad)."""
    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)


def euler_zyx_from_matrix(R: FloatArray) -> tuple[float, float, float]:
    """Inverse of :func:`matrix_from_euler_zyx`. Returns ``(roll, pitch, yaw)`` [rad].

    Pitch is in ``[-pi/2, pi/2]``. At gimbal lock (|pitch| = pi/2) roll is set to 0.
    """
    sp = -float(R[2, 0])
    sp = min(1.0, max(-1.0, sp))
    pitch = float(np.arcsin(sp))
    if abs(sp) < 1.0 - 1e-12:
        roll = float(np.arctan2(R[2, 1], R[2, 2]))
        yaw = float(np.arctan2(R[1, 0], R[0, 0]))
    else:
        roll = 0.0
        yaw = float(np.arctan2(-R[0, 1], R[1, 1]))
    return roll, pitch, yaw


def matrix_from_quat(q: npt.ArrayLike) -> FloatArray:
    """Rotation matrix from a quaternion ``(w, x, y, z)`` (normalised internally)."""
    qa = np.asarray(q, dtype=np.float64)
    n = float(np.linalg.norm(qa))
    if n == 0.0:
        raise ValueError("zero quaternion")
    w, x, y, z = qa / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def matrix_from_rotvec(rotvec: npt.ArrayLike) -> FloatArray:
    """Rotation matrix from a rotation vector (axis * angle [rad]), Rodrigues' formula."""
    v = np.asarray(rotvec, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(v))
    if theta < 1e-12:
        kx = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
        return np.asarray(np.eye(3) + kx, dtype=np.float64)
    k = v / theta
    kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.asarray(
        np.eye(3) + math.sin(theta) * kx + (1.0 - math.cos(theta)) * (kx @ kx), dtype=np.float64
    )


def quat_from_matrix(R: FloatArray) -> FloatArray:
    """Quaternion ``(w, x, y, z)`` with ``w >= 0`` from a rotation matrix (Shepperd)."""
    m = np.asarray(R, dtype=np.float64)
    tr = float(np.trace(m))
    if tr > 0.0:
        s = 2.0 * np.sqrt(tr + 1.0)
        q = np.array(
            [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
        )
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = np.array(
            [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
        )
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = np.array(
            [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
        )
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = np.array(
            [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
        )
    q = q / np.linalg.norm(q)
    if q[0] < 0.0:
        q = -q
    return np.asarray(q, dtype=np.float64)


# Camera optical frame (OpenCV) axes expressed in a FLU frame:
#   x_cam (right)  -> -y_flu,  y_cam (down) -> -z_flu,  z_cam (forward) -> +x_flu
R_BODY_FROM_CAMERA: FloatArray = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ]
)
R_BODY_FROM_CAMERA.setflags(write=False)
R_CAMERA_FROM_BODY: FloatArray = R_BODY_FROM_CAMERA.T.copy()
R_CAMERA_FROM_BODY.setflags(write=False)


def rotvec_from_matrix(R: FloatArray) -> FloatArray:
    """Rotation vector (axis · angle [rad], angle in [0, π]) of a rotation matrix."""
    q = quat_from_matrix(R)
    v = q[1:]
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.asarray(2.0 * v, dtype=np.float64)
    return np.asarray(v / n * (2.0 * math.atan2(n, float(q[0]))), dtype=np.float64)


def azel_to_enu(azimuth: npt.ArrayLike, elevation: npt.ArrayLike) -> FloatArray:
    """Unit direction(s) in ENU from azimuth (from North, clockwise) and elevation [rad].

    Returns an array of shape ``(..., 3)``.
    """
    az = np.asarray(azimuth, dtype=np.float64)
    el = np.asarray(elevation, dtype=np.float64)
    ce = np.cos(el)
    return np.stack([ce * np.sin(az), ce * np.cos(az), np.sin(el)], axis=-1)


def enu_to_azel(v: npt.ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Azimuth in ``[0, 2pi)`` and elevation in ``[-pi/2, pi/2]`` [rad] of ENU vector(s)."""
    va = np.asarray(v, dtype=np.float64)
    x, y, z = va[..., 0], va[..., 1], va[..., 2]
    az = np.mod(np.arctan2(x, y), 2.0 * np.pi)
    el = np.arctan2(z, np.hypot(x, y))
    return np.asarray(az, dtype=np.float64), np.asarray(el, dtype=np.float64)


@dataclass(frozen=True)
class Pose:
    """Rigid transform ``T_a_b`` = (R, t): ``p_a = R @ p_b + t``.

    ``R`` is a 3x3 rotation matrix, ``t`` a translation in metres.
    """

    R: FloatArray
    t: FloatArray

    def __post_init__(self) -> None:
        R = np.array(self.R, dtype=np.float64).reshape(3, 3)
        t = np.array(self.t, dtype=np.float64).reshape(3)
        R.setflags(write=False)
        t.setflags(write=False)
        object.__setattr__(self, "R", R)
        object.__setattr__(self, "t", t)

    @classmethod
    def identity(cls) -> Pose:
        """Identity transform."""
        return cls(np.eye(3), np.zeros(3))

    @classmethod
    def from_matrix(cls, T: npt.ArrayLike) -> Pose:
        """From a 4x4 homogeneous matrix."""
        Ta = np.asarray(T, dtype=np.float64)
        return cls(Ta[:3, :3], Ta[:3, 3])

    @classmethod
    def from_quat(cls, q_wxyz: npt.ArrayLike, t: npt.ArrayLike) -> Pose:
        """From quaternion ``(w, x, y, z)`` and translation [m]."""
        return cls(matrix_from_quat(q_wxyz), np.asarray(t, dtype=np.float64))

    @classmethod
    def from_euler_zyx(
        cls, roll: float, pitch: float, yaw: float, t: npt.ArrayLike = (0.0, 0.0, 0.0)
    ) -> Pose:
        """From ZYX Euler angles [rad] and translation [m]."""
        return cls(matrix_from_euler_zyx(roll, pitch, yaw), np.asarray(t, dtype=np.float64))

    def matrix(self) -> FloatArray:
        """4x4 homogeneous matrix."""
        T = np.eye(4)
        T[:3, :3] = self.R
        T[:3, 3] = self.t
        return T

    def quat(self) -> FloatArray:
        """Rotation as quaternion ``(w, x, y, z)``."""
        return quat_from_matrix(self.R)

    def inverse(self) -> Pose:
        """``T_b_a`` given ``T_a_b``."""
        Rt = self.R.T
        return Pose(Rt, -Rt @ self.t)

    def __matmul__(self, other: Pose) -> Pose:
        """Composition: ``T_a_b @ T_b_c = T_a_c``."""
        return Pose(self.R @ other.R, self.R @ other.t + self.t)

    def apply(self, points: npt.ArrayLike) -> FloatArray:
        """Transform point(s) of shape ``(..., 3)``."""
        p = np.asarray(points, dtype=np.float64)
        return np.asarray(p @ self.R.T + self.t, dtype=np.float64)

    def rotate(self, vectors: npt.ArrayLike) -> FloatArray:
        """Rotate direction vector(s) of shape ``(..., 3)`` (no translation)."""
        v = np.asarray(vectors, dtype=np.float64)
        return np.asarray(v @ self.R.T, dtype=np.float64)
