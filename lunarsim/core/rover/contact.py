"""Static terrain contact: attitude and height from the wheel contact points.

Each wheel rests on the highest terrain point of its contact footprint (3 x 3 samples
over ±0.4 R along and ±W/2 across the wheel), so sub-wheel roughness does not make the
attitude jitter. These heights are fitted by a plane (least squares), which is
the static equilibrium of a chassis whose suspension equalises the wheel loads
(differential / rocker-bogie-like). Wheels are treated as points at their lowest
footprint maximum; wheel-terrain penetration and rigid 3-point support are not modelled.

Attitude convention (ZYX Euler, FLU body): pitch > 0 = nose down, roll > 0 = left
side up. Static tip-over: the gravity direction in the body frame must stay within
the support rectangle seen from the centre of gravity.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from lunarsim.config.schema import RoverConfig
from lunarsim.core.frames import Pose, matrix_from_euler_zyx

__all__ = ["ContactState", "HeightFn", "settle", "wheel_offsets"]

FloatArray = npt.NDArray[np.float64]
HeightFn = Callable[[FloatArray, FloatArray], FloatArray]


def wheel_offsets(cfg: RoverConfig) -> FloatArray:
    """Wheel contact positions (x forward, y left) in the body frame [m], ``(n, 2)``."""
    hx, hy = 0.5 * cfg.wheelbase_m, 0.5 * cfg.track_m
    xs = [hx, -hx] if cfg.wheels == 4 else [hx, 0.0, -hx]
    return np.array([(x, y) for x in xs for y in (hy, -hy)])


@dataclass(frozen=True)
class ContactState:
    """Rover resting on the terrain.

    Attributes:
        pose: ``T_world_body``.
        roll, pitch, yaw: Attitude [rad] (yaw from East, counter-clockwise).
        wheel_points: Wheel contact points in world ENU ``(n, 3)`` [m].
        slope: Tilt of the body z axis from vertical [rad].
        lateral_margin, longitudinal_margin: Static tip-over margins [rad] (< 0: tipped).
        fit_residual_m: RMS distance of the contact points from the fitted plane [m].
    """

    pose: Pose
    roll: float
    pitch: float
    yaw: float
    wheel_points: FloatArray
    slope: float
    lateral_margin: float
    longitudinal_margin: float
    fit_residual_m: float


def settle(
    height: HeightFn, cfg: RoverConfig, x: float, y: float, yaw: float, iterations: int = 4
) -> ContactState:
    """Rest the rover with contact-centre ground track ``(x, y)`` [m] and heading ``yaw``."""
    w = wheel_offsets(cfg)
    lx = cfg.contact_half_length_ratio * cfg.wheel_radius_m
    ly = 0.5 * cfg.wheel_width_m
    fu, fv = np.meshgrid([-lx, 0.0, lx], [-ly, 0.0, ly])
    foot = np.stack([fu.ravel(), fv.ravel()], axis=1)  # footprint offsets (body x, y)
    n_w = len(w)
    contacts_body = np.zeros((n_w, 3))
    contacts_body[:, :2] = w
    samp = (w[:, None, :] + foot[None, :, :]).reshape(-1, 2)
    samp_body = np.zeros((len(samp), 3))
    samp_body[:, :2] = samp
    roll = pitch = 0.0
    cy, sy = math.cos(yaw), math.sin(yaw)
    for _ in range(iterations):
        R = matrix_from_euler_zyx(roll, pitch, yaw)
        rel = contacts_body @ R.T  # contact offsets from the contact centre, world
        wx, wy = x + rel[:, 0], y + rel[:, 1]
        srel = samp_body @ R.T
        wz = height(x + srel[:, 0], y + srel[:, 1]).reshape(n_w, -1).max(axis=1)
        # Yaw-aligned horizontal coordinates (u forward, v left).
        u = (wx - x) * cy + (wy - y) * sy
        v = -(wx - x) * sy + (wy - y) * cy
        A = np.stack([np.ones_like(u), u, v], axis=1)
        a, b, c = np.linalg.solve(A.T @ A, A.T @ wz)  # least-squares plane (normal eqs.)
        nrm = math.sqrt(1.0 + b * b + c * c)
        roll_new = math.asin(c / nrm)
        pitch_new = math.atan2(-b, 1.0)
        done = abs(roll_new - roll) < 1e-13 and abs(pitch_new - pitch) < 1e-13
        roll, pitch = roll_new, pitch_new
        if done:  # converged: further iterations would sample the same points
            break
    resid = float(np.sqrt(np.mean((A @ np.array([a, b, c]) - wz) ** 2)))
    R = matrix_from_euler_zyx(roll, pitch, yaw)
    origin = np.array([x, y, a]) + cfg.origin_height_m * R[:, 2]
    g_body = R.T @ np.array([0.0, 0.0, -1.0])
    tilt_lat = abs(math.atan2(g_body[1], -g_body[2]))
    tilt_lon = abs(math.atan2(g_body[0], -g_body[2]))
    lat_lim = math.atan2(0.5 * cfg.track_m, cfg.cg_height_m)
    lon_lim = math.atan2(0.5 * cfg.wheelbase_m, cfg.cg_height_m)
    return ContactState(
        pose=Pose(R, origin),
        roll=roll,
        pitch=pitch,
        yaw=yaw,
        wheel_points=np.stack([wx, wy, wz], axis=1),
        slope=math.acos(max(-1.0, min(1.0, R[2, 2]))),
        lateral_margin=lat_lim - tilt_lat,
        longitudinal_margin=lon_lim - tilt_lon,
        fit_residual_m=resid,
    )
