"""Sample GNC algorithms: IMU + wheel-odometry dead reckoning and a waypoint follower on top.

Usage::

    lunarsim gnc                                   # = lunarsim.gnc.examples:WaypointGnc
    lunarsim gnc --gnc lunarsim.gnc.examples:DeadReckoningGnc
"""

from __future__ import annotations

import math

import numpy as np

from .base import Gnc
from .types import Command, Observation, Pose, SimInfo


def wrap_pi(a: float) -> float:
    """Wrap an angle [rad] to [-pi, pi)."""
    return (a + math.pi) % (2 * math.pi) - math.pi


class DeadReckoning:
    """Dead reckoning from the gyro (heading), accelerometer (tilt) and wheel speed (distance).

    Assumes the IMU is mounted without rotation (IMU axes = body FLU). Between IMU samples the
    world-frame yaw rate is ``(R_wb ω_b)_z`` with roll / pitch from the gravity direction of the
    accelerometer, and the rover moves ``v cos(pitch)`` along the heading. Wheel slip and gyro
    bias are not observable here, so the estimate drifts — that is the point of the sample.
    """

    def __init__(self, pose: Pose) -> None:
        self.reset(pose)

    def reset(self, pose: Pose) -> None:
        self.x, self.y, self.z, self.yaw = pose.x, pose.y, pose.z, pose.yaw
        self.roll, self.pitch = pose.roll, pose.pitch
        self.t: float | None = None

    @property
    def pose(self) -> Pose:
        return Pose(self.x, self.y, self.z, self.yaw, self.roll, self.pitch)

    def update(self, obs: Observation) -> Pose:
        imu, v = obs.imu, obs.wheel_v
        if len(imu) == 0:  # no IMU: integrate the wheel odometry only
            self._move(v, obs.wheel_omega, obs.dt)
            return self.pose
        for k in range(len(imu)):
            t = float(imu.t[k])
            if self.t is None:
                self.t = t
                continue
            dt, self.t = t - self.t, t
            a, w = imu.accel[k], imu.gyro[k]
            self.roll = math.atan2(a[1], a[2])
            self.pitch = math.atan2(-a[0], math.hypot(a[1], a[2]))
            sr, cr = math.sin(self.roll), math.cos(self.roll)
            sp, cp = math.sin(self.pitch), math.cos(self.pitch)
            yaw_rate = -sp * w[0] + cp * sr * w[1] + cp * cr * w[2]
            self._move(v, yaw_rate, dt)
        return self.pose

    def _move(self, v: float, yaw_rate: float, dt: float) -> None:
        half = 0.5 * yaw_rate * dt  # exact arc for constant rates (as the simulator's rover)
        yaw_mid, chord = self.yaw + half, math.sin(half) / half if abs(half) > 1e-12 else 1.0
        vh = v * math.cos(self.pitch)
        self.x += vh * math.cos(yaw_mid) * dt * chord
        self.y += vh * math.sin(yaw_mid) * dt * chord
        self.z -= v * math.sin(self.pitch) * dt
        self.yaw = wrap_pi(self.yaw + yaw_rate * dt)


class DeadReckoningGnc(Gnc):
    """Estimate only; drive the rover with the waypoints from the browser UI if any (else stop)."""

    name = "dead-reckoning"

    def reset(self, info: SimInfo) -> None:
        super().reset(info)
        self.nav = DeadReckoning(info.initial_pose)

    def step(self, obs: Observation) -> Command:
        return Command(0.0, 0.0, self.nav.update(obs), "estimate only")


class WaypointGnc(Gnc):
    """Follow the browser's waypoints using the dead-reckoning estimate (not the truth).

    Control law as the simulator's built-in follower: ω = k ε, v = v_ref max(0, cos ε), where ε is
    the heading error to the next waypoint; a waypoint counts as reached within ``tolerance`` [m].
    """

    name = "waypoint"

    def __init__(
        self, speed: float | None = None, tolerance: float = 0.3, heading_gain: float = 1.5
    ) -> None:
        self.speed, self.tolerance, self.heading_gain = speed, tolerance, heading_gain

    def reset(self, info: SimInfo) -> None:
        super().reset(info)
        self.nav = DeadReckoning(info.initial_pose)
        self.goals: list[tuple[float, float]] = []
        self.index = 0

    def step(self, obs: Observation) -> Command:
        est = self.nav.update(obs)
        if obs.waypoints != self.goals:  # new route from the UI
            self.goals, self.index = list(obs.waypoints), 0
        while (
            self.index < len(self.goals)
            and math.dist(self.goals[self.index], (est.x, est.y)) <= self.tolerance
        ):
            self.index += 1
        if self.index >= len(self.goals):
            return Command(0.0, 0.0, est, "idle" if not self.goals else "route done")
        gx, gy = self.goals[self.index]
        err = wrap_pi(math.atan2(gy - est.y, gx - est.x) - est.yaw)
        speed = self.info.max_speed if self.speed is None else self.speed
        v = speed * max(0.0, math.cos(err))
        omega = float(
            np.clip(self.heading_gain * err, -self.info.max_yaw_rate, self.info.max_yaw_rate)
        )
        return Command(v, omega, est, f"wp {self.index + 1}/{len(self.goals)}")
