"""Waypoint following (simple heading controller), producing rover velocity commands.

For each step: aim at the current waypoint; heading error ε (rad) gives
``ω = clip(k ε)`` and ``v = v_ref · max(0, cos ε)`` (slow down / turn in place when
misaligned). Ackermann rovers get ``δ = atan(L ω / v)``. A waypoint is reached when
the ground track is within ``tolerance_m``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

__all__ = ["WaypointFollower"]


@dataclass
class WaypointFollower:
    """Sequential waypoint follower (positions in world ENU [m])."""

    waypoints: list[tuple[float, float]]
    speed_m_s: float
    tolerance_m: float = 0.3
    heading_gain: float = 1.5
    index: int = field(default=0)

    @property
    def done(self) -> bool:
        """True when all waypoints have been reached."""
        return self.index >= len(self.waypoints)

    def command(
        self, x: float, y: float, yaw: float, kinematics: str, wheelbase_m: float
    ) -> tuple[float, float, float]:
        """``(v, omega, steer)`` for the current ground track and heading [rad]."""
        while not self.done:
            tx, ty = self.waypoints[self.index]
            if math.hypot(tx - x, ty - y) > self.tolerance_m:
                break
            self.index += 1
        if self.done:
            return 0.0, 0.0, 0.0
        tx, ty = self.waypoints[self.index]
        err = math.atan2(ty - y, tx - x) - yaw
        err = (err + math.pi) % (2.0 * math.pi) - math.pi
        omega = self.heading_gain * err
        v = self.speed_m_s * max(0.0, math.cos(err))
        if kinematics == "ackermann":
            steer = math.atan2(wheelbase_m * omega, max(v, 1e-3))
            return max(v, 0.05 * self.speed_m_s), 0.0, steer
        return v, omega, 0.0
