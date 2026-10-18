"""Rover state, kinematic models and time stepping with slope-dependent slip.

Kinematics (commands in the body frame):
* ``skid_steer``: forward speed v [m/s] and yaw rate ω [rad/s].
* ``ackermann``: forward speed v and steering angle δ; ω = v tan(δ) / wheelbase.

Commands are clipped to the configured limits. Each sub-step integrates the ground
track (x, y, yaw) with the slipped body velocity, then re-settles the rover on the
terrain (static contact, see :mod:`.contact`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from lunarsim.config.schema import RoverConfig, SlipConfig
from lunarsim.core.frames import Pose
from lunarsim.core.rover.contact import ContactState, HeightFn, settle

__all__ = ["Rover", "RoverCommand", "RoverStatus", "longitudinal_slip"]


@dataclass(frozen=True)
class RoverCommand:
    """Velocity command. ``omega`` is used by skid-steer, ``steer`` by Ackermann [rad]."""

    v: float = 0.0
    omega: float = 0.0
    steer: float = 0.0


@dataclass(frozen=True)
class RoverStatus:
    """Outcome of the last step.

    Attributes:
        slip: Longitudinal slip ratio (>0: slower than commanded, <0: skidding).
        lateral_velocity: Side-slope drift [m/s] (body y).
        body_velocity: Actual body-frame velocity (vx, vy) [m/s].
        yaw_rate: Actual yaw rate [rad/s].
        slope_deg: Body tilt from vertical [deg].
        warnings: Active warnings (``slope_exceeded``, ``tipover_risk``, ``tipped_over``,
            ``stalled``).
    """

    slip: float = 0.0
    lateral_velocity: float = 0.0
    body_velocity: tuple[float, float] = (0.0, 0.0)
    yaw_rate: float = 0.0
    slope_deg: float = 0.0
    warnings: tuple[str, ...] = ()


def longitudinal_slip(uphill_angle: float, s: SlipConfig) -> float:
    """Slip ratio for an uphill (positive) / downhill (negative) angle [rad]."""
    crit = math.radians(s.critical_slope_deg)
    if uphill_angle >= 0.0:
        up = s.slip_flat + (1.0 - s.slip_flat) * (uphill_angle / crit) ** s.exponent
        return float(min(1.0, up))
    return float(-min(1.0, s.downhill_skid * (-uphill_angle / crit) ** s.exponent))


@dataclass
class Rover:
    """Rover on a terrain given by a vectorised height function ``height(x, y)``."""

    cfg: RoverConfig
    height: HeightFn
    x: float = field(init=False)
    y: float = field(init=False)
    yaw: float = field(init=False)
    command: RoverCommand = field(init=False, default_factory=RoverCommand)
    contact: ContactState = field(init=False)
    status: RoverStatus = field(init=False, default_factory=RoverStatus)
    odometer_m: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        self.reset(self.cfg.x_m, self.cfg.y_m, math.radians(self.cfg.yaw_deg))

    def reset(self, x: float, y: float, yaw: float) -> None:
        """Place the rover at ground track ``(x, y)`` [m] with heading ``yaw`` [rad]."""
        self.x, self.y, self.yaw = float(x), float(y), float(yaw)
        self.command = RoverCommand()
        self.contact = settle(self.height, self.cfg, self.x, self.y, self.yaw)
        self.status = self._status(0.0, 0.0, 0.0, 0.0, 0.0)

    def reset_contact(self) -> None:
        """Re-settle on the terrain at the current ground track (e.g. after re-meshing)."""
        self.contact = settle(self.height, self.cfg, self.x, self.y, self.yaw)

    @property
    def pose(self) -> Pose:
        """``T_world_body``."""
        return self.contact.pose

    def set_command(self, v: float, omega: float = 0.0, steer: float = 0.0) -> RoverCommand:
        """Set (and clip) the velocity command."""
        c = self.cfg
        vmax = c.max_speed_m_s
        wmax = math.radians(c.max_yaw_rate_deg_s)
        smax = math.radians(c.max_steer_deg)
        self.command = RoverCommand(
            v=float(np.clip(v, -vmax, vmax)),
            omega=float(np.clip(omega, -wmax, wmax)),
            steer=float(np.clip(steer, -smax, smax)),
        )
        return self.command

    def _yaw_rate(self) -> float:
        cmd = self.command
        if self.cfg.kinematics == "skid_steer":
            return cmd.omega
        w = cmd.v * math.tan(cmd.steer) / self.cfg.wheelbase_m
        wmax = math.radians(self.cfg.max_yaw_rate_deg_s)
        return float(np.clip(w, -wmax, wmax))

    def _status(self, slip: float, vy: float, vx: float, wz: float, v_cmd: float) -> RoverStatus:
        c, ct = self.cfg, self.contact
        warn = []
        if ct.slope > math.radians(c.max_slope_deg):
            warn.append("slope_exceeded")
        margin = min(ct.lateral_margin, ct.longitudinal_margin)
        if margin < 0.0:
            warn.append("tipped_over")
        elif margin < math.radians(c.tipover_warning_margin_deg):
            warn.append("tipover_risk")
        if v_cmd != 0.0 and slip >= 1.0:
            warn.append("stalled")
        return RoverStatus(slip, vy, (vx, vy), wz, math.degrees(ct.slope), tuple(warn))

    def step(self, dt: float) -> RoverStatus:
        """Advance by ``dt`` [s] (sub-stepped) and return the status."""
        if dt < 0.0:
            raise ValueError("dt must be non-negative")
        n = max(1, math.ceil(dt / self.cfg.max_substep_s)) if dt > 0 else 0
        h = dt / n if n else 0.0
        slip = vy = vx = wz = 0.0
        for _ in range(n):
            if "tipped_over" in self.status.warnings:
                break  # a tipped rover does not move
            v = self.command.v
            ct = self.contact
            uphill = -ct.pitch if v >= 0.0 else ct.pitch
            slip = longitudinal_slip(uphill, self.cfg.slip) if v != 0.0 else 0.0
            vx = v * (1.0 - slip)
            vy = -self.cfg.slip.lateral_coefficient * abs(v) * math.sin(ct.roll)
            wz = self._yaw_rate()
            # Horizontal ground-track velocity (body x projected by the pitch); exact arc
            # for constant velocities over the sub-step: chord = v h sinc(ω h / 2).
            half = 0.5 * wz * h
            yaw_mid = self.yaw + half
            chord = math.sin(half) / half if abs(half) > 1e-12 else 1.0
            vh = vx * math.cos(ct.pitch)
            dx = (vh * math.cos(yaw_mid) - vy * math.sin(yaw_mid)) * h * chord
            dy = (vh * math.sin(yaw_mid) + vy * math.cos(yaw_mid)) * h * chord
            self.x += dx
            self.y += dy
            self.yaw = (self.yaw + wz * h + math.pi) % (2.0 * math.pi) - math.pi
            self.odometer_m += abs(vx) * h
            self.contact = settle(self.height, self.cfg, self.x, self.y, self.yaw)
            self.status = self._status(slip, vy, vx, wz, v)
        return self.status

    def body_twist_world(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Linear [m/s] and angular [rad/s] velocity of the body origin in world ENU."""
        R = self.pose.R
        vx, vy = self.status.body_velocity
        lin = R @ np.array([vx, vy, 0.0])
        ang = np.array([0.0, 0.0, self.status.yaw_rate])
        return lin, ang
