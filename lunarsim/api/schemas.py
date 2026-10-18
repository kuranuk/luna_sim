"""Request bodies of the REST / WebSocket API (degrees only in ``*_deg`` fields)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "CreateWorld",
    "LightSwitch",
    "RoverCommandBody",
    "StepBody",
    "SunBody",
    "WaypointsBody",
]


class _Body(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CreateWorld(_Body):
    """World from a JSON config object or a YAML string (exactly one), optional seed."""

    config: dict[str, Any] | None = None
    config_yaml: str | None = None
    seed: int | None = Field(None, ge=0)

    @model_validator(mode="after")
    def _one(self) -> CreateWorld:
        if (self.config is None) == (self.config_yaml is None):
            raise ValueError("give exactly one of 'config' or 'config_yaml'")
        return self


class SunBody(_Body):
    """Manual (azimuth/elevation) or astronomical (UTC at the configured site)."""

    azimuth_deg: float | None = Field(None, ge=0.0, lt=360.0)
    elevation_deg: float | None = Field(None, ge=-90.0, le=90.0)
    utc: str | None = None

    @model_validator(mode="after")
    def _mode(self) -> SunBody:
        manual = self.azimuth_deg is not None and self.elevation_deg is not None
        if manual == (self.utc is not None):
            raise ValueError("give either azimuth_deg + elevation_deg, or utc")
        return self


class RoverCommandBody(_Body):
    """Velocity command (clipped to the rover limits)."""

    v_m_s: float = 0.0
    omega_rad_s: float = 0.0
    steer_rad: float = 0.0


class WaypointsBody(_Body):
    """Waypoints in world ENU [m] followed at ``speed_m_s``."""

    points: list[tuple[float, float]] = Field(min_length=1)
    speed_m_s: float = Field(gt=0.0)
    tolerance_m: float = Field(0.3, gt=0.0)


class StepBody(_Body):
    """Advance simulation time."""

    dt_s: float = Field(gt=0.0, le=3600.0)


class LightSwitch(_Body):
    """Switch a light."""

    on: bool


CameraFormat = Literal["png", "png16", "npz"]
LidarFormat = Literal["npz", "ply", "pcd", "json", "bin"]
