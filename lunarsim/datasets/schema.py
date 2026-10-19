"""Dataset job description (YAML), validated with pydantic."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = ["DatasetConfig", "SunVariation", "TrajectoryConfig", "load_dataset_config"]


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class VelocitySegment(_M):
    """Constant velocity command from ``t_s`` on."""

    t_s: float = Field(ge=0.0)
    v_m_s: float = 0.0
    omega_rad_s: float = 0.0
    steer_rad: float = 0.0


class TrajectoryConfig(_M):
    """Either waypoints (followed at ``speed_m_s``) or a piecewise-constant command program."""

    waypoints: list[tuple[float, float]] | None = None
    speed_m_s: float = Field(0.2, gt=0.0)
    commands: list[VelocitySegment] | None = None

    @model_validator(mode="after")
    def _one(self) -> TrajectoryConfig:
        if (self.waypoints is None) == (self.commands is None):
            raise ValueError("give exactly one of 'waypoints' or 'commands'")
        return self


class SunVariation(_M):
    """One run of the same trajectory under a different illumination."""

    name: str
    azimuth_deg: float | None = Field(None, ge=0.0, lt=360.0)
    elevation_deg: float | None = Field(None, ge=-90.0, le=90.0)
    utc: str | None = None
    lights_on: bool | None = None  # None: as configured

    @model_validator(mode="after")
    def _mode(self) -> SunVariation:
        manual = self.azimuth_deg is not None or self.elevation_deg is not None
        if manual and (self.azimuth_deg is None or self.elevation_deg is None):
            raise ValueError("give both azimuth_deg and elevation_deg")
        if manual and self.utc is not None:
            raise ValueError("give either azimuth/elevation or utc")
        return self


class DatasetConfig(_M):
    """Dataset job: scenario + trajectory + sensors + illumination variations."""

    scenario: str  # path to a scenario YAML (relative to the dataset YAML)
    seed: int | None = Field(None, ge=0)
    output_dir: str = "out/datasets/run"
    duration_s: float = Field(gt=0.0)
    frame_rate_hz: float = Field(2.0, gt=0.0, le=100.0)
    trajectory: TrajectoryConfig
    cameras: list[str] = []
    stereo: bool = True  # also write the right image for stereo cameras
    camera_buffers: list[Literal["depth", "labels", "shadow"]] = ["depth", "labels"]
    lidars: list[str] = []
    lidar_formats: list[Literal["bin", "ply", "pcd", "npz"]] = ["bin"]
    imus: list[str] = []
    variations: list[SunVariation] = [SunVariation(name="default")]

    @model_validator(mode="after")
    def _names(self) -> DatasetConfig:
        names = [v.name for v in self.variations]
        if len(set(names)) != len(names) or any(not n or "/" in n for n in names):
            raise ValueError("variation names must be unique, non-empty and without '/'")
        return self


def load_dataset_config(path: str | Path) -> tuple[DatasetConfig, Path]:
    """Load a dataset YAML; returns the config and the resolved scenario path."""
    p = Path(path)
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    cfg = DatasetConfig.model_validate(data)
    scenario = Path(cfg.scenario)
    if not scenario.is_absolute():
        cand = (p.parent / scenario).resolve()
        scenario = cand if cand.exists() else scenario.resolve()
    return cfg, scenario
