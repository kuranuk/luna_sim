"""Rover spotlights: angular profile, radiant intensity and pose in the world."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from lunarsim.config.schema import SpotlightConfig
from lunarsim.core.frames import Pose

__all__ = ["LightState", "light_state", "spot_profile"]

FloatArray = npt.NDArray[np.float64]


def spot_profile(theta: npt.ArrayLike, inner: float, outer: float) -> FloatArray:
    """Relative intensity vs off-axis angle [rad].

    1 inside ``inner``, smoothstep fall-off to 0 at ``outer``.
    """
    th = np.asarray(theta, dtype=np.float64)
    t = np.clip((th - inner) / (outer - inner), 0.0, 1.0)
    return np.asarray(1.0 - t * t * (3.0 - 2.0 * t), dtype=np.float64)


def _profile_solid_angle(inner: float, outer: float) -> float:
    """``∫ f(θ) dΩ = 2π ∫_0^outer f(θ) sin θ dθ`` [sr]: exact inner cone + numerical ramp."""
    th = np.linspace(inner, outer, 4001)
    y = spot_profile(th, inner, outer) * np.sin(th)
    ramp = (th[1] - th[0]) * (y.sum() - 0.5 * (y[0] + y[-1]))
    return float(2.0 * math.pi * (1.0 - math.cos(inner) + ramp))


@dataclass(frozen=True)
class LightState:
    """A spotlight placed in the world.

    Attributes:
        position: World ENU position [m].
        axis: Unit beam axis in world ENU.
        intensity: On-axis in-band radiant intensity [W/sr].
        inner, outer: Beam / field half-angles [rad].
    """

    position: FloatArray
    axis: FloatArray
    intensity: float
    inner: float
    outer: float

    def intensity_towards(self, direction_from_light: FloatArray) -> FloatArray:
        """Radiant intensity [W/sr] towards unit directions (``(N, 3)``, from the light)."""
        c = np.clip(direction_from_light @ self.axis, -1.0, 1.0)
        return self.intensity * spot_profile(np.arccos(c), self.inner, self.outer)


def light_state(cfg: SpotlightConfig, T_world_light: Pose) -> LightState:
    """World placement and on-axis intensity of a spotlight (beam along the light's +x)."""
    inner = math.radians(cfg.beam_half_angle_deg)
    outer = math.radians(cfg.field_half_angle_deg)
    power = cfg.flux_lm / cfg.luminous_efficacy_lm_per_w * cfg.camera_band_fraction  # [W]
    i0 = power / _profile_solid_angle(inner, outer)
    axis = T_world_light.rotate([1.0, 0.0, 0.0])
    return LightState(T_world_light.t.copy(), axis, i0, inner, outer)
