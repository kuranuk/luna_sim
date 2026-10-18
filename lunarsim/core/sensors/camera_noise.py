"""Camera sensor radiometry and noise.

Pipeline (per pixel):
    electrons = L_band · G · t_exp       (G from optics, see CameraSensorConfig)
              + dark_current · t_exp
    shot noise ~ Poisson(electrons), read noise ~ N(0, σ_read)
    clip at full well -> DN = e / gain + black level -> round -> clip to 2^bits - 1

Clipping at full well and at 0 reproduces blown-out sunlit slopes and crushed
shadows of the extreme lunar contrast. Optional auto exposure picks t_exp so that
a percentile of the noise-free signal reaches a target fraction of full well.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from lunarsim.config.schema import CameraSensorConfig

__all__ = ["SensorImage", "auto_exposure", "electrons_per_radiance", "expose"]

FloatArray = npt.NDArray[np.float64]
PLANCK = 6.62607015e-34  # [J s]
C_LIGHT = 299_792_458.0  # [m/s]


@dataclass(frozen=True)
class SensorImage:
    """Digitised image.

    Attributes:
        dn: Digital numbers (uint16), shape ``(H, W)``.
        electrons_mean: Expected signal before noise [e-].
        exposure_s: Exposure time used [s].
        saturated: True where the pixel hit full well or the ADC maximum.
    """

    dn: npt.NDArray[np.uint16]
    electrons_mean: FloatArray
    exposure_s: float
    saturated: npt.NDArray[np.bool_]


def electrons_per_radiance(s: CameraSensorConfig) -> float:
    """Electrons per second per unit in-band radiance [e- / (s · W m^-2 sr^-1)]."""
    etendue = math.pi / (4.0 * s.f_number**2) * s.pixel_pitch_m**2  # [m^2 sr]
    e_per_joule = s.quantum_efficiency * s.effective_wavelength_m / (PLANCK * C_LIGHT)
    return etendue * s.transmission * e_per_joule


def gain(s: CameraSensorConfig) -> float:
    """Conversion gain [e-/DN]."""
    if s.gain_e_per_dn is not None:
        return float(s.gain_e_per_dn)
    return float(s.full_well_e / (2**s.bits - 1))


def auto_exposure(radiance: FloatArray, s: CameraSensorConfig) -> float:
    """Exposure time [s] putting the ``ae_percentile`` of the signal at the target level."""
    rate = radiance * electrons_per_radiance(s) + s.dark_current_e_per_s
    ref = float(np.percentile(rate, s.ae_percentile))
    if ref <= 0.0:
        return s.ae_max_exposure_s
    return min(s.ae_max_exposure_s, s.ae_target_fraction * s.full_well_e / ref)


def expose(
    radiance: FloatArray,
    s: CameraSensorConfig,
    rng: np.random.Generator,
    exposure_s: float | None = None,
) -> SensorImage:
    """Simulate the sensor for an in-band radiance image [W m^-2 sr^-1]."""
    t = (
        exposure_s
        if exposure_s is not None
        else (auto_exposure(radiance, s) if s.auto_exposure else s.exposure_s)
    )
    mean_e = radiance * electrons_per_radiance(s) * t + s.dark_current_e_per_s * t
    e = rng.poisson(mean_e).astype(np.float64)
    if s.read_noise_e > 0.0:
        e += rng.normal(0.0, s.read_noise_e, size=e.shape)
    sat_well = e >= s.full_well_e
    e = np.clip(e, 0.0, s.full_well_e)
    max_dn = 2**s.bits - 1
    dn = np.rint(e / gain(s)) + s.black_level_dn
    sat = sat_well | (dn >= max_dn)
    dn = np.clip(dn, 0, max_dn).astype(np.uint16)
    return SensorImage(dn=dn, electrons_mean=mean_e, exposure_s=t, saturated=sat)
