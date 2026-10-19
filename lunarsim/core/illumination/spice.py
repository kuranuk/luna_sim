"""High-precision Sun/Earth geometry with SPICE (optional ``spiceypy`` + local kernels).

Kernels are never downloaded: the caller lists local files (or a meta-kernel). A
typical set is an LSK (``naif0012.tls``), a planetary SPK (``de440s.bsp`` or
``de421.bsp``) and a PCK defining the lunar body-fixed frame - ``pck00011.tpc`` for
``IAU_MOON``, or the lunar PA binary PCK + ``moon_*.tf`` for ``MOON_ME``.

Geometry: positions of the Sun and Earth relative to the Moon's centre in the lunar
body-fixed frame (aberration ``LT+S``), minus the site position on the sphere of radius
``MOON_MEAN_RADIUS_M`` at the planetocentric latitude / east longitude, rotated into the
site's local ENU frame.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any

import numpy as np

from lunarsim.core.frames import enu_to_azel
from lunarsim.core.illumination.sun import (
    AU_M,
    MOON_MEAN_RADIUS_M,
    SOLAR_IRRADIANCE_1AU,
    SunState,
    _enu_basis,
    _selenographic_unit,
)

__all__ = ["spice_available", "sun_spice"]

_LOCK = threading.Lock()  # the SPICE kernel pool and error state are process-global
_LOADED: set[str] = set()


def _spiceypy() -> Any:
    import spiceypy

    return spiceypy


def spice_available(kernels: Sequence[str]) -> bool:
    """True if spiceypy imports and every kernel file exists."""
    from pathlib import Path

    if not kernels:
        return False
    try:
        _spiceypy()
    except ImportError:
        return False
    return all(Path(k).is_file() for k in kernels)


def _furnsh(sp: Any, kernels: Sequence[str]) -> None:
    for k in kernels:
        if k not in _LOADED:
            sp.furnsh(k)
            _LOADED.add(k)


def sun_spice(
    latitude_deg: float,
    longitude_deg: float,
    utc: datetime,
    kernels: Sequence[str],
    frame: str = "IAU_MOON",
    irradiance_1au: float = SOLAR_IRRADIANCE_1AU,
) -> SunState:
    """Sun and Earth direction at a selenographic site from SPICE kernels.

    Args:
        latitude_deg: Planetocentric latitude of the site [deg].
        longitude_deg: East longitude [deg].
        utc: Time (naive datetimes are UTC).
        kernels: Local kernel files (loaded once per process).
        frame: Lunar body-fixed frame (``IAU_MOON`` or ``MOON_ME``).
        irradiance_1au: Solar irradiance at 1 AU [W/m²].
    """
    if not -90.0 < latitude_deg < 90.0:
        raise ValueError("latitude must be strictly between -90 and 90 deg")
    sp = _spiceypy()
    if utc.tzinfo is not None:
        utc = utc.astimezone(timezone.utc).replace(tzinfo=None)
    with _LOCK:
        _furnsh(sp, kernels)
        et = sp.str2et(utc.strftime("%Y-%m-%dT%H:%M:%S.%f"))
        sun_km, _ = sp.spkpos("SUN", et, frame, "LT+S", "MOON")
        earth_km, _ = sp.spkpos("EARTH", et, frame, "LT+S", "MOON")
    site = MOON_MEAN_RADIUS_M * _selenographic_unit(latitude_deg, longitude_deg)
    basis = _enu_basis(latitude_deg, longitude_deg)
    s_vec = np.asarray(sun_km, dtype=np.float64) * 1e3 - site
    e_vec = np.asarray(earth_km, dtype=np.float64) * 1e3 - site
    s_enu, e_enu = basis @ s_vec, basis @ e_vec
    az, el = enu_to_azel(s_enu)
    r_au = float(np.linalg.norm(s_vec)) / AU_M
    return SunState(
        direction=s_enu / np.linalg.norm(s_enu),
        azimuth=float(az),
        elevation=float(el),
        irradiance=irradiance_1au / r_au**2,
        distance_au=r_au,
        earth_direction=e_enu / np.linalg.norm(e_enu),
        earth_distance_m=float(np.linalg.norm(e_enu)),
    )


def angular_separation_deg(a: SunState, b: SunState) -> tuple[float, float]:
    """Angles [deg] between the Sun directions and between the Earth directions."""
    s = math.degrees(math.acos(float(np.clip(np.dot(a.direction, b.direction), -1.0, 1.0))))
    e = float("nan")
    if a.earth_direction is not None and b.earth_direction is not None:
        c = float(np.clip(np.dot(a.earth_direction, b.earth_direction), -1.0, 1.0))
        e = math.degrees(math.acos(c))
    return s, e
