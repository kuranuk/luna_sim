"""Sun (and Earth) direction at a lunar surface site.

Two modes:

* **manual** – azimuth/elevation given directly (tests, controlled datasets).
* **astronomical (simple analytic)** – from site latitude/longitude and UTC, using
  low-precision mean elements:

  - Sun apparent ecliptic longitude λ☉ and distance R (Astronomical Almanac
    low-precision formulae, valid ~0.01° for 1950-2050).
  - Moon mean longitude L' and mean longitude of the ascending node Ω
    (Meeus, *Astronomical Algorithms*, 2nd ed., ch. 47).
  - Sub-solar longitude = 180° − (L' − λ☉): the lunar prime meridian faces the
    mean Earth direction and the rotation is uniform with L'. Advances by 360°
    per synodic month (29.53 d).
  - Sub-solar latitude = asin(sin I · sin(λ☉ − Ω)) with I = 1.54242° the
    inclination of the lunar equator to the ecliptic (Cassini's laws; Meeus
    ch. 53). Period = draconic ("eclipse") year ≈ 346.6 d.

  The selenocentric Sun = geocentric Sun + Moon->Earth vector (removes the ≤ 0.15°
  parallax). Neglected: physical libration (~0.03°), UTC vs TT (~70 s). Compared
  against SPICE (DE421, IAU_MOON) over 2026: see ``docs/references.md``.

* **SPICE** – :mod:`lunarsim.core.illumination.spice` (optional ``spiceypy``).

The sun is treated as a point source at infinity here; the renderer samples the
solar disc for penumbrae (``render.penumbra_samples``). Earth (direction, distance)
is also provided for earthshine (:func:`earthshine_irradiance`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import numpy.typing as npt

from lunarsim.core.frames import azel_to_enu, enu_to_azel

__all__ = [
    "MOON_MEAN_RADIUS_M",
    "SOLAR_IRRADIANCE_1AU",
    "SunState",
    "earth_selenographic",
    "earthshine_irradiance",
    "julian_date",
    "subsolar_point",
    "sun_astronomical",
    "sun_manual",
]

FloatArray = npt.NDArray[np.float64]

# Total solar irradiance at 1 AU [W/m^2] (Kopp & Lean 2011, GRL 38, L01706: 1360.8 ± 0.5).
SOLAR_IRRADIANCE_1AU = 1361.0
# IAU mean radius of the Moon [m].
MOON_MEAN_RADIUS_M = 1737.4e3
AU_M = 149_597_870_700.0  # IAU 2012 [m]
EARTH_MEAN_RADIUS_M = 6371.0e3  # IUGG mean radius
LUNAR_EQUATOR_INCLINATION_DEG = 1.54242  # to the ecliptic (Meeus ch. 53)


@dataclass(frozen=True)
class SunState:
    """Illumination geometry at the site.

    Attributes:
        direction: Unit vector towards the Sun in world ENU.
        azimuth: From North, clockwise [rad].
        elevation: Above the local horizon [rad].
        irradiance: Solar irradiance normal to the beam [W/m^2].
        distance_au: Sun distance [AU].
        earth_direction: Unit vector towards Earth in ENU (None in manual mode).
        earth_distance_m: Distance to Earth's centre [m] (None in manual mode).
    """

    direction: FloatArray
    azimuth: float
    elevation: float
    irradiance: float
    distance_au: float
    earth_direction: FloatArray | None = None
    earth_distance_m: float | None = None

    @property
    def azimuth_deg(self) -> float:
        """Azimuth [deg]."""
        return math.degrees(self.azimuth)

    @property
    def elevation_deg(self) -> float:
        """Elevation [deg]."""
        return math.degrees(self.elevation)


def sun_manual(
    azimuth_deg: float,
    elevation_deg: float,
    distance_au: float = 1.0,
    irradiance_1au: float = SOLAR_IRRADIANCE_1AU,
    earth_azimuth_deg: float | None = None,
    earth_elevation_deg: float | None = None,
    earth_distance_m: float = 3.844e8,
) -> SunState:
    """Sun at a given azimuth/elevation [deg]; irradiance scaled by 1/distance².

    Earth is placed at ``earth_azimuth_deg`` / ``earth_elevation_deg`` if given.
    """
    az, el = math.radians(azimuth_deg), math.radians(elevation_deg)
    e_dir = None
    if earth_azimuth_deg is not None and earth_elevation_deg is not None:
        e_dir = azel_to_enu(math.radians(earth_azimuth_deg), math.radians(earth_elevation_deg))
    return SunState(
        direction=azel_to_enu(az, el),
        azimuth=az % (2 * math.pi),
        elevation=el,
        irradiance=irradiance_1au / distance_au**2,
        distance_au=distance_au,
        earth_direction=e_dir,
        earth_distance_m=earth_distance_m if e_dir is not None else None,
    )


def earthshine_irradiance(sun: SunState, geometric_albedo: float = 0.367) -> float:
    """Irradiance from the sunlit Earth normal to its direction [W/m²] (0 without Earth).

    ``E = S · p · (R_E / d)² · Φ(α)``: S the solar irradiance (≈ the same at Earth and
    Moon), p the geometric albedo, d the Earth distance and Φ the Lambert-sphere phase
    function ``(sin α + (π - α) cos α) / π`` (Φ(0) = 1) of the Earth phase angle α
    (angle Sun-Earth-Moon; with the Sun at infinity cos α = -ŝ·ê). Full Earth gives
    ≈ 0.14 W/m² (≈ 1e-4 of sunlight), consistent with the full-Earth magnitude ≈ -17.
    """
    if sun.earth_direction is None or sun.earth_distance_m is None:
        return 0.0
    cos_a = float(np.clip(-np.dot(sun.direction, sun.earth_direction), -1.0, 1.0))
    a = math.acos(cos_a)
    phase = (math.sin(a) + (math.pi - a) * cos_a) / math.pi
    return float(
        sun.irradiance * geometric_albedo * (EARTH_MEAN_RADIUS_M / sun.earth_distance_m) ** 2
        * phase
    )  # fmt: skip


def julian_date(utc: datetime) -> float:
    """Julian date of a UTC datetime (naive datetimes are taken as UTC)."""
    if utc.tzinfo is None:
        utc = utc.replace(tzinfo=timezone.utc)
    return 2440587.5 + utc.timestamp() / 86400.0


def _elements(jd: float) -> dict[str, float]:
    """Low-precision solar and lunar mean elements [deg, AU]."""
    d = jd - 2451545.0
    T = d / 36525.0
    g = math.radians((357.528 + 0.9856003 * d) % 360.0)
    lam_sun = (280.460 + 0.9856474 * d + 1.915 * math.sin(g) + 0.020 * math.sin(2 * g)) % 360.0
    r_au = 1.00014 - 0.01671 * math.cos(g) - 0.00014 * math.cos(2 * g)
    Lp = (218.3164477 + 481267.88123421 * T) % 360.0  # Moon mean longitude
    D = (297.8501921 + 445267.1114034 * T) % 360.0  # mean elongation
    M = (357.5291092 + 35999.0502909 * T) % 360.0  # Sun mean anomaly
    Mp = (134.9633964 + 477198.8675055 * T) % 360.0  # Moon mean anomaly
    F = (93.2720950 + 483202.0175233 * T) % 360.0  # argument of latitude
    Om = (125.0445479 - 1934.1362891 * T) % 360.0  # ascending node
    return {"lam_sun": lam_sun, "r_au": r_au, "Lp": Lp, "D": D, "M": M, "Mp": Mp, "F": F, "Om": Om}


def subsolar_point(utc: datetime) -> tuple[float, float, float]:
    """Selenographic sub-solar point ``(lat_deg, lon_deg_east, sun_distance_au)``."""
    e = _elements(julian_date(utc))
    lon = (180.0 - (e["Lp"] - e["lam_sun"])) % 360.0
    if lon > 180.0:
        lon -= 360.0
    lat = math.degrees(
        math.asin(
            math.sin(math.radians(LUNAR_EQUATOR_INCLINATION_DEG))
            * math.sin(math.radians(e["lam_sun"] - e["Om"]))
        )
    )
    return lat, lon, e["r_au"]


def earth_selenographic(utc: datetime) -> tuple[float, float, float]:
    """Sub-Earth point ``(lat_deg, lon_deg_east, distance_m)`` (optical libration).

    Truncated lunar theory (terms > 0.03° in longitude, > 0.017° in latitude of Meeus
    ch. 47, table 47.A/B) and the optical
    libration formulae of Meeus ch. 53. Accuracy ~0.1-0.3° (TODO(validation)).
    """
    e = _elements(julian_date(utc))
    r = math.radians
    D, M, Mp, F = r(e["D"]), r(e["M"]), r(e["Mp"]), r(e["F"])
    lam = e["Lp"] + (
        6.288774 * math.sin(Mp)
        + 1.274027 * math.sin(2 * D - Mp)
        + 0.658314 * math.sin(2 * D)
        + 0.213618 * math.sin(2 * Mp)
        - 0.185116 * math.sin(M)
        - 0.114332 * math.sin(2 * F)
        + 0.058793 * math.sin(2 * D - 2 * Mp)
        + 0.057066 * math.sin(2 * D - M - Mp)
        + 0.053322 * math.sin(2 * D + Mp)
        + 0.045758 * math.sin(2 * D - M)
        - 0.040923 * math.sin(M - Mp)
        - 0.034720 * math.sin(D)
        - 0.030383 * math.sin(M + Mp)
    )
    beta = (
        5.128122 * math.sin(F)
        + 0.280602 * math.sin(Mp + F)
        + 0.277693 * math.sin(Mp - F)
        + 0.173237 * math.sin(2 * D - F)
        + 0.055413 * math.sin(2 * D - Mp + F)
        + 0.046271 * math.sin(2 * D - Mp - F)
        + 0.032573 * math.sin(2 * D + F)
        + 0.017198 * math.sin(2 * Mp + F)
    )
    dist_km = 385000.56 - 20905.355 * math.cos(Mp) - 3699.111 * math.cos(2 * D - Mp)
    dist_km -= 2955.968 * math.cos(2 * D) + 569.925 * math.cos(2 * Mp)
    inc = r(LUNAR_EQUATOR_INCLINATION_DEG)
    W = r(lam - e["Om"])
    b = r(beta)
    A = math.atan2(
        math.sin(W) * math.cos(b) * math.cos(inc) - math.sin(b) * math.sin(inc),
        math.cos(W) * math.cos(b),
    )
    lon = (math.degrees(A) - e["F"]) % 360.0
    if lon > 180.0:
        lon -= 360.0
    lat = math.degrees(
        math.asin(-math.sin(W) * math.cos(b) * math.sin(inc) - math.sin(b) * math.cos(inc))
    )
    return lat, lon, dist_km * 1e3


def _selenographic_unit(lat_deg: float, lon_deg: float) -> FloatArray:
    la, lo = math.radians(lat_deg), math.radians(lon_deg)
    return np.array([math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)])


def _enu_basis(lat_deg: float, lon_deg: float) -> FloatArray:
    """Rows: East, North, Up unit vectors of the site in Moon-fixed coordinates."""
    la, lo = math.radians(lat_deg), math.radians(lon_deg)
    east = [-math.sin(lo), math.cos(lo), 0.0]
    north = [-math.sin(la) * math.cos(lo), -math.sin(la) * math.sin(lo), math.cos(la)]
    up = [math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)]
    return np.array([east, north, up])


def sun_astronomical(
    latitude_deg: float,
    longitude_deg: float,
    utc: datetime,
    irradiance_1au: float = SOLAR_IRRADIANCE_1AU,
) -> SunState:
    """Sun (and Earth) direction at selenographic site (lat, east lon) [deg] at ``utc``."""
    if not -90.0 < latitude_deg < 90.0:
        raise ValueError(
            "latitude must be strictly between -90 and 90 deg (ENU undefined at poles)"
        )
    basis = _enu_basis(latitude_deg, longitude_deg)
    slat, slon, r_au = subsolar_point(utc)
    elat, elon, edist = earth_selenographic(utc)
    moon_to_earth = edist * _selenographic_unit(elat, elon)
    # Geocentric Sun vector + Moon->Earth vector = selenocentric Sun (parallax ≤ 0.15°).
    sun_vec = r_au * AU_M * _selenographic_unit(slat, slon) + moon_to_earth
    site = MOON_MEAN_RADIUS_M * _selenographic_unit(latitude_deg, longitude_deg)
    s_enu = basis @ (sun_vec - site)
    az, el = enu_to_azel(s_enu)
    r_sun = float(np.linalg.norm(sun_vec - site)) / AU_M
    e_enu = basis @ (moon_to_earth - site)
    return SunState(
        direction=s_enu / np.linalg.norm(s_enu),
        azimuth=float(az),
        elevation=float(el),
        irradiance=irradiance_1au / r_sun**2,
        distance_au=r_sun,
        earth_direction=e_enu / np.linalg.norm(e_enu),
        earth_distance_m=float(np.linalg.norm(e_enu)),
    )
