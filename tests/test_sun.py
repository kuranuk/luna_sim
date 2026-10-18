from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from lunarsim.core.illumination.sun import (
    SOLAR_IRRADIANCE_1AU,
    earth_selenographic,
    subsolar_point,
    sun_astronomical,
    sun_manual,
)

T0 = datetime(2026, 3, 1, tzinfo=timezone.utc)
SYNODIC = 29.530589  # days (mean synodic month)
DRACONIC_YEAR = 346.620  # days


def test_manual_directions() -> None:
    np.testing.assert_allclose(sun_manual(0.0, 0.0).direction, [0, 1, 0], atol=1e-15)
    np.testing.assert_allclose(sun_manual(90.0, 0.0).direction, [1, 0, 0], atol=1e-15)
    np.testing.assert_allclose(sun_manual(123.0, 90.0).direction, [0, 0, 1], atol=1e-15)
    s = sun_manual(200.0, 10.0, distance_au=2.0)
    assert s.irradiance == pytest.approx(SOLAR_IRRADIANCE_1AU / 4)
    assert s.azimuth_deg == pytest.approx(200.0) and s.elevation_deg == pytest.approx(10.0)


def test_subsolar_longitude_period() -> None:
    _, lon0, _ = subsolar_point(T0)
    _, lon1, _ = subsolar_point(T0 + timedelta(days=SYNODIC))
    dlon = (lon1 - lon0 + 180.0) % 360.0 - 180.0
    assert abs(dlon) < 2.0  # mean month; equation of centre of the Sun varies ~±1.9°
    # Westward drift: half a month later the sub-solar point is on the opposite side.
    _, lon_h, _ = subsolar_point(T0 + timedelta(days=SYNODIC / 2))
    assert abs(((lon_h - lon0) % 360.0) - 180.0) < 3.0


def test_subsolar_latitude_amplitude_and_period() -> None:
    lats = [subsolar_point(T0 + timedelta(days=float(d)))[0] for d in np.arange(0, 400, 2.0)]
    assert max(np.abs(lats)) <= 1.5425
    assert max(lats) > 1.5 and min(lats) < -1.5
    a = subsolar_point(T0)[0]
    b = subsolar_point(T0 + timedelta(days=DRACONIC_YEAR))[0]
    assert a == pytest.approx(b, abs=0.03)


def test_new_moon_subsolar_far_side() -> None:
    # Meeus' reference mean new moon: 2000-01-06 18:14 (TT); sub-solar point on the far side.
    _, lon, r = subsolar_point(datetime(2000, 1, 6, 18, 14, tzinfo=timezone.utc))
    assert abs(abs(lon) - 180.0) < 8.0
    assert r == pytest.approx(0.983, abs=0.002)  # near perihelion (early January)


def test_earth_near_subearth_point() -> None:
    for d in range(0, 60, 7):
        lat, lon, dist = earth_selenographic(T0 + timedelta(days=d))
        assert abs(lat) < 7.0 and abs(lon) < 8.5  # optical libration limits
        assert 356e6 < dist < 407e6


def test_south_pole_low_sun() -> None:
    els = [
        sun_astronomical(-89.5, 0.0, T0 + timedelta(days=float(d))).elevation_deg
        for d in np.arange(0, 360, 3.0)
    ]
    assert max(np.abs(els)) < 0.5 + 1.55 + 0.05
    s = sun_astronomical(-89.5, 0.0, T0)
    assert s.earth_direction is not None and np.isclose(np.linalg.norm(s.earth_direction), 1.0)


def test_equator_noon_high_sun() -> None:
    lat, lon, _ = subsolar_point(T0)
    s = sun_astronomical(0.0, lon, T0)
    assert s.elevation_deg > 88.0 - abs(lat)
    with pytest.raises(ValueError):
        sun_astronomical(-90.0, 0.0, T0)
