from __future__ import annotations

import math

import numpy as np
import pytest

from lunarsim.config import parse_config
from lunarsim.config.schema import RoverConfig
from lunarsim.core.rover import Rover, settle, wheel_offsets
from lunarsim.core.rover.rover import longitudinal_slip
from lunarsim.core.world import World


def plane(slope_deg: float, downhill_az_deg: float = 0.0):
    """Height function of a plane descending towards azimuth (from North, clockwise)."""
    g = math.tan(math.radians(slope_deg))
    az = math.radians(downhill_az_deg)
    dx, dy = math.sin(az), math.cos(az)
    return lambda x, y: -g * (np.asarray(x) * dx + np.asarray(y) * dy)


def test_wheel_offsets() -> None:
    assert wheel_offsets(RoverConfig()).shape == (4, 2)
    w6 = wheel_offsets(RoverConfig(wheels=6))
    assert w6.shape == (6, 2) and set(np.round(w6[:, 0], 6)) == {0.5, 0.0, -0.5}


def test_settle_flat() -> None:
    cfg = RoverConfig()
    c = settle(lambda x, y: np.full_like(np.asarray(x, dtype=float), 2.0), cfg, 1.0, 2.0, 0.3)
    assert c.roll == pytest.approx(0.0) and c.pitch == pytest.approx(0.0)
    np.testing.assert_allclose(c.pose.t, [1.0, 2.0, 2.0 + cfg.origin_height_m])
    np.testing.assert_allclose(c.wheel_points[:, 2], 2.0)
    assert c.fit_residual_m < 1e-12


@pytest.mark.parametrize(
    "yaw_deg,exp_roll,exp_pitch",
    [
        (90.0, 0.0, 10.0),  # facing North = downhill -> nose down (pitch > 0)
        (-90.0, 0.0, -10.0),  # facing South = uphill -> nose up
        (0.0, -10.0, 0.0),  # facing East: left (North) side is lower -> roll < 0
    ],
)
def test_settle_on_slope(yaw_deg: float, exp_roll: float, exp_pitch: float) -> None:
    point_wheels = RoverConfig(contact_half_length_ratio=0.0, wheel_width_m=1e-9)
    c = settle(plane(10.0, 0.0), point_wheels, 0.0, 0.0, math.radians(yaw_deg))
    assert math.degrees(c.roll) == pytest.approx(exp_roll, abs=1e-9)
    assert math.degrees(c.pitch) == pytest.approx(exp_pitch, abs=1e-9)
    assert math.degrees(c.slope) == pytest.approx(10.0, abs=1e-9)
    # Body z axis = plane normal; origin at origin_height along it above the plane.
    n = np.array([0.0, math.sin(math.radians(10)), math.cos(math.radians(10))])  # tilts North
    np.testing.assert_allclose(c.pose.R[:, 2], n, atol=1e-12)
    np.testing.assert_allclose(c.pose.t, 0.5 * n, atol=1e-12)


def test_wheel_footprint_rests_on_highest_point() -> None:
    cfg = RoverConfig()  # footprint ±0.06 m along, ±0.1 m across each wheel
    c = settle(plane(10.0, 0.0), cfg, 0.0, 0.0, math.radians(90.0))  # facing downhill
    # Uphill sample at body x = -0.4 R; tilted with the body -> horizontal 0.4 R cos(10°).
    lift = math.sin(math.radians(10.0)) * 0.4 * cfg.wheel_radius_m
    assert math.degrees(c.pitch) == pytest.approx(10.0, abs=1e-9)
    n = np.array([0.0, math.sin(math.radians(10)), math.cos(math.radians(10))])
    np.testing.assert_allclose(c.pose.t, 0.5 * n + [0.0, 0.0, lift], atol=1e-12)

    # A narrow spike under one wheel lifts that wheel only to the spike top.
    def spiky(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x, y = np.asarray(x), np.asarray(y)
        return np.where((np.abs(x - 0.5) < 1e-3) & (np.abs(y - 0.4) < 1e-3), 0.05, 0.0)

    cs = settle(spiky, cfg, 0.0, 0.0, 0.0)
    assert cs.wheel_points[0, 2] == pytest.approx(0.05) and cs.roll != 0.0


def test_skid_steer_straight_and_circle() -> None:
    flat = plane(0.0)
    r = Rover(RoverConfig(), flat)
    r.set_command(0.2, 0.0)
    r.step(10.0)
    assert r.x == pytest.approx(2.0 * (1 - 0.02)) and r.y == pytest.approx(0.0)
    r2 = Rover(RoverConfig(slip={"slip_flat": 0.0}), flat)
    w = math.radians(10.0)
    r2.set_command(0.1, w)
    r2.step(2 * math.pi / w)
    assert r2.x == pytest.approx(0.0, abs=1e-6) and r2.y == pytest.approx(0.0, abs=1e-6)
    assert r2.odometer_m == pytest.approx(0.1 * 2 * math.pi / w)


def test_ackermann_turn_radius() -> None:
    cfg = RoverConfig(kinematics="ackermann", slip={"slip_flat": 0.0}, max_yaw_rate_deg_s=90)
    r = Rover(cfg, plane(0.0))
    steer = math.radians(20.0)
    r.set_command(0.2, steer=steer)
    radius = cfg.wheelbase_m / math.tan(steer)
    r.step(0.5 * math.pi * radius / 0.2)  # quarter circle
    assert r.x == pytest.approx(radius, abs=1e-6) and r.y == pytest.approx(radius, abs=1e-6)
    assert math.degrees(r.yaw) == pytest.approx(90.0, abs=1e-6)


def test_command_clipping() -> None:
    r = Rover(RoverConfig(max_speed_m_s=0.2, max_yaw_rate_deg_s=20), plane(0.0))
    c = r.set_command(5.0, 10.0, 3.0)
    assert c.v == 0.2 and c.omega == pytest.approx(math.radians(20.0))
    assert c.steer == pytest.approx(math.radians(30.0))


def test_slip_uphill_downhill_and_stall() -> None:
    s = RoverConfig().slip
    assert longitudinal_slip(0.0, s) == pytest.approx(0.02)
    up = longitudinal_slip(math.radians(10.0), s)
    assert up == pytest.approx(0.02 + 0.98 * (10 / 25) ** 2)
    assert longitudinal_slip(math.radians(-10.0), s) == pytest.approx(-0.05 * 0.16)
    # Driving uphill (South) on a 10° slope: horizontal progress v (1 - s) cos(10°) t.
    r = Rover(RoverConfig(), plane(10.0, 0.0))
    r.reset(0.0, 0.0, math.radians(-90.0))
    r.set_command(0.2)
    r.step(5.0)
    assert -r.y == pytest.approx(0.2 * (1 - up) * math.cos(math.radians(10)) * 5.0, rel=1e-9)
    assert r.status.slip == pytest.approx(up)
    steep = Rover(RoverConfig(max_slope_deg=40.0), plane(26.0, 0.0))
    steep.reset(0.0, 0.0, math.radians(-90.0))
    steep.set_command(0.2)
    steep.step(1.0)
    assert "stalled" in steep.status.warnings and steep.y == pytest.approx(0.0, abs=1e-12)


def test_lateral_drift_downhill() -> None:
    r = Rover(RoverConfig(), plane(10.0, 0.0))  # facing East, North side lower
    r.set_command(0.2)
    r.step(5.0)
    assert r.y > 0.0  # drifts to the left = North = downhill
    assert r.status.lateral_velocity > 0.0


def test_slope_and_tipover_warnings() -> None:
    cfg = RoverConfig()
    lim = math.degrees(math.atan2(0.4, 0.45))  # lateral tip angle ~41.6°
    for slope, expected in [
        (15.0, ()),
        (25.0, ("slope_exceeded",)),
        (lim - 3.0, ("slope_exceeded", "tipover_risk")),
        (lim + 2.0, ("slope_exceeded", "tipped_over")),
    ]:
        r = Rover(cfg, plane(slope, 0.0))  # side slope for an East-facing rover
        assert r.status.warnings == expected, slope
    tipped = Rover(cfg, plane(lim + 2.0, 0.0))
    tipped.set_command(0.2)
    tipped.step(1.0)
    assert tipped.x == 0.0  # does not move


def _world(**extra: object) -> World:
    return World(
        parse_config(
            {
                "seed": 4,
                "terrain": {
                    "size_x_m": 100,
                    "size_y_m": 100,
                    "resolution_m": 0.25,
                    "fractal": {"enabled": False},
                },
                "sun": {"elevation_deg": -10.0},
                **extra,
            }
        )
    )


def test_world_step_and_mounted_poses() -> None:
    w = _world(
        cameras=[
            {"name": "nav", "stereo_baseline_m": 0.3, "intrinsics": {"width": 64, "height": 48}}
        ]
    )
    T = w.camera_pose("nav")
    np.testing.assert_allclose(T.t, [0.5, 0.0, 1.5], atol=1e-12)
    pitch = math.radians(15.0)
    np.testing.assert_allclose(
        T.rotate([0, 0, 1]), [math.cos(pitch), 0, -math.sin(pitch)], atol=1e-12
    )
    Tr = w.camera_pose("nav", "right")
    np.testing.assert_allclose(Tr.t - T.t, 0.3 * T.rotate([1, 0, 0]), atol=1e-12)
    np.testing.assert_allclose(Tr.t - T.t, [0.0, -0.3, 0.0], atol=1e-12)  # right of East-facing
    left, right = w.render_stereo("nav")
    assert left.image.exposure_s == right.image.exposure_s
    w.set_rover_command(0.2, math.radians(5))
    for _ in range(10):
        w.step(0.5)
    assert w.time_s == pytest.approx(5.0)
    assert w.camera_pose("nav").t[0] > 1.0


def test_world_deterministic_steps() -> None:
    a, b = _world(), _world()
    for w in (a, b):
        w.set_rover_command(0.2, 0.1)
        for _ in range(7):
            w.step(0.3)
    np.testing.assert_array_equal(a.rover.pose.matrix(), b.rover.pose.matrix())


def test_near_patch_follows_rover() -> None:
    w = World(
        parse_config(
            {
                "seed": 1,
                "terrain": {
                    "size_x_m": 400,
                    "size_y_m": 400,
                    "resolution_m": 1.0,
                    "near": {
                        "enabled": True,
                        "size_m": 20.0,
                        "resolution_m": 0.1,
                        "tile_cells": 128,
                        "overlap_cells": 16,
                    },
                },
                "rover": {"max_speed_m_s": 2.0},
                "cameras": [],
                "lidars": [],
            }
        )
    )
    x0 = w.heightmap.extent[0]
    w.set_rover_command(2.0)
    w.step(5.0)  # 10 m East: beyond 1/4 of the patch size
    assert w.heightmap.extent[0] > x0 + 5.0
    assert w.heightmap.contains(w.rover.x, w.rover.y)
