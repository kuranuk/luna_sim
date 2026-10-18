from __future__ import annotations

import math

import numpy as np
import pytest

from lunarsim.config import parse_config
from lunarsim.config.schema import SpotlightConfig
from lunarsim.core.frames import Pose
from lunarsim.core.illumination.lights import _profile_solid_angle, light_state, spot_profile
from lunarsim.core.sensors.camera import camera_pose_from_body
from lunarsim.core.world import World


def test_profile_and_power() -> None:
    inner, outer = math.radians(15), math.radians(30)
    assert spot_profile(0.1, inner, outer) == 1.0
    assert spot_profile(outer, inner, outer) == 0.0
    cfg = SpotlightConfig()
    st = light_state(cfg, Pose.identity())
    power = cfg.flux_lm / cfg.luminous_efficacy_lm_per_w * cfg.camera_band_fraction
    assert st.intensity * _profile_solid_angle(inner, outer) == pytest.approx(power)
    # Narrow uniform cone: solid angle -> 2π(1 - cos θ).
    a = _profile_solid_angle(math.radians(10), math.radians(10.0001))
    assert a == pytest.approx(2 * math.pi * (1 - math.cos(math.radians(10))), rel=1e-4)
    np.testing.assert_allclose(st.axis, [1, 0, 0])


def _night_world(light_height: float, on: bool = True) -> World:
    return World(
        parse_config(
            {
                "seed": 0,
                "terrain": {
                    "size_x_m": 40,
                    "size_y_m": 40,
                    "resolution_m": 0.1,
                    "fractal": {"enabled": False},
                },
                "sun": {"elevation_deg": -20.0},
                "surface": {"model": "lambert", "lambert_albedo": 0.2},
                "rover": {"origin_height_m": 0.0},
                "cameras": [
                    {"name": "c", "intrinsics": {"width": 41, "height": 41, "hfov_deg": 20}}
                ],
                "lights": [
                    {"name": "L", "enabled": on, "mount": {"z_m": light_height, "pitch_deg": 90.0}}
                ],
            }
        )
    )


def test_inverse_square_and_lambert() -> None:
    pose = camera_pose_from_body([0.0, 0.0, 6.0], pitch=math.pi / 2)
    rad = []
    for h in (2.0, 4.0):
        w = _night_world(h)
        b = w.render_camera("c", pose).buffers
        st = w.active_lights()[0]
        centre = b.radiance[20, 20]
        assert centre == pytest.approx(0.2 / math.pi * st.intensity / h**2, rel=1e-6)
        rad.append(centre)
    assert rad[0] / rad[1] == pytest.approx(4.0, rel=1e-6)


def test_light_off_and_switch() -> None:
    w = _night_world(2.0, on=False)
    pose = camera_pose_from_body([0.0, 0.0, 6.0], pitch=math.pi / 2)
    assert (w.render_camera("c", pose).buffers.radiance == 0).all()
    w.set_light("L", True)
    assert w.render_camera("c", pose).buffers.radiance.max() > 0
    with pytest.raises(KeyError):
        w.set_light("nope", True)


def test_light_shadow() -> None:
    w = _night_world(2.0)
    # A 0.5 m post right below the light shadows the ground directly beneath it.
    hm = w.terrain.heightmap
    post = (np.abs(hm.x)[None, :] <= 0.25) & (np.abs(hm.y - 0.0)[:, None] <= 0.25)
    hm.z[post] = 1.0
    w._build_caster()
    # Light 2 m above ground; post top at 1 m spans the nodes |x|, |y| <= 0.2 m, so its
    # shadow reaches 0.4 m along the axes (similar triangles) and further on diagonals.
    pose = camera_pose_from_body([0.0, 0.0, 6.0], pitch=math.pi / 2)
    b = w.render_camera("c", pose).buffers
    pts = b.points.reshape(-1, 3)
    rad = b.radiance.ravel()
    ground = pts[:, 2] < 0.01
    r_xy = np.hypot(pts[:, 0], pts[:, 1])
    behind = ground & (r_xy > 0.3) & (r_xy < 0.38)
    open_ground = ground & (r_xy > 0.8) & (r_xy < 1.0)
    assert behind.any() and open_ground.any()
    assert rad[behind].max() == 0.0
    assert rad[open_ground].min() > 0.0
