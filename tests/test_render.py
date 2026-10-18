from __future__ import annotations

import math

import numpy as np
import pytest

from lunarsim.config import parse_config
from lunarsim.core.raycast import HAS_NUMBA
from lunarsim.core.sensors.camera import camera_pose_from_body
from lunarsim.core.terrain import Label
from lunarsim.core.world import World


def _flat_world(model: str = "lambert", el: float = 90.0, backend: str = "auto") -> World:
    return World(
        parse_config(
            {
                "seed": 0,
                "terrain": {
                    "size_x_m": 40,
                    "size_y_m": 40,
                    "resolution_m": 0.25,
                    "fractal": {"enabled": False},
                },
                "sun": {"azimuth_deg": 0.0, "elevation_deg": el},
                "surface": {"model": model, "lambert_albedo": 0.2},
                "cameras": [
                    {"name": "c", "intrinsics": {"width": 80, "height": 60, "hfov_deg": 60}}
                ],
                "render": {"backend": backend},
            }
        )
    )


def test_flat_lambert_radiance_and_depth() -> None:
    w = _flat_world()
    pose = camera_pose_from_body([0, 0, 5.0], pitch=math.pi / 2)  # looking straight down
    b = w.render_camera("c", pose).buffers
    assert np.isfinite(b.depth).all()
    np.testing.assert_allclose(b.depth, 5.0, atol=1e-9)
    np.testing.assert_allclose(b.radiance, 1361.0 * 0.2 / math.pi, rtol=1e-12)
    assert not b.shadow.any() and (b.labels == Label.PLAIN).all()
    np.testing.assert_allclose(b.normals[..., 2], 1.0)


def test_sky_and_horizon() -> None:
    w = _flat_world(el=30.0)
    pose = camera_pose_from_body([0, 0, 2.0], pitch=0.0)  # horizontal view: top half is sky
    b = w.render_camera("c", pose).buffers
    assert (b.labels[:20] == Label.SKY).all() and np.isinf(b.depth[:20]).all()
    assert (b.radiance[:20] == 0).all()
    assert np.isfinite(b.depth[-1]).all()


def test_sun_below_horizon_all_dark() -> None:
    w = _flat_world(el=-5.0)
    b = w.render_camera("c", camera_pose_from_body([0, 0, 5.0], pitch=math.pi / 2)).buffers
    assert (b.radiance == 0).all() and b.shadow.all()


def test_render_reproducible_and_noise_by_frame() -> None:
    cfg = parse_config(
        {
            "seed": 9,
            "terrain": {
                "size_x_m": 60,
                "size_y_m": 60,
                "resolution_m": 0.25,
                "craters": {"enabled": True},
            },
            "sun": {"azimuth_deg": 100, "elevation_deg": 12},
            "cameras": [{"name": "c", "intrinsics": {"width": 96, "height": 72}}],
        }
    )
    w1, w2 = World(cfg), World(cfg)
    pose = camera_pose_from_body([0, 0, w1.height_at(0, 0) + 1.5], pitch=0.3)
    a = w1.render_camera("c", pose, frame_index=3)
    b = w2.render_camera("c", pose, frame_index=3)
    assert a.image.dn.tobytes() == b.image.dn.tobytes()
    assert a.buffers.radiance.tobytes() == b.buffers.radiance.tobytes()
    c = w1.render_camera("c", pose, frame_index=4)
    assert c.buffers.radiance.tobytes() == a.buffers.radiance.tobytes()
    assert c.image.dn.tobytes() != a.image.dn.tobytes()


@pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
def test_backends_render_alike() -> None:
    base = {
        "seed": 1,
        "terrain": {
            "size_x_m": 60,
            "size_y_m": 60,
            "resolution_m": 0.25,
            "craters": {"enabled": True},
        },
        "sun": {"azimuth_deg": 70, "elevation_deg": 10},
        "cameras": [{"name": "c", "intrinsics": {"width": 64, "height": 48}}],
    }
    wa = World(parse_config({**base, "render": {"backend": "numba"}}))
    wb = World(parse_config({**base, "render": {"backend": "numpy"}}))
    pose = camera_pose_from_body([0, 0, wa.height_at(0, 0) + 1.5], pitch=0.3)
    ra = wa.render_camera("c", pose).buffers
    rb = wb.render_camera("c", pose).buffers
    np.testing.assert_array_equal(ra.shadow, rb.shadow)
    np.testing.assert_allclose(ra.radiance, rb.radiance, rtol=1e-9, atol=1e-12)
