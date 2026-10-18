from __future__ import annotations

import numpy as np
import pytest

from lunarsim.config.schema import CameraIntrinsicsConfig, CameraSensorConfig
from lunarsim.core.frames import Pose
from lunarsim.core.rng import RngFactory
from lunarsim.core.sensors.camera import CameraModel, camera_pose_from_body
from lunarsim.core.sensors.camera_noise import (
    auto_exposure,
    electrons_per_radiance,
    expose,
    gain,
)

DIST = dict(k1=-0.28, k2=0.07, p1=1e-3, p2=-5e-4, k3=0.0)


def test_intrinsics_from_hfov() -> None:
    cam = CameraModel.from_config(CameraIntrinsicsConfig(width=640, height=480, hfov_deg=90.0))
    assert cam.fx == pytest.approx(320.0) and cam.cx == 319.5 and cam.cy == 239.5
    d = cam.unproject(np.array([[cam.cx, cam.cy]]))
    np.testing.assert_allclose(d[0], [0, 0, 1], atol=1e-15)


@pytest.mark.parametrize("dist", [{}, DIST])
def test_project_unproject_roundtrip(dist: dict[str, float]) -> None:
    cam = CameraModel.from_config(CameraIntrinsicsConfig(**dist))
    rng = np.random.default_rng(0)
    px = np.c_[rng.uniform(0, 639, 2000), rng.uniform(0, 479, 2000)]
    rays = cam.unproject(px)
    back, ok = cam.project(rays * rng.uniform(0.5, 50, (2000, 1)))
    assert ok.all()
    np.testing.assert_allclose(back, px, atol=1e-6)


def test_distortion_inverse() -> None:
    cam = CameraModel.from_config(CameraIntrinsicsConfig(**DIST))
    x = np.linspace(-0.5, 0.5, 51)
    X, Y = np.meshgrid(x, x * 0.75)
    xd, yd = cam.distort(X, Y)
    xu, yu = cam.undistort(xd, yd)
    np.testing.assert_allclose(xu, X, atol=1e-10)
    np.testing.assert_allclose(yu, Y, atol=1e-10)


def test_project_behind_and_outside() -> None:
    cam = CameraModel.from_config(CameraIntrinsicsConfig())
    _, ok = cam.project(np.array([[0, 0, -1.0], [100.0, 0, 1.0], [0.1, 0.1, 1.0]]))
    np.testing.assert_array_equal(ok, [False, False, True])


def test_pose_helper_axes() -> None:
    T = camera_pose_from_body([1, 2, 3], yaw=np.pi / 2)  # facing North
    np.testing.assert_allclose(T.rotate([0, 0, 1]), [0, 1, 0], atol=1e-15)  # optical axis
    np.testing.assert_allclose(T.rotate([1, 0, 0]), [1, 0, 0], atol=1e-15)  # image right = East
    np.testing.assert_allclose(T.rotate([0, 1, 0]), [0, 0, -1], atol=1e-15)  # image down
    T2 = camera_pose_from_body([0, 0, 0], pitch=np.radians(30))
    assert T2.rotate([0, 0, 1])[2] == pytest.approx(-0.5)  # nose down
    assert isinstance(T, Pose)


def test_sensor_statistics() -> None:
    s = CameraSensorConfig(read_noise_e=3.0, dark_current_e_per_s=0.0)
    L = 5.0
    rate = L * electrons_per_radiance(s)
    t = 0.3 * s.full_well_e / rate
    img = expose(np.full((200, 200), L), s, RngFactory(1).generator("c"), exposure_s=t)
    e_mean = 0.3 * s.full_well_e
    dn = img.dn.astype(float)
    assert dn.mean() == pytest.approx(e_mean / gain(s), rel=0.01)
    expected_std = np.sqrt(e_mean + 3.0**2) / gain(s)
    assert dn.std() == pytest.approx(expected_std, rel=0.05)
    assert not img.saturated.any()


def test_saturation_black_and_reproducible() -> None:
    s = CameraSensorConfig()
    rad = np.array([[0.0, 1e6], [1e-9, 1e6]])
    a = expose(rad, s, RngFactory(2).generator("c"))
    assert a.dn[0, 1] == 2**s.bits - 1 and a.saturated[0, 1]
    assert a.dn[0, 0] < 10  # dark + read noise only
    b = expose(rad, s, RngFactory(2).generator("c"))
    np.testing.assert_array_equal(a.dn, b.dn)


def test_auto_exposure_target() -> None:
    s = CameraSensorConfig(
        auto_exposure=True,
        ae_percentile=100.0,
        ae_target_fraction=0.5,
        dark_current_e_per_s=0.0,
        read_noise_e=0.0,
    )
    rad = np.linspace(0.0, 20.0, 1000).reshape(10, 100)
    t = auto_exposure(rad, s)
    assert 20.0 * electrons_per_radiance(s) * t == pytest.approx(0.5 * s.full_well_e)
    img = expose(rad, s, RngFactory(3).generator("c"))
    assert img.exposure_s == pytest.approx(t)
