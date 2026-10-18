from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from lunarsim.core import frames as fr

angles = st.floats(-np.pi, np.pi, allow_nan=False)
pitches = st.floats(-np.pi / 2 + 1e-3, np.pi / 2 - 1e-3, allow_nan=False)
coords = st.floats(-1e4, 1e4, allow_nan=False)


def _is_rotation(R: np.ndarray) -> bool:
    return np.allclose(R @ R.T, np.eye(3), atol=1e-12) and np.isclose(np.linalg.det(R), 1.0)


@given(angles, pitches, angles)
def test_euler_roundtrip(roll: float, pitch: float, yaw: float) -> None:
    R = fr.matrix_from_euler_zyx(roll, pitch, yaw)
    assert _is_rotation(R)
    r2, p2, y2 = fr.euler_zyx_from_matrix(R)
    np.testing.assert_allclose(fr.matrix_from_euler_zyx(r2, p2, y2), R, atol=1e-10)
    np.testing.assert_allclose([r2, p2, y2], [roll, pitch, yaw], atol=1e-8)


@given(angles, st.floats(-np.pi / 2, np.pi / 2), angles)
def test_quat_roundtrip(roll: float, pitch: float, yaw: float) -> None:
    R = fr.matrix_from_euler_zyx(roll, pitch, yaw)
    q = fr.quat_from_matrix(R)
    assert np.isclose(np.linalg.norm(q), 1.0) and q[0] >= 0.0
    np.testing.assert_allclose(fr.matrix_from_quat(q), R, atol=1e-12)


def test_quat_known() -> None:
    q = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])  # +90 deg about z
    np.testing.assert_allclose(fr.matrix_from_quat(q) @ [1, 0, 0], [0, 1, 0], atol=1e-15)
    np.testing.assert_allclose(fr.matrix_from_quat(q), fr.rot_z(np.pi / 2), atol=1e-15)


@settings(max_examples=50)
@given(angles, pitches, angles, coords, coords, coords)
def test_pose_inverse_and_compose(
    r: float, p: float, y: float, a: float, b: float, c: float
) -> None:
    T = fr.Pose.from_euler_zyx(r, p, y, (a, b, c))
    pts = np.array([[1.0, 2.0, 3.0], [-4.0, 0.5, 9.0]])
    np.testing.assert_allclose(T.inverse().apply(T.apply(pts)), pts, atol=1e-8)
    np.testing.assert_allclose((T @ T.inverse()).matrix(), np.eye(4), atol=1e-9)
    np.testing.assert_allclose(fr.Pose.from_matrix(T.matrix()).matrix(), T.matrix())
    np.testing.assert_allclose(fr.Pose.from_quat(T.quat(), T.t).R, T.R, atol=1e-12)


def test_camera_axes() -> None:
    R = fr.R_BODY_FROM_CAMERA
    assert _is_rotation(R)
    np.testing.assert_array_equal(R @ [0, 0, 1], [1, 0, 0])  # optical axis -> forward
    np.testing.assert_array_equal(R @ [1, 0, 0], [0, -1, 0])  # image right -> body right
    np.testing.assert_array_equal(R @ [0, 1, 0], [0, 0, -1])  # image down -> body down
    np.testing.assert_array_equal(fr.R_CAMERA_FROM_BODY @ R, np.eye(3))


def test_azel_conventions() -> None:
    np.testing.assert_allclose(fr.azel_to_enu(0.0, 0.0), [0, 1, 0], atol=1e-15)  # North
    np.testing.assert_allclose(fr.azel_to_enu(np.pi / 2, 0.0), [1, 0, 0], atol=1e-15)  # East
    np.testing.assert_allclose(fr.azel_to_enu(1.0, np.pi / 2), [0, 0, 1], atol=1e-15)  # Zenith


@given(st.floats(0.0, 2 * np.pi - 1e-9), st.floats(-np.pi / 2 + 1e-6, np.pi / 2 - 1e-6))
def test_azel_roundtrip(az: float, el: float) -> None:
    az2, el2 = fr.enu_to_azel(fr.azel_to_enu(az, el))
    assert np.isclose(el2, el, atol=1e-12)
    d = np.angle(np.exp(1j * (az2 - az)))
    assert abs(d) < 1e-9
