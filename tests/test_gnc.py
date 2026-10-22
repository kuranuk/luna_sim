"""Python GNC interface (lunarsim.gnc): protocol, session, samples, and a closed loop with the
browser simulator's JS World run headless in Node (web/tools/headless.mjs)."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import threading
from pathlib import Path

import numpy as np
import pytest

from lunarsim.gnc import Command, Gnc, ImuBatch, Observation, Pose, SimInfo
from lunarsim.gnc.examples import DeadReckoning, WaypointGnc
from lunarsim.gnc.protocol import PROTOCOL_VERSION, decode, encode
from lunarsim.gnc.server import is_loopback, load_gnc_factory
from lunarsim.gnc.session import GncSession

ROOT = Path(__file__).parents[1]
NODE = shutil.which("node")


def _info(**kw: object) -> dict[str, object]:
    info: dict[str, object] = {
        "controlPeriod": 0.1,
        "initialPose": {"x": 1.0, "y": 2.0, "z": 0.3, "yaw": 0.5, "roll": 0.0, "pitch": 0.0},
        "waypoints": [],
        "scenario": {
            "imu": {"rateHz": 100, "mount": {"x": 0, "y": 0, "z": 0.3}},
            "lidars": [{"name": "spad_flash", "mount": {"x": 0.55, "z": 0.8, "pitchDeg": 10}}],
            "rover": {"maxSpeed": 0.2, "maxYawRateDeg": 20},
        },
    }
    info.update(kw)
    return info


def test_protocol_round_trip_and_bounds() -> None:
    blobs = {
        "a": np.arange(5, dtype=np.float64),
        "b": np.arange(6, dtype=np.float32).reshape(2, 3),
        "c": np.array([1, 2, 3], dtype=np.uint8),
        "d": np.zeros(0, dtype=np.uint16),
    }
    header, got = decode(encode({"type": "obs", "seq": 1}, blobs))
    assert header == {"type": "obs", "seq": 1}
    for k, v in blobs.items():
        assert got[k].dtype == v.dtype and got[k].shape == v.shape
        np.testing.assert_array_equal(got[k], v)
    data = bytearray(encode({"x": 1}, {"a": np.zeros(4)}))
    with pytest.raises(ValueError):
        decode(bytes(data[:-8]))  # truncated blob
    with pytest.raises(ValueError):
        decode(b"XXXX" + bytes(data[4:]))
    with pytest.raises(ValueError):
        encode({}, {"bad": np.zeros(2, dtype=np.complex64)})


def test_pose_quaternion_matches_frames() -> None:
    from lunarsim.core.frames import matrix_from_euler_zyx, quat_from_matrix

    p = Pose(0, 0, 0, yaw=1.1, roll=0.2, pitch=-0.3)
    np.testing.assert_allclose(
        p.quaternion(), quat_from_matrix(matrix_from_euler_zyx(0.2, -0.3, 1.1)), atol=1e-12
    )


def test_dead_reckoning_is_exact_without_noise() -> None:
    """Level ground, constant v and yaw rate, perfect IMU: the arc is integrated exactly."""
    dr = DeadReckoning(Pose(0, 0, 0, 0))
    v, w, n = 0.2, 0.1, 1001
    t = np.arange(n) * 0.01
    gyro = np.tile([0.0, 0.0, w], (n, 1))
    accel = np.tile([0.0, 0.0, 1.62], (n, 1))
    est = dr.update(Observation(10.0, 10.0, ImuBatch(t, gyro, accel), [], v, w, []))
    r = v / w
    assert est.yaw == pytest.approx(1.0, abs=1e-12)
    assert est.x == pytest.approx(r * math.sin(1.0), abs=1e-9)
    assert est.y == pytest.approx(r * (1 - math.cos(1.0)), abs=1e-9)


def test_session_hello_step_errors_and_logs(tmp_path: Path) -> None:
    class Boom(Gnc):
        def step(self, obs: Observation) -> Command:
            raise RuntimeError("boom")

    s = GncSession(WaypointGnc(), tmp_path, log=lambda _: None)
    assert s.handle_binary(encode({"type": "obs", "seq": 1, "t": 0, "dt": 0.1}, {}))["type"] == (
        "error"
    )
    bad = s.handle_text(json.dumps({"type": "hello", "protocol": 99, "info": _info()}))
    assert bad is not None and bad["type"] == "error"
    ready = s.handle_text(
        json.dumps({"type": "hello", "protocol": PROTOCOL_VERSION, "info": _info()})
    )
    assert ready == {"type": "ready", "name": "waypoint"}
    obs = {"type": "obs", "seq": 1, "t": 0.1, "dt": 0.1, "wheel": {"v": 0, "omega": 0},
           "waypoints": [[5.0, 2.0]], "lidar": []}  # fmt: skip
    truth = np.array([[0.1, 1.0, 2.0, 0.3, 1.0, 0.0, 0.0, 0.0]])
    reply = s.handle_binary(encode(obs, {"truth/poses": truth}))
    assert reply["type"] == "cmd" and reply["seq"] == 1 and reply["v"] > 0
    assert reply["estimate"]["x"] == pytest.approx(1.0)
    s.close()
    run = next(tmp_path.iterdir())
    assert (run / "truth_tum.txt").read_text().split()[:2] == ["0.100000", "1.000000"]
    assert len((run / "estimate_tum.txt").read_text().splitlines()) == 1
    assert (run / "commands.csv").read_text().splitlines()[1].startswith("0.100000,")

    s2 = GncSession(Boom(), None, log=lambda _: None)
    s2.handle_text(json.dumps({"type": "hello", "protocol": PROTOCOL_VERSION, "info": _info()}))
    err = s2.handle_binary(encode(obs, {}))
    assert err["type"] == "error" and "boom" in err["message"]


def test_sim_info_and_loader(tmp_path: Path) -> None:
    info = SimInfo.from_dict(_info(waypoints=[[1, 2]]))
    assert info.imu_rate_hz == 100 and info.waypoints == [(1.0, 2.0)]
    assert info.lidar_mounts["spad_flash"].rpy_deg == (0.0, 10.0, 0.0)
    assert info.max_yaw_rate == pytest.approx(math.radians(20))
    f = tmp_path / "my_gnc.py"
    f.write_text("from lunarsim.gnc import Gnc\nclass Mine(Gnc):\n    name = 'mine'\n")
    assert load_gnc_factory(f"{f}:Mine")().name == "mine"
    assert load_gnc_factory("lunarsim.gnc.examples:WaypointGnc")().name == "waypoint"
    with pytest.raises(ValueError):
        load_gnc_factory("no_colon")
    assert is_loopback("127.0.0.1") and is_loopback("localhost") and not is_loopback("0.0.0.0")


@pytest.mark.skipif(NODE is None, reason="Node.js not installed")
@pytest.mark.parametrize(("token", "lidar"), [(None, True), ("s3cret", False)])
def test_closed_loop_with_headless_browser_sim(
    tmp_path: Path, token: str | None, lidar: bool
) -> None:
    pytest.importorskip("websockets")
    from lunarsim.gnc.server import create_server

    logs: list[str] = []
    server = create_server(WaypointGnc, port=0, token=token, log_dir=tmp_path, log=logs.append)
    port = server.socket.getsockname()[1]
    th = threading.Thread(target=server.serve_forever, daemon=True)
    th.start()
    try:
        url = f"ws://127.0.0.1:{port}/" + (f"?token={token}" if token else "")
        cmd = [NODE, str(ROOT / "web/tools/headless.mjs"), "--url", url, "--seconds", "40",
               "--n", "256", "--res", "0.25", "--waypoints", "3,0;3,3"]  # fmt: skip
        if not lidar:
            cmd.append("--no-lidar")
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        assert r.returncode == 0, r.stderr + "\n".join(logs)
        res = json.loads(r.stdout.strip().splitlines()[-1])
        if token:
            bad = subprocess.run([*cmd[:3], f"ws://127.0.0.1:{port}/?token=wrong"],
                                 capture_output=True, text=True, timeout=120)  # fmt: skip
            assert bad.returncode != 0
    finally:
        server.shutdown()
        th.join(5)
    assert res["gnc"] == "waypoint" and res["steps"] == 400
    assert res["status"] == "route done"
    # The estimate reached the last waypoint; the truth is nearby (dead-reckoning drift, slip).
    assert math.dist((res["estimate"]["x"], res["estimate"]["y"]), (3, 3)) <= 0.3 + 1e-6
    assert math.dist((res["truth"]["x"], res["truth"]["y"]), (3, 3)) < 1.0
    run = next(tmp_path.iterdir())
    truth = np.loadtxt(run / "truth_tum.txt")
    est = np.loadtxt(run / "estimate_tum.txt")
    assert truth.shape[1] == 8 and est.shape == (400, 8)
    assert truth[-1, 0] == pytest.approx(40.0)


def test_session_passes_camera_images() -> None:
    seen: list[Observation] = []

    class Rec(Gnc):
        def step(self, obs: Observation) -> Command:
            seen.append(obs)
            return Command(0.0, 0.0)

    s = GncSession(Rec(), None, log=lambda _: None)
    info = _info(camerasAvailable=True)
    info["scenario"]["cameras"] = [
        {"name": "navcam", "mount": {"x": 0.5, "z": 1.0, "pitchDeg": 12}}
    ]  # type: ignore[index]
    s.handle_text(json.dumps({"type": "hello", "protocol": PROTOCOL_VERSION, "info": info}))
    assert s.gnc.info.cameras_available
    assert s.gnc.info.camera_mounts["navcam"].rpy_deg == (0.0, 12.0, 0.0)
    img = np.arange(12, dtype=np.uint16).reshape(3, 4)
    cam = {"name": "navcam", "frame": 7, "t": 0.5, "width": 4, "height": 3, "bits": 12,
           "exposure": 0.004, "saturated": 1, "fx": 2.0, "fy": 2.0, "cx": 1.5,
           "cy": 1.0}  # fmt: skip
    obs = {"type": "obs", "seq": 1, "t": 0.5, "dt": 0.1, "cameras": [cam]}
    assert s.handle_binary(encode(obs, {"camera/navcam/7/image": img}))["type"] == "cmd"
    c = seen[0].cameras[0]
    assert (c.name, c.frame, c.bits, c.saturated) == ("navcam", 7, 12, 1)
    np.testing.assert_array_equal(c.image, img)
    np.testing.assert_array_equal(c.K, [[2, 0, 1.5], [0, 2, 1.0], [0, 0, 1]])


def test_session_passes_lidar_beam_and_pattern() -> None:
    seen: list[Observation] = []

    class Rec(Gnc):
        def step(self, obs: Observation) -> Command:
            seen.append(obs)
            return Command(0.0, 0.0)

    s = GncSession(Rec(), None, log=lambda _: None)
    s.handle_text(json.dumps({"type": "hello", "protocol": PROTOCOL_VERSION, "info": _info()}))
    pattern = {"kind": "flash", "width": 160, "height": 120, "hfovDeg": 100.0, "vfovDeg": 75.0}
    scan = {
        "name": "spad_flash",
        "frame": 3,
        "t": 1.5,
        "count": 2,
        "beams": 19200,
        "pattern": pattern,
    }
    p = "lidar/spad_flash/3/"
    blobs = {
        p + "points": np.zeros((2, 4), np.float32), p + "labels": np.zeros(2, np.uint8),
        p + "time": np.zeros(2, np.float32), p + "ring": np.array([0, 119], np.uint16),
        p + "beam": np.array([5, 19199], np.uint32),
    }  # fmt: skip
    obs = {"type": "obs", "seq": 1, "t": 1.5, "dt": 0.1, "lidar": [scan]}
    assert s.handle_binary(encode(obs, blobs))["type"] == "cmd"
    sc = seen[0].lidar[0]
    assert sc.pattern["hfovDeg"] == 100.0
    np.testing.assert_array_equal(sc.beam, [5, 19199])
