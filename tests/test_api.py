from __future__ import annotations

import io
import json
import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient
from PIL import Image

from lunarsim.api import ServerSettings, create_app
from lunarsim.config import parse_config
from lunarsim.core.world import World
from lunarsim.io.bundle import decode_npz
from lunarsim.io.pointcloud import read_pcd, read_ply

CFG = {
    "seed": 5,
    "terrain": {
        "size_x_m": 40,
        "size_y_m": 40,
        "resolution_m": 0.25,
        "craters": {"enabled": True, "sfd": {"d_min_m": 0.6, "d_max_m": 8.0}},
    },
    "sun": {"azimuth_deg": 120.0, "elevation_deg": 10.0},
    "cameras": [
        {"name": "cam", "intrinsics": {"width": 64, "height": 48}, "stereo_baseline_m": 0.3}
    ],
    "lidars": [
        {"name": "spad", "pattern": {"kind": "flash", "flash_width": 40, "flash_height": 30}}
    ],
    "lights": [{"name": "head"}],
}


@pytest.fixture()
def client() -> Iterator[TestClient]:
    with TestClient(create_app(ServerSettings(max_worlds=2, max_json_points=500))) as c:
        yield c


def _create(c: TestClient, **extra: object) -> str:
    r = c.post("/worlds", json={"config": CFG, **extra})
    assert r.status_code == 201, r.text
    return str(r.json()["world_id"])


def test_health_version_and_docs(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}
    assert "version" in client.get("/version").json()
    spec = client.get("/openapi.json").json()
    assert "/worlds/{world_id}/sensors/lidar/{name}" in spec["paths"]
    assert client.get("/docs").status_code == 200


def test_create_from_yaml_and_validation(client: TestClient) -> None:
    yml = Path(__file__).parents[1].joinpath("configs", "flat_test.yaml").read_text()
    r = client.post("/worlds", json={"config_yaml": yml, "seed": 3})
    assert r.status_code == 201 and r.json()["metadata"]["seed"] == 3
    assert client.post("/worlds", json={"config": {"terrain": {"bogus": 1}}}).status_code == 422
    assert client.post("/worlds", json={"config_yaml": "- a\n- b"}).status_code == 422
    assert client.post("/worlds", json={}).status_code == 422
    _create(client)
    assert client.post("/worlds", json={"config": CFG}).status_code == 429  # max_worlds=2


def test_terrain_endpoints_match_core(client: TestClient) -> None:
    wid = _create(client)
    ref = World(parse_config(CFG))
    r = client.get(f"/worlds/{wid}/heightmap")
    z = np.load(io.BytesIO(r.content), allow_pickle=False)
    np.testing.assert_array_equal(z, ref.heightmap.z)
    assert float(r.headers["x-resolution-m"]) == 0.25
    png = client.get(f"/worlds/{wid}/heightmap", params={"format": "png16"})
    assert Image.open(io.BytesIO(png.content)).size == (161, 161)
    assert client.get(f"/worlds/{wid}/heightmap", params={"layer": "far"}).status_code == 422
    cr = client.get(f"/worlds/{wid}/craters").json()
    assert len(cr) == len(ref.terrain.craters) and "diameter_m" in cr[0]
    assert client.get(f"/worlds/{wid}/rocks").status_code == 200
    meta = client.get(f"/worlds/{wid}").json()
    assert meta["config_hash"] == ref.config_hash and "state" in meta


def test_control_and_step_match_core(client: TestClient) -> None:
    wid = _create(client)
    ref = World(parse_config(CFG))
    assert client.put(
        f"/worlds/{wid}/sun", json={"azimuth_deg": 200, "elevation_deg": 5}
    ).is_success
    ref.set_sun_manual(200, 5)
    assert client.put(f"/worlds/{wid}/sun", json={"azimuth_deg": 1}).status_code == 422
    client.post(f"/worlds/{wid}/rover/cmd", json={"v_m_s": 0.2, "omega_rad_s": 0.05})
    ref.set_rover_command(0.2, 0.05)
    for _ in range(3):
        st = client.post(f"/worlds/{wid}/step", json={"dt_s": 0.7}).json()
        ref.step(0.7)
    assert st == json.loads(json.dumps(ref.state()))
    assert client.post(f"/worlds/{wid}/step", json={"dt_s": -1}).status_code == 422
    lights = client.put(f"/worlds/{wid}/lights/head", json={"on": False}).json()
    assert lights == {"head": False}
    assert client.put(f"/worlds/{wid}/lights/nope", json={"on": True}).status_code == 404


def test_waypoints(client: TestClient) -> None:
    wid = _create(client)
    client.post(
        f"/worlds/{wid}/rover/waypoints",
        json={"points": [[2.0, 0.0], [2.0, 2.0]], "speed_m_s": 0.2},
    )
    st = client.post(f"/worlds/{wid}/step", json={"dt_s": 60.0}).json()
    x, y = st["rover"]["ground_track_m"]
    assert st["rover"]["waypoints_remaining"] == 0
    assert np.hypot(x - 2.0, y - 2.0) < 0.35


def test_camera_endpoint(client: TestClient) -> None:
    wid = _create(client)
    ref = World(parse_config(CFG))
    png = client.get(f"/worlds/{wid}/sensors/camera/cam")
    assert png.headers["content-type"] == "image/png"
    assert Image.open(io.BytesIO(png.content)).size == (64, 48)
    r = client.get(
        f"/worlds/{wid}/sensors/camera/cam",
        params={"format": "npz", "buffers": "depth,labels", "frame_index": 2},
    )
    d = decode_npz(r.content)
    np.testing.assert_array_equal(d["image"], ref.render_camera("cam", frame_index=2).image.dn)
    assert d["depth"].shape == (48, 64) and d["K"].shape == (3, 3)
    right = decode_npz(
        client.get(
            f"/worlds/{wid}/sensors/camera/cam", params={"format": "npz", "eye": "right"}
        ).content
    )
    assert not np.allclose(right["T_world_camera"], d["T_world_camera"])
    raw16 = client.get(f"/worlds/{wid}/sensors/camera/cam", params={"format": "png16"})
    assert np.asarray(Image.open(io.BytesIO(raw16.content))).dtype in (np.uint16, np.int32)
    bad = client.get(
        f"/worlds/{wid}/sensors/camera/cam", params={"format": "npz", "buffers": "nope"}
    )
    assert bad.status_code == 422
    assert client.get(f"/worlds/{wid}/sensors/camera/none").status_code == 404


def test_lidar_endpoint(client: TestClient) -> None:
    wid = _create(client)
    ref = World(parse_config(CFG))
    d = decode_npz(client.get(f"/worlds/{wid}/sensors/lidar/spad").content)
    np.testing.assert_array_equal(d["points"], ref.render_lidar("spad").points)
    ply = read_ply_bytes(
        client.get(f"/worlds/{wid}/sensors/lidar/spad", params={"format": "ply"}).content
    )
    assert ply.shape[0] == d["points"].shape[0]
    assert client.get(f"/worlds/{wid}/sensors/lidar/spad", params={"format": "pcd"}).is_success
    big = client.get(f"/worlds/{wid}/sensors/lidar/spad", params={"format": "json"})
    assert big.status_code == 413  # 1200 beams > max_json_points=500


def read_ply_bytes(data: bytes) -> np.ndarray:
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".ply") as f:
        f.write(data)
        f.flush()
        return read_ply(f.name)


def test_delete(client: TestClient) -> None:
    wid = _create(client)
    assert client.delete(f"/worlds/{wid}").status_code == 204
    assert client.get(f"/worlds/{wid}").status_code == 404
    assert client.delete(f"/worlds/{wid}").status_code == 404


def test_auth() -> None:
    with TestClient(create_app(ServerSettings(token="s3cret"))) as c:
        assert c.get("/health").status_code == 200  # liveness stays open
        assert c.post("/worlds", json={"config": CFG}).status_code == 401
        h = {"Authorization": "Bearer wrong"}
        assert c.get("/worlds", headers=h).status_code == 401
        h = {"Authorization": "Bearer s3cret"}
        wid = c.post("/worlds", json={"config": CFG}, headers=h).json()["world_id"]
        from starlette.websockets import WebSocketDisconnect

        with pytest.raises(WebSocketDisconnect), c.websocket_connect(f"/ws/{wid}") as ws:
            ws.receive_text()
        with c.websocket_connect(f"/ws/{wid}?token=s3cret") as ws:
            ws.send_text(json.dumps({"type": "state"}))
            assert json.loads(ws.receive_text())["type"] == "state"


def test_exposure_check() -> None:
    ServerSettings().check_exposure()
    ServerSettings(host="::1").check_exposure()
    with pytest.raises(ValueError):
        ServerSettings(host="0.0.0.0").check_exposure()
    ServerSettings(host="0.0.0.0", token="t").check_exposure()
    ServerSettings(host="0.0.0.0", allow_insecure=True).check_exposure()


def test_websocket_protocol(client: TestClient) -> None:
    wid = _create(client)
    with client.websocket_connect(f"/ws/{wid}") as ws:
        ws.send_text(json.dumps({"type": "bogus"}))
        assert json.loads(ws.receive_text())["type"] == "error"
        ws.send_text(
            json.dumps({"type": "subscribe", "cameras": [{"name": "cam"}], "lidars": ["spad"]})
        )
        assert json.loads(ws.receive_text())["type"] == "subscribed"
        ws.send_text(json.dumps({"type": "cmd", "v_m_s": 0.2}))
        ws.send_text(json.dumps({"type": "step", "dt_s": 0.5}))
        st = json.loads(ws.receive_text())
        assert st["type"] == "state" and st["time_s"] == 0.5
        hdr = json.loads(ws.receive_text())
        assert hdr["type"] == "camera" and hdr["frame_index"] == 1
        cam = decode_npz(ws.receive_bytes())
        assert cam["image"].shape == (48, 64) and "depth" in cam
        hdr = json.loads(ws.receive_text())
        assert hdr["type"] == "lidar"
        assert decode_npz(ws.receive_bytes())["points"].shape[1] == 4
        ws.send_text(json.dumps({"type": "subscribe"}))
        json.loads(ws.receive_text())
        ws.send_text(json.dumps({"type": "run", "dt_s": 0.1, "rate_hz": 50}))
        times = [json.loads(ws.receive_text())["time_s"] for _ in range(3)]
        ws.send_text(json.dumps({"type": "stop"}))
        assert times == sorted(times) and times[0] > 0.5


# ----------------------------------------------------------------------------- live server


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture()
def live_server() -> Iterator[str]:
    uvicorn = pytest.importorskip("uvicorn")
    port = _free_port()
    config = uvicorn.Config(
        create_app(ServerSettings(token="tok")), host="127.0.0.1", port=port, log_level="warning"
    )
    server = uvicorn.Server(config)
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    th.join(timeout=10)


def test_python_client_against_live_server(live_server: str) -> None:
    pytest.importorskip("websockets")
    from lunarsim.client import LunarSimClient, LunarSimError

    with pytest.raises(LunarSimError) as ei:
        LunarSimClient(live_server).list_worlds()
    assert ei.value.status == 401
    c = LunarSimClient(live_server, token="tok")
    assert c.health() == {"status": "ok"}
    wid = c.create_world(CFG)["world_id"]
    z, geo = c.heightmap(wid)
    assert z.shape == (161, 161) and geo["resolution_m"] == 0.25
    c.set_sun(wid, azimuth_deg=90.0, elevation_deg=20.0)
    c.rover_cmd(wid, v_m_s=0.2)
    st = c.step(wid, 1.0)
    assert st["time_s"] == 1.0
    cam = c.camera(wid, "cam", buffers=["labels"])
    assert cam["image"].shape == (48, 64) and cam["labels"].dtype == np.uint8
    scan = c.lidar(wid, "spad")
    assert scan["points"].shape[1] == 4
    assert "ply" in c.lidar(wid, "spad", fmt="ply")
    with c.stream(wid) as s:
        s.send(type="subscribe", lidars=["spad"])
        assert s.receive()["type"] == "subscribed"
        st2 = s.step(0.5)
        assert st2["time_s"] == 1.5
        msg = s.receive()
        assert msg["type"] == "lidar" and msg["data"]["points"].shape[1] == 4
    c.delete_world(wid)
    assert c.list_worlds() == []
    _ = read_pcd  # imported helper kept for symmetry with PLY


def test_viewer_formats(client: TestClient) -> None:
    wid = _create(client)
    ref = World(parse_config(CFG))
    r = client.get(f"/worlds/{wid}/heightmap", params={"format": "f32", "max_size": 41})
    rows, cols = int(r.headers["x-rows"]), int(r.headers["x-cols"])
    z = np.frombuffer(r.content, dtype="<f4").reshape(rows, cols)
    stride = -(-max(ref.heightmap.z.shape) // 41)
    np.testing.assert_allclose(z, ref.heightmap.z[::stride, ::stride], atol=1e-4)
    assert max(rows, cols) <= 41 and float(r.headers["x-resolution-m"]) == 0.25 * stride
    r = client.get(f"/worlds/{wid}/sensors/lidar/spad", params={"format": "bin"})
    pts = np.frombuffer(r.content, dtype="<f4").reshape(-1, 4)
    assert pts.shape[0] == int(r.headers["x-points"]) > 0
    assert np.asarray(json.loads(r.headers["x-pose"])).shape == (4, 4)


def test_viewer_static_and_preload(tmp_path: Path) -> None:
    import yaml

    p = tmp_path / "w.yaml"
    p.write_text(yaml.safe_dump(CFG))
    app = create_app(ServerSettings(token="tok", preload_worlds=[str(p)]))
    with TestClient(app) as c:
        r = c.get("/", follow_redirects=False)
        assert r.status_code in (302, 307) and r.headers["location"] == "/viewer/"
        page = c.get("/viewer/")  # static files need no token; data endpoints do
        assert page.status_code == 200 and "importmap" in page.text
        js = c.get("/viewer/vendor/three.module.js")
        assert js.status_code == 200 and "http" not in page.text.split("importmap")[1][:200]
        assert c.get("/worlds").status_code == 401
        lst = c.get("/worlds", headers={"Authorization": "Bearer tok"}).json()
        assert len(lst) == 1
