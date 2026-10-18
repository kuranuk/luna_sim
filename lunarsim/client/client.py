"""Thin client for the LunarSim REST/WebSocket API.

REST uses :mod:`urllib.request` (no third-party HTTP stack); WebSocket streaming uses
``websockets.sync.client`` (optional, only needed for :meth:`LunarSimClient.stream`).
Sensor bundles are ``.npz`` archives decoded with pickle disabled.

Example::

    from lunarsim.client import LunarSimClient
    c = LunarSimClient("http://127.0.0.1:8000")
    wid = c.create_world("configs/crater_field.yaml", seed=1)["world_id"]
    c.rover_cmd(wid, v_m_s=0.2)
    c.step(wid, 1.0)
    img = c.camera(wid, "cam0", fmt="npz", buffers=["depth"])["image"]
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from lunarsim.io.bundle import decode_npz

__all__ = ["LunarSimClient", "LunarSimError", "StreamSession"]


class LunarSimError(RuntimeError):
    """HTTP error returned by the server (``status`` and server ``detail``)."""

    def __init__(self, status: int, detail: Any) -> None:
        super().__init__(f"HTTP {status}: {detail}")
        self.status = status
        self.detail = detail


class LunarSimClient:
    """REST client. ``token`` is sent as ``Authorization: Bearer``."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        token: str | None = None,
        timeout: float = 600.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout

    # ------------------------------------------------------------------ transport
    def _request(
        self, method: str, path: str, body: Any = None, params: dict[str, Any] | None = None
    ) -> tuple[bytes, dict[str, str]]:
        url = self.base_url + path
        if params:
            url += "?" + urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
        data = None
        headers = {"Accept": "*/*"}
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        req = urllib.request.Request(url, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return resp.read(), dict(resp.headers)
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            try:
                detail = json.loads(raw).get("detail", raw.decode())
            except (ValueError, AttributeError):
                detail = raw.decode(errors="replace")
            raise LunarSimError(exc.code, detail) from None

    def _json(self, method: str, path: str, body: Any = None, **params: Any) -> Any:
        raw, _ = self._request(method, path, body, params or None)
        return json.loads(raw) if raw else None

    # ------------------------------------------------------------------ service
    def health(self) -> dict[str, str]:
        """``GET /health``."""
        return dict(self._json("GET", "/health"))

    def version(self) -> dict[str, Any]:
        """``GET /version``."""
        return dict(self._json("GET", "/version"))

    # ------------------------------------------------------------------ worlds
    def create_world(
        self, config: str | Path | dict[str, Any], seed: int | None = None
    ) -> dict[str, Any]:
        """Create a world from a YAML file path, a YAML string or a config dict."""
        body: dict[str, Any] = {"seed": seed}
        if isinstance(config, dict):
            body["config"] = config
        else:
            p = Path(config)
            body["config_yaml"] = p.read_text(encoding="utf-8") if p.exists() else str(config)
        return dict(self._json("POST", "/worlds", body))

    def list_worlds(self) -> list[str]:
        """Registered world ids."""
        return list(self._json("GET", "/worlds")["world_ids"])

    def world(self, world_id: str) -> dict[str, Any]:
        """Metadata and state."""
        return dict(self._json("GET", f"/worlds/{world_id}"))

    def delete_world(self, world_id: str) -> None:
        """Delete a world."""
        self._request("DELETE", f"/worlds/{world_id}")

    def heightmap(
        self, world_id: str, layer: str = "near"
    ) -> tuple[npt.NDArray[np.float64], dict[str, float]]:
        """Heightmap ``z[i, j]`` (rows South->North) and grid geometry."""
        import io

        raw, hdr = self._request("GET", f"/worlds/{world_id}/heightmap", params={"layer": layer})
        z = np.load(io.BytesIO(raw), allow_pickle=False)
        h = {k.lower(): v for k, v in hdr.items()}
        geo = {
            "resolution_m": float(h["x-resolution-m"]),
            "x0_m": float(h["x-origin-x-m"]),
            "y0_m": float(h["x-origin-y-m"]),
        }
        return np.asarray(z, dtype=np.float64), geo

    def craters(self, world_id: str, layer: str = "near") -> list[dict[str, Any]]:
        """Crater ground truth."""
        return list(self._json("GET", f"/worlds/{world_id}/craters", layer=layer))

    def rocks(self, world_id: str) -> list[dict[str, Any]]:
        """Rock ground truth (near layer)."""
        return list(self._json("GET", f"/worlds/{world_id}/rocks"))

    # ------------------------------------------------------------------ control
    def set_sun(
        self,
        world_id: str,
        azimuth_deg: float | None = None,
        elevation_deg: float | None = None,
        utc: str | None = None,
    ) -> dict[str, Any]:
        """Manual sun (azimuth/elevation) or astronomical (UTC)."""
        body = {"azimuth_deg": azimuth_deg, "elevation_deg": elevation_deg, "utc": utc}
        return dict(
            self._json(
                "PUT", f"/worlds/{world_id}/sun", {k: v for k, v in body.items() if v is not None}
            )
        )

    def set_light(self, world_id: str, name: str, on: bool) -> dict[str, bool]:
        """Switch a rover light."""
        return dict(self._json("PUT", f"/worlds/{world_id}/lights/{name}", {"on": on}))

    def rover_cmd(
        self, world_id: str, v_m_s: float = 0.0, omega_rad_s: float = 0.0, steer_rad: float = 0.0
    ) -> dict[str, Any]:
        """Velocity command."""
        body = {"v_m_s": v_m_s, "omega_rad_s": omega_rad_s, "steer_rad": steer_rad}
        return dict(self._json("POST", f"/worlds/{world_id}/rover/cmd", body))

    def waypoints(
        self, world_id: str, points: Sequence[tuple[float, float]], speed_m_s: float,
        tolerance_m: float = 0.3,
    ) -> dict[str, Any]:  # fmt: skip
        """Follow waypoints."""
        body = {"points": [list(p) for p in points], "speed_m_s": speed_m_s,
                "tolerance_m": tolerance_m}  # fmt: skip
        return dict(self._json("POST", f"/worlds/{world_id}/rover/waypoints", body))

    def step(self, world_id: str, dt_s: float) -> dict[str, Any]:
        """Advance time; returns the new state."""
        return dict(self._json("POST", f"/worlds/{world_id}/step", {"dt_s": dt_s}))

    def state(self, world_id: str) -> dict[str, Any]:
        """Current state."""
        return dict(self._json("GET", f"/worlds/{world_id}/state"))

    # ------------------------------------------------------------------ sensors
    def camera(
        self,
        world_id: str,
        name: str,
        frame_index: int = 0,
        eye: str = "left",
        fmt: str = "npz",
        buffers: Sequence[str] = (),
    ) -> dict[str, Any]:
        """Camera frame: ``npz`` -> dict of arrays; ``png``/``png16`` -> ``{"png": bytes}``."""
        params = {"frame_index": frame_index, "eye": eye, "format": fmt,
                  "buffers": ",".join(buffers)}  # fmt: skip
        raw, hdr = self._request("GET", f"/worlds/{world_id}/sensors/camera/{name}", params=params)
        if fmt == "npz":
            return dict(decode_npz(raw))
        return {"png": raw, "headers": hdr}

    def lidar(
        self, world_id: str, name: str, frame_index: int = 0, fmt: str = "npz"
    ) -> dict[str, Any]:
        """LiDAR frame: ``npz`` -> dict of arrays, ``json`` -> dict, ``ply``/``pcd`` -> bytes."""
        params = {"frame_index": frame_index, "format": fmt}
        raw, hdr = self._request("GET", f"/worlds/{world_id}/sensors/lidar/{name}", params=params)
        if fmt == "npz":
            return dict(decode_npz(raw))
        if fmt == "json":
            return dict(json.loads(raw))
        return {fmt: raw, "headers": hdr}

    # ------------------------------------------------------------------ websocket
    @contextmanager
    def stream(self, world_id: str) -> Iterator[StreamSession]:
        """WebSocket session (requires the ``websockets`` package)."""
        from websockets.sync.client import connect

        ws_url = self.base_url.replace("http://", "ws://").replace("https://", "wss://")
        url = f"{ws_url}/ws/{world_id}"
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else None
        with connect(url, additional_headers=headers, max_size=None) as conn:
            yield StreamSession(conn)


class StreamSession:
    """Synchronous WebSocket session (see the protocol in :class:`lunarsim.api.app.WsSession`)."""

    def __init__(self, conn: Any) -> None:
        self.conn = conn

    def send(self, **msg: Any) -> None:
        """Send a JSON control message (``type`` plus fields)."""
        self.conn.send(json.dumps(msg))

    def receive(self, timeout: float | None = 60.0) -> dict[str, Any]:
        """Next message; sensor headers are joined with their binary npz payload."""
        msg = json.loads(self.conn.recv(timeout=timeout))
        if msg.get("type") in ("camera", "lidar"):
            payload = self.conn.recv(timeout=timeout)
            msg["data"] = decode_npz(payload)
        return dict(msg)

    def step(self, dt_s: float) -> dict[str, Any]:
        """Step and return the state message (sensor messages remain queued)."""
        self.send(type="step", dt_s=dt_s)
        while True:
            m = self.receive()
            if m["type"] in ("state", "error"):
                return m
