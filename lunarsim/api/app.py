"""FastAPI application: REST + WebSocket around :class:`lunarsim.core.world.World`.

All simulation work runs in worker threads (``run_in_threadpool``) under the
world's lock, so the event loop never blocks on rendering or terrain generation.
No physics lives here.
"""

from __future__ import annotations

import asyncio
import json
import secrets
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any, TypeVar

import yaml
from fastapi import Depends, FastAPI, Header, HTTPException, Query, WebSocket
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool
from starlette.websockets import WebSocketDisconnect

from lunarsim.api.registry import Registry, WorldEntry
from lunarsim.api.schemas import (
    CameraFormat,
    CreateWorld,
    LidarFormat,
    LightSwitch,
    RoverCommandBody,
    StepBody,
    SunBody,
    WaypointsBody,
)
from lunarsim.api.settings import ServerSettings
from lunarsim.config.load import parse_config
from lunarsim.core.raycast import HAS_NUMBA
from lunarsim.core.sensors.imu import ImuSamples
from lunarsim.core.world import CameraFrame, World
from lunarsim.io.bundle import (
    camera_npz,
    lidar_json,
    lidar_npz,
    lidar_pcd,
    lidar_ply,
)
from lunarsim.io.raster import encode_npy, encode_png8, encode_png16
from lunarsim.version import __version__

__all__ = ["create_app"]

T = TypeVar("T")
NPY = "application/x-npy"
NPZ = "application/x-npz"


def create_app(settings: ServerSettings | None = None) -> FastAPI:
    """Build the FastAPI app (one in-memory world registry per app)."""
    cfg = settings or ServerSettings()
    registry = Registry(cfg.max_worlds, cfg.render_cache_size)
    app = FastAPI(
        title="LunarSim",
        version=__version__,
        description="Offline lunar rover simulator API (terrain, sun, camera, LiDAR, rover).",
    )
    app.state.registry = registry
    app.state.settings = cfg
    for path in cfg.preload_worlds:
        from lunarsim.config.load import load_config as _load

        registry.add(World(_load(path)))

    # ------------------------------------------------------------------ helpers
    def check_token(supplied: str | None) -> None:
        if cfg.token and not (supplied and secrets.compare_digest(supplied, cfg.token)):
            raise HTTPException(401, "invalid or missing token")

    def auth(authorization: Annotated[str | None, Header()] = None) -> None:
        token = None
        if authorization and authorization.lower().startswith("bearer "):
            token = authorization[7:].strip()
        check_token(token)

    def entry(world_id: str) -> WorldEntry:
        try:
            return registry.get(world_id)
        except KeyError:
            raise HTTPException(404, f"unknown world {world_id!r}") from None

    async def locked(e: WorldEntry, fn: Callable[[World], T]) -> T:
        def run() -> T:
            with e.lock:
                return fn(e.world)

        try:
            return await run_in_threadpool(run)
        except KeyError as exc:
            raise HTTPException(404, f"unknown name {exc}") from None
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None

    deps = [Depends(auth)]

    static_dir = Path(__file__).resolve().parents[1] / "viewer" / "static"
    if static_dir.is_dir():  # browser viewer (static files only; data needs the token)
        index_html = (static_dir / "index.html").read_text(encoding="utf-8")
        if not cfg.viewer_cdn:
            index_html = index_html.replace('<html lang="ja">', '<html lang="ja" data-cdn="off">')

        @app.get("/viewer/", include_in_schema=False)
        @app.get("/viewer/index.html", include_in_schema=False)
        def viewer_index() -> HTMLResponse:
            return HTMLResponse(index_html)

        app.mount("/viewer", StaticFiles(directory=static_dir, html=True), name="viewer")

        @app.get("/", include_in_schema=False)
        def root() -> RedirectResponse:
            return RedirectResponse("/viewer/")

    # ------------------------------------------------------------------ service
    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/version")
    def version() -> dict[str, Any]:
        return {"version": __version__, "numba": HAS_NUMBA}

    # ------------------------------------------------------------------ worlds
    @app.post("/worlds", dependencies=deps, status_code=201)
    async def create_world(body: CreateWorld) -> dict[str, Any]:
        try:
            registry.reserve()
        except OverflowError as exc:
            raise HTTPException(429, str(exc)) from None
        if body.config_yaml is not None:
            try:
                data = yaml.safe_load(body.config_yaml)
            except yaml.YAMLError as exc:
                raise HTTPException(422, f"invalid YAML: {exc}") from None
            if data is not None and not isinstance(data, dict):
                raise HTTPException(422, "YAML top level must be a mapping")
        else:
            data = body.config
        try:
            config = parse_config(data, seed=body.seed)
        except ValidationError as exc:
            raise HTTPException(422, json.loads(exc.json())) from None
        world = await run_in_threadpool(World, config)
        try:
            wid = registry.add(world)
        except OverflowError as exc:
            raise HTTPException(429, str(exc)) from None
        return {"world_id": wid, "metadata": world.metadata()}

    @app.get("/worlds", dependencies=deps)
    def list_worlds() -> dict[str, list[str]]:
        return {"world_ids": registry.ids()}

    @app.get("/worlds/{world_id}", dependencies=deps)
    async def get_world(world_id: str) -> dict[str, Any]:
        e = entry(world_id)
        return await locked(e, lambda w: w.metadata() | {"state": w.state()})

    @app.delete("/worlds/{world_id}", dependencies=deps, status_code=204)
    def delete_world(world_id: str) -> Response:
        try:
            registry.remove(world_id)
        except KeyError:
            raise HTTPException(404, f"unknown world {world_id!r}") from None
        return Response(status_code=204)

    @app.get("/worlds/{world_id}/config", dependencies=deps)
    async def get_config(world_id: str) -> dict[str, Any]:
        return await locked(entry(world_id), lambda w: w.config.model_dump(mode="json"))

    # ------------------------------------------------------------------ terrain
    def _layer(w: World, layer: str) -> Any:
        if layer == "near":
            return w.terrain
        far = w.terrain_far
        if far is None:
            raise ValueError("this world has a single terrain layer (use layer=near)")
        return far

    @app.get("/worlds/{world_id}/heightmap", dependencies=deps)
    async def heightmap(
        world_id: str,
        fmt: Annotated[str, Query(alias="format", pattern="^(npy|png16|f32)$")] = "npy",
        layer: Annotated[str, Query(pattern="^(near|far)$")] = "near",
        max_size: Annotated[
            int | None, Query(ge=2, description="decimate to <= N per side")
        ] = None,
    ) -> Response:
        def make(w: World) -> tuple[bytes, dict[str, str], str]:
            hm = _layer(w, layer).heightmap
            z, res = hm.z, hm.resolution
            if max_size is not None and max(z.shape) > max_size:
                stride = -(-max(z.shape) // max_size)  # ceil division; keeps the origin node
                z, res = z[::stride, ::stride], res * stride
            hdr = {
                "X-Resolution-M": repr(res),
                "X-Origin-X-M": repr(hm.x0),
                "X-Origin-Y-M": repr(hm.y0),
                "X-Rows": str(z.shape[0]),
                "X-Cols": str(z.shape[1]),
                "X-Row-Order": "north-up" if fmt == "png16" else "south-to-north",
            }
            if fmt == "npy":
                return encode_npy(z), hdr, NPY
            if fmt == "f32":  # raw little-endian float32, row-major (for the viewer)
                return z.astype("<f4").tobytes(), hdr, "application/octet-stream"
            data, meta = encode_png16(z, extra_meta={"resolution_m": res})
            hdr["X-Offset-M"] = repr(meta["offset"])
            hdr["X-Scale-M"] = repr(meta["scale"])
            return data, hdr, "image/png"

        data, hdr, media = await locked(entry(world_id), make)
        return Response(data, media_type=media, headers=hdr)

    @app.get("/worlds/{world_id}/craters", dependencies=deps)
    async def craters(
        world_id: str, layer: Annotated[str, Query(pattern="^(near|far)$")] = "near"
    ) -> list[dict[str, Any]]:
        return await locked(entry(world_id), lambda w: _layer(w, layer).craters.to_records())

    @app.get("/worlds/{world_id}/rocks", dependencies=deps)
    async def rocks(world_id: str) -> list[dict[str, Any]]:
        return await locked(entry(world_id), lambda w: w.terrain.rocks.to_records())

    # ------------------------------------------------------------------ sun / lights
    @app.put("/worlds/{world_id}/sun", dependencies=deps)
    async def set_sun(world_id: str, body: SunBody) -> dict[str, Any]:
        def apply(w: World) -> dict[str, Any]:
            if body.utc is not None:
                w.set_sun_utc(body.utc)
            else:
                assert body.azimuth_deg is not None and body.elevation_deg is not None
                w.set_sun_manual(body.azimuth_deg, body.elevation_deg)
            sun = w.state()["sun"]
            assert isinstance(sun, dict)
            return sun

        return await locked(entry(world_id), apply)

    @app.put("/worlds/{world_id}/lights/{name}", dependencies=deps)
    async def set_light(world_id: str, name: str, body: LightSwitch) -> dict[str, bool]:
        def apply(w: World) -> dict[str, bool]:
            w.set_light(name, body.on)
            return dict(w.light_on)

        return await locked(entry(world_id), apply)

    # ------------------------------------------------------------------ rover / time
    @app.post("/worlds/{world_id}/rover/cmd", dependencies=deps)
    async def rover_cmd(world_id: str, body: RoverCommandBody) -> dict[str, Any]:
        def apply(w: World) -> dict[str, Any]:
            w.set_rover_command(body.v_m_s, body.omega_rad_s, body.steer_rad)
            return w.state()

        return await locked(entry(world_id), apply)

    @app.post("/worlds/{world_id}/rover/waypoints", dependencies=deps)
    async def rover_waypoints(world_id: str, body: WaypointsBody) -> dict[str, Any]:
        def apply(w: World) -> dict[str, Any]:
            w.set_waypoints(list(body.points), body.speed_m_s, body.tolerance_m)
            return w.state()

        return await locked(entry(world_id), apply)

    @app.post("/worlds/{world_id}/step", dependencies=deps)
    async def step(world_id: str, body: StepBody) -> dict[str, Any]:
        def apply(w: World) -> dict[str, Any]:
            w.step(body.dt_s)
            return w.state()

        return await locked(entry(world_id), apply)

    @app.get("/worlds/{world_id}/state", dependencies=deps)
    async def state(world_id: str) -> dict[str, Any]:
        return await locked(entry(world_id), lambda w: w.state())

    @app.get("/worlds/{world_id}/sensors/imu/{name}", dependencies=deps)
    async def imu(
        world_id: str,
        name: str,
        since_s: Annotated[float | None, Query(description="samples with t > since_s")] = None,
    ) -> dict[str, Any]:
        def make(w: World) -> dict[str, Any]:
            if name not in w.imus:
                raise HTTPException(404, f"unknown imu {name!r}")
            return w.imu_samples(name, since_s).to_dict() | {"time_s": w.time_s}

        return await locked(entry(world_id), make)

    # ------------------------------------------------------------------ sensors
    def _camera(e: WorldEntry, name: str, frame_index: int, eye: str) -> CameraFrame:
        w = e.world
        if name not in w.cameras:
            raise KeyError(name)
        frame: CameraFrame = e.cached(
            ("camera", name, eye, frame_index),
            lambda: w.render_camera(name, None, frame_index, None, eye),
        )
        return frame

    @app.get("/worlds/{world_id}/sensors/camera/{name}", dependencies=deps)
    async def camera(
        world_id: str,
        name: str,
        frame_index: Annotated[int, Query(ge=0)] = 0,
        eye: Annotated[str, Query(pattern="^(left|right)$")] = "left",
        fmt: Annotated[CameraFormat, Query(alias="format")] = "png",
        buffers: Annotated[str, Query(description="comma list, npz only")] = "",
    ) -> Response:
        e = entry(world_id)

        def make() -> tuple[bytes, str, dict[str, str]]:
            with e.lock:
                frame = _camera(e, name, frame_index, eye)
                cfg_cam, model = e.world.cameras[name]
            hdr = {
                "X-Exposure-S": repr(frame.image.exposure_s),
                "X-Frame-Index": str(frame_index),
                "X-Pose": json.dumps(frame.pose.matrix().tolist()),
            }
            if fmt == "png":
                img = frame.image.dn / float(2**cfg_cam.sensor.bits - 1)
                return encode_png8(img), "image/png", hdr
            if fmt == "png16":
                from io import BytesIO

                from PIL import Image

                buf = BytesIO()
                Image.fromarray(frame.image.dn).save(buf, format="PNG")
                return buf.getvalue(), "image/png", hdr
            names = [b for b in buffers.split(",") if b]
            return camera_npz(frame, model, names), NPZ, hdr

        try:
            data, media, hdr = await run_in_threadpool(make)
        except KeyError as exc:
            raise HTTPException(404, f"unknown camera {exc}") from None
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None
        return Response(data, media_type=media, headers=hdr)

    @app.get("/worlds/{world_id}/sensors/lidar/{name}", dependencies=deps, response_model=None)
    async def lidar(
        world_id: str,
        name: str,
        frame_index: Annotated[int, Query(ge=0)] = 0,
        fmt: Annotated[LidarFormat, Query(alias="format")] = "npz",
    ) -> Response | JSONResponse:
        e = entry(world_id)

        def make() -> Any:
            with e.lock:
                w = e.world
                if name not in w.lidars:
                    raise KeyError(name)
                return e.cached(
                    ("lidar", name, frame_index), lambda: w.render_lidar(name, None, frame_index)
                )

        try:
            scan = await run_in_threadpool(make)
        except KeyError as exc:
            raise HTTPException(404, f"unknown lidar {exc}") from None
        hdr = {
            "X-Points": str(len(scan)),
            "X-Frame-Index": str(frame_index),
            "X-Pose": json.dumps(scan.pose.matrix().tolist()),
        }
        if fmt == "bin":  # KITTI velodyne layout: float32 x, y, z, intensity (sensor frame)
            data = scan.points.astype("<f4").tobytes()
            return Response(data, media_type="application/octet-stream", headers=hdr)
        if fmt == "json":
            if len(scan) > cfg.max_json_points:
                raise HTTPException(413, f"{len(scan)} points: use format=npz/ply/pcd")
            return JSONResponse(lidar_json(scan), headers=hdr)
        if fmt == "ply":
            return Response(lidar_ply(scan), media_type="application/octet-stream", headers=hdr)
        if fmt == "pcd":
            return Response(lidar_pcd(scan), media_type="application/octet-stream", headers=hdr)
        return Response(lidar_npz(scan), media_type=NPZ, headers=hdr)

    # ------------------------------------------------------------------ websocket
    @app.websocket("/ws/{world_id}")
    async def ws(websocket: WebSocket, world_id: str, token: str | None = None) -> None:
        header = websocket.headers.get("authorization", "")
        supplied = token or (header[7:].strip() if header.lower().startswith("bearer ") else None)
        if cfg.token and not (supplied and secrets.compare_digest(supplied, cfg.token)):
            await websocket.close(code=1008)
            return
        try:
            e = registry.get(world_id)
        except KeyError:
            await websocket.close(code=1008)
            return
        await websocket.accept()
        await WsSession(e, websocket).run()

    return app


class WsSession:
    """One WebSocket client: JSON control messages, JSON state + binary sensor frames.

    Client -> server (JSON): ``cmd``, ``waypoints``, ``step``, ``state``, ``subscribe``,
    ``run`` (auto-step at ``rate_hz`` with ``dt_s``), ``stop``.
    Server -> client: ``{"type": "state", ...}`` after every step; for each subscribed
    sensor a JSON header ``{"type": "camera"|"lidar", "name", "frame_index", "bytes"}``
    followed by one binary message (npz); for each subscribed IMU (every step) a JSON
    ``{"type": "imu", "name", "t", "gyro", "accel", ...}`` with the samples since the last
    message. Errors: ``{"type": "error", "detail"}``.
    """

    def __init__(self, entry: WorldEntry, websocket: WebSocket) -> None:
        self.entry = entry
        self.ws = websocket
        self.cameras: list[tuple[str, str]] = []
        self.lidars: list[str] = []
        self.imus: dict[str, float | None] = {}  # name -> time of the last sample sent
        self.every = 1
        self.steps = 0
        self.runner: asyncio.Task[None] | None = None
        self.send_lock = asyncio.Lock()

    async def _call(self, fn: Callable[[World], T]) -> T:
        def run() -> T:
            with self.entry.lock:
                return fn(self.entry.world)

        return await run_in_threadpool(run)

    async def _send_json(self, msg: dict[str, Any]) -> None:
        async with self.send_lock:
            await self.ws.send_text(json.dumps(msg))

    async def _step(self, dt: float) -> None:
        state = await self._call(lambda w: (w.step(dt), w.state())[1])
        self.steps += 1
        await self._send_json({"type": "state", **state})
        for name, last in list(self.imus.items()):

            def imu_batch(w: World, n: str = name, t: float | None = last) -> ImuSamples:
                return w.imu_samples(n, t)

            batch = await self._call(imu_batch)
            if len(batch):
                self.imus[name] = float(batch.t[-1])
                await self._send_json({"type": "imu", "name": name, **batch.to_dict()})
        if self.steps % self.every:
            return
        idx = self.steps
        for name, eye in self.cameras:

            def cam(w: World, name: str = name, eye: str = eye) -> bytes:
                frame = w.render_camera(name, None, idx, None, eye)
                return camera_npz(frame, w.cameras[name][1], ["depth", "labels"])

            data = await self._call(cam)
            async with self.send_lock:
                await self.ws.send_text(
                    json.dumps(
                        {
                            "type": "camera",
                            "name": name,
                            "eye": eye,
                            "frame_index": idx,
                            "bytes": len(data),
                        }
                    )
                )
                await self.ws.send_bytes(data)
        for name in self.lidars:

            def lid(w: World, name: str = name) -> bytes:
                return lidar_npz(w.render_lidar(name, None, idx))

            data = await self._call(lid)
            async with self.send_lock:
                await self.ws.send_text(
                    json.dumps(
                        {"type": "lidar", "name": name, "frame_index": idx, "bytes": len(data)}
                    )
                )
                await self.ws.send_bytes(data)

    async def _run_loop(self, dt: float, rate_hz: float) -> None:
        period = 1.0 / rate_hz
        loop = asyncio.get_running_loop()
        while True:
            t0 = loop.time()
            await self._step(dt)
            await asyncio.sleep(max(0.0, period - (loop.time() - t0)))

    async def _handle(self, msg: dict[str, Any]) -> None:
        kind = msg.get("type")
        if kind == "cmd":
            body = RoverCommandBody.model_validate({k: v for k, v in msg.items() if k != "type"})
            await self._call(
                lambda w: w.set_rover_command(body.v_m_s, body.omega_rad_s, body.steer_rad)
            )
        elif kind == "waypoints":
            wb = WaypointsBody.model_validate({k: v for k, v in msg.items() if k != "type"})
            await self._call(
                lambda w: w.set_waypoints(list(wb.points), wb.speed_m_s, wb.tolerance_m)
            )
        elif kind == "step":
            await self._step(StepBody.model_validate({"dt_s": msg.get("dt_s")}).dt_s)
        elif kind == "state":
            await self._send_json({"type": "state", **(await self._call(lambda w: w.state()))})
        elif kind == "subscribe":
            world = self.entry.world
            cams = [(c["name"], c.get("eye", "left")) for c in msg.get("cameras", [])]
            for name, _ in cams:
                if name not in world.cameras:
                    raise ValueError(f"unknown camera {name!r}")
            lids = list(msg.get("lidars", []))
            for name in lids:
                if name not in world.lidars:
                    raise ValueError(f"unknown lidar {name!r}")
            imus = list(msg.get("imus", []))
            for name in imus:
                if name not in world.imus:
                    raise ValueError(f"unknown imu {name!r}")
            self.cameras, self.lidars = cams, lids
            now = world.time_s
            self.imus = {n: now for n in imus}
            self.every = max(1, int(msg.get("every_n_steps", 1)))
            await self._send_json(
                {"type": "subscribed", "cameras": cams, "lidars": lids, "imus": imus}
            )
        elif kind == "run":
            await self._stop()
            dt = StepBody.model_validate({"dt_s": msg.get("dt_s", 0.1)}).dt_s
            rate = float(msg.get("rate_hz", 10.0))
            if not 0.0 < rate <= 100.0:
                raise ValueError("rate_hz must be in (0, 100]")
            self.runner = asyncio.create_task(self._run_loop(dt, rate))
        elif kind == "stop":
            await self._stop()
        else:
            raise ValueError(f"unknown message type {kind!r}")

    async def _stop(self) -> None:
        if self.runner is not None:
            self.runner.cancel()
            try:
                await self.runner
            except asyncio.CancelledError:
                pass
            self.runner = None

    async def run(self) -> None:
        try:
            while True:
                text = await self.ws.receive_text()
                try:
                    msg = json.loads(text)
                    if not isinstance(msg, dict):
                        raise ValueError("message must be a JSON object")
                    await self._handle(msg)
                except (ValueError, ValidationError, KeyError) as exc:
                    await self._send_json({"type": "error", "detail": str(exc)})
        except WebSocketDisconnect:
            pass
        finally:
            await self._stop()
