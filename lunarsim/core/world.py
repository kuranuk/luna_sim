"""World: the single container of simulation state.

A :class:`World` is fully determined by its validated :class:`WorldConfig` (which
includes the seed) plus the explicit state changes applied to it (sun updates,
rover commands and time steps, light switching). The near patch of a two-layer
terrain follows the rover automatically.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import numpy.typing as npt

from lunarsim.config.load import load_config
from lunarsim.config.schema import (
    CameraConfig,
    LidarConfig,
    MountConfig,
    SpotlightConfig,
    WorldConfig,
    config_hash,
)
from lunarsim.core.frames import R_BODY_FROM_CAMERA, Pose, enu_to_azel
from lunarsim.core.illumination.horizon import (
    IlluminationMap,
    horizon_map,
    illumination_map,
    sun_track,
)
from lunarsim.core.illumination.lights import LightState, light_state
from lunarsim.core.illumination.spice import spice_available, sun_spice
from lunarsim.core.illumination.sun import (
    SunState,
    earthshine_irradiance,
    sun_astronomical,
    sun_manual,
)
from lunarsim.core.illumination.visibility import SkyLighting
from lunarsim.core.raycast.heightfield import HeightfieldCaster
from lunarsim.core.raycast.layered import LayeredCaster
from lunarsim.core.render import RenderBuffers, render_rays
from lunarsim.core.rng import RngFactory
from lunarsim.core.rover import Rover, RoverStatus, WaypointFollower
from lunarsim.core.sensors.camera import CameraModel
from lunarsim.core.sensors.camera_noise import SensorImage, expose
from lunarsim.core.sensors.imu import ImuSamples, ImuSimulator
from lunarsim.core.sensors.lidar import LidarScan, SensorMotion, render_lidar
from lunarsim.core.terrain import Heightmap, Terrain, build_terrain
from lunarsim.core.terrain.dem import BaseSurface
from lunarsim.core.terrain.layers import TwoLayerTerrain

__all__ = ["CameraFrame", "World", "parse_utc"]


def parse_utc(text: str) -> datetime:
    """Parse an ISO 8601 UTC timestamp (``Z`` suffix allowed)."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


@dataclass(frozen=True)
class CameraFrame:
    """One rendered camera frame with ground truth.

    Attributes:
        name: Camera name.
        frame_index: Frame index (selects the noise stream).
        pose: ``T_world_camera`` (OpenCV optical frame).
        buffers: Radiance, depth, normals, labels, shadow mask, ...
        image: Digitised sensor output.
    """

    name: str
    frame_index: int
    pose: Pose
    buffers: RenderBuffers
    image: SensorImage


class World:
    """Terrain layers, sun and sensors built from config and seed."""

    def __init__(self, config: WorldConfig, workers: int | None = None) -> None:
        self.config = config
        self.rngs = RngFactory(config.seed)
        self._two_layer: TwoLayerTerrain | None = None
        self.base_surface = BaseSurface.from_config(config.terrain, config.site)
        if config.terrain.near.enabled:
            self._two_layer = TwoLayerTerrain(
                config.terrain, self.rngs, workers=workers, base=self.base_surface
            )
        else:
            self._single = build_terrain(
                config.terrain, self.rngs, workers=workers, base=self.base_surface
            )
        self._build_caster()
        self.sun: SunState = self._sun_from_config()
        self.cameras: dict[str, tuple[CameraConfig, CameraModel]] = {
            c.name: (c, CameraModel.from_config(c.intrinsics)) for c in config.cameras
        }
        self.lidars: dict[str, LidarConfig] = {s.name: s for s in config.lidars}
        self.lights: dict[str, SpotlightConfig] = {lt.name: lt for lt in config.lights}
        self.light_on: dict[str, bool] = {lt.name: lt.enabled for lt in config.lights}
        self.time_s = 0.0
        self.rover = Rover(config.rover, self._heights)
        self.follower: WaypointFollower | None = None
        self.imus: dict[str, ImuSimulator] = {
            m.name: ImuSimulator(m, self.rngs, config.site.latitude_deg) for m in config.imus
        }
        self._push_imus()
        # Incremented on every change that can alter sensor output (cache key for servers).
        self.version = 0

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_yaml(cls, path: str | Path, seed: int | None = None) -> World:
        """Build a world from a scenario YAML file (``seed`` overrides the file)."""
        return cls(load_config(path, seed=seed))

    def _build_caster(self) -> None:
        backend = self.config.render.backend
        if self._two_layer is not None:
            self.caster: HeightfieldCaster | LayeredCaster = LayeredCaster(
                HeightfieldCaster(self._two_layer.near.heightmap, backend),
                HeightfieldCaster(self._two_layer.far.heightmap, backend),
            )
        else:
            self.caster = HeightfieldCaster(self._single.heightmap, backend)

    def _sun_from_config(self) -> SunState:
        s = self.config.sun
        if s.mode == "manual":
            return self._manual_sun(s.azimuth_deg, s.elevation_deg)
        return self._astronomical_sun(parse_utc(s.utc))

    def _manual_sun(self, azimuth_deg: float, elevation_deg: float) -> SunState:
        s = self.config.sun
        return sun_manual(
            azimuth_deg,
            elevation_deg,
            s.distance_au,
            s.irradiance_1au_w_m2,
            s.earth_azimuth_deg,
            s.earth_elevation_deg,
            s.earth_distance_m,
        )

    @property
    def ephemeris(self) -> str:
        """Ephemeris used in astronomical mode: ``"spice"`` or ``"analytic"``."""
        s = self.config.sun
        if s.ephemeris == "spice":
            return "spice"
        if s.ephemeris == "auto" and spice_available(s.spice_kernels):
            return "spice"
        return "analytic"

    def _astronomical_sun(self, utc: datetime) -> SunState:
        s, site = self.config.sun, self.config.site
        if self.ephemeris == "spice":
            return sun_spice(
                site.latitude_deg,
                site.longitude_deg,
                utc,
                s.spice_kernels,
                s.spice_frame,
                s.irradiance_1au_w_m2,
            )
        return sun_astronomical(site.latitude_deg, site.longitude_deg, utc, s.irradiance_1au_w_m2)

    @property
    def sky(self) -> SkyLighting:
        """Distant light sources with the configured penumbra / curvature / earthshine."""
        rc, tc = self.config.render, self.config.terrain
        return SkyLighting(
            self.sun,
            penumbra_samples=rc.penumbra_samples,
            sun_radius_m=rc.sun_radius_m,
            limb_darkening_u=rc.limb_darkening_u,
            curvature_radius_m=tc.moon_radius_m if tc.curvature else None,
            earthshine_w_m2=self.earthshine_w_m2,
        )

    @property
    def earthshine_w_m2(self) -> float:
        """Earthshine irradiance normal to the Earth direction [W/m²] (0 if disabled)."""
        s = self.config.sun
        if not s.earthshine:
            return 0.0
        return earthshine_irradiance(self.sun, s.earth_geometric_albedo)

    # ------------------------------------------------------------------ terrain access
    @property
    def terrain(self) -> Terrain:
        """Finest terrain layer (the near patch in two-layer mode)."""
        return self._two_layer.near if self._two_layer is not None else self._single

    @property
    def terrain_far(self) -> Terrain | None:
        """Coarse far layer (None in single-layer mode)."""
        return self._two_layer.far if self._two_layer is not None else None

    @property
    def layers(self) -> list[Terrain]:
        """Layers in ray-caster index order (0 = finest)."""
        far = self.terrain_far
        return [self.terrain] if far is None else [self.terrain, far]

    @property
    def heightmap(self) -> Heightmap:
        """Finest terrain elevation grid."""
        return self.terrain.heightmap

    def recenter_near(self, x: float, y: float) -> None:
        """Regenerate the near patch around ``(x, y)`` [m] (two-layer mode only)."""
        if self._two_layer is None:
            raise RuntimeError("recenter_near requires terrain.near.enabled")
        self._two_layer.recenter(x, y)
        self._build_caster()
        if hasattr(self, "version"):
            self.version += 1

    def _heights(
        self, x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Vectorised terrain height on the triangulated surface, finest layer first."""
        xa, ya = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        layers = self.layers
        z = layers[-1].heightmap.height_at_mesh(xa, ya)
        if len(layers) > 1:
            near = layers[0].heightmap
            inside = near.contains(xa, ya)
            if inside.any():
                z = np.where(inside, near.height_at_mesh(xa, ya), z)
        return np.asarray(z, dtype=np.float64)

    def height_at(self, x: float, y: float) -> float:
        """Terrain elevation [m] on the rendered (triangulated) surface, finest layer first."""
        for layer in self.layers:
            if bool(layer.heightmap.contains(x, y)):
                return float(layer.heightmap.height_at_mesh(x, y))
        raise ValueError(f"({x}, {y}) is outside the terrain")

    # ------------------------------------------------------------------ sun
    def set_sun_manual(self, azimuth_deg: float, elevation_deg: float) -> SunState:
        """Set the sun direction directly (Earth as configured for manual mode)."""
        self.sun = self._manual_sun(azimuth_deg, elevation_deg)
        self.version += 1
        return self.sun

    def set_sun_utc(self, utc: datetime | str) -> SunState:
        """Set the sun (and Earth) from the site location and a UTC time."""
        t = parse_utc(utc) if isinstance(utc, str) else utc
        self.sun = self._astronomical_sun(t)
        self.version += 1
        return self.sun

    def reset_dynamic(self) -> None:
        """Return time, rover, sun, lights and the near patch to their configured initial state.

        Terrain is not regenerated (it is fully determined by the config), so repeated runs
        after a reset are identical to runs on a freshly built world.
        """
        self.time_s = 0.0
        self.follower = None
        self.sun = self._sun_from_config()
        self.light_on = {lt.name: lt.enabled for lt in self.config.lights}
        nr = self.config.terrain.near
        if self._two_layer is not None:
            hm = self._two_layer.near.heightmap
            cx = 0.5 * (hm.extent[0] + hm.extent[1])
            cy = 0.5 * (hm.extent[2] + hm.extent[3])
            r = nr.resolution_m
            if abs(cx - nr.center_x_m) > r or abs(cy - nr.center_y_m) > r:
                self.recenter_near(nr.center_x_m, nr.center_y_m)
        rc = self.config.rover
        self.rover.odometer_m = 0.0
        self.rover.reset(rc.x_m, rc.y_m, math.radians(rc.yaw_deg))
        for imu in self.imus.values():
            imu.reset()
        self._push_imus()
        self.version += 1

    def long_term_illumination(
        self,
        start: datetime | str,
        end: datetime | str,
        step: timedelta,
        layer: str = "auto",
        n_azimuth: int = 64,
        stride: int = 1,
    ) -> IlluminationMap:
        """Illumination / PSR statistics over ``[start, end]`` (astronomical Sun at the site).

        ``layer``: ``"near"``, ``"far"`` or ``"auto"`` (far layer if present). The horizon
        map spans that layer only (terrain beyond it does not block).
        """
        t0 = parse_utc(start) if isinstance(start, str) else start
        t1 = parse_utc(end) if isinstance(end, str) else end
        far = self.terrain_far
        terrain = far if (layer in ("auto", "far") and far is not None) else self.terrain
        if layer == "far" and far is None:
            raise ValueError("this world has no far layer")
        hor = horizon_map(terrain.heightmap, n_azimuth, stride, backend=self.config.render.backend)
        suns = sun_track(self._astronomical_sun, t0, t1, step)
        return illumination_map(hor, suns, self.config.render.sun_radius_m)

    # ------------------------------------------------------------------ rover / time
    def set_rover_command(self, v: float, omega: float = 0.0, steer: float = 0.0) -> None:
        """Velocity command: v [m/s], yaw rate ω [rad/s] (skid-steer) or steer [rad].

        Cancels any active waypoint following.
        """
        self.follower = None
        self.rover.set_command(v, omega, steer)

    def set_waypoints(
        self, waypoints: list[tuple[float, float]], speed_m_s: float, tolerance_m: float = 0.3
    ) -> None:
        """Follow waypoints (world ENU [m]) at ``speed_m_s``; replaces velocity commands."""
        pts = [(float(p[0]), float(p[1])) for p in waypoints]
        self.follower = WaypointFollower(pts, speed_m_s, tolerance_m)

    def step(self, dt: float, control_period_s: float = 0.1) -> RoverStatus:
        """Advance simulation time by ``dt`` [s] (deterministic).

        With waypoints active, the command is updated every ``control_period_s`` (starting
        at the beginning of the call). With IMUs, rover sub-steps end on every IMU sample
        instant ``k / rate_hz`` where the IMU poses are recorded.
        """
        if dt < 0.0:
            raise ValueError("dt must be non-negative")
        status = self.rover.status
        t = self.time_s
        remaining = dt
        ctrl_left = 0.0  # time until the next waypoint-controller update
        while remaining > 1e-12:
            h = remaining
            if self.follower is not None:
                if ctrl_left <= 1e-12:
                    r = self.rover
                    v, w, st = self.follower.command(
                        r.x, r.y, r.yaw, self.config.rover.kinematics,
                        self.config.rover.wheelbase_m,
                    )  # fmt: skip
                    r.set_command(v, w, st)
                    ctrl_left = control_period_s
                h = min(h, ctrl_left)
            for imu in self.imus.values():
                h = min(h, max(imu.next_time - t, 0.0))
            status = self.rover.step(h)
            remaining -= h
            ctrl_left -= h
            t += h
            for imu in self.imus.values():  # snap to the sample instant (no drift)
                if abs(imu.next_time - t) < 1e-9:
                    t = imu.next_time
            self._follow_rover()
            self._push_imus(t)
        self.time_s += dt
        self.version += 1
        return status

    def _push_imus(self, t: float | None = None) -> None:
        """Record IMU samples whose instant is reached (at ``t``, default current time)."""
        now = self.time_s if t is None else t
        for name, imu in self.imus.items():
            while imu.next_time <= now + 1e-9:
                cfg = self.config.imus[[m.name for m in self.config.imus].index(name)]
                imu.push(self.rover.pose @ self.mount_pose(cfg.mount))

    def imu_samples(self, name: str, since_s: float | None = None) -> ImuSamples:
        """Buffered samples of IMU ``name`` with ``t > since_s`` [s]."""
        return self.imus[name].samples(since_s)

    def _follow_rover(self) -> None:
        """Re-centre the near patch when the rover nears its border (two-layer mode)."""
        if self._two_layer is None:
            return
        nr = self.config.terrain.near
        hm = self._two_layer.near.heightmap
        cx = 0.5 * (hm.extent[0] + hm.extent[1])
        cy = 0.5 * (hm.extent[2] + hm.extent[3])
        if max(abs(self.rover.x - cx), abs(self.rover.y - cy)) > 0.25 * nr.size_m:
            self.recenter_near(self.rover.x, self.rover.y)
            self.rover.reset_contact()

    # ------------------------------------------------------------------ mounts
    @staticmethod
    def mount_pose(m: MountConfig) -> Pose:
        """``T_body_mount`` from a mount config."""
        return Pose.from_euler_zyx(
            math.radians(m.roll_deg),
            math.radians(m.pitch_deg),
            math.radians(m.yaw_deg),
            (m.x_m, m.y_m, m.z_m),
        )

    def camera_pose(self, name: str, eye: str = "left") -> Pose:
        """``T_world_camera`` (optical frame) of a rover-mounted camera."""
        cfg, _ = self.cameras[name]
        T = self.rover.pose @ self.mount_pose(cfg.mount) @ Pose(R_BODY_FROM_CAMERA, np.zeros(3))
        if eye == "right":
            if cfg.stereo_baseline_m is None:
                raise ValueError(f"camera {name!r} is not a stereo pair")
            T = T @ Pose(np.eye(3), np.array([cfg.stereo_baseline_m, 0.0, 0.0]))
        elif eye != "left":
            raise ValueError("eye must be 'left' or 'right'")
        return T

    def lidar_pose(self, name: str) -> Pose:
        """``T_world_sensor`` of a rover-mounted LiDAR."""
        return self.rover.pose @ self.mount_pose(self.lidars[name].mount)

    def set_light(self, name: str, on: bool) -> None:
        """Switch a rover light on/off."""
        if name not in self.lights:
            raise KeyError(name)
        self.light_on[name] = bool(on)
        self.version += 1

    def active_lights(self) -> list[LightState]:
        """Lights that are on, placed at their rover mounts."""
        return [
            light_state(cfg, self.rover.pose @ self.mount_pose(cfg.mount))
            for name, cfg in self.lights.items()
            if self.light_on[name]
        ]

    # ------------------------------------------------------------------ sensors
    def render_camera(
        self,
        name: str,
        T_world_camera: Pose | None = None,
        frame_index: int = 0,
        exposure_s: float | None = None,
        eye: str = "left",
    ) -> CameraFrame:
        """Render camera ``name``; pose defaults to its rover mount (``eye`` for stereo)."""
        cfg, model = self.cameras[name]
        pose = T_world_camera if T_world_camera is not None else self.camera_pose(name, eye)
        buffers = render_rays(
            self.layers,
            self.caster,
            self.sun,
            self.config.surface,
            model,
            pose,
            self.config.render.shadow_bias_m,
            sun_band_fraction=cfg.sensor.band_fraction,
            lights=self.active_lights(),
            sky=self.sky,
        )
        stream = f"sensor/camera/{name}" + ("/right" if eye == "right" else "")
        rng = self.rngs.generator(f"{stream}/frame/{frame_index:06d}")
        image = expose(buffers.radiance, cfg.sensor, rng, exposure_s)
        return CameraFrame(name, frame_index, pose, buffers, image)

    def render_stereo(
        self, name: str, frame_index: int = 0, exposure_s: float | None = None
    ) -> tuple[CameraFrame, CameraFrame]:
        """Left and right frames of a mounted stereo camera (independent noise).

        With auto exposure the right camera reuses the left exposure time (synchronised).
        """
        left = self.render_camera(name, None, frame_index, exposure_s, "left")
        right = self.render_camera(name, None, frame_index, left.image.exposure_s, "right")
        return left, right

    def render_lidar(
        self,
        name: str,
        T_world_sensor: Pose | None = None,
        frame_index: int = 0,
        motion: SensorMotion | None = None,
    ) -> LidarScan:
        """Simulate LiDAR ``name``; pose defaults to its rover mount.

        For a mounted LiDAR without explicit ``motion`` the rover's current velocity is
        used (motion distortion of spinning/rosette scans).
        """
        if T_world_sensor is None:
            T_world_sensor = self.lidar_pose(name)
            if motion is None and (self.rover.command.v != 0.0 or self.rover.status.yaw_rate):
                lin, ang = self.rover.body_twist_world()
                r = T_world_sensor.t - self.rover.pose.t
                motion = SensorMotion(linear=lin + np.cross(ang, r), angular=ang)
        rng = self.rngs.generator(f"sensor/lidar/{name}/frame/{frame_index:06d}")
        return render_lidar(
            self.layers,
            self.caster,
            self.lidars[name],
            T_world_sensor,
            rng,
            motion,
            sun=self.sun,
            shadow_bias_m=self.config.render.shadow_bias_m,
            sky=self.sky,
        )

    # ------------------------------------------------------------------ metadata
    @property
    def seed(self) -> int:
        """Root seed."""
        return self.config.seed

    @property
    def config_hash(self) -> str:
        """SHA-256 of the canonical config (identifies the world reproducibly)."""
        return config_hash(self.config)

    def state(self) -> dict[str, object]:
        """JSON-serialisable dynamic state (time, rover pose and status, sun)."""
        r = self.rover
        c = r.contact
        st = r.status
        return {
            "time_s": self.time_s,
            "version": self.version,
            "rover": {
                "position_m": [float(v) for v in r.pose.t],
                "quaternion_wxyz": [float(v) for v in r.pose.quat()],
                "roll_deg": math.degrees(c.roll),
                "pitch_deg": math.degrees(c.pitch),
                "yaw_deg": math.degrees(r.yaw),
                "ground_track_m": [r.x, r.y],
                "odometer_m": r.odometer_m,
                "slip": st.slip,
                "body_velocity_m_s": list(st.body_velocity),
                "yaw_rate_rad_s": st.yaw_rate,
                "slope_deg": st.slope_deg,
                "warnings": list(st.warnings),
                "command": {
                    "v_m_s": r.command.v,
                    "omega_rad_s": r.command.omega,
                    "steer_rad": r.command.steer,
                },
                "waypoints_remaining": (
                    0
                    if self.follower is None
                    else len(self.follower.waypoints) - self.follower.index
                ),
            },
            "sun": {
                "azimuth_deg": self.sun.azimuth_deg,
                "elevation_deg": self.sun.elevation_deg,
                "irradiance_w_m2": self.sun.irradiance,
            },
            "earth": self._earth_state(),
            "lights": dict(self.light_on),
        }

    def _earth_state(self) -> dict[str, float] | None:
        e = self.sun.earth_direction
        if e is None:
            return None
        az, el = enu_to_azel(e)
        return {
            "azimuth_deg": math.degrees(float(az)),
            "elevation_deg": math.degrees(float(el)),
            "distance_m": float(self.sun.earth_distance_m or 0.0),
            "earthshine_w_m2": self.earthshine_w_m2,
        }

    def metadata(self) -> dict[str, object]:
        """JSON-serialisable summary."""
        hm = self.heightmap
        xmin, xmax, ymin, ymax = hm.extent
        meta: dict[str, object] = {
            "name": self.config.name,
            "seed": self.seed,
            "config_hash": self.config_hash,
            "grid_shape": list(hm.shape),
            "resolution_m": hm.resolution,
            "extent_m": {"x_min": xmin, "x_max": xmax, "y_min": ymin, "y_max": ymax},
            "elevation_m": {"min": float(hm.z.min()), "max": float(hm.z.max())},
            "num_craters": len(self.terrain.craters),
            "num_rocks": len(self.terrain.rocks),
            "sun": {
                "azimuth_deg": self.sun.azimuth_deg,
                "elevation_deg": self.sun.elevation_deg,
                "irradiance_w_m2": self.sun.irradiance,
            },
            "raycast_backend": self.caster.backend,
            "ephemeris": self.ephemeris if self.config.sun.mode == "astronomical" else "manual",
            "time_s": self.time_s,
            "rover": {
                "x_m": self.rover.x,
                "y_m": self.rover.y,
                "yaw_deg": math.degrees(self.rover.yaw),
                "roll_deg": math.degrees(self.rover.contact.roll),
                "pitch_deg": math.degrees(self.rover.contact.pitch),
                "z_m": float(self.rover.pose.t[2]),
                "warnings": list(self.rover.status.warnings),
            },
        }
        far = self.terrain_far
        if far is not None:
            fx = far.heightmap.extent
            meta["far_layer"] = {
                "grid_shape": list(far.heightmap.shape),
                "resolution_m": far.heightmap.resolution,
                "extent_m": {"x_min": fx[0], "x_max": fx[1], "y_min": fx[2], "y_max": fx[3]},
                "num_large_craters": len(far.craters),
            }
        return meta

    def __repr__(self) -> str:
        return f"World(name={self.config.name!r}, seed={self.seed}, shape={self.heightmap.shape})"
