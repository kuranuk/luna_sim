"""World: the single container of simulation state.

A :class:`World` is fully determined by its validated :class:`WorldConfig` (which
includes the seed) plus the explicit state changes applied to it (sun updates,
near-patch re-centring; rover commands in later phases).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from lunarsim.config.load import load_config
from lunarsim.config.schema import CameraConfig, LidarConfig, WorldConfig, config_hash
from lunarsim.core.frames import Pose
from lunarsim.core.illumination.sun import SunState, sun_astronomical, sun_manual
from lunarsim.core.raycast.heightfield import HeightfieldCaster
from lunarsim.core.raycast.layered import LayeredCaster
from lunarsim.core.render import RenderBuffers, render_rays
from lunarsim.core.rng import RngFactory
from lunarsim.core.sensors.camera import CameraModel
from lunarsim.core.sensors.camera_noise import SensorImage, expose
from lunarsim.core.sensors.lidar import LidarScan, SensorMotion, render_lidar
from lunarsim.core.terrain import Heightmap, Terrain, build_terrain
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
        if config.terrain.near.enabled:
            self._two_layer = TwoLayerTerrain(config.terrain, self.rngs, workers=workers)
        else:
            self._single = build_terrain(config.terrain, self.rngs, workers=workers)
        self._build_caster()
        self.sun: SunState = self._sun_from_config()
        self.cameras: dict[str, tuple[CameraConfig, CameraModel]] = {
            c.name: (c, CameraModel.from_config(c.intrinsics)) for c in config.cameras
        }
        self.lidars: dict[str, LidarConfig] = {s.name: s for s in config.lidars}

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
            return sun_manual(s.azimuth_deg, s.elevation_deg, s.distance_au, s.irradiance_1au_w_m2)
        site = self.config.site
        return sun_astronomical(
            site.latitude_deg, site.longitude_deg, parse_utc(s.utc), s.irradiance_1au_w_m2
        )

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

    def height_at(self, x: float, y: float) -> float:
        """Terrain elevation [m] on the rendered (triangulated) surface, finest layer first."""
        for layer in self.layers:
            if bool(layer.heightmap.contains(x, y)):
                return float(layer.heightmap.height_at_mesh(x, y))
        raise ValueError(f"({x}, {y}) is outside the terrain")

    # ------------------------------------------------------------------ sun
    def set_sun_manual(self, azimuth_deg: float, elevation_deg: float) -> SunState:
        """Set the sun direction directly."""
        s = self.config.sun
        self.sun = sun_manual(azimuth_deg, elevation_deg, s.distance_au, s.irradiance_1au_w_m2)
        return self.sun

    def set_sun_utc(self, utc: datetime | str) -> SunState:
        """Set the sun from the site location and a UTC time (astronomical mode)."""
        t = parse_utc(utc) if isinstance(utc, str) else utc
        site = self.config.site
        self.sun = sun_astronomical(
            site.latitude_deg, site.longitude_deg, t, self.config.sun.irradiance_1au_w_m2
        )
        return self.sun

    # ------------------------------------------------------------------ sensors
    def render_camera(
        self,
        name: str,
        T_world_camera: Pose,
        frame_index: int = 0,
        exposure_s: float | None = None,
    ) -> CameraFrame:
        """Render camera ``name`` at pose ``T_world_camera``; noise from ``frame_index``."""
        cfg, model = self.cameras[name]
        buffers = render_rays(
            self.layers,
            self.caster,
            self.sun,
            self.config.surface,
            model,
            T_world_camera,
            self.config.render.shadow_bias_m,
        )
        rng = self.rngs.generator(f"sensor/camera/{name}/frame/{frame_index:06d}")
        image = expose(buffers.radiance, cfg.sensor, rng, exposure_s)
        return CameraFrame(name, frame_index, T_world_camera, buffers, image)

    def render_lidar(
        self,
        name: str,
        T_world_sensor: Pose,
        frame_index: int = 0,
        motion: SensorMotion | None = None,
    ) -> LidarScan:
        """Simulate LiDAR ``name`` at pose ``T_world_sensor`` (FLU sensor frame)."""
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
