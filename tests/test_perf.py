"""Performance benchmarks (run with LUNARSIM_PERF=1)."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from lunarsim.core.world import World

ROOT = Path(__file__).parents[1]


@pytest.mark.perf
def test_terrain_1km_quarter_metre() -> None:
    t0 = time.perf_counter()
    w = World.from_yaml(ROOT / "configs" / "bench_1km.yaml")
    dt = time.perf_counter() - t0
    md = w.metadata()
    print(
        f"\n1 km^2 @ 0.25 m {w.heightmap.shape}, {md['num_craters']} craters, "
        f"{md['num_rocks']} rocks: {dt:.2f} s"
    )
    assert dt < 60.0


@pytest.mark.perf
@pytest.mark.parametrize("backend", ["numba", "numpy"])
def test_camera_640x480_with_shadows(backend: str) -> None:
    import math

    from lunarsim.config import load_config
    from lunarsim.core.raycast import HAS_NUMBA
    from lunarsim.core.sensors.camera import camera_pose_from_body

    if backend == "numba" and not HAS_NUMBA:
        pytest.skip("numba not installed")
    cfg = load_config(ROOT / "configs" / "south_pole_lowsun_stereo.yaml")
    cfg = cfg.model_copy(update={"render": cfg.render.model_copy(update={"backend": backend})})
    w = World(cfg)
    for light in w.lights:
        w.set_light(light, False)
    pose = camera_pose_from_body(
        [0, 0, w.height_at(0, 0) + 1.5], pitch=math.radians(8), yaw=math.radians(90 - 139)
    )
    w.render_camera("navcam", pose)  # warm-up / JIT
    t0 = time.perf_counter()
    w.render_camera("navcam", pose)
    dt = time.perf_counter() - t0
    print(f"\n640x480 camera + shadows, two-layer south pole, {backend}: {dt:.2f} s")
    if backend == "numba":
        assert dt < 1.0


@pytest.mark.perf
@pytest.mark.parametrize("which", ["spad_flash", "spin32"])
def test_lidar_scan(which: str) -> None:
    import math

    from lunarsim.config import load_config
    from lunarsim.core.frames import Pose

    cfg = load_config(ROOT / "configs" / "south_pole_lowsun_stereo.yaml")
    w = World(cfg)
    pose = Pose.from_euler_zyx(0.0, math.radians(8), 0.0, (0, 0, w.height_at(0, 0) + 1.5))
    w.render_lidar(which, pose)  # warm-up
    t0 = time.perf_counter()
    s = w.render_lidar(which, pose)
    dt = time.perf_counter() - t0
    print(f"\nLiDAR {which}: {len(s)} points, {dt:.3f} s")
    assert dt < 0.2
