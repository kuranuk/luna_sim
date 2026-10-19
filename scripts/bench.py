"""Benchmark the main operations and write out/bench/bench.{json,md}.

Usage: python scripts/bench.py [--quick]

Measures (best of N after a warm-up, so numba compilation is excluded):
terrain generation (1 km² @ 0.25 m), south-polar world build, 640x480 camera with
hard shadows / penumbra / earthshine / lights, SPAD flash and 32-line LiDAR, horizon
map + one-year illumination (PSR) map, rover stepping with a 200 Hz IMU.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from lunarsim.config import load_config
from lunarsim.config.schema import ImuConfig, WorldConfig
from lunarsim.core.illumination.horizon import horizon_map, illumination_map, sun_track
from lunarsim.core.raycast import HAS_NUMBA
from lunarsim.core.world import World
from lunarsim.version import __version__

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "bench"


def best(fn: Callable[[], Any], repeat: int) -> float:
    fn()  # warm-up (JIT, caches)
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)


def once(fn: Callable[[], Any]) -> tuple[float, Any]:
    t0 = time.perf_counter()
    r = fn()
    return time.perf_counter() - t0, r


def with_render(cfg: WorldConfig, **upd: Any) -> WorldConfig:
    return cfg.model_copy(update={"render": cfg.render.model_copy(update=upd)})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="skip the 1 km terrain and PSR year")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    def add(item: str, seconds: float, target: str = "", note: str = "") -> None:
        rows.append({"item": item, "seconds": round(seconds, 3), "target": target, "note": note})
        print(f"{item:<62s} {seconds:8.3f} s  {note}")

    if not args.quick:
        dt, w = once(lambda: World.from_yaml(ROOT / "configs" / "bench_1km.yaml"))
        md = w.metadata()
        add(
            "terrain 1 km² @ 0.25 m (4001²) + craters + rocks",
            dt,
            "数十秒",
            f"{md['num_craters']} craters, {md['num_rocks']} rocks",
        )
        del w

    base = load_config(ROOT / "configs" / "south_pole_lowsun_stereo.yaml")
    dt, world = once(lambda: World(base))
    add("south-pole world build (2 km/1 m + 100 m/5 cm)", dt)
    for name in world.lights:
        world.set_light(name, False)
    add("camera 640x480, hard shadows", best(lambda: world.render_camera("navcam"), 3), "1 s")

    pen = World(with_render(base, penumbra_samples=16))
    for name in pen.lights:
        pen.set_light(name, False)
    add("camera 640x480, penumbra (16 disc samples)", best(lambda: pen.render_camera("navcam"), 3))
    es_cfg = with_render(base, penumbra_samples=16).model_copy(
        update={"sun": base.sun.model_copy(update={"earthshine": True})}
    )
    es = World(es_cfg)
    add(
        "camera 640x480, penumbra + earthshine + 2 lights",
        best(lambda: es.render_camera("navcam"), 3),
    )
    add("stereo pair 640x480, hard shadows", best(lambda: world.render_stereo("navcam"), 2))
    add("LiDAR SPAD flash 160x120", best(lambda: world.render_lidar("spad_flash"), 3))
    add("LiDAR spinning 32 lines", best(lambda: world.render_lidar("spin32"), 3), "0.2 s")

    far = world.terrain_far
    assert far is not None
    hm = far.heightmap
    stride = 4
    dt, hor = once(lambda: horizon_map(hm, 64, stride))
    add(f"horizon map {hor.z.shape[0]}x{hor.z.shape[1]} nodes x 64 azimuths", dt)
    if not args.quick:
        t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
        site = base.site
        from lunarsim.core.illumination.sun import sun_astronomical

        suns = sun_track(
            lambda t: sun_astronomical(site.latitude_deg, site.longitude_deg, t),
            t0,
            t0 + timedelta(days=346.62),
            timedelta(hours=2),
        )
        dt, _ = once(lambda: illumination_map(hor, suns))
        add(f"illumination/PSR map, draconic year @ 2 h ({len(suns)} samples)", dt)

    imu_cfg = base.model_copy(update={"imus": [ImuConfig(name="imu", rate_hz=200.0)]})
    wi = World(imu_cfg)
    wi.set_rover_command(0.2, 0.05)
    dt_imu = best(lambda: wi.step(10.0), 2)
    w0 = World(base)
    w0.set_rover_command(0.2, 0.05)
    dt_plain = best(lambda: w0.step(10.0), 2)
    add("rover step 10 s (no IMU, 20 Hz sub-steps)", dt_plain)
    add("rover step 10 s with a 200 Hz IMU", dt_imu, note="2000 IMU samples")

    env = {
        "lunarsim": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpus": os.cpu_count(),
        "numba": HAS_NUMBA,
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (OUT / "bench.json").write_text(json.dumps({"env": env, "results": rows}, indent=2))
    md = ["| 項目 | 時間 [s] | 目標 | 備考 |", "|---|---|---|---|"]
    md += [f"| {r['item']} | {r['seconds']:.3f} | {r['target']} | {r['note']} |" for r in rows]
    md.append("")
    md.append(
        f"環境: Python {env['python']}, {env['cpus']} CPU, numba={env['numba']}, {env['platform']}"
    )
    (OUT / "bench.md").write_text("\n".join(md) + "\n")
    print(f"\nwrote {OUT / 'bench.json'} and {OUT / 'bench.md'}")


if __name__ == "__main__":
    main()
