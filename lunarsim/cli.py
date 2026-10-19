"""Command-line entry point: ``lunarsim <command> ...``."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

__all__ = ["main"]


def _cmd_gen_world(args: argparse.Namespace) -> int:
    from lunarsim.core.world import World
    from lunarsim.io.groundtruth import write_catalog, write_label_png
    from lunarsim.io.raster import save_npy, write_png16

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    world = World.from_yaml(args.config, seed=args.seed)
    dt = time.perf_counter() - t0
    hm = world.heightmap
    save_npy(out / "heightmap.npy", hm.z)
    write_png16(out / "heightmap.png", hm.z, extra_meta={"resolution_m": hm.resolution})
    write_label_png(out / "labels.png", world.terrain.labels)
    write_catalog(out / "craters", world.terrain.craters.to_records())
    write_catalog(out / "rocks", world.terrain.rocks.to_records())
    far = world.terrain_far
    if far is not None:
        save_npy(out / "far_heightmap.npy", far.heightmap.z)
        write_png16(
            out / "far_heightmap.png", far.heightmap.z,
            extra_meta={"resolution_m": far.heightmap.resolution},
        )  # fmt: skip
        write_label_png(out / "far_labels.png", far.labels)
        write_catalog(out / "far_craters", far.craters.to_records())
    meta = world.metadata() | {"generation_time_s": dt}
    (out / "world.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn

        from lunarsim.api import ServerSettings, create_app
    except ImportError as exc:  # pragma: no cover
        print(f"server dependencies missing ({exc}); install the 'server' extra")
        return 2
    settings = ServerSettings.from_env(
        host=args.host, port=args.port, token=args.token, allow_insecure=args.allow_insecure,
        max_worlds=args.max_worlds, preload_worlds=args.world or None,
        viewer_cdn=False if args.no_cdn else None,
    )  # fmt: skip
    try:
        settings.check_exposure()
    except ValueError as exc:
        print(f"error: {exc}")
        return 2
    print(f"LunarSim viewer: http://{settings.host}:{settings.port}/viewer/"
          + ("#token=<your token>" if settings.token else ""))  # fmt: skip
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port, log_level="info")
    return 0


def _cmd_psr(args: argparse.Namespace) -> int:
    from datetime import timedelta

    import numpy as np

    from lunarsim.core.terrain import Heightmap
    from lunarsim.core.world import World, parse_utc
    from lunarsim.io.preview import hillshade
    from lunarsim.io.raster import save_npy, write_png8

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    world = World.from_yaml(args.config)
    t0 = parse_utc(args.start)
    t1 = t0 + timedelta(days=args.days)
    tic = time.perf_counter()
    m = world.long_term_illumination(
        t0, t1, timedelta(hours=args.step_hours), args.layer, args.n_azimuth, args.stride
    )
    dt = time.perf_counter() - tic
    save_npy(out / "mean_sun_fraction.npy", m.mean_sun_fraction)
    save_npy(out / "max_sun_fraction.npy", m.max_sun_fraction)
    save_npy(out / "earth_visible_fraction.npy", m.earth_visible_fraction)
    save_npy(out / "psr.npy", m.psr)
    write_png8(out / "mean_sun_fraction.png", m.mean_sun_fraction)
    write_png8(out / "earth_visible_fraction.png", np.nan_to_num(m.earth_visible_fraction))
    res = float(m.x[1] - m.x[0]) if m.x.size > 1 else 1.0
    far = world.terrain_far
    terrain = far if (args.layer in ("auto", "far") and far is not None) else world.terrain
    z = terrain.heightmap.z[:: args.stride, :: args.stride]
    shade = hillshade(Heightmap(np.ascontiguousarray(z), res, 0.0, 0.0))
    rgb = np.stack([shade, shade, shade], axis=-1) * 0.8
    rgb[m.psr] = [0.1, 0.3, 1.0]
    write_png8(out / "psr_overlay.png", rgb)
    summary = {
        "config": str(args.config),
        "site": world.config.site.model_dump(),
        "ephemeris": world.ephemeris,
        "start_utc": t0.isoformat(),
        "end_utc": t1.isoformat(),
        "step_hours": args.step_hours,
        "samples": m.n_samples,
        "grid_shape": list(m.mean_sun_fraction.shape),
        "grid_spacing_m": res,
        "psr_area_fraction": float(m.psr.mean()),
        "mean_illumination": float(m.mean_sun_fraction.mean()),
        "max_mean_illumination": float(m.mean_sun_fraction.max()),
        "compute_time_s": dt,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


def _cmd_dataset(args: argparse.Namespace) -> int:
    from lunarsim.datasets import generate_dataset, load_dataset_config

    cfg, scenario = load_dataset_config(args.config)
    t0 = time.perf_counter()
    out = generate_dataset(cfg, scenario, args.out)
    print(f"dataset written to {out} in {time.perf_counter() - t0:.1f} s")
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI main."""
    parser = argparse.ArgumentParser(prog="lunarsim")
    sub = parser.add_subparsers(dest="command", required=True)
    g = sub.add_parser("gen-world", help="generate terrain from a scenario YAML")
    g.add_argument("config", help="scenario YAML")
    g.add_argument("--seed", type=int, default=None, help="override seed")
    g.add_argument("--out", default="out/world", help="output directory")
    g.set_defaults(func=_cmd_gen_world)
    ps = sub.add_parser("psr", help="long-term illumination / PSR map of a scenario")
    ps.add_argument("config", help="scenario YAML (site latitude/longitude are used)")
    ps.add_argument("--start", default="2026-01-01T00:00:00Z")
    ps.add_argument("--days", type=float, default=346.62, help="span (default: draconic year)")
    ps.add_argument("--step-hours", type=float, default=2.0)
    ps.add_argument("--layer", choices=["auto", "near", "far"], default="auto")
    ps.add_argument("--n-azimuth", type=int, default=64)
    ps.add_argument("--stride", type=int, default=1, help="use every N-th grid node")
    ps.add_argument("--out", default="out/psr")
    ps.set_defaults(func=_cmd_psr)
    ds = sub.add_parser("dataset", help="generate a dataset from a dataset YAML")
    ds.add_argument("config", help="dataset YAML (see configs/dataset_*.yaml)")
    ds.add_argument("--out", default=None, help="override output_dir")
    ds.set_defaults(func=_cmd_dataset)
    sv = sub.add_parser("serve", help="run the REST/WebSocket server")
    sv.add_argument("--host", default=None, help="bind address (default 127.0.0.1)")
    sv.add_argument("--port", type=int, default=None)
    sv.add_argument("--token", default=None, help="API token (or env LUNARSIM_TOKEN)")
    sv.add_argument("--allow-insecure", action="store_true",
                    help="allow a non-loopback bind without a token")  # fmt: skip
    sv.add_argument("--max-worlds", type=int, default=None)
    sv.add_argument("--world", action="append", default=[], metavar="YAML",
                    help="create a world from this scenario at start-up (repeatable)")  # fmt: skip
    sv.add_argument("--no-cdn", action="store_true",
                    help="viewer uses only the vendored Three.js (no CDN requests)")  # fmt: skip
    sv.set_defaults(func=_cmd_serve)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
