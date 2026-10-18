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
        max_worlds=args.max_worlds,
    )  # fmt: skip
    try:
        settings.check_exposure()
    except ValueError as exc:
        print(f"error: {exc}")
        return 2
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port, log_level="info")
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
    sv = sub.add_parser("serve", help="run the REST/WebSocket server")
    sv.add_argument("--host", default=None, help="bind address (default 127.0.0.1)")
    sv.add_argument("--port", type=int, default=None)
    sv.add_argument("--token", default=None, help="API token (or env LUNARSIM_TOKEN)")
    sv.add_argument("--allow-insecure", action="store_true",
                    help="allow a non-loopback bind without a token")  # fmt: skip
    sv.add_argument("--max-worlds", type=int, default=None)
    sv.set_defaults(func=_cmd_serve)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
