"""Command-line entry point: ``lunarsim <command> ...``."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

__all__ = ["main"]


def _cmd_gen_world(args: argparse.Namespace) -> int:
    from lunarsim.core.world import World
    from lunarsim.io.raster import save_npy, write_png16

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    world = World.from_yaml(args.config, seed=args.seed)
    dt = time.perf_counter() - t0
    hm = world.heightmap
    save_npy(out / "heightmap.npy", hm.z)
    write_png16(out / "heightmap.png", hm.z, extra_meta={"resolution_m": hm.resolution})
    meta = world.metadata() | {"generation_time_s": dt}
    (out / "world.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
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
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
