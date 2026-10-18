"""Phase 7 demo: dataset export (same path, several suns) and the browser viewer.

1. Generates a short version of ``configs/dataset_south_pole_sun_sweep.yaml`` into
   out/phase7/dataset and writes a contact sheet (one row per sun elevation).
2. If Playwright is installed (optional, not in the lock), starts ``lunarsim serve``
   with a token, opens the viewer headless, drives with the W key and saves screenshots.

Usage: python scripts/demo_phase7.py [--frames 9] [--no-viewer]
"""

from __future__ import annotations

import argparse
import os
import secrets
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

from lunarsim.datasets import generate_dataset, load_dataset_config

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "phase7"


def contact_sheet(ds: Path, frames: list[int]) -> Path:
    rows = []
    for vdir in sorted(p for p in ds.iterdir() if (p / "cameras").is_dir()):
        cam = next((vdir / "cameras").iterdir())
        tiles = []
        for k in frames:
            dn = np.asarray(Image.open(cam / "left" / f"{k:06d}.png"), dtype=np.float64)
            hi = max(np.percentile(dn, 99.5), 1.0)
            tiles.append((np.clip(dn / hi, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8))
        rows.append(np.hstack(tiles))
        print(f"  {vdir.name}: mean DN {np.mean(rows[-1]):.1f}")
    path = OUT / "sun_sweep_contact.png"
    Image.fromarray(np.vstack(rows)).save(path)
    return path


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def viewer_screens(world_yaml: Path) -> None:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("playwright not installed: viewer screenshots skipped")
        return
    port, token = free_port(), secrets.token_urlsafe(12)
    env = dict(os.environ, LUNARSIM_TOKEN=token)
    cmd = [sys.executable, "-m", "lunarsim.cli", "serve", "--port", str(port), "--world",
           str(world_yaml)]  # fmt: skip
    srv = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    try:
        for _ in range(900):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
                break
            except OSError:
                time.sleep(0.1)
        exe = os.environ.get("CHROMIUM", "/opt/pw-browsers/chromium")
        with sync_playwright() as p:
            b = p.chromium.launch(
                executable_path=exe if Path(exe).exists() else None,
                args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
            )
            pg = b.new_page(viewport={"width": 1400, "height": 820})
            pg.goto(f"http://127.0.0.1:{port}/viewer/#token={token}")
            pg.wait_for_function(
                "window.lunarsimViewer && window.lunarsimViewer.state", timeout=90000
            )
            pg.wait_for_timeout(4000)
            pg.screenshot(path=str(OUT / "viewer.png"))
            pg.click("canvas")
            pg.keyboard.down("w")
            pg.wait_for_timeout(3000)
            pg.keyboard.up("w")
            pg.keyboard.press(" ")
            pg.wait_for_timeout(2500)
            st = pg.evaluate("window.lunarsimViewer.state")
            pg.screenshot(path=str(OUT / "viewer_drive.png"))
            b.close()
        print(f"viewer: drove {st['rover']['odometer_m']:.2f} m; screenshots in {OUT}")
    finally:
        srv.terminate()
        srv.wait(10)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=9)
    ap.add_argument("--no-viewer", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cfg, scenario = load_dataset_config(ROOT / "configs" / "dataset_south_pole_sun_sweep.yaml")
    cfg = cfg.model_copy(update={"duration_s": (args.frames - 1) / cfg.frame_rate_hz})
    t0 = time.perf_counter()
    ds = generate_dataset(cfg, scenario, OUT / "dataset")
    n = args.frames * len(cfg.variations)
    dt = time.perf_counter() - t0
    print(f"dataset: {n} frames in {dt:.1f} s ({dt / n:.2f} s/frame) -> {ds}")
    print(f"contact sheet: {contact_sheet(ds, [0, args.frames // 2, args.frames - 1])}")
    if not args.no_viewer:
        viewer_screens(ROOT / "configs" / "south_pole_lowsun.yaml")


if __name__ == "__main__":
    main()
