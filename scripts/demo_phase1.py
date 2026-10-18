"""Phase 1 demo: fractal terrain -> heightmap / hillshade / slope images in out/phase1/.

Usage: python scripts/demo_phase1.py [--config configs/south_pole_slope.yaml] [--seed N]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from lunarsim.core.terrain.fractal import rms_slope_at_baseline
from lunarsim.core.world import World
from lunarsim.io.preview import hillshade, normalize
from lunarsim.io.raster import save_npy, write_png8, write_png16

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        nargs="*",
        default=[
            str(ROOT / "configs" / "mare_plain.yaml"),
            str(ROOT / "configs" / "south_pole_slope.yaml"),
        ],
    )
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default=str(ROOT / "out" / "phase1"))
    args = ap.parse_args()

    for cfg_path in args.config:
        t0 = time.perf_counter()
        world = World.from_yaml(cfg_path, seed=args.seed)
        dt = time.perf_counter() - t0
        hm = world.heightmap
        out = Path(args.out) / world.config.name
        out.mkdir(parents=True, exist_ok=True)

        save_npy(out / "heightmap.npy", hm.z)
        write_png16(out / "heightmap16.png", hm.z, extra_meta={"resolution_m": hm.resolution})
        write_png8(out / "heightmap_preview.png", normalize(hm.z))
        write_png8(out / "hillshade.png", hillshade(hm, azimuth_deg=135.0, elevation_deg=20.0))
        slope_deg = np.rad2deg(hm.slope())
        write_png8(out / "slope_0-30deg.png", normalize(slope_deg, 0.0, 30.0))

        # Sanity: RMS slope vs baseline should follow s(L) ∝ L^(H-1) (below roll-off).
        lags = [1, 2, 4, 8, 16]
        s = [rms_slope_at_baseline(hm.z, hm.resolution, n) for n in lags]
        h_est = 1.0 + np.polyfit(np.log(lags), np.log(s), 1)[0]
        stats = world.metadata() | {
            "generation_time_s": round(dt, 3),
            "slope_deg": {
                "mean": float(slope_deg.mean()),
                "p95": float(np.percentile(slope_deg, 95)),
                "max": float(slope_deg.max()),
            },
            "rms_slope_deg_by_baseline_m": {
                f"{n * hm.resolution:g}": float(np.rad2deg(np.arctan(v)))
                for n, v in zip(lags, s, strict=True)
            },
            "hurst_estimated_from_lags": float(h_est),
        }
        (out / "stats.json").write_text(json.dumps(stats, indent=2))
        print(json.dumps(stats, indent=2))
        print(f"-> {out}")


if __name__ == "__main__":
    main()
