"""Phase 2 demo: craters + rocks -> images, labels and ground truth in out/phase2/.

Usage: python scripts/demo_phase2.py [--config configs/crater_field.yaml] [--seed N]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from lunarsim.core.terrain.craters import sfd_cdf
from lunarsim.core.terrain.rocks import cumulative_area_fraction
from lunarsim.core.world import World
from lunarsim.io.groundtruth import write_catalog, write_label_png
from lunarsim.io.preview import hillshade, normalize
from lunarsim.io.raster import save_npy, write_png8, write_png16

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "crater_field.yaml"))
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default=str(ROOT / "out" / "phase2"))
    args = ap.parse_args()

    t0 = time.perf_counter()
    world = World.from_yaml(args.config, seed=args.seed)
    dt = time.perf_counter() - t0
    terrain, hm, cfg = world.terrain, world.heightmap, world.config.terrain
    out = Path(args.out) / world.config.name
    out.mkdir(parents=True, exist_ok=True)

    save_npy(out / "heightmap.npy", hm.z)
    save_npy(out / "labels.npy", terrain.labels)
    write_png16(out / "heightmap16.png", hm.z, extra_meta={"resolution_m": hm.resolution})
    write_png8(out / "hillshade_low_sun.png", hillshade(hm, azimuth_deg=90.0, elevation_deg=15.0))
    write_png8(out / "slope_0-30deg.png", normalize(np.rad2deg(hm.slope()), 0.0, 30.0))
    write_label_png(out / "labels.png", terrain.labels)
    write_label_png(out / "labels_color.png", terrain.labels, color=True)
    write_catalog(out / "craters", terrain.craters.to_records())
    write_catalog(out / "rocks", terrain.rocks.to_records())

    # Statistical sanity checks on the ground truth.
    cat, sfd = terrain.craters, cfg.craters.sfd
    xmin, xmax, ymin, ymax = hm.extent
    on_grid = (cat.x >= xmin) & (cat.x <= xmax) & (cat.y >= ymin) & (cat.y <= ymax)
    area = (xmax - xmin) * (ymax - ymin)
    sfd_check = {}
    for d in (1, 2, 5, 10, 20, 50):
        if d < sfd.d_min_m or d > sfd.d_max_m:
            continue
        expected = area * sfd.coefficient * (d**-sfd.exponent - sfd.d_max_m**-sfd.exponent)
        sfd_check[f"N(>{d}m)"] = {
            "observed": int(np.sum(cat.diameter[on_grid] > d)),
            "expected": round(float(expected), 1),
        }
    rocks, rc = terrain.rocks, cfg.rocks
    bg = rocks.source_crater < 0
    rock_check = {}
    for d in (0.5, 1.0, 2.0):
        if d < rc.d_min_m:
            continue
        sel = bg & (rocks.diameter >= d)
        rock_check[f"F(>={d}m)"] = {
            "observed": float(np.sum(0.25 * np.pi * rocks.diameter[sel] ** 2) / area),
            "model": float(
                cumulative_area_fraction(d, rc.k) - cumulative_area_fraction(rc.d_max_m, rc.k)
            ),
        }
    label_frac = np.bincount(terrain.labels.ravel(), minlength=5) / terrain.labels.size
    stats = world.metadata() | {
        "generation_time_s": round(dt, 3),
        "crater_sfd_on_grid": sfd_check,
        "ks_cdf_at_2m": float(sfd_cdf(2.0, sfd)),
        "rock_area_fraction_background": rock_check,
        "num_ejecta_rocks": int(np.sum(~bg)),
        "label_fraction": {
            "plain": float(label_frac[0]),
            "crater_interior": float(label_frac[1]),
            "crater_rim": float(label_frac[2]),
            "rock": float(label_frac[3]),
        },
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
