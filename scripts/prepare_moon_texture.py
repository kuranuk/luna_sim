"""Prepare the whole-Moon texture for the browser simulator's globe view.

Input: an equirectangular lunar colour map, longitude -180..180 left to right, north up, e.g. the
NASA SVS "CGI Moon Kit" (https://svs.gsfc.nasa.gov/4720, public domain, credit NASA's Scientific
Visualization Studio): lroc_color_poles_1k.jpg / _2k.tif / _4k.tif (16-bit TIFFs are supported).
Output: web/assets/moon/color.jpg (8-bit sRGB JPEG), loaded automatically by web/js/globe.js.

    python scripts/prepare_moon_texture.py lroc_color_poles_4k.tif --width 4096
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "web" / "assets" / "moon" / "color.jpg"


def load_rgb8(path: Path) -> Image.Image:
    """Read a colour map as 8-bit RGB (16-bit TIFFs scaled by 1/257, via tifffile if needed)."""
    if path.suffix.lower() in (".tif", ".tiff"):
        try:
            import tifffile
        except ImportError:  # pragma: no cover - optional dependency (pip install lunarsim[dem])
            img = Image.open(path)
        else:
            a = np.asarray(tifffile.imread(path))
            if a.dtype == np.uint16:
                a = (a.astype(np.float64) / 257.0).round().astype(np.uint8)
            if a.ndim == 2:
                a = np.repeat(a[:, :, None], 3, axis=2)
            return Image.fromarray(a[:, :, :3].astype(np.uint8), "RGB")
    else:
        img = Image.open(path)
    return img.convert("RGB")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("source", type=Path, help="equirectangular colour map (jpg / png / tif)")
    ap.add_argument(
        "--width", type=int, default=2048, help="output width [px] (height = width / 2)"
    )
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    img = load_rgb8(args.source)
    w, h = img.size
    if abs(w - 2 * h) > 1:
        raise SystemExit(f"expected an equirectangular 2:1 image, got {w}x{h}")
    img = img.resize((args.width, args.width // 2), Image.LANCZOS)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    img.save(args.out, quality=90)
    print(f"wrote {args.out} ({args.width}x{args.width // 2})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
