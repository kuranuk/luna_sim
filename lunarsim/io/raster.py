"""Raster I/O: float arrays as ``.npy`` and quantised 16-bit PNG with scale/offset.

PNG images are written North-up (row 0 = northern edge), i.e. the heightmap's
South-to-North row order is flipped. Readers flip back.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from PIL import Image
from PIL.PngImagePlugin import PngInfo

__all__ = [
    "load_npy",
    "read_png16",
    "save_npy",
    "write_png8",
    "write_png16",
    "write_uint16_png",
]

_META_KEY = "lunarsim"


def save_npy(path: str | Path, array: npt.ArrayLike) -> None:
    """Save an array as ``.npy`` (no pickled objects)."""
    np.save(Path(path), np.asarray(array), allow_pickle=False)


def load_npy(path: str | Path) -> npt.NDArray[Any]:
    """Load a ``.npy`` file with pickle disabled."""
    return np.asarray(np.load(Path(path), allow_pickle=False))


def write_png16(
    path: str | Path,
    values: npt.NDArray[np.float64],
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    flip_north_up: bool = True,
    extra_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Quantise ``values`` to uint16 and write a PNG plus a JSON sidecar.

    ``value = offset + scale * pixel``. The same metadata is stored in a PNG text
    chunk and in ``<path>.json``. Returns the metadata dict.
    """
    v = np.asarray(values, dtype=np.float64)
    lo = float(np.nanmin(v)) if vmin is None else float(vmin)
    hi = float(np.nanmax(v)) if vmax is None else float(vmax)
    scale = (hi - lo) / 65535.0 if hi > lo else 1.0
    q = np.clip(np.rint((v - lo) / scale), 0, 65535).astype(np.uint16)
    if flip_north_up:
        q = q[::-1]
    meta: dict[str, Any] = {
        "offset": lo,
        "scale": scale,
        "north_up": flip_north_up,
        **(extra_meta or {}),
    }
    info = PngInfo()
    info.add_text(_META_KEY, json.dumps(meta, sort_keys=True))
    p = Path(path)
    Image.fromarray(np.ascontiguousarray(q)).save(p, pnginfo=info)
    p.with_suffix(p.suffix + ".json").write_text(json.dumps(meta, indent=2, sort_keys=True))
    return meta


def read_png16(path: str | Path) -> tuple[npt.NDArray[np.float64], dict[str, Any]]:
    """Read a PNG written by :func:`write_png16`; returns ``(values, meta)``."""
    with Image.open(Path(path)) as im:
        meta = json.loads(im.text[_META_KEY])
        q = np.asarray(im, dtype=np.float64)
    if meta.get("north_up", True):
        q = q[::-1]
    return np.ascontiguousarray(meta["offset"] + meta["scale"] * q), meta


def write_png8(path: str | Path, image: npt.ArrayLike, *, flip_north_up: bool = True) -> None:
    """Write an 8-bit grayscale (H, W) or RGB (H, W, 3) image given in [0, 1]."""
    a = np.clip(np.asarray(image, dtype=np.float64), 0.0, 1.0)
    q = np.rint(a * 255.0).astype(np.uint8)
    if flip_north_up:
        q = q[::-1]
    Image.fromarray(np.ascontiguousarray(q)).save(Path(path))


def write_uint16_png(path: str | Path, image: npt.ArrayLike) -> None:
    """Write integer data (e.g. camera DN) as a 16-bit grayscale PNG, rows as given."""
    a = np.asarray(image)
    if a.min() < 0 or a.max() > 65535:
        raise ValueError("values must fit in uint16")
    Image.fromarray(np.ascontiguousarray(a.astype(np.uint16))).save(Path(path))
