"""scripts/prepare_moon_texture.py: equirectangular map -> web/assets/moon/color.jpg."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("prep", ROOT / "scripts" / "prepare_moon_texture.py")
assert spec is not None and spec.loader is not None
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)


def test_prepare_from_16bit_tiff_and_jpeg(tmp_path: Path) -> None:
    tifffile = pytest.importorskip("tifffile")
    a = np.zeros((64, 128, 3), dtype=np.uint16)
    a[:, :64] = 65535  # western half white
    tifffile.imwrite(tmp_path / "m.tif", a)
    out = tmp_path / "color.jpg"
    assert prep.main([str(tmp_path / "m.tif"), "--width", "256", "--out", str(out)]) == 0
    img = np.asarray(Image.open(out))
    assert img.shape == (128, 256, 3)
    assert img[:, :100].mean() > 240 and img[:, 160:].mean() < 15
    g = np.tile(np.linspace(40, 90, 64)[None, :], (32, 1)).astype(np.uint8)  # elevation ramp
    Image.fromarray(g).save(tmp_path / "dem.png")
    elev = tmp_path / "elev.jpg"
    assert (
        prep.main(
            [str(tmp_path / "dem.png"), "--kind", "elevation", "--width", "64", "--out", str(elev)]
        )
        == 0
    )
    e = np.asarray(Image.open(elev), dtype=float)
    assert e.ndim == 2 and e[:, :3].mean() < 10 and e[:, -3:].mean() > 245  # stretched, west low
    Image.fromarray(np.full((10, 30, 3), 9, np.uint8)).save(tmp_path / "bad.jpg")
    with pytest.raises(SystemExit):
        prep.main([str(tmp_path / "bad.jpg"), "--out", str(out)])
