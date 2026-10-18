from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lunarsim.io.raster import load_npy, read_png16, save_npy, write_png8, write_png16


def test_npy_roundtrip(tmp_path: Path) -> None:
    a = np.random.default_rng(0).normal(size=(7, 9))
    save_npy(tmp_path / "a.npy", a)
    np.testing.assert_array_equal(load_npy(tmp_path / "a.npy"), a)


def test_npy_refuses_objects(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        save_npy(tmp_path / "o.npy", np.array([{"a": 1}], dtype=object))


def test_png16_roundtrip(tmp_path: Path) -> None:
    z = np.random.default_rng(1).normal(scale=10.0, size=(33, 47))
    meta = write_png16(tmp_path / "h.png", z, extra_meta={"resolution_m": 0.25})
    z2, meta2 = read_png16(tmp_path / "h.png")
    assert meta2 == meta and meta2["resolution_m"] == 0.25
    assert np.max(np.abs(z2 - z)) <= 0.5 * meta["scale"] + 1e-12
    assert (tmp_path / "h.png.json").exists()


def test_png8(tmp_path: Path) -> None:
    write_png8(tmp_path / "g.png", np.linspace(0, 1, 12).reshape(3, 4))
    write_png8(tmp_path / "c.png", np.zeros((3, 4, 3)))
