"""Point cloud writers/readers: binary little-endian PLY and binary PCD (v0.7)."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

__all__ = ["pcd_bytes", "ply_bytes", "read_pcd", "read_ply", "write_pcd", "write_ply"]

_PLY_TYPES: dict[np.dtype[Any], str] = {
    np.dtype("float32"): "float",
    np.dtype("float64"): "double",
    np.dtype("uint8"): "uchar",
    np.dtype("uint16"): "ushort",
    np.dtype("int32"): "int",
    np.dtype("uint32"): "uint",
}
_PLY_BACK: dict[str, np.dtype[Any]] = {v: k for k, v in _PLY_TYPES.items()}
_PCD_TYPES = {"f": "F", "u": "U", "i": "I"}


def _structured(fields: Mapping[str, npt.ArrayLike]) -> npt.NDArray[np.void]:
    arrays = {k: np.asarray(v) for k, v in fields.items()}
    n = {a.shape[0] for a in arrays.values()}
    if len(n) != 1:
        raise ValueError("all fields must have the same length")
    dt = np.dtype([(k, a.dtype.newbyteorder("<")) for k, a in arrays.items()])
    out = np.empty(n.pop(), dtype=dt)
    for k, a in arrays.items():
        out[k] = a
    return out


def ply_bytes(fields: Mapping[str, npt.ArrayLike]) -> bytes:
    """Binary little-endian PLY (one vertex element with the given fields)."""
    data = _structured(fields)
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {data.shape[0]}"]
    for name in data.dtype.names or ():
        header.append(f"property {_PLY_TYPES[np.dtype(data.dtype[name]).newbyteorder('=')]} {name}")
    header.append("end_header")
    return ("\n".join(header) + "\n").encode("ascii") + data.tobytes()


def write_ply(path: str | Path, fields: Mapping[str, npt.ArrayLike]) -> None:
    """Write a binary little-endian PLY with one vertex element of the given fields."""
    Path(path).write_bytes(ply_bytes(fields))


def read_ply(path: str | Path) -> npt.NDArray[np.void]:
    """Read a PLY written by :func:`write_ply`."""
    raw = Path(path).read_bytes()
    end = raw.index(b"end_header\n") + len(b"end_header\n")
    lines = raw[:end].decode("ascii").splitlines()
    n = 0
    props = []
    for ln in lines:
        p = ln.split()
        if p[:2] == ["element", "vertex"]:
            n = int(p[2])
        elif p[0] == "property":
            props.append((p[2], _PLY_BACK[p[1]].newbyteorder("<")))
    return np.frombuffer(raw[end:], dtype=np.dtype(props), count=n)


def pcd_bytes(fields: Mapping[str, npt.ArrayLike]) -> bytes:
    """Binary PCD v0.7 (unorganised)."""
    data = _structured(fields)
    names = list(data.dtype.names or ())
    sizes = [str(data.dtype[k].itemsize) for k in names]
    types = [_PCD_TYPES[data.dtype[k].kind] for k in names]
    n = data.shape[0]
    header = [
        "# .PCD v0.7 - Point Cloud Data file format",
        "VERSION 0.7",
        "FIELDS " + " ".join(names),
        "SIZE " + " ".join(sizes),
        "TYPE " + " ".join(types),
        "COUNT " + " ".join("1" for _ in names),
        f"WIDTH {n}",
        "HEIGHT 1",
        "VIEWPOINT 0 0 0 1 0 0 0",
        f"POINTS {n}",
        "DATA binary",
    ]
    return ("\n".join(header) + "\n").encode("ascii") + data.tobytes()


def write_pcd(path: str | Path, fields: Mapping[str, npt.ArrayLike]) -> None:
    """Write a binary PCD v0.7 file (unorganised)."""
    Path(path).write_bytes(pcd_bytes(fields))


def read_pcd(path: str | Path) -> npt.NDArray[np.void]:
    """Read a binary PCD written by :func:`write_pcd`."""
    raw = Path(path).read_bytes()
    end = raw.index(b"DATA binary\n") + len(b"DATA binary\n")
    meta: dict[str, list[str]] = {}
    for ln in raw[:end].decode("ascii").splitlines():
        if ln and not ln.startswith("#"):
            k, *v = ln.split()
            meta[k] = v
    back = {"F": "f", "U": "u", "I": "i"}
    dt = np.dtype(
        [
            (name, f"<{back[t]}{s}")
            for name, s, t in zip(meta["FIELDS"], meta["SIZE"], meta["TYPE"], strict=True)
        ]
    )
    return np.frombuffer(raw[end:], dtype=dt, count=int(meta["POINTS"][0]))
