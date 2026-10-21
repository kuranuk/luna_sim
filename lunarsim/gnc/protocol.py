"""Binary message format shared with the browser simulator (``web/js/gnc/protocol.js``).

Layout (little-endian)::

    b"LSG1" | uint32 header_length | header (UTF-8 JSON) | zero padding to 8 bytes | blobs

The header's ``blobs`` list describes each array: ``{"name", "dtype", "shape", "offset"}`` with
``offset`` counted from the start of the blob section (each blob 8-byte aligned). dtypes are numpy
type strings without byte order (``f4``, ``f8``, ``u1``, ``u2``, ``i4``, ``u4``). JSON control
messages (``hello``, ``ready``, ``cmd``, ``error``) are sent as WebSocket text frames.
"""

from __future__ import annotations

import json
import struct
from typing import Any

import numpy as np
import numpy.typing as npt

MAGIC = b"LSG1"
PROTOCOL_VERSION = 1
_DTYPES = {"f4", "f8", "u1", "u2", "i4", "u4"}


def _align8(n: int) -> int:
    return (n + 7) & ~7


def encode(header: dict[str, Any], blobs: dict[str, npt.NDArray[Any]]) -> bytes:
    """Pack a JSON header and named arrays into one binary message."""
    descs, parts, off = [], [], 0
    for name, arr in blobs.items():
        a = np.ascontiguousarray(arr)
        dt = a.dtype.newbyteorder("<").str[1:] if a.dtype.itemsize > 1 else a.dtype.str[1:]
        if dt not in _DTYPES:
            raise ValueError(f"unsupported dtype {a.dtype} for blob {name!r}")
        raw = a.astype(a.dtype.newbyteorder("<"), copy=False).tobytes()
        descs.append({"name": name, "dtype": dt, "shape": list(a.shape), "offset": off})
        parts.append(raw + b"\0" * (_align8(len(raw)) - len(raw)))
        off += _align8(len(raw))
    head = json.dumps({**header, "blobs": descs}, separators=(",", ":")).encode()
    pre = MAGIC + struct.pack("<I", len(head)) + head
    return pre + b"\0" * (_align8(len(pre)) - len(pre)) + b"".join(parts)


def decode(data: bytes) -> tuple[dict[str, Any], dict[str, npt.NDArray[Any]]]:
    """Unpack a binary message into (header, {name: array}). Arrays are read-only views."""
    if len(data) < 8 or data[:4] != MAGIC:
        raise ValueError("not a LunarSim GNC message (bad magic)")
    (hlen,) = struct.unpack_from("<I", data, 4)
    if 8 + hlen > len(data):
        raise ValueError("truncated header")
    header = json.loads(data[8 : 8 + hlen].decode())
    base = _align8(8 + hlen)
    blobs: dict[str, npt.NDArray[Any]] = {}
    for d in header.pop("blobs", []):
        if d["dtype"] not in _DTYPES:
            raise ValueError(f"unsupported dtype {d['dtype']!r}")
        dt = np.dtype("<" + d["dtype"])
        shape = tuple(int(s) for s in d["shape"])
        count = int(np.prod(shape, dtype=np.int64)) if shape else 1
        start = base + int(d["offset"])
        if start < base or start + count * dt.itemsize > len(data):
            raise ValueError(f"blob {d['name']!r} out of bounds")
        blobs[d["name"]] = np.frombuffer(data, dtype=dt, count=count, offset=start).reshape(shape)
    return header, blobs
