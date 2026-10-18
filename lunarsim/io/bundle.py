"""Sensor-frame serialisation shared by the server and the client.

Bundles are ``.npz`` archives of plain numeric arrays (``np.load(..., allow_pickle=False)``).
"""

from __future__ import annotations

import io
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from lunarsim.io.pointcloud import pcd_bytes, ply_bytes

if TYPE_CHECKING:
    from lunarsim.core.sensors.camera import CameraModel
    from lunarsim.core.sensors.lidar import LidarScan
    from lunarsim.core.world import CameraFrame

__all__ = [
    "CAMERA_BUFFERS",
    "camera_npz",
    "decode_npz",
    "encode_npz",
    "lidar_fields",
    "lidar_json",
    "lidar_npz",
    "lidar_pcd",
    "lidar_ply",
]

CAMERA_BUFFERS = ("radiance", "depth", "range", "normals", "labels", "shadow", "layer", "points")


def encode_npz(arrays: Mapping[str, npt.ArrayLike]) -> bytes:
    """Uncompressed ``.npz`` bytes of numeric arrays."""
    buf = io.BytesIO()
    savez: Any = np.savez  # numpy's stub mistypes **kwargs
    savez(buf, **{k: np.asarray(v) for k, v in arrays.items()})
    return buf.getvalue()


def decode_npz(data: bytes) -> dict[str, npt.NDArray[Any]]:
    """Decode bytes from :func:`encode_npz` (pickle disabled)."""
    with np.load(io.BytesIO(data), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def camera_npz(frame: CameraFrame, model: CameraModel, buffers: Iterable[str] = ()) -> bytes:
    """Image (DN), pose, intrinsics, exposure and selected ground-truth buffers."""
    b = frame.buffers
    src: dict[str, npt.NDArray[Any]] = {
        "radiance": b.radiance.astype(np.float32),
        "depth": b.depth.astype(np.float32),
        "range": b.range_m.astype(np.float32),
        "normals": b.normals.astype(np.float32),
        "labels": b.labels,
        "shadow": b.shadow,
        "layer": b.layer.astype(np.int8),
        "points": b.points.astype(np.float32),
    }
    out: dict[str, npt.ArrayLike] = {
        "image": frame.image.dn,
        "saturated": frame.image.saturated,
        "exposure_s": np.float64(frame.image.exposure_s),
        "T_world_camera": frame.pose.matrix(),
        "K": model.K,
        "distortion": np.asarray(model.dist),
        "frame_index": np.int64(frame.frame_index),
    }
    for name in buffers:
        if name not in src:
            raise ValueError(f"unknown camera buffer {name!r}; choose from {CAMERA_BUFFERS}")
        out[name] = src[name]
    return encode_npz(out)


def lidar_fields(scan: LidarScan) -> dict[str, npt.NDArray[Any]]:
    """Per-point fields (sensor frame) for point-cloud files."""
    return {
        "x": scan.points[:, 0],
        "y": scan.points[:, 1],
        "z": scan.points[:, 2],
        "intensity": scan.points[:, 3],
        "ring": scan.ring,
        "time": scan.time,
        "label": scan.labels,
    }


def lidar_npz(scan: LidarScan) -> bytes:
    """All per-point arrays plus pose and organised range image."""
    out: dict[str, npt.ArrayLike] = {
        "points": scan.points,
        "ring": scan.ring,
        "time": scan.time,
        "range_m": scan.range_m.astype(np.float32),
        "labels": scan.labels,
        "false_alarm": scan.false_alarm,
        "points_world_gt": scan.points_world_gt.astype(np.float32),
        "beam_index": scan.beam_index,
        "T_world_sensor": scan.pose.matrix(),
    }
    if scan.photon_counts is not None:
        out["photon_counts"] = scan.photon_counts.astype(np.int32)
    if scan.range_image is not None:
        out["range_image"] = scan.range_image.astype(np.float32)
    return encode_npz(out)


def lidar_ply(scan: LidarScan) -> bytes:
    """Binary PLY of the scan (sensor frame)."""
    return ply_bytes(lidar_fields(scan))


def lidar_pcd(scan: LidarScan) -> bytes:
    """Binary PCD of the scan (sensor frame)."""
    return pcd_bytes(lidar_fields(scan))


def lidar_json(scan: LidarScan) -> dict[str, Any]:
    """JSON-friendly dict (use only for small scans)."""
    return {
        "points": scan.points.tolist(),
        "ring": scan.ring.tolist(),
        "time": scan.time.tolist(),
        "labels": scan.labels.tolist(),
        "T_world_sensor": scan.pose.matrix().tolist(),
    }
