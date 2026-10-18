from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from lunarsim.core.frames import Pose
from lunarsim.datasets import DatasetConfig, generate_dataset, load_dataset_config

SCENARIO = {
    "name": "ds_test",
    "seed": 7,
    "terrain": {
        "size_x_m": 40,
        "size_y_m": 40,
        "resolution_m": 0.25,
        "craters": {"enabled": True, "sfd": {"d_min_m": 0.6, "d_max_m": 6.0}},
    },
    "sun": {"azimuth_deg": 100, "elevation_deg": 10},
    "cameras": [
        {"name": "nav", "intrinsics": {"width": 48, "height": 32}, "stereo_baseline_m": 0.3}
    ],
    "lidars": [
        {"name": "spad", "pattern": {"kind": "flash", "flash_width": 20, "flash_height": 15}}
    ],
}
JOB = {
    "scenario": "scenario.yaml",
    "duration_s": 3.0,
    "frame_rate_hz": 2.0,
    "trajectory": {"waypoints": [[1.0, 0.0], [1.0, 1.0]], "speed_m_s": 0.2},
    "cameras": ["nav"],
    "camera_buffers": ["depth", "labels", "shadow"],
    "lidars": ["spad"],
    "lidar_formats": ["bin", "ply"],
    "variations": [
        {"name": "el05", "azimuth_deg": 100, "elevation_deg": 5},
        {"name": "el30", "azimuth_deg": 100, "elevation_deg": 30},
    ],
}


@pytest.fixture(scope="module")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    d = tmp_path_factory.mktemp("ds")
    (d / "scenario.yaml").write_text(yaml.safe_dump(SCENARIO))
    (d / "job.yaml").write_text(yaml.safe_dump(JOB))
    cfg, scen = load_dataset_config(d / "job.yaml")
    return generate_dataset(cfg, scen, d / "out")


def _kitti(path: Path) -> np.ndarray:
    return np.loadtxt(path).reshape(-1, 3, 4)


def test_layout_and_counts(dataset: Path) -> None:
    n = 7  # 3 s at 2 Hz, inclusive
    for v in ("el05", "el30"):
        r = dataset / v
        assert len((r / "times.txt").read_text().split()) == n
        for sub in ("left", "right", "depth", "labels", "shadow"):
            assert len(list((r / "cameras/nav" / sub).iterdir())) == n
        assert len(list((r / "lidars/spad").glob("*.bin"))) == n
        assert len(list((r / "lidars/spad").glob("*.ply"))) == n
        assert (r / "calib/calib.yaml").exists() and (r / "rover_state.csv").exists()
    for f in ("dataset.yaml", "scenario.yaml", "summary.json", "terrain/near_heightmap.npy"):
        assert (dataset / f).exists()
    img = np.asarray(Image.open(dataset / "el05/cameras/nav/left/000000.png"))
    assert img.shape == (32, 48)
    assert np.load(dataset / "el05/cameras/nav/depth/000003.npy").dtype == np.float32
    pts = np.fromfile(dataset / "el05/lidars/spad/000000.bin", dtype="<f4").reshape(-1, 4)
    assert pts.shape[0] > 0


def test_kitti_and_tum_poses(dataset: Path) -> None:
    P = _kitti(dataset / "el05/poses/nav_kitti.txt")
    np.testing.assert_allclose(P[0], np.hstack([np.eye(3), np.zeros((3, 1))]), atol=1e-12)
    assert np.linalg.norm(P[-1][:, 3]) > 0.5  # the camera moved
    for R in P[:, :, :3]:
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-9)
    tum = np.loadtxt(dataset / "el05/poses/nav_tum.txt")
    assert tum.shape == (7, 8)
    np.testing.assert_allclose(np.linalg.norm(tum[:, 4:], axis=1), 1.0, atol=1e-9)

    # KITTI relative poses are consistent with the TUM world poses.
    def pose(row: np.ndarray) -> Pose:
        return Pose.from_quat([row[7], row[4], row[5], row[6]], row[1:4])

    rel = (pose(tum[0]).inverse() @ pose(tum[-1])).matrix()[:3, :]
    np.testing.assert_allclose(rel, P[-1], atol=1e-6)


def test_same_path_different_sun(dataset: Path) -> None:
    a = (dataset / "el05/poses/body_tum.txt").read_text()
    b = (dataset / "el30/poses/body_tum.txt").read_text()
    assert a == b
    ia = np.asarray(Image.open(dataset / "el05/cameras/nav/left/000002.png"))
    ib = np.asarray(Image.open(dataset / "el30/cameras/nav/left/000002.png"))
    assert not np.array_equal(ia, ib)


def test_calibration(dataset: Path) -> None:
    lines = dict(
        ln.split(": ") for ln in (dataset / "el05/calib/nav_kitti.txt").read_text().splitlines()
    )
    P0 = np.array(lines["P0"].split(), float).reshape(3, 4)
    P1 = np.array(lines["P1"].split(), float).reshape(3, 4)
    assert P1[0, 3] == pytest.approx(-P0[0, 0] * 0.3)
    Tr = np.array(lines["Tr"].split(), float).reshape(3, 4)
    np.testing.assert_allclose(Tr[:, :3] @ Tr[:, :3].T, np.eye(3), atol=1e-12)
    calib = yaml.safe_load((dataset / "el05/calib/calib.yaml").read_text())
    assert calib["cameras"]["nav"]["stereo_baseline_m"] == 0.3
    assert "T_body_lidar" in calib["lidars"]["spad"]


def test_reproducible(dataset: Path, tmp_path: Path) -> None:
    (tmp_path / "scenario.yaml").write_text(yaml.safe_dump(SCENARIO))
    job = {**JOB, "variations": [JOB["variations"][0]], "lidar_formats": ["bin"]}
    (tmp_path / "job.yaml").write_text(yaml.safe_dump(job))
    cfg, scen = load_dataset_config(tmp_path / "job.yaml")
    out = generate_dataset(cfg, scen, tmp_path / "out")
    for rel in (
        "el05/cameras/nav/left/000004.png",
        "el05/lidars/spad/000004.bin",
        "el05/poses/nav_kitti.txt",
    ):
        assert (out / rel).read_bytes() == (dataset / rel).read_bytes()


def test_schema_validation() -> None:
    with pytest.raises(ValueError):
        DatasetConfig.model_validate({**JOB, "trajectory": {}})
    with pytest.raises(ValueError):
        DatasetConfig.model_validate({**JOB, "variations": [{"name": "a"}, {"name": "a"}]})
    with pytest.raises(ValueError):
        DatasetConfig.model_validate({**JOB, "variations": [{"name": "a", "azimuth_deg": 3}]})
