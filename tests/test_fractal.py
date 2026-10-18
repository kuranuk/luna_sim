from __future__ import annotations

import numpy as np
import pytest

from lunarsim.config import FractalConfig, TerrainConfig, WorldConfig
from lunarsim.core.rng import RngFactory
from lunarsim.core.terrain import build_terrain
from lunarsim.core.terrain.fractal import rms_slope_at_baseline, synthesize_fractal
from lunarsim.core.world import World


def _radial_psd_slope(z: np.ndarray, res: float) -> float:
    ny, nx = z.shape
    p = np.abs(np.fft.rfft2(z)) ** 2
    k = np.hypot(np.fft.fftfreq(ny, res)[:, None], np.fft.rfftfreq(nx, res)[None, :])
    kmin, kmax = 4.0 / (nx * res), 0.25 / res  # avoid the largest scales and near-Nyquist
    edges = np.geomspace(kmin, kmax, 20)
    idx = np.digitize(k.ravel(), edges)
    pr = p.ravel()
    kc, pc = [], []
    for i in range(1, len(edges)):
        m = idx == i
        if m.any():
            kc.append(k.ravel()[m].mean())
            pc.append(pr[m].mean())
    return float(np.polyfit(np.log(kc), np.log(pc), 1)[0])


@pytest.mark.parametrize("hurst", [0.5, 0.8, 0.95])
def test_psd_exponent(hurst: float) -> None:
    cfg = FractalConfig(hurst=hurst, rms_slope_deg=5.0, pad_factor=1.0)
    slopes = [
        _radial_psd_slope(
            synthesize_fractal((512, 512), 0.25, cfg, RngFactory(s).generator("f"), workers=1),
            0.25,
        )
        for s in range(4)
    ]
    assert np.mean(slopes) == pytest.approx(-(2 * hurst + 2), abs=0.1)


@pytest.mark.parametrize("deg,baseline,res", [(3.0, 1.0, 0.25), (10.0, 2.0, 0.1), (5.0, 0.5, 0.5)])
def test_rms_slope_exact(deg: float, baseline: float, res: float) -> None:
    cfg = FractalConfig(rms_slope_deg=deg, baseline_m=baseline)
    z = synthesize_fractal((300, 400), res, cfg, RngFactory(1).generator("f"))
    lag = round(baseline / res)
    assert rms_slope_at_baseline(z, res, lag) == pytest.approx(np.tan(np.deg2rad(deg)), rel=1e-9)
    assert abs(z.mean()) < 1e-9


def test_disabled_or_zero_is_flat() -> None:
    z = synthesize_fractal(
        (32, 32), 0.25, FractalConfig(enabled=False), RngFactory(0).generator("f")
    )
    assert not z.any()


def test_tilt() -> None:
    cfg = TerrainConfig(
        size_x_m=40,
        size_y_m=40,
        resolution_m=0.5,
        base_elevation_m=100.0,
        fractal={"enabled": False},
        tilt={"slope_deg": 10.0, "downhill_azimuth_deg": 90.0},
    )
    hm = build_terrain(cfg, RngFactory(0)).heightmap
    np.testing.assert_allclose(hm.slope(), np.deg2rad(10.0), atol=1e-12)
    np.testing.assert_allclose(hm.aspect(), np.pi / 2, atol=1e-12)  # descends to the East
    assert hm.height_at(0.0, 0.0) == pytest.approx(100.0)


def test_world_bit_reproducible() -> None:
    cfg = WorldConfig(seed=3, terrain=TerrainConfig(size_x_m=60, size_y_m=40, resolution_m=0.2))
    a = World(cfg, workers=1).heightmap.z
    b = World(cfg, workers=4).heightmap.z
    assert a.tobytes() == b.tobytes()
    c = World(cfg.model_copy(update={"seed": 4})).heightmap.z
    assert not np.array_equal(a, c)
