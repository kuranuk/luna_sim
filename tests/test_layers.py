from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from lunarsim.config import TerrainConfig, parse_config
from lunarsim.core.rng import RngFactory
from lunarsim.core.terrain.fractal import rms_slope_at_baseline
from lunarsim.core.terrain.layers import TwoLayerTerrain
from lunarsim.core.world import World

BASE = {
    "size_x_m": 400.0,
    "size_y_m": 400.0,
    "resolution_m": 1.0,
    "fractal": {
        "hurst": 0.85,
        "rms_slope_deg": 5.0,
        "baseline_m": 1.0,
        "rolloff_wavelength_m": 200,
    },
    "near": {
        "enabled": True,
        "size_m": 40.0,
        "resolution_m": 0.05,
        "tile_cells": 256,
        "overlap_cells": 32,
    },
}


def _cfg(**over: object) -> TerrainConfig:
    d = {**BASE, **over}
    return TerrainConfig.model_validate(d)


def test_validation() -> None:
    with pytest.raises(ValidationError):
        _cfg(near={**BASE["near"], "resolution_m": 2.0})  # not finer
    with pytest.raises(ValidationError):
        _cfg(near={**BASE["near"], "center_x_m": 190.0})  # outside far terrain
    with pytest.raises(ValidationError):
        _cfg(craters={"enabled": True, "sfd": {"d_min_m": 0.05}})  # < 2 fine cells


@pytest.mark.parametrize("seed", [1, 2])
def test_rms_slope_two_layer(seed: int) -> None:
    t = TwoLayerTerrain(_cfg(), RngFactory(seed))
    hm = t.near.heightmap
    s = np.degrees(np.arctan(rms_slope_at_baseline(hm.z, hm.resolution, 20)))
    assert s == pytest.approx(5.0, abs=0.6)


def test_patch_consistency_across_centres() -> None:
    cfg = _cfg(
        craters={"enabled": True, "sfd": {"d_min_m": 0.2, "d_max_m": 30.0}},
        rocks={
            "enabled": True,
            "k": 0.08,
            "d_min_m": 0.15,
            "ejecta": {"enabled": True, "min_crater_diameter_m": 5.0},
        },
    )
    t = TwoLayerTerrain(cfg, RngFactory(4))
    a = t.build_near(0.0, 0.0)
    b = t.build_near(6.0, -3.0)  # integer number of fine cells
    ha, hb = a.heightmap, b.heightmap
    # Overlap region away from both patch borders (beyond the largest crater reach).
    di = round((hb.y0 - ha.y0) / 0.05)
    dj = round((hb.x0 - ha.x0) / 0.05)
    za = ha.z[di + 60 : -60, dj + 60 : -60]
    zb = hb.z[60 : ha.shape[0] - di - 60, 60 : ha.shape[1] - dj - 60]
    assert za.shape == zb.shape and za.size > 0
    np.testing.assert_allclose(za, zb, atol=1e-9)


def test_seam_small_without_small_features() -> None:
    t = TwoLayerTerrain(_cfg(), RngFactory(5))
    hm, far = t.near.heightmap, t.far.heightmap
    x = hm.x
    edge = hm.z[0, :] - far.height_at_mesh(x, np.full_like(x, hm.y[0]))
    assert np.sqrt(np.mean(edge**2)) < 0.06  # ~ fine-detail amplitude


def test_world_two_layer_render_and_recenter() -> None:
    cfg = parse_config(
        {
            "seed": 2,
            "terrain": BASE,
            "sun": {"azimuth_deg": 90, "elevation_deg": 8},
            "cameras": [{"name": "c", "intrinsics": {"width": 64, "height": 48}}],
        }
    )
    w = World(cfg)
    assert len(w.layers) == 2
    from lunarsim.core.sensors.camera import camera_pose_from_body

    pose = camera_pose_from_body([0, 0, w.height_at(0, 0) + 1.5], pitch=np.radians(2.0))
    f = w.render_camera("c", pose)
    lay = f.buffers.layer
    assert (lay == 0).any() and (lay == 1).any()  # near ground and far horizon both visible
    w.recenter_near(10.0, 0.0)
    assert w.heightmap.extent[0] == pytest.approx(-10.0)
