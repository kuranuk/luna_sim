from __future__ import annotations

import numpy as np

from lunarsim.core.terrain import Heightmap


def _plane(a: float, b: float, c: float, n: int = 41, res: float = 0.5) -> Heightmap:
    hm0 = Heightmap.centered(np.zeros((n, n + 6)), res)
    z = a * hm0.x[None, :] + b * hm0.y[:, None] + c
    return Heightmap.centered(z, res)


def test_geometry() -> None:
    hm = Heightmap.centered(np.zeros((5, 9)), 0.5)
    assert hm.extent == (-2.0, 2.0, -1.0, 1.0)
    assert hm.contains(0.0, 0.0) and not hm.contains(2.1, 0.0)


def test_bilinear_exact_on_plane() -> None:
    hm = _plane(0.1, -0.2, 3.0)
    rng = np.random.default_rng(0)
    xmin, xmax, ymin, ymax = hm.extent
    x = rng.uniform(xmin, xmax, 1000)
    y = rng.uniform(ymin, ymax, 1000)
    np.testing.assert_allclose(hm.height_at(x, y), 0.1 * x - 0.2 * y + 3.0, atol=1e-12)
    np.testing.assert_allclose(hm.height_at(xmax, ymax), 0.1 * xmax - 0.2 * ymax + 3.0)


def test_normals_slope_aspect_on_plane() -> None:
    a, b = 0.3, 0.4  # grade 0.5, uphill towards (+x, +y)
    hm = _plane(a, b, 0.0)
    n = hm.normals()
    expected = np.array([-a, -b, 1.0]) / np.sqrt(a * a + b * b + 1)
    np.testing.assert_allclose(n, np.broadcast_to(expected, n.shape), atol=1e-12)
    np.testing.assert_allclose(hm.slope(), np.arctan(0.5), atol=1e-12)
    # downhill direction is (-0.6, -0.8): azimuth atan2(-0.6, -0.8) from North
    np.testing.assert_allclose(hm.aspect(), np.mod(np.arctan2(-a, -b), 2 * np.pi), atol=1e-12)
