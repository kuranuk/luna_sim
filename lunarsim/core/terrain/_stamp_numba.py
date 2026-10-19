"""numba kernels for crater and rock stamping (same algorithm as the numpy loops).

The numpy implementations in :mod:`.craters` / :mod:`.rocks` loop over features in
Python and pay numpy call overhead on tiny patches; these kernels run the identical
per-cell arithmetic sequentially (age order matters for craters) without that
overhead. Results agree with the numpy path to floating-point rounding (tested).
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

try:  # optional accelerator
    import numba

    HAS_NUMBA = True
except ImportError:  # pragma: no cover
    numba = None
    HAS_NUMBA = False

__all__ = [
    "HAS_NUMBA",
    "height_mesh_numba",
    "max_pool_numba",
    "stamp_craters_numba",
    "stamp_rocks_numba",
]

FloatArray = npt.NDArray[np.float64]


def _craters_kernel(
    z: FloatArray,
    x0: float,
    y0: float,
    res: float,
    cxs: FloatArray,
    cys: FloatArray,
    radii: FloatArray,
    gs: FloatArray,
    table: FloatArray,
    rmax: float,
    planes_in: FloatArray,
    planes_out: FloatArray,
    labels: npt.NDArray[np.uint8],
    use_labels: bool,
    lab_min: float,
    ring_in: float,
    ring_out: float,
    rep_in: float,
    rep_out: float,
    rim_in: float,
    rim_out: float,
    lab_interior: int,
    lab_rim: int,
) -> None:  # pragma: no cover - compiled by numba
    ny, nx = z.shape
    ng, nr = table.shape
    prof = np.empty(nr)
    drho = rmax / (nr - 1)
    for k in range(cxs.size):
        cx, cy, R = cxs[k], cys[k], radii[k]
        reach = rmax * R
        j0 = max(0, math.floor((cx - reach - x0) / res))
        j1 = min(nx, math.ceil((cx + reach - x0) / res) + 1)
        i0 = max(0, math.floor((cy - reach - y0) / res))
        i1 = min(ny, math.ceil((cy + reach - y0) / res) + 1)
        if j0 >= j1 or i0 >= i1:
            continue
        # Profile / R for degradation g (linear interpolation between table rows).
        t = min(max(gs[k], 0.0), 1.0) * (ng - 1)
        ti = min(int(t), ng - 2)
        tf = t - ti
        for m in range(nr):
            prof[m] = (1.0 - tf) * table[ti, m] + tf * table[ti + 1, m]
        # Reference plane: given, or least squares on the ring (normal equations).
        a = planes_in[k, 0]
        b = planes_in[k, 1]
        c = planes_in[k, 2]
        if not (math.isfinite(a) and math.isfinite(b) and math.isfinite(c)):
            n = 0
            sx = sy = sxx = sxy = syy = sz = sxz = syz = 0.0
            pz = 0.0
            for i in range(i0, i1):
                dy = y0 + res * i - cy
                for j in range(j0, j1):
                    dx = x0 + res * j - cx
                    pz += z[i, j]
                    rho = math.hypot(dx, dy) / R
                    if ring_in <= rho <= ring_out:
                        zz = z[i, j]
                        n += 1
                        sx += dx
                        sy += dy
                        sxx += dx * dx
                        sxy += dx * dy
                        syy += dy * dy
                        sz += zz
                        sxz += dx * zz
                        syz += dy * zz
            n_full = math.pi * (ring_out**2 - ring_in**2) * (R / res) ** 2
            fitted = False
            if n >= 3 and n >= 0.5 * n_full:
                # Solve [[n, sx, sy], [sx, sxx, sxy], [sy, sxy, syy]] (a, b, c) = (sz, sxz, syz).
                m00, m01, m02 = float(n), sx, sy
                m11, m12, m22 = sxx, sxy, syy
                c00 = m11 * m22 - m12 * m12
                c01 = m02 * m12 - m01 * m22
                c02 = m01 * m12 - m02 * m11
                det = m00 * c00 + m01 * c01 + m02 * c02
                scale = abs(m00 * m11 * m22) + 1e-300
                if abs(det) > 1e-12 * scale:
                    c11 = m00 * m22 - m02 * m02
                    c12 = m01 * m02 - m00 * m12
                    c22 = m00 * m11 - m01 * m01
                    a = (c00 * sz + c01 * sxz + c02 * syz) / det
                    b = (c01 * sz + c11 * sxz + c12 * syz) / det
                    c = (c02 * sz + c12 * sxz + c22 * syz) / det
                    fitted = True
            if not fitted:
                a = sz / n if n > 0 else pz / ((i1 - i0) * (j1 - j0))
                b = 0.0
                c = 0.0
        planes_out[k, 0] = a
        planes_out[k, 1] = b
        planes_out[k, 2] = c
        do_lab = use_labels and 2.0 * R >= lab_min
        for i in range(i0, i1):
            dy = y0 + res * i - cy
            for j in range(j0, j1):
                dx = x0 + res * j - cx
                rho = math.hypot(dx, dy) / R
                if rho > rmax:
                    delta = 0.0
                else:
                    u = rho / drho
                    m = min(int(u), nr - 2)
                    f = u - m
                    delta = R * ((1.0 - f) * prof[m] + f * prof[m + 1])
                w = min(max((rep_out - rho) / (rep_out - rep_in), 0.0), 1.0)
                zz = z[i, j]
                z[i, j] = zz + (delta + w * (a + b * dx + c * dy - zz))
                if do_lab:
                    if rho < rim_in:
                        labels[i, j] = lab_interior
                    elif rho < rim_out:
                        labels[i, j] = lab_rim


def _rocks_kernel(
    z: FloatArray,
    x0: float,
    y0: float,
    res: float,
    cxs: FloatArray,
    cys: FloatArray,
    diam: FloatArray,
    ratio: FloatArray,
    orient: FloatArray,
    height: FloatArray,
    labels: npt.NDArray[np.uint8],
    use_labels: bool,
    lab_rock: int,
) -> None:  # pragma: no cover - compiled by numba
    ny, nx = z.shape
    for k in range(cxs.size):
        cx, cy = cxs[k], cys[k]
        a = 0.5 * diam[k]
        b = a * ratio[k]
        j0 = max(0, math.floor((cx - a - x0) / res))
        j1 = min(nx, math.ceil((cx + a - x0) / res) + 1)
        i0 = max(0, math.floor((cy - a - y0) / res))
        i1 = min(ny, math.ceil((cy + a - y0) / res) + 1)
        co, si = math.cos(orient[k]), math.sin(orient[k])
        h = height[k]
        for i in range(i0, i1):
            dy = y0 + res * i - cy
            for j in range(j0, j1):
                dx = x0 + res * j - cx
                u = (dx * co + dy * si) / a
                v = (-dx * si + dy * co) / b
                s = 1.0 - u * u - v * v
                if s > 0.0:
                    z[i, j] += h * math.sqrt(s)
                    if use_labels:
                        labels[i, j] = lab_rock


def _height_mesh_kernel(
    z: FloatArray, x0: float, y0: float, res: float, x: FloatArray, y: FloatArray, out: FloatArray
) -> None:  # pragma: no cover - compiled by numba
    """Same arithmetic as :meth:`Heightmap.height_at_mesh` (scalar loop)."""
    ny, nx = z.shape
    for k in range(x.size):
        u = min(max((x[k] - x0) / res, 0.0), nx - 1.0)
        v = min(max((y[k] - y0) / res, 0.0), ny - 1.0)
        j = min(math.floor(u), nx - 2)
        i = min(math.floor(v), ny - 2)
        fu = u - j
        fv = v - i
        z00 = z[i, j]
        if fu >= fv:
            zx = z[i, j + 1]
            out[k] = z00 + fu * (zx - z00) + fv * (z[i + 1, j + 1] - zx)
        else:
            zy = z[i + 1, j]
            out[k] = z00 + fv * (zy - z00) + fu * (z[i + 1, j + 1] - zy)


def _max_pool_kernel(src: FloatArray, out: FloatArray, stride: int) -> None:  # pragma: no cover
    """2x2 max pooling with ``stride`` (2: halve, edges padded with -inf; 1: quad max)."""
    h, w = src.shape
    oh, ow = out.shape
    for i in numba.prange(oh):
        for j in range(ow):
            m = -math.inf
            for di in range(2):
                ii = stride * i + di
                if ii < h:
                    for dj in range(2):
                        jj = stride * j + dj
                        if jj < w and src[ii, jj] > m:
                            m = src[ii, jj]
            out[i, j] = m


if HAS_NUMBA:
    height_mesh_numba = numba.njit(cache=True)(_height_mesh_kernel)
    max_pool_numba = numba.njit(cache=True, parallel=True)(_max_pool_kernel)
    stamp_craters_numba = numba.njit(cache=True)(_craters_kernel)
    stamp_rocks_numba = numba.njit(cache=True)(_rocks_kernel)
else:  # pragma: no cover
    height_mesh_numba = None
    max_pool_numba = None
    stamp_craters_numba = None
    stamp_rocks_numba = None
