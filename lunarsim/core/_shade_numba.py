"""numba kernels for per-pixel shading (same formulas as the numpy implementations).

* :func:`hapke_vectors` - Hapke reflectance from unit normal / to-light / to-viewer vectors
  (:func:`lunarsim.core.illumination.brdf.hapke` with :func:`geometry_from_vectors`).
* :func:`surface_normals` - bilinear node-gradient normals and nearest-node labels
  (:func:`lunarsim.core.render.sample_surface` for one layer).
* :func:`box_interval` - ray parameter interval inside an xy box (layered caster).

Each kernel is a scalar loop over rays (``prange``), so the result per element does not
depend on the thread count. Agreement with numpy: floating-point rounding (tested).
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

__all__ = ["HAS_NUMBA", "box_interval", "hapke_vectors", "surface_normals"]

FloatArray = npt.NDArray[np.float64]


def _h(x: float, w: float, r0: float) -> float:  # pragma: no cover - compiled
    xs = max(x, 1e-12)
    return 1.0 / (1.0 - w * xs * (r0 + 0.5 * (1.0 - 2.0 * r0 * xs) * math.log((1.0 + xs) / xs)))


def _hapke_one(
    nx: float,
    ny: float,
    nz: float,
    lx: float,
    ly: float,
    lz: float,
    vx: float,
    vy: float,
    vz: float,
    w: float,
    b: float,
    c: float,
    bs0: float,
    hs: float,
    bc0: float,
    hc: float,
    theta_bar: float,
    K: float,
) -> float:  # pragma: no cover - compiled
    mu0 = nx * lx + ny * ly + nz * lz
    mu = nx * vx + ny * vy + nz * vz
    if not (mu0 > 0.0 and mu > 0.0):
        return 0.0
    cos_g = min(max(lx * vx + ly * vy + lz * vz, -1.0), 1.0)
    si_r = math.sqrt(max(1.0 - mu0 * mu0, 0.0))
    se_r = math.sqrt(max(1.0 - mu * mu, 0.0))
    den = si_r * se_r
    cpsi = (cos_g - mu0 * mu) / den if den > 1e-12 else 1.0
    psi = math.acos(min(max(cpsi, -1.0), 1.0))
    g = math.acos(cos_g)
    m0 = min(max(mu0, 1e-9), 1.0)
    m = min(max(mu, 1e-9), 1.0)
    if theta_bar <= 0.0:
        mu0e, mue, S = m0, m, 1.0
    else:
        tt = math.tan(theta_bar)
        chi = 1.0 / math.sqrt(1.0 + math.pi * tt * tt)
        i = math.acos(m0)
        e = math.acos(m)
        si, se = math.sin(i), math.sin(e)
        ti = max(math.tan(i), 1e-12)
        te = max(math.tan(e), 1e-12)
        e1i = math.exp(-2.0 / (math.pi * tt * ti))
        e1e = math.exp(-2.0 / (math.pi * tt * te))
        e2i = math.exp(-1.0 / (math.pi * tt * tt * ti * ti))
        e2e = math.exp(-1.0 / (math.pi * tt * tt * te * te))
        s2 = math.sin(psi / 2.0) ** 2
        cp = math.cos(psi)
        f = math.exp(-2.0 * math.tan(min(psi, math.pi - 1e-9) / 2.0))
        eta_i = chi * (math.cos(i) + si * tt * e2i / (2.0 - e1i))
        eta_e = chi * (math.cos(e) + se * tt * e2e / (2.0 - e1e))
        if i <= e:
            d = 2.0 - e1e - (psi / math.pi) * e1i
            mu0e = chi * (m0 + si * tt * (cp * e2e + s2 * e2i) / d)
            mue = chi * (m + se * tt * (e2e - s2 * e2i) / d)
            S = mue / eta_e * m0 / eta_i * chi / (1.0 - f + f * chi * m0 / eta_i)
        else:
            d = 2.0 - e1i - (psi / math.pi) * e1e
            mu0e = chi * (m0 + si * tt * (e2i - s2 * e2e) / d)
            mue = chi * (m + se * tt * (cp * e2i + s2 * e2e) / d)
            S = mue / eta_e * m0 / eta_i * chi / (1.0 - f + f * chi * m / eta_e)
    ph = (
        0.5 * (1.0 + c) * (1.0 - b * b) / (1.0 - 2.0 * b * cos_g + b * b) ** 1.5
        + 0.5 * (1.0 - c) * (1.0 - b * b) / (1.0 + 2.0 * b * cos_g + b * b) ** 1.5
    )
    tg2 = math.tan(g / 2.0)
    b_sh = 1.0 + bs0 / (1.0 + tg2 / hs)
    x = tg2 / hc
    cb = (1.0 - math.exp(-x)) / x if x > 1e-12 else 1.0
    b_cb = 1.0 + bc0 * 0.5 * (1.0 + cb) / (1.0 + x) ** 2
    gamma = math.sqrt(1.0 - w)
    r0 = (1.0 - gamma) / (1.0 + gamma)
    M = _h(mu0e / K, w, r0) * _h(mue / K, w, r0) - 1.0
    return float(K * w / (4.0 * math.pi) * mu0e / (mu0e + mue) * (ph * b_sh + M) * b_cb * S)


def _hapke_kernel(
    n: FloatArray, lt: FloatArray, v: FloatArray, prm: FloatArray, out: FloatArray
) -> None:  # pragma: no cover - compiled
    w, b, c, bs0, hs, bc0, hc, theta_bar, K = (
        prm[0], prm[1], prm[2], prm[3], prm[4], prm[5], prm[6], prm[7], prm[8],
    )  # fmt: skip
    for k in numba.prange(n.shape[0]):
        out[k] = _hapke_one(
            n[k, 0], n[k, 1], n[k, 2], lt[k, 0], lt[k, 1], lt[k, 2], v[k, 0], v[k, 1], v[k, 2],
            w, b, c, bs0, hs, bc0, hc, theta_bar, K,
        )  # fmt: skip


def _normals_kernel(
    z: FloatArray,
    grid_labels: npt.NDArray[np.uint8],
    res: float,
    x0: float,
    y0: float,
    px: FloatArray,
    py: FloatArray,
    qi: npt.NDArray[np.int64],
    qj: npt.NDArray[np.int64],
    out_n: FloatArray,
    out_lab: npt.NDArray[np.uint8],
) -> None:  # pragma: no cover - compiled
    ny, nx = z.shape
    for k in numba.prange(px.size):
        i, j = qi[k], qj[k]
        fu = min(max((px[k] - x0) / res - j, 0.0), 1.0)
        fv = min(max((py[k] - y0) / res - i, 0.0), 1.0)
        gx = 0.0
        gy = 0.0
        for di in range(2):
            for dj in range(2):
                ii, jj = i + di, j + dj
                jm, jp = max(jj - 1, 0), min(jj + 1, nx - 1)
                im, ip = max(ii - 1, 0), min(ii + 1, ny - 1)
                wu = fu if dj == 1 else 1.0 - fu
                wv = fv if di == 1 else 1.0 - fv
                wgt = wu * wv
                gx += wgt * ((z[ii, jp] - z[ii, jm]) / ((jp - jm) * res))
                gy += wgt * ((z[ip, jj] - z[im, jj]) / ((ip - im) * res))
        nrm = math.sqrt(gx * gx + gy * gy + 1.0)
        out_n[k, 0] = -gx / nrm
        out_n[k, 1] = -gy / nrm
        out_n[k, 2] = 1.0 / nrm
        out_lab[k] = grid_labels[i + (1 if fv >= 0.5 else 0), j + (1 if fu >= 0.5 else 0)]


def _box_kernel(
    o: FloatArray,
    d: FloatArray,
    xlo: float,
    xhi: float,
    ylo: float,
    yhi: float,
    ta: FloatArray,
    tb: FloatArray,
) -> None:  # pragma: no cover - compiled
    for k in numba.prange(o.shape[0]):
        a, b = -math.inf, math.inf
        for axis in range(2):
            lo = xlo if axis == 0 else ylo
            hi = xhi if axis == 0 else yhi
            p, dd = o[k, axis], d[k, axis]
            if abs(dd) < 1e-15:
                inside = lo <= p <= hi
                tmn = -math.inf if inside else math.inf
                tmx = math.inf if inside else -math.inf
            else:
                t1 = (lo - p) / dd
                t2 = (hi - p) / dd
                tmn, tmx = min(t1, t2), max(t1, t2)
            a = max(a, tmn)
            b = min(b, tmx)
        ta[k] = a
        tb[k] = b


if HAS_NUMBA:
    _h = numba.njit(cache=True, inline="always")(_h)
    _hapke_one = numba.njit(cache=True)(_hapke_one)
    _hapke_nb = numba.njit(parallel=True, cache=True)(_hapke_kernel)
    _normals_nb = numba.njit(parallel=True, cache=True)(_normals_kernel)
    _box_nb = numba.njit(parallel=True, cache=True)(_box_kernel)


def hapke_vectors(
    n: FloatArray, to_light: FloatArray, v: FloatArray, prm: FloatArray
) -> FloatArray:
    """Hapke reflectance [1/sr] for ``(N, 3)`` unit vectors; ``prm`` = (w, b, c, bs0, hs,
    bc0, hc, theta_bar [rad], K)."""
    out = np.empty(n.shape[0])
    _hapke_nb(
        np.ascontiguousarray(n, np.float64), np.ascontiguousarray(to_light, np.float64),
        np.ascontiguousarray(v, np.float64), prm, out,
    )  # fmt: skip
    return out


def surface_normals(
    z: FloatArray,
    grid_labels: npt.NDArray[np.uint8],
    res: float,
    x0: float,
    y0: float,
    pts: FloatArray,
    qi: npt.NDArray[np.int64],
    qj: npt.NDArray[np.int64],
) -> tuple[FloatArray, npt.NDArray[np.uint8]]:
    """Smooth normals ``(M, 3)`` and nearest-node labels at hit points of one layer."""
    m = pts.shape[0]
    out_n = np.empty((m, 3))
    out_lab = np.empty(m, dtype=np.uint8)
    _normals_nb(
        z, grid_labels, res, x0, y0, np.ascontiguousarray(pts[:, 0]),
        np.ascontiguousarray(pts[:, 1]), np.ascontiguousarray(qi, np.int64),
        np.ascontiguousarray(qj, np.int64), out_n, out_lab,
    )  # fmt: skip
    return out_n, out_lab


def box_interval(
    o: FloatArray, d: FloatArray, box: tuple[float, float, float, float]
) -> tuple[FloatArray, FloatArray]:
    """``(ta, tb)`` where the ray's xy track is inside ``box`` (ta > tb: never)."""
    ta = np.empty(o.shape[0])
    tb = np.empty(o.shape[0])
    _box_nb(
        np.ascontiguousarray(o, np.float64), np.ascontiguousarray(d, np.float64),
        box[0], box[1], box[2], box[3], ta, tb,
    )  # fmt: skip
    return ta, tb
