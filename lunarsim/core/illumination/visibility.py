"""Visibility of the Sun (and Earth) discs from surface points: hard shadows and penumbrae.

The Sun subtends a radius ``α = asin(R_sun / d)`` ≈ 0.267° at 1 AU. With ``samples > 0``
the disc is sampled with a deterministic Vogel (golden-angle) spiral of equal-area points
weighted by linear limb darkening ``1 - u (1 - μ)``; each sample is visible if it is in
front of the surface, above the local horizon plane and not occluded by terrain.
Adaptive: four rim samples (top, bottom, left, right of the disc) are tested first and
only points where they disagree get the full sample set, so the cost outside penumbra
and partial-facing regions is four shadow rays per point.

With ``samples == 0`` the source is a point (the disc centre): visibility is 0 or 1,
identical to the classic single shadow ray.

Local horizon: the tangent-plane world has no curvature unless ``curvature_radius_m``
is given, in which case the local vertical at ``p`` is ``(x/R, y/R, 1)`` normalised
(sphere centre at ``(0, 0, -R)``). Without it the global elevation must be > 0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt

from lunarsim.core.illumination.sun import AU_M, EARTH_MEAN_RADIUS_M, SunState

__all__ = ["DiscSource", "SkyLighting", "disc_visibility", "earth_source", "sun_source"]

FloatArray = npt.NDArray[np.float64]
GOLDEN_ANGLE = math.pi * (3.0 - math.sqrt(5.0))


class _Occluder(Protocol):
    def occluded(
        self,
        origins: npt.ArrayLike,
        directions: npt.ArrayLike,
        t_max: npt.ArrayLike = ...,
        t_min: npt.ArrayLike = ...,
    ) -> npt.NDArray[np.bool_]: ...


@dataclass(frozen=True)
class DiscSource:
    """A distant disc-shaped light source.

    Attributes:
        direction: Unit vector to the disc centre (world ENU).
        radius: Angular radius [rad].
        limb_u: Linear limb-darkening coefficient (0 = uniform disc).
    """

    direction: FloatArray
    radius: float
    limb_u: float = 0.0

    def samples(self, n: int) -> tuple[FloatArray, FloatArray, FloatArray]:
        """``(directions (n, 3), weights (n,), rim directions (4, 3))`` over the disc."""
        s = self.direction
        ref = np.array([0.0, 0.0, 1.0]) if abs(s[2]) < 0.99 else np.array([1.0, 0.0, 0.0])
        up = ref - s * (ref @ s)  # disc "up" = towards increasing elevation
        up /= np.linalg.norm(up)
        side = np.cross(s, up)
        a = self.radius

        def dirs(r: FloatArray, th: FloatArray) -> FloatArray:
            t = np.tan(r)
            d = s[None, :] + t[:, None] * (np.cos(th)[:, None] * up + np.sin(th)[:, None] * side)
            return np.asarray(d / np.linalg.norm(d, axis=1, keepdims=True), dtype=np.float64)

        k = np.arange(n, dtype=np.float64)
        rn = np.sqrt((k + 0.5) / n)  # equal-area radii in units of the disc radius
        w = 1.0 - self.limb_u * (1.0 - np.sqrt(np.clip(1.0 - rn * rn, 0.0, 1.0)))
        rim = dirs(np.full(4, a), np.array([0.0, math.pi, 0.5 * math.pi, 1.5 * math.pi]))
        return dirs(rn * a, k * GOLDEN_ANGLE), w / w.sum(), rim


def sun_source(sun: SunState, sun_radius_m: float = 6.957e8, limb_u: float = 0.6) -> DiscSource:
    """The solar disc as seen from the Moon."""
    return DiscSource(sun.direction, math.asin(sun_radius_m / (sun.distance_au * AU_M)), limb_u)


def earth_source(sun: SunState) -> DiscSource | None:
    """The Earth disc (uniform; None without an Earth direction)."""
    if sun.earth_direction is None or sun.earth_distance_m is None:
        return None
    return DiscSource(sun.earth_direction, math.asin(EARTH_MEAN_RADIUS_M / sun.earth_distance_m))


@dataclass(frozen=True)
class SkyLighting:
    """Distant light sources and how to sample them.

    Attributes:
        sun: Sun (and Earth) geometry.
        penumbra_samples: Disc samples (0 = point sources, hard shadows).
        sun_radius_m: Solar radius [m].
        limb_darkening_u: Solar limb darkening coefficient.
        curvature_radius_m: Moon radius for the local vertical (None = flat world).
        earthshine_w_m2: Earthshine irradiance normal to the Earth direction (0 = off).
    """

    sun: SunState
    penumbra_samples: int = 0
    sun_radius_m: float = 6.957e8
    limb_darkening_u: float = 0.6
    curvature_radius_m: float | None = None
    earthshine_w_m2: float = 0.0

    def local_up(self, points: FloatArray) -> FloatArray | None:
        """Local vertical at ``points`` (None = global +z)."""
        if self.curvature_radius_m is None:
            return None
        u = np.empty_like(points)
        u[:, 0] = points[:, 0] / self.curvature_radius_m
        u[:, 1] = points[:, 1] / self.curvature_radius_m
        u[:, 2] = 1.0
        return np.asarray(u / np.linalg.norm(u, axis=1, keepdims=True), dtype=np.float64)

    def sun_visibility(
        self, caster: _Occluder, points: FloatArray, normals: FloatArray, bias_m: float
    ) -> FloatArray:
        """Visible fraction of the solar disc in [0, 1] per point."""
        src = sun_source(self.sun, self.sun_radius_m, self.limb_darkening_u)
        return disc_visibility(
            caster, points, normals, src, bias_m, self.penumbra_samples, self.local_up(points)
        )

    def earth_visibility(
        self, caster: _Occluder, points: FloatArray, normals: FloatArray, bias_m: float
    ) -> FloatArray:
        """Visibility (0/1) of the Earth centre (zeros without Earth).

        Earth is treated as a point source: its ~1° disc would make penumbrae wide and
        expensive for a source ~1e-4 as bright as the Sun.
        """
        src = earth_source(self.sun)
        if src is None:
            return np.zeros(points.shape[0])
        return disc_visibility(caster, points, normals, src, bias_m, 0, self.local_up(points))


def _visible(
    caster: _Occluder,
    points: FloatArray,
    normals: FloatArray,
    d: FloatArray,
    bias_m: float,
    up: FloatArray | None,
) -> npt.NDArray[np.bool_]:
    """Facing, above the local horizon and unoccluded towards direction ``d`` (3,)."""
    ok = normals @ d > 0.0
    ok &= (up @ d > 0.0) if up is not None else bool(d[2] > 0.0)
    idx = np.flatnonzero(ok)
    if idx.size:
        o = points[idx] + bias_m * normals[idx]
        ok[idx[caster.occluded(o, np.broadcast_to(d, o.shape))]] = False
    return ok


def disc_visibility(
    caster: _Occluder,
    points: FloatArray,
    normals: FloatArray,
    src: DiscSource,
    bias_m: float,
    samples: int = 0,
    local_up: FloatArray | None = None,
) -> FloatArray:
    """Visible (weighted) fraction of a disc source from surface points, in [0, 1]."""
    m = points.shape[0]
    if samples <= 0:
        return _visible(caster, points, normals, src.direction, bias_m, local_up).astype(np.float64)
    out = np.zeros(m)
    # Points that cannot see any part of the disc: facing / horizon beyond the radius.
    sin_a = math.sin(src.radius)
    cand = normals @ src.direction > -sin_a
    cand &= (
        (local_up @ src.direction > -sin_a)
        if local_up is not None
        else bool(src.direction[2] > -sin_a)
    )
    idx = np.flatnonzero(cand)
    if idx.size == 0:
        return out
    dirs, w, rim = src.samples(samples)
    P, N = points[idx], normals[idx]
    U = local_up[idx] if local_up is not None else None
    rim_vis = np.stack([_visible(caster, P, N, d, bias_m, U) for d in rim], axis=1)
    n_lit = rim_vis.sum(axis=1)
    out[idx[n_lit == 4]] = 1.0
    mixed = np.flatnonzero((n_lit > 0) & (n_lit < 4))
    # Rim all dark but the disc centre facing: the disc may still peek past a thin occluder;
    # treated as dark (thin features smaller than the disc are missed by the rim test).
    if mixed.size:
        Pm, Nm = P[mixed], N[mixed]
        Um = U[mixed] if U is not None else None
        acc = np.zeros(mixed.size)
        for d, wk in zip(dirs, w, strict=True):
            acc += wk * _visible(caster, Pm, Nm, d, bias_m, Um)
        out[idx[mixed]] = acc
    return out
