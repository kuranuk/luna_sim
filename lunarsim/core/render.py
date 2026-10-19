"""CPU ray-cast rendering of terrain radiance and ground-truth buffers.

For every pixel: primary ray -> terrain hit -> shading normal -> sun shadow ray ->
reflectance model -> radiance [W m^-2 sr^-1] (in the camera band). No ambient term: the Moon has
no atmosphere. Optional: penumbrae (solar disc sampling), earthshine (Earth as a weak
second disc source) and lunar curvature (local horizon), see :class:`SkyLighting`.
Inter-reflection between terrain facets is not modelled.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt

from lunarsim.config.schema import ReflectanceConfig
from lunarsim.core.frames import Pose
from lunarsim.core.illumination.brdf import reflectance
from lunarsim.core.illumination.lights import LightState
from lunarsim.core.illumination.sun import SunState
from lunarsim.core.illumination.visibility import SkyLighting
from lunarsim.core.raycast.heightfield import RayHits
from lunarsim.core.sensors.camera import CameraModel
from lunarsim.core.terrain.generate import Terrain
from lunarsim.core.terrain.labels import Label

__all__ = ["Caster", "RenderBuffers", "SurfaceSample", "render_rays", "sample_surface"]

FloatArray = npt.NDArray[np.float64]


class Caster(Protocol):
    """Ray caster interface (see :mod:`lunarsim.core.raycast`)."""

    def intersect(
        self,
        origins: npt.ArrayLike,
        directions: npt.ArrayLike,
        t_min: npt.ArrayLike = ...,
        t_max: npt.ArrayLike = ...,
    ) -> RayHits: ...

    def occluded(
        self,
        origins: npt.ArrayLike,
        directions: npt.ArrayLike,
        t_max: npt.ArrayLike = ...,
        t_min: npt.ArrayLike = ...,
    ) -> npt.NDArray[np.bool_]: ...


@dataclass(frozen=True)
class SurfaceSample:
    """Surface attributes at ray hits (rows of hit rays only).

    Attributes:
        points: Hit points in world ENU [m], ``(M, 3)``.
        normals: Smooth shading normals: central-difference gradients at the grid
            nodes, bilinearly interpolated (the geometry itself stays faceted), ``(M, 3)``.
        labels: Terrain label of the nearest grid node, ``(M,)``.
    """

    points: FloatArray
    normals: FloatArray
    labels: npt.NDArray[np.uint8]


def _node_gradient_bilinear(
    z: FloatArray,
    res: float,
    i: npt.NDArray[np.int64],
    j: npt.NDArray[np.int64],
    fu: FloatArray,
    fv: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Bilinear interpolation of node gradients (central differences, one-sided at edges)."""
    ny, nx = z.shape

    def grad(ii: npt.NDArray[np.int64], jj: npt.NDArray[np.int64]) -> tuple[FloatArray, FloatArray]:
        jm, jp = np.maximum(jj - 1, 0), np.minimum(jj + 1, nx - 1)
        im, ip = np.maximum(ii - 1, 0), np.minimum(ii + 1, ny - 1)
        gx = (z[ii, jp] - z[ii, jm]) / ((jp - jm) * res)
        gy = (z[ip, jj] - z[im, jj]) / ((ip - im) * res)
        return gx, gy

    g00, g01, g10, g11 = grad(i, j), grad(i, j + 1), grad(i + 1, j), grad(i + 1, j + 1)
    w00, w01 = (1 - fu) * (1 - fv), fu * (1 - fv)
    w10, w11 = (1 - fu) * fv, fu * fv
    gx = w00 * g00[0] + w01 * g01[0] + w10 * g10[0] + w11 * g11[0]
    gy = w00 * g00[1] + w01 * g01[1] + w10 * g10[1] + w11 * g11[1]
    return np.asarray(gx), np.asarray(gy)


def sample_surface(
    layers: Sequence[Terrain], o: FloatArray, d: FloatArray, hits: RayHits
) -> SurfaceSample:
    """Points, normals and labels at the hits (``hits.hit`` rows of ``o``/``d``)."""
    m = hits.hit
    pts = o[m] + d[m] * hits.t[m, None]
    nrm = np.zeros_like(pts)
    lab = np.zeros(pts.shape[0], dtype=np.uint8)
    lay = hits.layer[m]
    qi, qj = hits.qi[m], hits.qj[m]
    for li, terrain in enumerate(layers):
        sel = lay == li
        if not sel.any():
            continue
        hm = terrain.heightmap
        i, j = qi[sel], qj[sel]
        fu = np.clip((pts[sel, 0] - hm.x0) / hm.resolution - j, 0.0, 1.0)
        fv = np.clip((pts[sel, 1] - hm.y0) / hm.resolution - i, 0.0, 1.0)
        gx, gy = _node_gradient_bilinear(hm.z, hm.resolution, i, j, fu, fv)
        n = np.stack([-gx, -gy, np.ones_like(gx)], axis=1)
        nrm[sel] = n / np.linalg.norm(n, axis=1, keepdims=True)
        ni = i + (fv >= 0.5).astype(np.int64)
        nj = j + (fu >= 0.5).astype(np.int64)
        lab[sel] = terrain.labels[ni, nj]
    return SurfaceSample(points=pts, normals=nrm, labels=lab)


@dataclass(frozen=True)
class RenderBuffers:
    """Per-pixel outputs (images of shape ``(H, W)`` unless noted).

    Attributes:
        radiance: In-band radiance [W m^-2 sr^-1] (sun + lights; 0 for sky).
        depth: Z in the camera frame [m] (``inf`` for sky).
        range_m: Distance along the ray [m] (``inf`` for sky).
        normals: World ENU shading normals, ``(H, W, 3)`` (0 for sky).
        labels: Terrain labels (``Label.SKY`` for sky).
        shadow: True where less than half of the solar disc is visible from the surface
            (point sun: no direct sunlight). Rover lights do not change this mask.
        layer: Terrain layer index hit (-1 for sky).
        points: World hit points ``(H, W, 3)`` [m] (nan for sky).
        sun_fraction: Visible fraction of the solar disc in [0, 1] (0 for sky).
    """

    radiance: FloatArray
    depth: FloatArray
    range_m: FloatArray
    normals: FloatArray
    labels: npt.NDArray[np.uint8]
    shadow: npt.NDArray[np.bool_]
    layer: npt.NDArray[np.int64]
    points: FloatArray
    sun_fraction: FloatArray


def _light_radiance(
    light: LightState,
    caster: Caster,
    surface: ReflectanceConfig,
    surf: SurfaceSample,
    to_viewer: FloatArray,
    bias: float,
) -> FloatArray:
    """In-band radiance [W m^-2 sr^-1] reflected from one spotlight."""
    vec = light.position - surf.points
    dist = np.linalg.norm(vec, axis=1)
    to_light = vec / np.maximum(dist, 1e-9)[:, None]
    intensity = light.intensity_towards(-to_light)
    mu0 = np.einsum("ij,ij->i", surf.normals, to_light)
    out = np.zeros(surf.points.shape[0])
    idx = np.flatnonzero((intensity > 0.0) & (mu0 > 0.0))
    if idx.size == 0:
        return out
    o = surf.points[idx] + bias * surf.normals[idx]
    blocked = caster.occluded(o, to_light[idx], np.maximum(dist[idx] - 2.0 * bias, 0.0))
    idx = idx[~blocked]
    r = reflectance(surface, surf.normals[idx], to_light[idx], to_viewer[idx])
    out[idx] = r * intensity[idx] / dist[idx] ** 2
    return out


def render_rays(
    layers: Sequence[Terrain],
    caster: Caster,
    sun: SunState,
    surface: ReflectanceConfig,
    camera: CameraModel,
    T_world_camera: Pose,
    shadow_bias_m: float,
    sun_band_fraction: float = 1.0,
    lights: Sequence[LightState] = (),
    sky: SkyLighting | None = None,
) -> RenderBuffers:
    """Render the in-band radiance and ground-truth buffers of one camera view.

    Sunlight is scaled by ``sun_band_fraction`` (fraction of the solar irradiance in the
    camera band) and by the visible fraction of the solar disc; earthshine (if
    ``sky.earthshine_w_m2 > 0``) adds ``r · E_earth · band_fraction · visible fraction``;
    each light adds ``r(i, e, g) · I(θ) / d²`` where visible from the light.
    ``sky`` defaults to a point sun without curvature or earthshine.
    """
    if sky is None:
        sky = SkyLighting(sun)
    h, w = camera.height, camera.width
    rays_cam = camera.pixel_rays()
    d = T_world_camera.rotate(rays_cam)
    o = np.broadcast_to(T_world_camera.t, d.shape).copy()
    hits = caster.intersect(o, d)
    m = hits.hit
    surf = sample_surface(layers, o, d, hits)

    n_pix = h * w
    radiance = np.zeros(n_pix)
    shadow = np.zeros(n_pix, dtype=bool)
    labels = np.full(n_pix, Label.SKY, dtype=np.uint8)
    normals = np.zeros((n_pix, 3))
    points = np.full((n_pix, 3), np.nan)
    labels[m] = surf.labels
    normals[m] = surf.normals
    points[m] = surf.points

    sun_frac = np.zeros(n_pix)
    if m.any():
        s = np.broadcast_to(sun.direction, surf.normals.shape)
        frac = sky.sun_visibility(caster, surf.points, surf.normals, shadow_bias_m)
        v = -d[m]
        r = reflectance(surface, surf.normals, s, v)
        rad = np.where(frac > 0.0, r * sun.irradiance * sun_band_fraction * frac, 0.0)
        if sky.earthshine_w_m2 > 0.0 and sun.earth_direction is not None:
            e_frac = sky.earth_visibility(caster, surf.points, surf.normals, shadow_bias_m)
            e = np.broadcast_to(sun.earth_direction, surf.normals.shape)
            r_e = reflectance(surface, surf.normals, e, v)
            rad += np.where(
                e_frac > 0.0, r_e * sky.earthshine_w_m2 * sun_band_fraction * e_frac, 0.0
            )
        for light in lights:
            rad += _light_radiance(light, caster, surface, surf, v, shadow_bias_m)
        radiance[m] = rad
        shadow[m] = frac < 0.5
        sun_frac[m] = frac

    rng = np.where(m, hits.t, np.inf)
    z_axis = T_world_camera.R[:, 2]
    depth = np.where(m, rng * (d @ z_axis), np.inf)
    return RenderBuffers(
        radiance=radiance.reshape(h, w),
        depth=depth.reshape(h, w),
        range_m=rng.reshape(h, w),
        normals=normals.reshape(h, w, 3),
        labels=labels.reshape(h, w),
        shadow=shadow.reshape(h, w),
        layer=hits.layer.reshape(h, w),
        points=points.reshape(h, w, 3),
        sun_fraction=sun_frac.reshape(h, w),
    )
