"""Regular-grid heightmap (DEM) in the world ENU frame.

Grid convention: ``z[i, j]`` is the elevation [m] of the cell centre at

    x_j = x0 + j * resolution   (j: West -> East)
    y_i = y0 + i * resolution   (i: South -> North)

so row 0 is the *southern* edge. Image writers flip rows so that North is up.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

__all__ = ["Heightmap"]

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class Heightmap:
    """Elevation grid with geometry helpers.

    Attributes:
        z: Elevation [m], shape ``(ny, nx)``, float64.
        resolution: Cell spacing [m].
        x0: World x [m] of column 0 cell centre.
        y0: World y [m] of row 0 cell centre.
    """

    z: FloatArray
    resolution: float
    x0: float
    y0: float

    def __post_init__(self) -> None:
        if self.z.ndim != 2:
            raise ValueError("z must be 2-D")
        if self.resolution <= 0.0:
            raise ValueError("resolution must be positive")

    @classmethod
    def centered(cls, z: FloatArray, resolution: float) -> Heightmap:
        """Heightmap whose grid centre lies at the world origin."""
        ny, nx = z.shape
        return cls(
            z=np.ascontiguousarray(z, dtype=np.float64),
            resolution=float(resolution),
            x0=-0.5 * (nx - 1) * resolution,
            y0=-0.5 * (ny - 1) * resolution,
        )

    @property
    def shape(self) -> tuple[int, int]:
        """``(ny, nx)``."""
        ny, nx = self.z.shape
        return int(ny), int(nx)

    @property
    def x(self) -> FloatArray:
        """Cell-centre x coordinates [m], shape ``(nx,)``."""
        return self.x0 + self.resolution * np.arange(self.shape[1], dtype=np.float64)

    @property
    def y(self) -> FloatArray:
        """Cell-centre y coordinates [m], shape ``(ny,)``."""
        return self.y0 + self.resolution * np.arange(self.shape[0], dtype=np.float64)

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """``(x_min, x_max, y_min, y_max)`` of cell centres [m]."""
        ny, nx = self.shape
        return (
            self.x0,
            self.x0 + (nx - 1) * self.resolution,
            self.y0,
            self.y0 + (ny - 1) * self.resolution,
        )

    def contains(self, x: npt.ArrayLike, y: npt.ArrayLike) -> npt.NDArray[np.bool_]:
        """True where ``(x, y)`` [m] lies inside the grid (inclusive of edges)."""
        xmin, xmax, ymin, ymax = self.extent
        xa = np.asarray(x, dtype=np.float64)
        ya = np.asarray(y, dtype=np.float64)
        return np.asarray((xa >= xmin) & (xa <= xmax) & (ya >= ymin) & (ya <= ymax))

    def height_at(self, x: npt.ArrayLike, y: npt.ArrayLike) -> FloatArray:
        """Bilinearly interpolated elevation [m] at world ``(x, y)`` [m].

        Points outside the grid are clamped to the border (use :meth:`contains`
        to detect them).
        """
        xa = np.asarray(x, dtype=np.float64)
        ya = np.asarray(y, dtype=np.float64)
        ny, nx = self.shape
        u = np.clip((xa - self.x0) / self.resolution, 0.0, nx - 1.0)
        v = np.clip((ya - self.y0) / self.resolution, 0.0, ny - 1.0)
        j0 = np.minimum(np.floor(u).astype(np.intp), nx - 2)
        i0 = np.minimum(np.floor(v).astype(np.intp), ny - 2)
        fu = u - j0
        fv = v - i0
        z = self.z
        z00 = z[i0, j0]
        z01 = z[i0, j0 + 1]
        z10 = z[i0 + 1, j0]
        z11 = z[i0 + 1, j0 + 1]
        out = (z00 * (1 - fu) + z01 * fu) * (1 - fv) + (z10 * (1 - fu) + z11 * fu) * fv
        return np.asarray(out, dtype=np.float64)

    def height_at_mesh(self, x: npt.ArrayLike, y: npt.ArrayLike) -> FloatArray:
        """Elevation [m] on the triangulated surface used by the ray caster.

        Each quad is split along its (i, j)-(i+1, j+1) diagonal. Points outside the
        grid are clamped to the border.
        """
        xa = np.asarray(x, dtype=np.float64)
        ya = np.asarray(y, dtype=np.float64)
        ny, nx = self.shape
        u = np.clip((xa - self.x0) / self.resolution, 0.0, nx - 1.0)
        v = np.clip((ya - self.y0) / self.resolution, 0.0, ny - 1.0)
        j = np.minimum(np.floor(u).astype(np.intp), nx - 2)
        i = np.minimum(np.floor(v).astype(np.intp), ny - 2)
        fu = u - j
        fv = v - i
        z = self.z
        z00 = z[i, j]
        zx = z[i, j + 1]
        zy = z[i + 1, j]
        z11 = z[i + 1, j + 1]
        lower = z00 + fu * (zx - z00) + fv * (z11 - zx)  # triangle (00, x+, 11), fu >= fv
        upper = z00 + fv * (zy - z00) + fu * (z11 - zy)  # triangle (00, 11, y+), fv > fu
        return np.asarray(np.where(fu >= fv, lower, upper), dtype=np.float64)

    def gradient(self) -> tuple[FloatArray, FloatArray]:
        """``(dz/dx, dz/dy)`` [m/m] by central differences (one-sided at the border)."""
        dzdy, dzdx = np.gradient(self.z, self.resolution)
        return np.asarray(dzdx, dtype=np.float64), np.asarray(dzdy, dtype=np.float64)

    def normals(self) -> FloatArray:
        """Unit surface normals in ENU, shape ``(ny, nx, 3)``."""
        gx, gy = self.gradient()
        n = np.stack([-gx, -gy, np.ones_like(gx)], axis=-1)
        n /= np.linalg.norm(n, axis=-1, keepdims=True)
        return n

    def slope(self) -> FloatArray:
        """Surface slope angle [rad], shape ``(ny, nx)``."""
        gx, gy = self.gradient()
        return np.asarray(np.arctan(np.hypot(gx, gy)), dtype=np.float64)

    def aspect(self) -> FloatArray:
        """Downhill direction as azimuth from North, clockwise [rad] in ``[0, 2pi)``.

        Undefined (returned as 0) where the surface is exactly flat.
        """
        gx, gy = self.gradient()
        return np.asarray(np.mod(np.arctan2(-gx, -gy), 2.0 * np.pi), dtype=np.float64)
