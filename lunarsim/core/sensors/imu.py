"""6-axis IMU (gyroscope + accelerometer) driven by the rover trajectory (SPEC 5.3).

Truth at sample k (period h = 1 / rate, sensor frame S, world ENU W):

* angular rate ``ω_S = log(R_WS,k-1ᵀ R_WS,k) / h + R_WS,kᵀ Ω_moon``, with the Moon's
  sidereal rotation ``Ω_moon = Ω (0, cos φ, sin φ)`` in the ENU frame of the site at
  latitude φ;
* specific force ``f_S = R_WS,kᵀ (a_W - g_W)`` with ``g_W = (0, 0, -g)`` and
  ``a_W = (p_k - 2 p_{k-1} + p_{k-2}) / h²`` from the sensor positions (lever arm
  included). Backward differences: the truth lags the pose by up to one period.

Optionally both are low-passed with ``filter_order`` cascaded first-order stages at
``bandwidth_hz`` (the rover
contact is rigid and static, so raw second differences contain contact-switching
spikes; the filter stands in for suspension and the sensor's digital filter).

Errors (Kalibr / EuRoC convention): ``y = truth + b + σ / sqrt(h) · n`` and
``b_k+1 = b_k + σ_rw · sqrt(h) · n'``, turn-on bias ~ N(0, σ_b). Noise comes in blocks of
``BLOCK`` samples from streams ``sensor/imu/{name}/block/{b:06d}``, so the samples depend
only on the sample index, not on how the simulation time is split into steps.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from lunarsim.config.schema import ImuConfig
from lunarsim.core.frames import Pose, rotvec_from_matrix
from lunarsim.core.rng import RngFactory

__all__ = ["ImuSamples", "ImuSimulator"]

FloatArray = npt.NDArray[np.float64]
BLOCK = 1024


@dataclass(frozen=True)
class ImuSamples:
    """A batch of IMU samples (all arrays share the first dimension N).

    Attributes:
        t: Sample times [s].
        gyro: Measured angular rate [rad/s], ``(N, 3)`` (sensor frame FLU).
        accel: Measured specific force [m/s²], ``(N, 3)``.
        gyro_true: Noise-free angular rate [rad/s].
        accel_true: Noise-free specific force [m/s²].
        gyro_bias: Gyro bias [rad/s].
        accel_bias: Accelerometer bias [m/s²].
    """

    t: FloatArray
    gyro: FloatArray
    accel: FloatArray
    gyro_true: FloatArray
    accel_true: FloatArray
    gyro_bias: FloatArray
    accel_bias: FloatArray

    def __len__(self) -> int:
        return int(self.t.shape[0])

    def to_dict(self) -> dict[str, list[float] | list[list[float]]]:
        """JSON-friendly representation."""
        return {k: getattr(self, k).tolist() for k in self.__dataclass_fields__}


class ImuSimulator:
    """Generates IMU samples from sensor poses pushed at the sample instants."""

    def __init__(self, cfg: ImuConfig, rngs: RngFactory, latitude_deg: float) -> None:
        self.cfg = cfg
        self.rngs = rngs
        self.period = 1.0 / cfg.rate_hz
        la = math.radians(latitude_deg)
        self._omega_moon = cfg.moon_rotation_rad_s * np.array([0.0, math.cos(la), math.sin(la)])
        self._buf: deque[FloatArray] = deque(maxlen=max(1, int(cfg.buffer_s * cfg.rate_hz)))
        self.reset()

    def reset(self) -> None:
        """Clear history, buffer and filter state; draw a new turn-on bias (same seed)."""
        self.k = 0
        self._pos: list[FloatArray] = []
        self._R_prev: FloatArray | None = None
        self._filt: list[FloatArray] | None = None
        self._buf.clear()
        n = self.cfg.noise
        z = self.rngs.generator(f"sensor/imu/{self.cfg.name}/bias").standard_normal(6)
        self._bg = n.gyro_turn_on_bias * z[:3]
        self._ba = n.accel_turn_on_bias * z[3:]
        self._block_index = -1
        self._block: FloatArray = np.zeros((BLOCK, 12))

    @property
    def next_time(self) -> float:
        """Time [s] of the next sample."""
        return self.k * self.period

    def _noise(self, k: int) -> FloatArray:
        b = k // BLOCK
        if b != self._block_index:
            g = self.rngs.generator(f"sensor/imu/{self.cfg.name}/block/{b:06d}")
            self._block = np.asarray(g.standard_normal((BLOCK, 12)), dtype=np.float64)
            self._block_index = b
        return np.asarray(self._block[k % BLOCK], dtype=np.float64)

    def push(self, T_world_sensor: Pose) -> FloatArray:
        """Record the sensor pose at ``next_time`` and return the sample row.

        Row layout: ``t, gyro(3), accel(3), gyro_true(3), accel_true(3), bg(3), ba(3)``.
        """
        h, c = self.period, self.cfg
        p, R = T_world_sensor.t.copy(), T_world_sensor.R
        self._pos.append(p)
        if len(self._pos) > 3:
            self._pos.pop(0)
        while len(self._pos) < 3:  # at rest before the first sample
            self._pos.insert(0, p)
        p2, p1, p0 = self._pos[-1], self._pos[-2], self._pos[-3]
        a_w = (p2 - 2.0 * p1 + p0) / (h * h)
        f_s = R.T @ (a_w + np.array([0.0, 0.0, c.gravity_m_s2]))
        w_rel = np.zeros(3) if self._R_prev is None else rotvec_from_matrix(self._R_prev.T @ R) / h
        w_s = w_rel + R.T @ self._omega_moon
        self._R_prev = R.copy()
        truth = np.concatenate([w_s, f_s])
        if c.bandwidth_hz is not None:
            if self._filt is None:
                self._filt = [truth.copy() for _ in range(c.filter_order)]
            else:
                alpha = 1.0 - math.exp(-2.0 * math.pi * c.bandwidth_hz * h)
                x = truth
                for i in range(c.filter_order):
                    self._filt[i] = self._filt[i] + alpha * (x - self._filt[i])
                    x = self._filt[i]
            truth = self._filt[-1]
        nz = c.noise
        n = self._noise(self.k)
        sq = math.sqrt(h)
        gyro = truth[:3] + self._bg + nz.gyro_noise_density / sq * n[0:3]
        accel = truth[3:] + self._ba + nz.accel_noise_density / sq * n[3:6]
        row: FloatArray = np.concatenate([[self.next_time], gyro, accel, truth, self._bg, self._ba])
        self._bg = self._bg + nz.gyro_random_walk * sq * n[6:9]
        self._ba = self._ba + nz.accel_random_walk * sq * n[9:12]
        self._buf.append(row)
        self.k += 1
        return row

    def samples(self, since_s: float | None = None) -> ImuSamples:
        """Buffered samples with ``t > since_s`` (all buffered samples if None)."""
        rows: FloatArray = np.array(self._buf, dtype=np.float64).reshape(-1, 19)
        if since_s is not None:
            rows = np.asarray(rows[rows[:, 0] > since_s + 1e-12], dtype=np.float64)
        return ImuSamples(
            rows[:, 0], rows[:, 1:4], rows[:, 4:7], rows[:, 7:10], rows[:, 10:13],
            rows[:, 13:16], rows[:, 16:19],
        )  # fmt: skip
