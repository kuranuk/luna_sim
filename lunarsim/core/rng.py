"""Deterministic, name-addressed random number streams.

Every stochastic component draws from its own :class:`numpy.random.Generator`
derived from the world seed and a *stream name* (e.g. ``"terrain/fractal"`` or
``"sensor/lidar/front/frame/000123"``). Streams are independent of each other and
of the order in which they are requested, so

* changing the crater configuration does not move the rocks, and
* rendering frame 12 twice (or out of order) gives identical noise.

The global ``np.random`` state is never used (enforced by ruff NPY002).

Reproducibility note: NumPy guarantees the bit stream of ``PCG64`` for a given
``SeedSequence``; the *distribution* methods of ``Generator`` may change between
NumPy feature releases. Bit-exact reproduction therefore assumes the versions in
``requirements.lock``.
"""

from __future__ import annotations

import hashlib

import numpy as np

__all__ = ["RngFactory", "stream_key"]


def stream_key(name: str) -> tuple[int, ...]:
    """Map a stream name to a stable spawn key (4 x uint32 from SHA-256).

    Python's built-in ``hash`` is salted per process, so it must not be used here.
    """
    if not name:
        raise ValueError("stream name must be a non-empty string")
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    return tuple(int.from_bytes(digest[i : i + 4], "little") for i in range(0, 16, 4))


class RngFactory:
    """Factory of independent generators keyed by name, rooted at one integer seed."""

    def __init__(self, seed: int) -> None:
        if seed < 0:
            raise ValueError("seed must be a non-negative integer")
        self._seed = int(seed)

    @property
    def seed(self) -> int:
        """Root seed."""
        return self._seed

    def seed_sequence(self, name: str) -> np.random.SeedSequence:
        """Return the :class:`~numpy.random.SeedSequence` of stream ``name``."""
        return np.random.SeedSequence(entropy=self._seed, spawn_key=stream_key(name))

    def generator(self, name: str) -> np.random.Generator:
        """Return a fresh generator for stream ``name`` (same name -> same stream)."""
        return np.random.Generator(np.random.PCG64(self.seed_sequence(name)))

    def __repr__(self) -> str:
        return f"RngFactory(seed={self._seed})"
