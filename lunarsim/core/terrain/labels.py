"""Semantic label classes for terrain cells and rendered pixels (uint8)."""

from __future__ import annotations

from enum import IntEnum

__all__ = ["Label"]


class Label(IntEnum):
    """Semantic classes. ``SHADOW`` is assigned by the illumination stage (Phase 3)."""

    PLAIN = 0
    CRATER_INTERIOR = 1
    CRATER_RIM = 2
    ROCK = 3
    SHADOW = 4
    NOISE = 254  # false LiDAR return (no surface at that range)
    SKY = 255  # no surface (rendered images only)
