"""Load and validate YAML scenario files (``yaml.safe_load`` only, never pickle)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from lunarsim.config.schema import WorldConfig

__all__ = ["load_config", "parse_config"]


def parse_config(data: dict[str, Any] | None, seed: int | None = None) -> WorldConfig:
    """Validate a plain mapping. ``seed`` (if given) overrides ``data['seed']``."""
    merged: dict[str, Any] = dict(data or {})
    if seed is not None:
        merged["seed"] = seed
    return WorldConfig.model_validate(merged)


def load_config(path: str | Path, seed: int | None = None) -> WorldConfig:
    """Read a YAML file and validate it as a :class:`WorldConfig`."""
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is not None and not isinstance(data, dict):
        raise ValueError(f"{path}: top level of a scenario file must be a mapping")
    return parse_config(data, seed=seed)
