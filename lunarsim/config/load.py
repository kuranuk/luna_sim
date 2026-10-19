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


def _resolve(base: Path, value: str) -> str:
    """``value`` relative to the YAML directory if it exists there, else unchanged."""
    p = Path(value)
    if p.is_absolute() or not (base / p).exists():
        return value
    return str((base / p).resolve())


def load_config(path: str | Path, seed: int | None = None) -> WorldConfig:
    """Read a YAML file and validate it as a :class:`WorldConfig`.

    Relative data paths (``terrain.dem.path``, ``sun.spice_kernels``) are resolved
    against the YAML file's directory when the file exists there.
    """
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is not None and not isinstance(data, dict):
        raise ValueError(f"{path}: top level of a scenario file must be a mapping")
    base = Path(path).resolve().parent
    if data:
        dem = (data.get("terrain") or {}).get("dem") or {}
        if isinstance(dem, dict) and isinstance(dem.get("path"), str):
            dem["path"] = _resolve(base, dem["path"])
        sun = data.get("sun") or {}
        if isinstance(sun, dict) and isinstance(sun.get("spice_kernels"), list):
            sun["spice_kernels"] = [
                _resolve(base, k) if isinstance(k, str) else k for k in sun["spice_kernels"]
            ]
    return parse_config(data, seed=seed)
