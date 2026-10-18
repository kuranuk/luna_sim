from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from lunarsim.config import TerrainConfig, WorldConfig, config_hash, load_config, parse_config

CONFIGS = sorted((Path(__file__).parents[1] / "configs").glob("*.yaml"))


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_shipped_configs_validate(path: Path) -> None:
    if path.name.startswith("dataset_"):  # dataset jobs reference a world scenario
        from lunarsim.datasets import load_dataset_config

        _, scenario = load_dataset_config(path)
        load_config(scenario)
    else:
        load_config(path)


def test_unknown_key_rejected() -> None:
    with pytest.raises(ValidationError):
        parse_config({"terrain": {"resolution": 0.1}})


def test_bounds() -> None:
    with pytest.raises(ValidationError):
        TerrainConfig(resolution_m=-1.0)
    with pytest.raises(ValidationError):
        TerrainConfig(size_x_m=2000, size_y_m=2000, resolution_m=0.05)  # 1.6e9 cells
    with pytest.raises(ValidationError):
        TerrainConfig(size_x_m=10, size_y_m=10, resolution_m=0.25, fractal={"baseline_m": 8.0})


def test_shape() -> None:
    assert TerrainConfig(size_x_m=100, size_y_m=50, resolution_m=0.5).shape == (101, 201)


def test_seed_override_and_hash() -> None:
    a = parse_config({"seed": 1})
    b = parse_config({"seed": 1}, seed=2)
    assert b.seed == 2
    assert config_hash(a) == config_hash(WorldConfig(seed=1))
    assert config_hash(a) != config_hash(b)


def test_non_mapping_yaml(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("- 1\n- 2\n")
    with pytest.raises(ValueError):
        load_config(p)
