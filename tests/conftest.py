from __future__ import annotations

import os

import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if os.environ.get("LUNARSIM_PERF") == "1":
        return
    skip = pytest.mark.skip(reason="set LUNARSIM_PERF=1 to run performance benchmarks")
    for item in items:
        if "perf" in item.keywords:
            item.add_marker(skip)
