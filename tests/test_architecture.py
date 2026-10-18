"""Structural rules: core must stay framework-free and network-free."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_core_imports_without_web_stack() -> None:
    code = (
        "import sys, lunarsim.core.world, lunarsim.io.raster, lunarsim.io.preview;"
        "bad = [m for m in ('fastapi','uvicorn','starlette','requests','httpx','urllib3',"
        "'open3d') if m in sys.modules];"
        "print(','.join(bad))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == ""


def test_no_global_numpy_random_or_pickle_in_sources() -> None:
    offenders = []
    for p in (ROOT / "lunarsim").rglob("*.py"):
        text = p.read_text(encoding="utf-8")
        for needle in (
            "np.random.seed(",
            "np.random.rand(",
            "np.random.normal(",
            "np.random.uniform(",
            "import pickle",
            "allow_pickle=True",
        ):
            if needle in text:
                offenders.append(f"{p.relative_to(ROOT)}: {needle}")
    assert offenders == []
