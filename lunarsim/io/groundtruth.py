"""Ground-truth writers: crater/rock catalogs (CSV + JSON) and label rasters."""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from PIL import Image

from lunarsim.core.terrain.labels import Label

__all__ = ["write_catalog", "write_label_png"]

# Fixed palette for label previews (index -> RGB).
LABEL_COLORS: dict[int, tuple[int, int, int]] = {
    Label.PLAIN: (90, 90, 90),
    Label.CRATER_INTERIOR: (40, 110, 200),
    Label.CRATER_RIM: (240, 200, 40),
    Label.ROCK: (220, 50, 50),
    Label.SHADOW: (0, 0, 0),
}


def write_catalog(path_stem: str | Path, records: Sequence[dict[str, Any]]) -> None:
    """Write records as ``<stem>.csv`` and ``<stem>.json``."""
    stem = Path(path_stem)
    stem.with_suffix(".json").write_text(json.dumps(list(records), indent=1), encoding="utf-8")
    with stem.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        if not records:
            return
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)


def write_label_png(
    path: str | Path,
    labels: npt.NDArray[np.uint8],
    *,
    color: bool = False,
    flip_north_up: bool = True,
) -> None:
    """Write labels as raw uint8 class ids (default) or a colour preview."""
    lab = labels[::-1] if flip_north_up else labels
    if color:
        lut = np.zeros((256, 3), dtype=np.uint8)
        for k, rgb in LABEL_COLORS.items():
            lut[int(k)] = rgb
        Image.fromarray(lut[lab]).save(Path(path))
    else:
        Image.fromarray(np.ascontiguousarray(lab)).save(Path(path))
