"""Download all wheels needed for an offline install into ``wheelhouse/<tag>/``.

Run on a networked machine (any OS), then copy the directory to the closed
network and install with::

    pip install --no-index --find-links wheelhouse/<tag> -r requirements.lock
    pip install --no-index --find-links wheelhouse/<tag> -e .

Example::

    python scripts/build_wheelhouse.py --python-version 3.10 --platform manylinux2014_x86_64
    python scripts/build_wheelhouse.py --python-version 3.12 --platform win_amd64
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--lock", default=str(ROOT / "requirements.lock"))
    ap.add_argument(
        "--python-version", default=f"{sys.version_info.major}.{sys.version_info.minor}"
    )
    ap.add_argument(
        "--platform",
        action="append",
        help="pip platform tag(s), e.g. manylinux2014_x86_64, manylinux_2_17_x86_64, win_amd64. "
        "Omit to target the current machine.",
    )
    ap.add_argument("--dest", default=None)
    args = ap.parse_args()

    tag = f"py{args.python_version.replace('.', '')}-{'_'.join(args.platform or ['native'])}"
    dest = Path(args.dest or ROOT / "wheelhouse" / tag)
    dest.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "pip", "download", "--only-binary=:all:", "-d", str(dest)]
    if args.platform:
        cmd += ["--python-version", args.python_version, "--implementation", "cp"]
        for p in args.platform:
            cmd += ["--platform", p]
    for reqs in (["-r", args.lock], ["setuptools>=64", "wheel"]):
        print("+", " ".join(cmd + reqs))
        subprocess.run(cmd + reqs, check=True)
    print(f"wheelhouse ready: {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
