#!/usr/bin/env bash
# Thin wrapper; see scripts/build_wheelhouse.py --help
set -euo pipefail
exec "${PYTHON:-python3}" "$(dirname "$0")/build_wheelhouse.py" "$@"
