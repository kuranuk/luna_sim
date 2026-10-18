"""FastAPI server (thin wrapper around :mod:`lunarsim.core`; requires the ``server`` extra)."""

from lunarsim.api.app import create_app
from lunarsim.api.settings import ServerSettings

__all__ = ["ServerSettings", "create_app"]
