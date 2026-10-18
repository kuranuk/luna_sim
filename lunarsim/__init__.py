"""LunarSim: a lightweight, offline, CPU-only lunar rover simulator.

``lunarsim.core`` is a pure library (no web framework dependency). The FastAPI
server, client and viewer live in separate subpackages and are optional.
"""

from lunarsim.version import __version__

__all__ = ["__version__"]
