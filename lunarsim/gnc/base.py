"""Base class for GNC algorithms connected to the browser simulator."""

from __future__ import annotations

from .types import Command, Observation, SimInfo


class Gnc:
    """Subclass and override :meth:`reset` and :meth:`step`.

    The simulator runs in lockstep: it advances one control period (``info.control_period`` [s]),
    sends the measurements as an :class:`Observation`, and waits for the :class:`Command`, so the
    run is deterministic regardless of how long ``step`` takes. Ground truth is never passed to
    the algorithm; it is only logged (``lunarsim gnc --log-dir``).
    """

    name = "gnc"

    def reset(self, info: SimInfo) -> None:
        """Called on connect and whenever the simulator is reset or the terrain is rebuilt."""
        self.info = info

    def step(self, obs: Observation) -> Command:
        """Return the command for the next control period."""
        raise NotImplementedError
