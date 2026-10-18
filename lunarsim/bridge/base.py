"""Bridge interface for middleware such as ROS 2 (future extension, no implementation).

A bridge adapts :class:`lunarsim.core.world.World` (or the REST/WebSocket API) to a
middleware. Suggested ROS 2 mapping (REP 103/105 frames: ``map``/``odom`` = world ENU,
``base_link`` = body FLU, ``<camera>_optical_frame`` = OpenCV):

* ``/clock`` (rosgraph_msgs/Clock) from ``World.time_s`` (simulated time).
* ``/tf`` world -> base_link from the rover pose; static mounts on ``/tf_static``.
* ``/<camera>/image_raw`` (sensor_msgs/Image, mono16) + ``camera_info`` (K, D plumb_bob).
* ``/<lidar>/points`` (sensor_msgs/PointCloud2: x, y, z, intensity, ring, time).
* ``/ground_truth/odom`` (nav_msgs/Odometry) from the true pose.
* ``/cmd_vel`` (geometry_msgs/Twist) -> ``World.set_rover_command(v, omega)``.

Implementations must keep the simulator deterministic: advance time only through
``World.step`` driven by the bridge, never from wall-clock callbacks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

__all__ = ["SimBridge", "TopicSpec"]


@dataclass(frozen=True)
class TopicSpec:
    """A published or subscribed topic."""

    name: str
    message_type: str
    frame_id: str
    rate_hz: float | None = None


class SimBridge(Protocol):
    """Lifecycle of a middleware bridge around a world."""

    def topics(self) -> list[TopicSpec]:
        """Topics this bridge publishes/subscribes."""
        ...

    def on_command(self, v_m_s: float, omega_rad_s: float) -> None:
        """Handle an incoming velocity command."""
        ...

    def publish(self) -> None:
        """Publish the current world state and sensor data."""
        ...

    def spin_once(self, dt_s: float) -> None:
        """Advance the simulation by ``dt_s`` and publish."""
        ...
