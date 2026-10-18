"""Server settings and the exposure safety check."""

from __future__ import annotations

import ipaddress
import os

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["ServerSettings", "is_loopback"]


class ServerSettings(BaseModel):
    """Runtime settings of the API server.

    The default binds to 127.0.0.1 only. Binding to any other address requires a
    token (``Authorization: Bearer <token>``; WebSocket: ``?token=`` or the header)
    unless ``allow_insecure`` is set explicitly.
    """

    model_config = ConfigDict(extra="forbid")

    host: str = "127.0.0.1"
    port: int = Field(8000, ge=1, le=65535)
    token: str | None = None
    allow_insecure: bool = False
    max_worlds: int = Field(4, ge=1)
    max_json_points: int = Field(50_000, ge=1)
    render_cache_size: int = Field(8, ge=0)

    @classmethod
    def from_env(cls, **overrides: object) -> ServerSettings:
        """Settings with ``LUNARSIM_TOKEN`` taken from the environment if not given."""
        data: dict[str, object] = {"token": os.environ.get("LUNARSIM_TOKEN") or None}
        data.update({k: v for k, v in overrides.items() if v is not None})
        return cls.model_validate(data)

    def check_exposure(self) -> None:
        """Refuse a non-loopback bind without a token (unless explicitly insecure)."""
        if not is_loopback(self.host) and not self.token and not self.allow_insecure:
            raise ValueError(
                f"refusing to bind to {self.host!r} without a token: set LUNARSIM_TOKEN / "
                "--token, or pass --allow-insecure explicitly"
            )


def is_loopback(host: str) -> bool:
    """True for localhost / loopback addresses."""
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False
