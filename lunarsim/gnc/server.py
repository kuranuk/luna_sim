"""WebSocket server the browser simulator connects to (``lunarsim gnc``).

The browser is the client (pages cannot listen), this process is the server on 127.0.0.1 by
default. Binding to a non-loopback address requires a token (``?token=`` in the URL), as for
``lunarsim serve``. Requires the ``websockets`` package (``pip install lunarsim[server]``).
"""

from __future__ import annotations

import hmac
import importlib
import importlib.util
import ipaddress
import json
import sys
from collections.abc import Callable
from http import HTTPStatus
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from .base import Gnc
from .session import GncSession

DEFAULT_GNC = "lunarsim.gnc.examples:WaypointGnc"
MAX_MESSAGE_BYTES = 64 * 1024 * 1024


def load_gnc_factory(spec: str) -> Callable[[], Gnc]:
    """``package.module:Class`` or ``path/to/file.py:Class`` -> zero-argument factory."""
    target, _, attr = spec.rpartition(":")
    if not target or not attr:
        raise ValueError(f"expected 'module:Class' or 'file.py:Class', got {spec!r}")
    if target.endswith(".py"):
        path = Path(target).resolve()
        mod_spec = importlib.util.spec_from_file_location(path.stem, path)
        if mod_spec is None or mod_spec.loader is None:
            raise ImportError(f"cannot load {path}")
        module = importlib.util.module_from_spec(mod_spec)
        sys.modules[path.stem] = module
        mod_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(target)
    factory = getattr(module, attr)
    if not callable(factory):
        raise TypeError(f"{spec} is not callable")
    return factory  # type: ignore[no-any-return]


def is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def create_server(
    factory: Callable[[], Gnc],
    host: str = "127.0.0.1",
    port: int = 8765,
    token: str | None = None,
    log_dir: Path | None = None,
    log: Callable[[str], None] = print,
) -> Any:
    """Bind the WebSocket server (``port=0`` = any free port); call ``serve_forever()`` on it.

    One :class:`Gnc` instance per browser connection.
    """
    try:
        from websockets.sync.server import ServerConnection, serve
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise SystemExit("lunarsim gnc needs 'websockets' (pip install lunarsim[server])") from e
    if not is_loopback(host) and not token:
        raise SystemExit(f"binding to {host} (not loopback) requires --token / LUNARSIM_TOKEN")

    def check(conn: ServerConnection, request: Any) -> Any:
        if token is None:
            return None
        got = parse_qs(urlsplit(request.path).query).get("token", [""])[0]
        if not hmac.compare_digest(got.encode(), token.encode()):
            return conn.respond(HTTPStatus.UNAUTHORIZED, "invalid token\n")
        return None

    def handler(conn: ServerConnection) -> None:
        origin = conn.request.headers.get("Origin", "-") if conn.request else "-"
        log(f"browser connected (origin {origin})")
        session = GncSession(factory(), log_dir, log)
        try:
            for msg in conn:
                if isinstance(msg, str):
                    reply = session.handle_text(msg)
                else:
                    reply = session.handle_binary(msg)
                if reply is not None:
                    conn.send(json.dumps(reply))
        finally:
            session.close()
            log(f"browser disconnected after {session.steps} steps")

    return serve(
        handler, host, port, process_request=check, max_size=MAX_MESSAGE_BYTES, compression=None
    )


def run(
    factory: Callable[[], Gnc],
    host: str = "127.0.0.1",
    port: int = 8765,
    token: str | None = None,
    log_dir: Path | None = None,
    log: Callable[[str], None] = print,
) -> None:
    """Serve until interrupted (``lunarsim gnc``)."""
    with create_server(factory, host, port, token, log_dir, log) as server:
        url = f"ws://{host}:{port}/" + (f"?token={token}" if token else "")
        log(f"GNC server listening on {url} — press 接続 in the browser simulator (航法誘導)")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            log("stopped")
