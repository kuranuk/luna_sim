"""In-memory registry of worlds with per-world locks and a small render cache."""

from __future__ import annotations

import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from lunarsim.core.world import World

__all__ = ["Registry", "WorldEntry"]


@dataclass
class WorldEntry:
    """A world plus the lock serialising all access to it (worker threads)."""

    world: World
    lock: threading.Lock = field(default_factory=threading.Lock)
    cache: OrderedDict[tuple[Any, ...], Any] = field(default_factory=OrderedDict)
    cache_size: int = 8

    def cached(self, key: tuple[Any, ...], make: Any) -> Any:
        """Memoise ``make()`` under ``key`` + current world version (call under lock)."""
        k = (*key, self.world.version)
        if k in self.cache:
            self.cache.move_to_end(k)
            return self.cache[k]
        value = make()
        if self.cache_size > 0:
            self.cache[k] = value
            while len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        return value


class Registry:
    """Thread-safe mapping ``world_id -> WorldEntry`` with a capacity limit."""

    def __init__(self, max_worlds: int, cache_size: int = 8) -> None:
        self._worlds: dict[str, WorldEntry] = {}
        self._lock = threading.Lock()
        self.max_worlds = max_worlds
        self.cache_size = cache_size

    def reserve(self) -> None:
        """Raise if no capacity is left (checked before building a world)."""
        with self._lock:
            if len(self._worlds) >= self.max_worlds:
                raise OverflowError(f"world limit reached ({self.max_worlds})")

    def add(self, world: World) -> str:
        """Register a world and return its id."""
        with self._lock:
            if len(self._worlds) >= self.max_worlds:
                raise OverflowError(f"world limit reached ({self.max_worlds})")
            wid = uuid.uuid4().hex
            self._worlds[wid] = WorldEntry(world, cache_size=self.cache_size)
            return wid

    def get(self, wid: str) -> WorldEntry:
        """Entry for ``wid`` (KeyError if unknown)."""
        with self._lock:
            return self._worlds[wid]

    def remove(self, wid: str) -> None:
        """Delete a world (KeyError if unknown)."""
        with self._lock:
            del self._worlds[wid]

    def ids(self) -> list[str]:
        """Registered world ids."""
        with self._lock:
            return list(self._worlds)
