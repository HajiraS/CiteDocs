"""Simple in-memory sliding-window rate limiter (per process, per client key)."""

import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable

_SWEEP_THRESHOLD = 1000  # prune idle keys once this many clients are tracked


class RateLimiter:
    def __init__(
        self,
        limit: int,
        window_seconds: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._limit = limit
        self._window = window_seconds
        self._clock = clock
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()  # sync endpoints run in a thread pool

    def check(self, key: str) -> float:
        """Record a request for `key`. Returns 0.0 if allowed, else seconds until a slot frees."""
        with self._lock:
            now = self._clock()
            hits = self._hits[key]
            while hits and hits[0] <= now - self._window:
                hits.popleft()
            if len(hits) >= self._limit:
                return hits[0] + self._window - now
            hits.append(now)
            if len(self._hits) > _SWEEP_THRESHOLD:
                self._sweep(now)
            return 0.0

    def _sweep(self, now: float) -> None:
        """Drop clients whose newest request has left the window, so memory stays bounded."""
        idle = [k for k, h in self._hits.items() if not h or h[-1] <= now - self._window]
        for key in idle:
            del self._hits[key]