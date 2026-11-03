"""In-memory sliding-window limiter for sign-in attempts (per account and per client address)."""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque


class LoginLimiter:
    def __init__(self, per_user: int = 5, per_ip: int = 20, window_s: int = 900):
        self.per_user, self.per_ip, self.window_s = per_user, per_ip, window_s
        self._fails: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def _prune(self, key: str, now: float) -> deque[float]:
        q = self._fails[key]
        while q and now - q[0] > self.window_s:
            q.popleft()
        return q

    def retry_after(self, username: str, ip: str | None) -> int:
        """Seconds until another attempt is allowed (0 = allowed now)."""
        now = time.monotonic()
        with self._lock:
            waits = []
            for key, limit in ((f"u:{username.lower()}", self.per_user), (f"i:{ip}", self.per_ip)):
                q = self._prune(key, now)
                if len(q) >= limit:
                    waits.append(int(self.window_s - (now - q[0])) + 1)
            return max(waits, default=0)

    def failed(self, username: str, ip: str | None) -> None:
        now = time.monotonic()
        with self._lock:
            for key in (f"u:{username.lower()}", f"i:{ip}"):
                self._prune(key, now).append(now)

    def succeeded(self, username: str) -> None:
        with self._lock:
            self._fails.pop(f"u:{username.lower()}", None)
