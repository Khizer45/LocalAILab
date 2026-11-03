"""One shared lock for all measurement jobs.

Benchmark, Model Arena and Agent Arena runs each have their own worker thread,
but they hold this lock while executing so that two jobs never run at the same
time and distort each other's latency or memory measurements. A run waits in
the ``queued`` state until the lock is free.
"""

from __future__ import annotations

import threading


class JobLock:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.holder: str | None = None

    def acquire(self, holder: str, stop: threading.Event | None = None) -> bool:
        """Block until the lock is free (or `stop` is set). Returns True when acquired."""
        while True:
            if self._lock.acquire(timeout=0.5):
                self.holder = holder
                return True
            if stop is not None and stop.is_set():
                return False

    def release(self) -> None:
        self.holder = None
        self._lock.release()

    def status(self) -> dict:
        return {"busy": self._lock.locked(), "holder": self.holder}
