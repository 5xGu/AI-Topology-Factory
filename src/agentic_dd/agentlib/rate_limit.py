from __future__ import annotations
import time
import threading
from collections import deque


class RateLimiter:
    ''' Thread-safe sliding-window rate limiter: blocks the calling thread
    until making a call would not exceed max_calls within window_s. Shared
    across all threads via a module-level singleton (see get_gwdg_limiter),
    so concurrent LangGraph-dispatched calls are coordinated correctly '''

    def __init__(self, max_calls: int, window_s: float):
        self.max_calls = max_calls
        self.window_s = window_s
        self._calls: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self):
        while True:
            with self._lock:
                now = time.monotonic()
                while self._calls and now - self._calls[0] >= self.window_s:
                    self._calls.popleft()
                if len(self._calls) < self.max_calls:
                    self._calls.append(now)
                    return
                wait = self.window_s - (now - self._calls[0])
            time.sleep(max(wait, 0.05))


# Two windows enforced together: 10/min burst cap AND 200/hour sustained cap.
_gwdg_minute_limiter = RateLimiter(max_calls=9, window_s=60)     # small safety margin below the stated 10/min
_gwdg_hour_limiter = RateLimiter(max_calls=195, window_s=3600)   # small safety margin below the stated 200/hr


def acquire_gwdg_slot():
    _gwdg_hour_limiter.acquire()
    _gwdg_minute_limiter.acquire()