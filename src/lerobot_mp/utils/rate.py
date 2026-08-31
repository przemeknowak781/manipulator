"""Pomiar i utrzymywanie tempa petli sterowania."""

from __future__ import annotations

import time
from collections import deque


class FpsMeter:
    """Srednia krocząca liczby klatek na sekunde."""

    def __init__(self, window: int = 30):
        self._times: deque[float] = deque(maxlen=window)
        self._last: float | None = None

    def tick(self, now: float | None = None) -> float:
        now = time.monotonic() if now is None else now
        if self._last is not None:
            delta = now - self._last
            if delta > 0:
                self._times.append(delta)
        self._last = now
        return self.fps

    @property
    def fps(self) -> float:
        if not self._times:
            return 0.0
        mean = sum(self._times) / len(self._times)
        return 1.0 / mean if mean > 0 else 0.0


class LoopRate:
    """Utrzymuje zadana czestotliwosc petli bez dryfu."""

    def __init__(self, hz: float):
        if hz <= 0:
            raise ValueError("Czestotliwosc petli musi byc dodatnia")
        self.period = 1.0 / hz
        self._next: float | None = None

    def sleep(self) -> None:
        now = time.monotonic()
        if self._next is None:
            self._next = now + self.period
            return
        remaining = self._next - now
        if remaining > 0:
            time.sleep(remaining)
            self._next += self.period
        else:
            # Spoznilismy sie - resetujemy harmonogram, zeby nie "gonic" petli.
            self._next = now + self.period
