"""Zrodlo pustych klatek - tlo podgladu, gdy kamera nie jest potrzebna.

Tryb `keys` steruje ramieniem z klawiatury i obrazu nie uzywa, ale okno
podgladu nadal musi na czyms rysowac HUD i panel ramienia. Zamiast obwarowywac
cala petle warunkami "jesli jest kamera", podstawiamy zrodlo o tym samym
interfejsie, ktore oddaje czarne tlo. Reszta aplikacji nie widzi roznicy -
dziala tez nagrywanie do pliku i skalowanie okna.

Klatek nie produkuje watek: nie ma tu na co czekac, wiec `read()` po prostu
zwraca kolejna. Numer klatki i tak rosnie, zeby licznik FPS pokazywal tempo
petli, a nie zero.
"""

from __future__ import annotations

import time

import numpy as np

from ..config import CameraConfig
from .camera import Frame


class BlankSource:
    """Podmiana `CameraStream` na czarne tlo o tych samych wymiarach."""

    def __init__(self, cfg: CameraConfig):
        self.cfg = cfg
        self._image = np.zeros((cfg.height, cfg.width, 3), dtype=np.uint8)
        self._index = 0
        self._open = False

    def open(self) -> "BlankSource":
        self._open = True
        return self

    def __enter__(self) -> "BlankSource":
        return self.open()

    def __exit__(self, *exc: object) -> None:
        self.close()

    def read(self) -> Frame | None:
        if not self._open:
            return None
        self._index += 1
        # Jeden wspolny bufor wystarczy: `_render` i tak zaczyna od `frame.copy()`,
        # wiec nikt nie rysuje po tym tle. Alokowanie 2,7 MB co klatke byloby
        # czysta strata.
        return Frame(image=self._image, timestamp=time.monotonic(), index=self._index)

    def wait_for_frame(self, timeout: float = 5.0) -> Frame:
        frame = self.read()
        if frame is None:
            raise RuntimeError("Zrodlo pustych klatek nie zostalo otwarte.")
        return frame

    @property
    def error(self) -> str | None:
        return None

    @property
    def is_running(self) -> bool:
        return self._open

    def close(self) -> None:
        self._open = False
