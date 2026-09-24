"""Kamery stanowiska: wykrywanie, strumienie na zywo i kamery symulowane - jeden interfejs.

`CameraHub.grab()` pasuje do `calib.session.Cameras`, wiec ta sama sesja
kalibracji jedzie na prawdziwych kamerach i na symulowanych.

Kamery prawdziwe uzywaja watku `vision.camera.CameraStream` (zawsze najnowsza
klatka, bez kolejki), z JEDNA roznica: bez lustrzanego odbicia. Aplikacja
sterowania dlonia odbija obraz, zeby ruch reki zgadzal sie z ekranem - dla
kalibracji lustrzany kadr to inna geometria i poza kamery wyszlaby z blednym
znakiem.
"""

from __future__ import annotations

import logging
import sys
import threading
from collections.abc import Callable

import cv2
import numpy as np

from ..config import CameraConfig
from ..vision.camera import CameraStream
from .workspace import CameraRecord, Workspace

logger = logging.getLogger(__name__)

#: Na Windows DirectShow otwiera kamere w ulamku sekundy; domyslny MSMF potrafi
#: kilka sekund, a przy indeksie bez urzadzenia wisi jeszcze dluzej.
_BACKEND = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY


def probe_devices(max_index: int = 6) -> list[dict]:
    """Kamery, ktore naprawde oddaja klatke: [{index, width, height}].

    Samo `isOpened()` nie wystarcza - na maszynach wirtualnych i przy zlym
    sterowniku kamera sie otwiera i zadnej klatki nie oddaje.
    """
    found = []
    for index in range(max_index):
        cap = cv2.VideoCapture(index, _BACKEND)
        try:
            if not cap.isOpened():
                continue
            ok, img = cap.read()
            if ok and img is not None:
                found.append(dict(index=index, width=int(img.shape[1]), height=int(img.shape[0])))
        finally:
            cap.release()
    return found


class _Live:
    def __init__(self, record: CameraRecord):
        cfg = CameraConfig(source=record.source, width=record.width, height=record.height,
                           fps=record.fps, mirror=False)
        self.stream = CameraStream(cfg)
        self.stream.open()

    def rgb(self) -> np.ndarray | None:
        frame = self.stream.read()
        return None if frame is None else cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        self.stream.close()


class CameraHub:
    """Otwarte kamery stanowiska.

    `sim_render(nazwa)` podaje kadr kamery symulowanej ze sceny blizniaka;
    bez niego kamery o zrodle "sim" po prostu nie maja obrazu.
    """

    def __init__(self, workspace: Workspace, sim_render: Callable[[str], np.ndarray] | None = None):
        self.workspace = workspace
        self.sim_render = sim_render
        self._live: dict[str, _Live] = {}
        self._errors: dict[str, str] = {}
        self._lock = threading.Lock()

    def sync(self) -> None:
        """Otwiera wlaczone kamery prawdziwe, zamyka usuniete i wylaczone."""
        wanted = {c.name: c for c in self.workspace.cameras if c.enabled and c.source != "sim"}
        with self._lock:
            for name in list(self._live):
                if name not in wanted:
                    self._live.pop(name).close()
            for name, rec in wanted.items():
                if name in self._live:
                    continue
                try:
                    self._live[name] = _Live(rec)
                    self._errors.pop(name, None)
                except Exception as exc:                  # brak urzadzenia, zajete, zly indeks
                    self._errors[name] = str(exc)
                    logger.warning("Kamera %s (%s) nie otworzyla sie: %s", name, rec.source, exc)

    def error(self, name: str) -> str | None:
        return self._errors.get(name)

    def frame(self, name: str) -> np.ndarray | None:
        """Najnowszy kadr RGB albo None."""
        rec = self.workspace.camera(name)
        if rec.source == "sim":
            return self.sim_render(name) if self.sim_render and rec.true_pose() is not None else None
        with self._lock:
            live = self._live.get(name)
        return live.rgb() if live else None

    def grab(self) -> dict[str, np.ndarray]:
        out = {}
        for rec in self.workspace.cameras:
            if not rec.enabled:
                continue
            img = self.frame(rec.name)
            if img is not None:
                out[rec.name] = img
        return out

    def close(self) -> None:
        with self._lock:
            for live in self._live.values():
                live.close()
            self._live.clear()
