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
import time
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

    def rgb_t(self) -> tuple[np.ndarray | None, float, str | None]:
        """(kadr RGB, chwila jego wykonania, blad strumienia).

        Watek `CameraStream` konczy sie po cichu, gdy `cap.read()` zawiedzie (na
        Shadow: zerwane przekazanie USB), a `read()` dalej oddaje ostatnia klatke -
        bez sprawdzenia bledu i wieku panel mial "kadr" zamrozony na zawsze.
        """
        err = self.stream.error
        if err is None and not self.stream.is_running:
            err = "watek kamery zatrzymany"
        if err is not None:
            return None, 0.0, err
        frame = self.stream.read()
        if frame is None:
            return None, 0.0, None
        return cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB), float(frame.timestamp), None

    def rgb(self) -> np.ndarray | None:
        return self.rgb_t()[0]

    def close(self) -> None:
        self.stream.close()


class CameraHub:
    """Otwarte kamery stanowiska.

    `sim_render(nazwa)` podaje kadr kamery symulowanej ze sceny blizniaka;
    bez niego kamery o zrodle "sim" po prostu nie maja obrazu.

    `frame_t` oddaje kadr razem z chwila jego wykonania (`time.monotonic`), a
    `frame` nie oddaje kadru starszego niz `stale_after` [s]: zamrozona kamera
    (strumien stanal, klatka ta sama) dawala fali kalibracyjnej i percepcji
    kostki ten sam obraz jako nowy.
    """

    def __init__(self, workspace: Workspace, sim_render: Callable[[str], np.ndarray] | None = None,
                 stale_after: float = 1.0):
        self.workspace = workspace
        self.sim_render = sim_render
        self.stale_after = stale_after
        self._live: dict[str, _Live] = {}
        self._errors: dict[str, str] = {}
        self._stream_errors: dict[str, str] = {}
        self._lock = threading.Lock()

    def sync(self) -> None:
        """Otwiera wlaczone kamery prawdziwe, zamyka usuniete i wylaczone."""
        wanted = {c.name: c for c in self.workspace.cameras if c.enabled and c.source != "sim"}
        with self._lock:
            for name in list(self._live):
                if name not in wanted:
                    self._live.pop(name).close()
                    self._stream_errors.pop(name, None)
            for name, rec in wanted.items():
                live = self._live.get(name)
                if live is not None:
                    if live.stream.error is None and live.stream.is_running:
                        continue
                    live.close()                          # strumien padl - sync otwiera go od nowa
                    self._live.pop(name)
                    self._stream_errors.pop(name, None)
                try:
                    self._live[name] = _Live(rec)
                    self._errors.pop(name, None)
                except Exception as exc:                  # brak urzadzenia, zajete, zly indeks
                    self._errors[name] = str(exc)
                    logger.warning("Kamera %s (%s) nie otworzyla sie: %s", name, rec.source, exc)

    def error(self, name: str) -> str | None:
        """Blad otwarcia kamery albo jej strumienia (zerwany, zamrozony)."""
        return self._errors.get(name) or self._stream_errors.get(name)

    def frame_t(self, name: str) -> tuple[np.ndarray | None, float]:
        """(najnowszy kadr RGB albo None, chwila jego wykonania wg `time.monotonic`; 0.0 = nie wiadomo).

        Kamera symulowana renderuje stan sceny z tej chwili - jej kadr jest zawsze swiezy.
        """
        rec = self.workspace.camera(name)
        if rec.source == "sim":
            if not (self.sim_render and rec.true_pose() is not None):
                return None, 0.0
            t = time.monotonic()
            return self.sim_render(name), t
        with self._lock:
            live = self._live.get(name)
        if live is None:
            return None, 0.0
        img, t, err = live.rgb_t()
        if err is not None:
            self._stream_errors[name] = f"strumien przerwany: {err}"
        elif img is not None and self.stale_after and time.monotonic() - t > self.stale_after:
            self._stream_errors[name] = f"brak nowych klatek od {time.monotonic() - t:.1f} s"
        else:
            self._stream_errors.pop(name, None)
        return img, t

    def frame(self, name: str) -> np.ndarray | None:
        """Najnowszy kadr RGB albo None - takze wtedy, gdy jest starszy niz `stale_after`."""
        img, t = self.frame_t(name)
        if img is not None and self.stale_after and t and time.monotonic() - t > self.stale_after:
            return None
        return img

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
