"""Renderowanie podgladu 3D w osobnym watku.

Zlozenie SO-101 to ~22 tys. trojkatow, czyli ok. 30 ms rysowania na klatke.
Robione wprost w petli sterowania zabieralo jej ten czas co druga iteracja:
petla zadana na 30 Hz schodzila do ~20 Hz, a podglad i tak zmienial sie
skokowo, bo odswiezal sie tylko wtedy, gdy petla akurat zdazyla.

Watek renderujacy rozdziela te dwie rzeczy. Petla sterowania zostawia najnowsza
poze i zabiera ostatni gotowy obraz - nigdy nie czekajac na rysowanie. Model
nadaza wiec za ruchem ramienia, a limity predkosci i watchdog tykaja rowno.

Ten sam wzorzec co `vision.camera.CameraStream`: trzymamy *tylko* najswiezsze
zadanie, wiec zaleglosci nie narastaja - gdy render trwa dluzej niz kolejne
zmiany pozy, po prostu pomijamy te posrednie.
"""

from __future__ import annotations

import logging
import threading
import time

import numpy as np

from .model import ArmModel
from .render import Renderer3D

logger = logging.getLogger(__name__)

Pose = dict[str, float]
Target = tuple[float, float, float] | None


class PreviewStream:
    """Nieblokujace zrodlo obrazow podgladu 3D."""

    def __init__(self, renderer: Renderer3D, model: ArmModel, hz: float = 30.0):
        self.renderer = renderer
        self.model = model
        self._period = 1.0 / max(hz, 1.0)

        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        self._pose: Pose = {}
        self._target: Target = None
        self._dirty = False
        # Obroty i przyblizenia z klawiatury przychodza z watku glownego, a
        # kamera nalezy do renderera - kolejkujemy je i stosujemy tam, gdzie
        # sie rysuje, zamiast ruszac kamera spod rak watkowi.
        self._camera_ops: list[tuple[str, float, float]] = []
        self._image: np.ndarray | None = None

    # ---------------------------------------------------------------- setup
    def start(self, pose: Pose, ee_target: Target = None) -> "PreviewStream":
        """Rysuje pierwsza klatke synchronicznie i rusza watek.

        Pierwsza klatka musi byc gotowa, zanim petla zlozy pierwszy kadr:
        inaczej okno (i plik nagrania) zaczelyby bez panelu i zmienily
        szerokosc w trakcie.
        """
        self._pose, self._target = dict(pose), ee_target
        self._image = self.renderer.render(self._pose, ee_target=self._target)

        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="preview", daemon=True)
        self._thread.start()
        return self

    # ----------------------------------------------------------------- uzycie
    def submit(self, pose: Pose, ee_target: Target = None) -> None:
        """Zostawia najnowsza poze do narysowania. Nie blokuje."""
        with self._lock:
            self._pose, self._target = pose, ee_target
            self._dirty = True
        self._wake.set()

    @property
    def image(self) -> np.ndarray | None:
        """Ostatnia gotowa klatka podgladu."""
        with self._lock:
            return self._image

    def orbit(self, d_azimuth: float, d_elevation: float) -> None:
        self._queue_camera(("orbit", d_azimuth, d_elevation))

    def zoom(self, factor: float) -> None:
        self._queue_camera(("zoom", factor, 0.0))

    def _queue_camera(self, op: tuple[str, float, float]) -> None:
        with self._lock:
            self._camera_ops.append(op)
            self._dirty = True
        self._wake.set()

    def close(self) -> None:
        self._stop.set()
        self._wake.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=2.0)

    # ------------------------------------------------------------------ watek
    def _run(self) -> None:
        while not self._stop.is_set():
            # Timeout, a nie czekanie bez konca: gdyby `set()` wypadlo tuz
            # przed `clear()`, watek obudzi sie sam i dokonczy zalegly render.
            self._wake.wait(timeout=0.1)
            self._wake.clear()
            if self._stop.is_set():
                break

            with self._lock:
                if not self._dirty:
                    continue
                pose, target = dict(self._pose), self._target
                ops, self._camera_ops = self._camera_ops, []
                self._dirty = False

            self._apply_camera(ops)

            started = time.monotonic()
            try:
                image = self.renderer.render(pose, ee_target=target)
            except Exception:
                logger.exception("Blad renderowania podgladu 3D")
                continue
            with self._lock:
                self._image = image

            # Limit czestotliwosci liczony od *konca* rysowania: na wolnej
            # maszynie render sam z siebie zejdzie ponizej `hz` i nie ma sensu
            # dokladac mu przerwy.
            remaining = self._period - (time.monotonic() - started)
            if remaining > 0:
                self._stop.wait(remaining)

    def _apply_camera(self, ops: list[tuple[str, float, float]]) -> None:
        camera = self.renderer.camera
        for name, first, second in ops:
            if name == "orbit":
                camera.orbit(first, second)
            else:
                camera.zoom(first)
