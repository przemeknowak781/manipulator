"""Symulator ramienia - pozwala uzywac aplikacji bez podlaczonego robota.

Model jest celowo prosty: kazdy staw dazy do zadanej pozycji z ograniczona
predkoscia i pewna bezwladnoscia (czlon inercyjny pierwszego rzedu). To
wystarcza, zeby zobaczyc, czy mapowanie gestow jest wygodne, i zeby przetestowac
cala petle sterowania bez ryzyka uszkodzenia sprzetu.
"""

from __future__ import annotations

import logging

from ..config import AppConfig, JOINT_NAMES
from .base import RobotBackend, RobotInfo

logger = logging.getLogger(__name__)


class SimulatedArm(RobotBackend):
    """Wirtualne ramie o dynamice zblizonej do serw STS3215."""

    def __init__(self, cfg: AppConfig, response_hz: float = 6.0):
        self.cfg = cfg
        #: Szybkosc nadazania za celem [Hz]; wieksza = sztywniejsze serwo.
        self.response_hz = response_hz
        self.info = RobotInfo(
            name="symulator",
            description="wirtualne ramie SO-101 (bez sprzetu)",
            simulated=True,
        )
        self._connected = False
        self._position: dict[str, float] = {}
        self._target: dict[str, float] = {}

    def connect(self) -> None:
        start = self.cfg.safety.home
        self._position = {}
        for name in JOINT_NAMES:
            jc = self.cfg.joint(name)
            value = min(max(float(start.get(name, 0.0)), jc.min), jc.max)
            self._position[name] = value
        self._target = dict(self._position)
        self._connected = True
        logger.info("Symulator ramienia gotowy (bez sprzetu).")

    def disconnect(self) -> None:
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def read_joints(self) -> dict[str, float]:
        return dict(self._position)

    def send_joints(self, targets: dict[str, float]) -> dict[str, float]:
        sent = {}
        for name in JOINT_NAMES:
            if name not in targets:
                continue
            jc = self.cfg.joint(name)
            value = min(max(float(targets[name]), jc.min), jc.max)
            self._target[name] = value
            sent[name] = value
        return sent

    def step(self, dt: float) -> None:
        """Przyblizenie dynamiki serwa: wykladnicze dazenie do celu z limitem predkosci."""
        if dt <= 0.0:
            return
        # Wspolczynnik z ciaglego czlonu inercyjnego - niezalezny od kroku czasu.
        alpha = 1.0 - pow(2.718281828459045, -self.response_hz * dt)
        for name, target in self._target.items():
            jc = self.cfg.joint(name)
            current = self._position[name]
            step = (target - current) * alpha
            max_step = jc.max_vel * dt
            if step > max_step:
                step = max_step
            elif step < -max_step:
                step = -max_step
            self._position[name] = current + step
