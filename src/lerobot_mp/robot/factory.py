"""Wybor backendu robota na podstawie konfiguracji."""

from __future__ import annotations

import importlib.util
import logging

from ..config import AppConfig
from .base import RobotBackend
from .sim import SimulatedArm

logger = logging.getLogger(__name__)


def lerobot_available() -> bool:
    return importlib.util.find_spec("lerobot") is not None


def create_backend(cfg: AppConfig) -> RobotBackend:
    """Tworzy backend: `sim`, `lerobot` albo `auto` (sprzet, jesli jest dostepny)."""
    backend = cfg.robot.backend.lower()

    if backend == "sim":
        return SimulatedArm(cfg)

    if backend == "lerobot":
        from .lerobot_backend import LeRobotArm

        return LeRobotArm(cfg)

    if backend == "auto":
        if cfg.robot.port and lerobot_available():
            from .lerobot_backend import LeRobotArm

            return LeRobotArm(cfg)
        reason = "nie podano portu" if not cfg.robot.port else "brak biblioteki lerobot"
        logger.info("Tryb auto: uruchamiam symulator (%s).", reason)
        return SimulatedArm(cfg)

    raise ValueError(f"Nieznany backend robota: {cfg.robot.backend!r} (sim|lerobot|auto)")
