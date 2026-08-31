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
    """Tworzy backend: `sim`, `lerobot`, `feetech` albo `auto` (co jest dostepne)."""
    backend = cfg.robot.backend.lower()

    if backend == "sim":
        return SimulatedArm(cfg)

    if backend == "lerobot":
        from .lerobot_backend import LeRobotArm

        return LeRobotArm(cfg)

    if backend == "feetech":
        from .feetech import FeetechArm

        return FeetechArm(cfg)

    if backend == "auto":
        if not cfg.robot.port:
            logger.info("Tryb auto: uruchamiam symulator (nie podano portu).")
            return SimulatedArm(cfg)
        if lerobot_available():
            from .lerobot_backend import LeRobotArm

            return LeRobotArm(cfg)
        # Bez LeRobota zostaje rozmowa wprost z serwami - do teleoperacji
        # wystarczy, a nie ciagnie za soba torcha.
        logger.info("Tryb auto: brak biblioteki lerobot, uzywam backendu `feetech`.")
        from .feetech import FeetechArm

        return FeetechArm(cfg)

    raise ValueError(
        f"Nieznany backend robota: {cfg.robot.backend!r} (sim|lerobot|feetech|auto)"
    )
