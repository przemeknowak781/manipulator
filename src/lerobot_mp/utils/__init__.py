"""Drobne narzedzia wspolne dla calej aplikacji."""

from .logging import setup_logging
from .rate import FpsMeter, LoopRate

__all__ = ["setup_logging", "FpsMeter", "LoopRate"]
