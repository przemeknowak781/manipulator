"""Sterowanie ramieniem LeRobot 101 (SO-101) gestami dloni przy uzyciu MediaPipe."""

from .config import AppConfig, JOINT_NAMES, load_config

__version__ = "0.1.0"
__all__ = ["AppConfig", "JOINT_NAMES", "load_config", "__version__"]
