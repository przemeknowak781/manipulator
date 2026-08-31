"""Konfiguracja logowania."""

from __future__ import annotations

import logging
import sys


def setup_logging(verbose: bool = False) -> None:
    """Wlacza czytelne logi na stderr (DEBUG przy `verbose`)."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(name)-28s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
        force=True,
    )
    # MediaPipe i biblioteki natywne potrafia zalac konsole.
    logging.getLogger("mediapipe").setLevel(logging.ERROR)
