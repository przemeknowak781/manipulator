"""Wspolna paleta i drobne pomocniki rysowania (OpenCV uzywa BGR)."""

from __future__ import annotations

import cv2
import numpy as np

BG = (28, 26, 24)
PANEL = (44, 40, 38)
TEXT = (236, 236, 236)
TEXT_DIM = (150, 150, 150)
ACCENT = (232, 168, 66)     # bursztyn - stan neutralny
OK = (110, 200, 120)        # zielony - sterowanie aktywne
WARN = (60, 190, 245)       # zolty/pomaranczowy - pauza
DANGER = (80, 80, 235)      # czerwony - stop awaryjny
GRID = (70, 65, 62)

FONT = cv2.FONT_HERSHEY_SIMPLEX

STATE_COLORS = {
    "STARTING": ACCENT,
    "IDLE": TEXT_DIM,
    "ACTIVE": OK,
    "HOLDING": WARN,
    "HOMING": ACCENT,
    "ESTOP": DANGER,
}


def panel(image: np.ndarray, x: int, y: int, w: int, h: int, alpha: float = 0.72) -> None:
    """Rysuje polprzezroczyste tlo panelu."""
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, image.shape[1]), min(y + h, image.shape[0])
    if x1 <= x0 or y1 <= y0:
        return
    roi = image[y0:y1, x0:x1]
    overlay = np.full_like(roi, PANEL)
    cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0.0, dst=roi)


def text(
    image: np.ndarray,
    value: str,
    org: tuple[int, int],
    scale: float = 0.5,
    color: tuple[int, int, int] = TEXT,
    thickness: int = 1,
) -> None:
    cv2.putText(image, value, org, FONT, scale, color, thickness, cv2.LINE_AA)


def bar(
    image: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    fraction: float,
    color: tuple[int, int, int],
    *,
    center_mark: bool = False,
) -> None:
    """Poziomy pasek wypelnienia z opcjonalnym znacznikiem srodka zakresu."""
    fraction = float(min(max(fraction, 0.0), 1.0))
    cv2.rectangle(image, (x, y), (x + w, y + h), GRID, -1)
    filled = int(w * fraction)
    if filled > 0:
        cv2.rectangle(image, (x, y), (x + filled, y + h), color, -1)
    cv2.rectangle(image, (x, y), (x + w, y + h), (90, 86, 82), 1)
    if center_mark:
        cx = x + w // 2
        cv2.line(image, (cx, y), (cx, y + h), (120, 116, 112), 1)
