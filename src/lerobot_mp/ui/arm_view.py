"""Podglad ramienia: widok z boku (plaszczyzna pionowa) i z gory.

W trybie symulacji to jedyne "okno na robota", ale przydaje sie takze przy
prawdziwym ramieniu - od razu widac, czy zadana poza ma sens i jak daleko
jest od granic przestrzeni roboczej.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from ..config import AppConfig
from ..control.kinematics import ArmKinematics
from .theme import ACCENT, GRID, OK, PANEL, TEXT, TEXT_DIM, WARN, text


def draw_arm_view(
    cfg: AppConfig,
    kinematics: ArmKinematics,
    joints: dict[str, float],
    size: tuple[int, int] = (320, 400),
    ee_target: tuple[float, float, float] | None = None,
) -> np.ndarray:
    """Rysuje panel z dwoma rzutami ramienia i zwraca go jako obraz BGR."""
    width, height = size
    canvas = np.full((height, width, 3), PANEL, dtype=np.uint8)

    half = height // 2
    _draw_side_view(canvas[:half], cfg, kinematics, joints, ee_target)
    _draw_top_view(canvas[half:], cfg, kinematics, joints, ee_target)
    cv2.line(canvas, (0, half), (width, half), GRID, 1)
    return canvas


def _scaler(width: int, height: int, span_m: float, origin_px: tuple[int, int]):
    """Zwraca funkcje metry -> piksele dla zadanej skali i poczatku ukladu."""
    px_per_m = min(width, height) / span_m
    ox, oy = origin_px

    def to_px(x: float, y: float) -> tuple[int, int]:
        return (int(ox + x * px_per_m), int(oy - y * px_per_m))

    return to_px, px_per_m


def _draw_side_view(
    canvas: np.ndarray,
    cfg: AppConfig,
    kin: ArmKinematics,
    joints: dict[str, float],
    ee_target: tuple[float, float, float] | None,
) -> None:
    h, w = canvas.shape[:2]
    span = 1.6 * (kin.max_reach + cfg.geometry.wrist_to_tip)
    to_px, px_per_m = _scaler(w, h, span, (int(w * 0.18), int(h * 0.80)))

    # Podloze i os pionowa.
    cv2.line(canvas, (0, to_px(0, 0)[1]), (w, to_px(0, 0)[1]), GRID, 1)
    text(canvas, "widok z boku", (8, 16), 0.38, TEXT_DIM)

    # Granice zasiegu (od barku).
    shoulder_px = to_px(0.0, cfg.geometry.base_height)
    for radius, color in ((kin.max_reach, GRID), (cfg.workspace.radius_max, GRID)):
        cv2.circle(canvas, shoulder_px, int(radius * px_per_m), color, 1)

    points = kin.chain_points(
        joints.get("shoulder_lift", 0.0),
        joints.get("elbow_flex", 0.0),
        joints.get("wrist_flex", 0.0),
    )
    pixels = [to_px(r, z) for r, z in points]

    for a, b in zip(pixels, pixels[1:]):
        cv2.line(canvas, a, b, ACCENT, 4, cv2.LINE_AA)
    for p in pixels[:-1]:
        cv2.circle(canvas, p, 5, TEXT, -1, cv2.LINE_AA)

    # Chwytak: szerokosc szczek proporcjonalna do zadanego otwarcia.
    grip = joints.get("gripper", 0.0) / 100.0
    tip, wrist = pixels[-1], pixels[-2]
    direction = np.array(tip, dtype=float) - np.array(wrist, dtype=float)
    norm = np.linalg.norm(direction)
    if norm > 1e-6:
        perp = np.array([-direction[1], direction[0]]) / norm
        jaw = perp * (4 + 12 * grip)
        for sign in (1, -1):
            offset = (tip[0] + int(sign * jaw[0]), tip[1] + int(sign * jaw[1]))
            cv2.line(canvas, tip, offset, OK, 3, cv2.LINE_AA)

    if ee_target is not None:
        tx, ty, tz = ee_target
        target_px = to_px(math.hypot(tx, ty), tz)
        cv2.drawMarker(canvas, target_px, WARN, cv2.MARKER_CROSS, 12, 2)


def _draw_top_view(
    canvas: np.ndarray,
    cfg: AppConfig,
    kin: ArmKinematics,
    joints: dict[str, float],
    ee_target: tuple[float, float, float] | None,
) -> None:
    h, w = canvas.shape[:2]
    span = 2.0 * (kin.max_reach + cfg.geometry.wrist_to_tip)
    to_px, px_per_m = _scaler(w, h, span, (w // 2, h // 2))
    text(canvas, "widok z gory", (8, 16), 0.38, TEXT_DIM)

    center = to_px(0, 0)
    cv2.circle(canvas, center, int(cfg.workspace.radius_max * px_per_m), GRID, 1)
    cv2.circle(canvas, center, int(cfg.workspace.radius_min * px_per_m), GRID, 1)

    pan = math.radians(joints.get("shoulder_pan", 0.0))
    points = kin.chain_points(
        joints.get("shoulder_lift", 0.0),
        joints.get("elbow_flex", 0.0),
        joints.get("wrist_flex", 0.0),
    )
    reach = points[-1][0]
    # Uwaga: os Y robota rosnie w lewo, a os Y obrazu w dol - stad minus.
    tip_px = to_px(reach * math.cos(pan), -reach * math.sin(pan))

    cv2.line(canvas, center, tip_px, ACCENT, 4, cv2.LINE_AA)
    cv2.circle(canvas, center, 6, TEXT, -1, cv2.LINE_AA)
    cv2.circle(canvas, tip_px, 5, OK, -1, cv2.LINE_AA)

    if ee_target is not None:
        tx, ty, _ = ee_target
        cv2.drawMarker(canvas, to_px(tx, -ty), WARN, cv2.MARKER_CROSS, 12, 2)


def compose(frame: np.ndarray, side_panel: np.ndarray) -> np.ndarray:
    """Skleja podglad kamery z panelem ramienia (dopasowuje wysokosci)."""
    h = frame.shape[0]
    pw = side_panel.shape[1]
    if side_panel.shape[0] != h:
        side_panel = cv2.resize(side_panel, (pw, h), interpolation=cv2.INTER_AREA)
    return np.hstack([frame, side_panel])
