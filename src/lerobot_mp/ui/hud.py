"""Nakladka informacyjna na podglad z kamery.

HUD ma odpowiadac na cztery pytania, ktore operator zadaje sobie w trakcie
sterowania: czy robot mnie widzi, czy sterowanie jest zalaczone, dokad jada
stawy i co zrobil nadzor bezpieczenstwa.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from ..config import AppConfig, JOINT_NAMES
from ..control.safety import SafetyReport, SafetyState
from ..vision.arm_features import ArmFeatures
from ..vision.features import HandFeatures
from ..vision.landmarks import HAND_CONNECTIONS, HandSample
from .theme import ACCENT, DANGER, OK, STATE_COLORS, TEXT, TEXT_DIM, WARN, bar, panel, text

#: Skroty klawiszowe pokazywane w stopce.
KEY_HELP = (
    "SPACJA sprzeglo | H dom | X stop awaryjny | C nowe zaczepienie | "
    "O/P kalibracja chwytaka | M tryb | Q wyjscie"
)


@dataclass
class HudData:
    """Wszystko, co HUD ma pokazac w jednej klatce."""

    state: SafetyState = SafetyState.IDLE
    engaged: bool = False
    hand_present: bool = False
    reason: str = ""
    features: HandFeatures = field(default_factory=HandFeatures.absent)
    arm: ArmFeatures = field(default_factory=ArmFeatures.absent)
    command: dict[str, float] = field(default_factory=dict)
    measured: dict[str, float] = field(default_factory=dict)
    report: SafetyReport = field(default_factory=SafetyReport)
    fps: float = 0.0
    latency_ms: float = 0.0
    backend: str = "symulator"
    tracker: str = "tasks"
    mode: str = "direct"
    message: str = ""
    ik_clamped: bool = False


def draw_arm_skeleton(image: np.ndarray, arm: ArmFeatures, active: bool) -> None:
    """Rysuje bark -> lokiec -> nadgarstek operatora (tryb `arm`)."""
    if not arm.present or len(arm.points_px) < 3:
        return
    color = OK if active else WARN
    points = [tuple(int(v) for v in p) for p in arm.points_px]
    for a, b in zip(points, points[1:]):
        cv2.line(image, a, b, color, 4, cv2.LINE_AA)
    for i, point in enumerate(points):
        cv2.circle(image, point, 8 if i == 1 else 6, TEXT, -1, cv2.LINE_AA)
    # Lokiec jest tu bohaterem - podpisujemy jego kat.
    text(image, f"{arm.elbow:.0f}", (points[1][0] + 12, points[1][1] - 10), 0.5, color, 2)


def draw_hand_skeleton(image: np.ndarray, hand: HandSample, active: bool) -> None:
    """Rysuje szkielet dloni; kolor zalezy od tego, czy sterowanie jest zalaczone."""
    h, w = image.shape[:2]
    pts = [(int(p[0] * w), int(p[1] * h)) for p in hand.landmarks[:, :2]]
    color = OK if active else WARN

    for a, b in HAND_CONNECTIONS:
        cv2.line(image, pts[a], pts[b], color, 2, cv2.LINE_AA)
    for i, p in enumerate(pts):
        radius = 5 if i in (4, 8) else 3  # kciuk i wskazujacy = chwytak
        cv2.circle(image, p, radius, TEXT, -1, cv2.LINE_AA)
    # Linia szczypniecia - wprost pokazuje, co steruje chwytakiem.
    cv2.line(image, pts[4], pts[8], ACCENT, 2, cv2.LINE_AA)


def draw_hud(image: np.ndarray, data: HudData, cfg: AppConfig) -> None:
    """Rysuje caly HUD na miejscu (modyfikuje `image`)."""
    h, w = image.shape[:2]
    _draw_status_bar(image, data, w)
    _draw_joint_panel(image, data, cfg, w, h)
    _draw_hand_panel(image, data, h)
    if data.arm.present:
        _draw_arm_panel(image, data, h)
    _draw_footer(image, data, w, h)


def _draw_status_bar(image: np.ndarray, data: HudData, w: int) -> None:
    panel(image, 0, 0, w, 46, alpha=0.78)
    color = STATE_COLORS.get(data.state.value, TEXT)

    cv2.circle(image, (24, 23), 9, color, -1, cv2.LINE_AA)
    text(image, data.state.value, (42, 29), 0.72, color, 2)

    label = "STEROWANIE" if data.engaged else "PAUZA"
    text(image, label, (190, 29), 0.6, OK if data.engaged else WARN, 2)

    info = f"{data.backend}  |  mapowanie: {data.mode}  |  mediapipe: {data.tracker}"
    text(image, info, (330, 21), 0.44, TEXT_DIM)
    text(image, f"{data.fps:5.1f} FPS   opoznienie {data.latency_ms:5.1f} ms", (330, 39), 0.44, TEXT_DIM)

    if data.state is SafetyState.ESTOP:
        cv2.rectangle(image, (0, 0), (w - 1, image.shape[0] - 1), DANGER, 4)


def _draw_joint_panel(image: np.ndarray, data: HudData, cfg: AppConfig, w: int, h: int) -> None:
    """Panel stawow: pasek zakresu, wartosc zadana i zmierzona."""
    rows = len(JOINT_NAMES)
    row_h, pad = 26, 12
    panel_w, panel_h = 320, rows * row_h + 2 * pad + 20
    x0, y0 = w - panel_w - 12, 58
    panel(image, x0, y0, panel_w, panel_h)
    text(image, "STAWY (zadane / zmierzone)", (x0 + pad, y0 + pad + 8), 0.44, TEXT_DIM)

    for i, name in enumerate(JOINT_NAMES):
        jc = cfg.joint(name)
        y = y0 + pad + 24 + i * row_h
        target = data.command.get(name)
        measured = data.measured.get(name)

        color = TEXT
        if name in data.report.at_limit:
            color = DANGER
        elif name in data.report.rate_limited:
            color = WARN

        text(image, name, (x0 + pad, y + 10), 0.4, color)

        bar_x, bar_w = x0 + 148, 96
        span = jc.max - jc.min
        if target is not None and span > 1e-6:
            bar(image, bar_x, y, bar_w, 12, (target - jc.min) / span, color, center_mark=True)
            if measured is not None:
                mx = bar_x + int(bar_w * min(max((measured - jc.min) / span, 0.0), 1.0))
                cv2.line(image, (mx, y - 2), (mx, y + 14), TEXT, 1)

        value = f"{target:6.1f}" if target is not None else "   ---"
        text(image, value, (x0 + panel_w - 62, y + 10), 0.42, color)


def _draw_hand_panel(image: np.ndarray, data: HudData, h: int) -> None:
    """Panel dloni: szczypniecie (chwytak) i wyprostowanie palcow (sprzeglo)."""
    panel_w, panel_h = 250, 108
    x0, y0 = 12, 58
    panel(image, x0, y0, panel_w, panel_h)
    f = data.features

    title = "DLON: brak" if not data.hand_present else f"DLON: {f.handedness} ({f.score:.0%})"
    text(image, title, (x0 + 12, y0 + 22), 0.46, TEXT if data.hand_present else TEXT_DIM)

    if not data.hand_present:
        # W trybie `arm` dlon jest dodatkiem, a nie warunkiem sterowania -
        # komunikat "pokaz dlon" bylby tam myleniem operatora.
        hint = (
            "nadgarstek i chwytak stoja"
            if data.mode.lower() == "arm"
            else "pokaz dlon kamerze"
        )
        text(image, hint, (x0 + 12, y0 + 48), 0.42, TEXT_DIM)
        return

    text(image, "chwytak", (x0 + 12, y0 + 46), 0.4, TEXT_DIM)
    grip = data.command.get("gripper", 0.0)
    bar(image, x0 + 90, y0 + 36, 110, 12, grip / 100.0, ACCENT)
    text(image, f"{grip:3.0f}%", (x0 + 206, y0 + 46), 0.4, TEXT)

    text(image, "palce", (x0 + 12, y0 + 70), 0.4, TEXT_DIM)
    curl_color = WARN if f.curled else OK
    bar(image, x0 + 90, y0 + 60, 110, 12, min(f.extension / 1.1, 1.0), curl_color)
    text(image, "pauza" if f.curled else "ruch", (x0 + 206, y0 + 70), 0.4, curl_color)

    text(
        image,
        f"glebokosc {f.scale:.3f}   obrot {np.degrees(f.roll):+6.0f}",
        (x0 + 12, y0 + 94),
        0.4,
        TEXT_DIM,
    )


def _draw_arm_panel(image: np.ndarray, data: HudData, h: int) -> None:
    """Panel z katami ramienia operatora - widoczny tylko w trybie `arm`."""
    panel_w, panel_h = 250, 96
    x0, y0 = 12, 174
    panel(image, x0, y0, panel_w, panel_h)
    arm = data.arm

    side = "prawe" if arm.side == "Right" else "lewe"
    text(image, f"RAMIE: {side} ({arm.visibility:.0%})", (x0 + 12, y0 + 22), 0.46, TEXT)

    rows = (
        ("uniesienie", arm.elevation, -90.0, 90.0),
        ("kierunek", arm.azimuth, -90.0, 90.0),
        ("lokiec", arm.elbow, 0.0, 150.0),
    )
    for i, (label, value, low, high) in enumerate(rows):
        y = y0 + 42 + i * 18
        text(image, label, (x0 + 12, y + 8), 0.4, TEXT_DIM)
        fraction = (value - low) / (high - low) if high > low else 0.0
        bar(image, x0 + 96, y, 96, 11, fraction, ACCENT)
        text(image, f"{value:+5.0f}", (x0 + 200, y + 8), 0.4, TEXT)


def _draw_footer(image: np.ndarray, data: HudData, w: int, h: int) -> None:
    panel(image, 0, h - 46, w, 46, alpha=0.78)

    note = data.message or data.reason
    if data.ik_clamped:
        note = note or "cel poza zasiegiem - przyciety"
    if note:
        color = WARN if not data.engaged else ACCENT
        text(image, note, (14, h - 26), 0.5, color)

    text(image, KEY_HELP, (14, h - 8), 0.4, TEXT_DIM)

    if data.report.seconds_without_hand > 0.5:
        text(
            image,
            f"brak dloni: {data.report.seconds_without_hand:.1f} s",
            (w - 190, h - 26),
            0.44,
            WARN,
        )
