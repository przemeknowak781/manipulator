"""Zamiana 21 punktow dloni na kilka czytelnych sygnalow sterujacych.

Cechy sa tak dobrane, zeby byly *niezalezne od odleglosci dloni od kamery*
tam, gdzie to mozliwe (dystanse dzielimy przez rozmiar dloni), a jednocześnie
zeby kazda z nich odpowiadala jednemu naturalnemu ruchowi reki.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .landmarks import (
    INDEX_MCP,
    INDEX_TIP,
    MIDDLE_MCP,
    PALM_POINTS,
    PINKY_MCP,
    THUMB_TIP,
    WRIST,
    CLUTCH_FINGERS,
    HandSample,
)


@dataclass
class HandFeatures:
    """Sygnaly sterujace wyliczone z jednej dloni.

    Attributes:
        x, y: srodek dloni w kadrze, 0..1 (0,0 = lewy gorny rog).
        scale: rozmiar dloni w kadrze (nadgarstek -> nasada srodkowego palca).
            Rosnie, gdy dlon zbliza sie do kamery - to nasz sygnal glebokosci.
        roll: obrot dloni w plaszczyznie obrazu [rad], w zakresie (-pi, pi].
        pitch: pochylenie dloni w przod/tyl [rad] (z punktow 3D, gdy sa dostepne).
        pinch: rozwarcie kciuk-wskazujacy znormalizowane rozmiarem dloni.
            ~0.1 przy zetknieciu palcow, ~1.0 przy szeroko rozwartych.
        extension: srednie wyprostowanie palcow srodkowego/serdecznego/malego.
        curled: True, gdy te trzy palce sa zwiniete (gest pauzy / sprzegla).
    """

    present: bool = False
    x: float = 0.5
    y: float = 0.5
    scale: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    pinch: float = 0.5
    extension: float = 1.0
    curled: bool = False
    handedness: str = "Unknown"
    score: float = 0.0
    #: Srodek dloni w pikselach - tylko do rysowania HUD.
    palm_px: tuple[int, int] = (0, 0)

    @classmethod
    def absent(cls) -> "HandFeatures":
        return cls(present=False)


#: Ponizej tej wartosci `extension` uznajemy palce za zwiniete (gest pauzy).
#: Zmierzone na otwartej dloni: ~0.75-0.80; piesc: ~0.35-0.45.
CURL_THRESHOLD = 0.58
#: Histereza, zeby stan nie migotal na granicy progu.
CURL_HYSTERESIS = 0.07


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def extract_features(
    hand: HandSample,
    frame_size: tuple[int, int],
    previous: HandFeatures | None = None,
    curl_threshold: float = CURL_THRESHOLD,
) -> HandFeatures:
    """Wylicza cechy sterujace dla jednej dloni.

    Args:
        hand: wykryta dlon.
        frame_size: (szerokosc, wysokosc) klatki w pikselach - potrzebne, bo
            wspolrzedne znormalizowane sa "sciśniete" w osi X dla kadru 16:9
            i bez korekty proporcji dystanse bylyby przeklamane.
        previous: poprzednie cechy - uzywane wylacznie do histerezy gestu.
        curl_threshold: prog rozpoznania zwinietych palcow (gest pauzy).
    """
    width, height = frame_size
    aspect = (width / height) if height else 1.0

    lm = hand.landmarks
    # Wspolrzedne "metryczne w jednostkach wysokosci kadru": korekta proporcji.
    flat = lm[:, :2].copy()
    flat[:, 0] *= aspect

    palm_center = lm[list(PALM_POINTS), :2].mean(axis=0)
    scale = _dist(flat[WRIST], flat[MIDDLE_MCP])
    if scale < 1e-6:
        return HandFeatures.absent()

    # --- obrot w plaszczyznie obrazu (roll) -------------------------------
    side = flat[INDEX_MCP] - flat[PINKY_MCP]
    if hand.handedness == "Left":
        # Lewa dlon jest LUSTRZANYM ODBICIEM prawej, a nie prawa obrocona
        # o 180 stopni. Odbijamy wiec wektor poprzeczny wzgledem osi dloni
        # (nadgarstek -> nasada srodkowego palca). Prostsze "dodaj 180 stopni"
        # zostawia systematyczny blad kilkunastu stopni miedzy rekami.
        forward = flat[MIDDLE_MCP] - flat[WRIST]
        norm_sq = float(forward @ forward)
        if norm_sq > 1e-12:
            side = 2.0 * (float(side @ forward) / norm_sq) * forward - side
    roll = math.atan2(float(side[1]), float(side[0]))

    # --- pochylenie (pitch) ------------------------------------------------
    pitch = _palm_pitch(hand)

    # --- chwytak: szczypniecie kciuk + wskazujacy --------------------------
    if hand.has_world:
        w = hand.world_landmarks
        palm_len = _dist(w[WRIST], w[MIDDLE_MCP])
        pinch = _dist(w[THUMB_TIP], w[INDEX_TIP]) / palm_len if palm_len > 1e-6 else 0.5
    else:
        pinch = _dist(flat[THUMB_TIP], flat[INDEX_TIP]) / scale

    # --- wyprostowanie palcow (gest pauzy) ---------------------------------
    extension = _finger_extension(hand, scale, flat)
    threshold = curl_threshold
    if previous is not None and previous.curled:
        threshold += CURL_HYSTERESIS  # trudniej "odkleic" stan niz go wlaczyc
    curled = extension < threshold

    return HandFeatures(
        present=True,
        x=float(palm_center[0]),
        y=float(palm_center[1]),
        scale=float(scale),
        roll=float(roll),
        pitch=float(pitch),
        pinch=float(pinch),
        extension=float(extension),
        curled=bool(curled),
        handedness=hand.handedness,
        score=hand.score,
        palm_px=(int(palm_center[0] * width), int(palm_center[1] * height)),
    )


def _palm_pitch(hand: HandSample) -> float:
    """Pochylenie dloni: kat osi nadgarstek->palce wzgledem plaszczyzny obrazu.

    Z punktow 3D (`world_landmarks`) dostajemy prawdziwy kat. Bez nich
    korzystamy z wzglednej glebokosci `z` punktow znormalizowanych, ktora jest
    mniej dokladna, ale zachowuje kierunek zmian.
    """
    if hand.has_world:
        w = hand.world_landmarks
        forward = w[MIDDLE_MCP] - w[WRIST]
    else:
        forward = hand.landmarks[MIDDLE_MCP] - hand.landmarks[WRIST]

    norm = float(np.linalg.norm(forward))
    if norm < 1e-9:
        return 0.0
    # Os Z w MediaPipe rosnie "od kamery"; minus daje intuicyjny znak:
    # palce skierowane w strone kamery -> pitch dodatni.
    return float(math.asin(max(-1.0, min(1.0, -forward[2] / norm))))


def _finger_extension(hand: HandSample, scale: float, flat: np.ndarray) -> float:
    """Srednie wyprostowanie palcow srodkowego/serdecznego/malego.

    ~1.0 dla wyprostowanych palcow, ~0.3-0.5 dla zwinietych w piesc.
    Kciuk i palec wskazujacy sa pominiete, bo obsluguja chwytak - dzieki temu
    szczypanie nie jest mylone z zaciskaniem piesci.
    """
    if hand.has_world:
        w = hand.world_landmarks
        palm_len = _dist(w[WRIST], w[MIDDLE_MCP])
        if palm_len < 1e-9:
            return 1.0
        ratios = [_dist(w[mcp], w[tip]) / palm_len for mcp, _pip, tip in CLUTCH_FINGERS]
    else:
        ratios = [_dist(flat[mcp], flat[tip]) / scale for mcp, _pip, tip in CLUTCH_FINGERS]
    return float(np.mean(ratios))
