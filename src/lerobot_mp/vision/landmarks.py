"""Wspolny format punktow charakterystycznych dloni.

MediaPipe ma dwa API (nowe `tasks` i stare `solutions`), ktore zwracaja
troche inne obiekty. Sprowadzamy oba do jednej, prostej reprezentacji
`HandSample`, dzieki czemu reszta aplikacji nie wie nic o MediaPipe.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Indeksy 21 punktow dloni wg MediaPipe.
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

NUM_LANDMARKS = 21

#: Punkty tworzace "dlon" (srodek dloni liczymy jako ich srednia).
PALM_POINTS = (WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)

#: Palce uzywane do wykrycia gestu pauzy - kciuk i wskazujacy sa zajete
#: przez chwytak, wiec sprzeglo obslugujemy pozostalymi trzema palcami.
CLUTCH_FINGERS = (
    (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP),
    (RING_MCP, RING_PIP, RING_TIP),
    (PINKY_MCP, PINKY_PIP, PINKY_TIP),
)

#: Polaczenia do rysowania szkieletu dloni.
HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)

Landmark = np.ndarray  # (3,) float32


@dataclass
class HandSample:
    """Jedna wykryta dlon w jednej klatce.

    Attributes:
        landmarks: (21, 3) wspolrzedne znormalizowane do kadru; x,y w [0, 1],
            z to wzgledna glebokosc wzgledem nadgarstka (mniejsze = blizej kamery).
        world_landmarks: (21, 3) wspolrzedne metryczne [m] w ukladzie dloni
            (srodek dloni w poczatku ukladu). Puste, gdy backend ich nie daje.
        handedness: "Left" / "Right" / "Unknown" - z perspektywy kamery.
        score: pewnosc klasyfikacji strony dloni.
    """

    landmarks: np.ndarray
    world_landmarks: np.ndarray | None = None
    handedness: str = "Unknown"
    score: float = 0.0

    def __post_init__(self) -> None:
        self.landmarks = np.asarray(self.landmarks, dtype=np.float32).reshape(NUM_LANDMARKS, 3)
        if self.world_landmarks is not None:
            self.world_landmarks = np.asarray(self.world_landmarks, dtype=np.float32).reshape(
                NUM_LANDMARKS, 3
            )

    @property
    def has_world(self) -> bool:
        return self.world_landmarks is not None


@dataclass
class TrackResult:
    """Wynik detekcji dla calej klatki."""

    hands: list[HandSample] = field(default_factory=list)
    #: Znacznik czasu klatki (monotoniczny, w sekundach).
    timestamp: float = 0.0

    def pick(self, preferred: str = "any") -> HandSample | None:
        """Wybiera dlon do sterowania.

        Gdy `preferred` to "Left"/"Right", bierzemy dlon o tej stronnosci;
        w przeciwnym razie te o najwyzszej pewnosci.
        """
        if not self.hands:
            return None
        if preferred and preferred.lower() not in ("any", ""):
            wanted = preferred.capitalize()
            matching = [h for h in self.hands if h.handedness == wanted]
            if matching:
                return max(matching, key=lambda h: h.score)
            return None
        return max(self.hands, key=lambda h: h.score)
