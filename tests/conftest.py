"""Wspolne narzedzia testow: syntetyczna dlon i skrocona konfiguracja."""

from __future__ import annotations

import math

import numpy as np
import pytest

from lerobot_mp.config import AppConfig, load_config
from lerobot_mp.vision.pose import (
    LEFT_ELBOW,
    LEFT_HIP,
    LEFT_SHOULDER,
    LEFT_WRIST,
    NOSE,
    NUM_POSE_LANDMARKS,
    RIGHT_ELBOW,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
    PoseSample,
)
from lerobot_mp.vision.landmarks import (
    INDEX_MCP,
    INDEX_TIP,
    MIDDLE_MCP,
    PINKY_MCP,
    RING_MCP,
    THUMB_TIP,
    WRIST,
    HandSample,
)


def make_hand(
    center: tuple[float, float] = (0.5, 0.5),
    scale: float = 0.12,
    roll_deg: float = 0.0,
    pinch: float = 0.6,
    curl: float = 0.85,
    handedness: str = "Right",
    depth: float = 0.0,
    aspect: float = 16.0 / 9.0,
) -> HandSample:
    """Buduje syntetyczna dlon o zadanych cechach.

    Dlon jest plaska: nadgarstek w srodku, palce "do gory" (w kierunku
    malejacego Y obrazu), obrocone o `roll_deg`. `curl` to stosunek dlugosci
    palca do dlugosci dloni - 0.85 to dlon otwarta, 0.35 to piesc.

    Geometria jest budowana w jednostkach FIZYCZNYCH (wysokosc kadru = 1),
    a dopiero na koniec sciskana w osi X o proporcje kadru - tak samo, jak
    robi to kamera. Bez tego kroku obrocona dlon zmienialaby rozmiar, a testy
    mierzylyby wade wlasnej atrapy zamiast kodu.

    Lewa dlon jest lustrzanym odbiciem prawej - takze to musi byc w atrapie,
    bo na tym opiera sie normalizacja obrotu dla obu rak.
    """
    points = np.zeros((21, 3), dtype=np.float32)
    angle = math.radians(roll_deg)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    mirror = -1.0 if handedness == "Left" else 1.0

    def place(index: int, along: float, across: float, z: float = 0.0) -> None:
        """`along` - w strone palcow, `across` - w poprzek dloni."""
        across = across * mirror
        x = across * cos_a - (-along) * sin_a
        y = across * sin_a + (-along) * cos_a
        points[index] = (center[0] + x * scale / aspect, center[1] + y * scale, z)

    place(WRIST, 0.0, 0.0)
    place(MIDDLE_MCP, 1.0, 0.0)              # dlugosc dloni = `scale`
    place(INDEX_MCP, 0.95, 0.30)
    place(PINKY_MCP, 0.85, -0.42)
    place(RING_MCP, 0.92, -0.16)

    # Palce sterujace sprzeglem: srodkowy, serdeczny, maly.
    for mcp, tip, across in ((MIDDLE_MCP, 12, 0.0), (RING_MCP, 16, -0.16), (PINKY_MCP, 20, -0.42)):
        base_along = {MIDDLE_MCP: 1.0, RING_MCP: 0.92, PINKY_MCP: 0.85}[mcp]
        place(tip, base_along + curl, across)
    for pip, mcp in ((10, MIDDLE_MCP), (14, RING_MCP), (18, PINKY_MCP)):
        points[pip] = (points[mcp] + points[{10: 12, 14: 16, 18: 20}[pip]]) / 2.0

    # Kciuk i palec wskazujacy: odleglosc miedzy koncowkami = `pinch` * scale.
    place(INDEX_TIP, 1.9, 0.30)
    place(THUMB_TIP, 1.9, 0.30 - pinch)
    for i in (1, 2, 3):
        points[i] = (points[WRIST] + points[THUMB_TIP]) / 2.0
    for i in (6, 7):
        points[i] = (points[INDEX_MCP] + points[INDEX_TIP]) / 2.0
    points[11] = points[10]
    points[15] = points[14]
    points[19] = points[18]

    points[:, 2] += depth
    # `world_landmarks` sa metryczne i bez sciskania osi X.
    world = points.copy()
    world[:, 0] *= aspect
    world[:, :2] -= world[WRIST, :2]
    return HandSample(points, world, handedness, 0.95)


@pytest.fixture
def cfg() -> AppConfig:
    return load_config()


@pytest.fixture
def frame_size() -> tuple[int, int]:
    return (1280, 720)


def make_pose(
    elevation_deg: float = -90.0,
    azimuth_deg: float = 0.0,
    elbow_deg: float = 0.0,
    side: str = "Right",
    visibility: float = 1.0,
    upper_arm: float = 0.30,
    forearm: float = 0.26,
) -> PoseSample:
    """Buduje syntetyczna sylwetke o zadanych katach ramienia.

    Operator stoi twarza do kamery. Uklad MediaPipe: X w prawo obrazu, Y w dol,
    Z w glab sceny - wiec strona LEWA operatora jest po prawej stronie obrazu,
    a bark jest nad biodrem, czyli ma MNIEJSZE Y.

    Katy sa zadane w tym samym ukladzie tulowia, ktorego uzywa
    `extract_arm_features`, wiec test moze porownac wejscie z wyjsciem.
    """
    world = np.zeros((NUM_POSE_LANDMARKS, 3), dtype=np.float32)
    world[LEFT_HIP] = (0.10, 0.0, 0.0)
    world[RIGHT_HIP] = (-0.10, 0.0, 0.0)
    world[LEFT_SHOULDER] = (0.20, -0.50, 0.0)
    world[RIGHT_SHOULDER] = (-0.20, -0.50, 0.0)
    world[NOSE] = (0.0, -0.70, -0.10)

    up = np.array([0.0, -1.0, 0.0])
    right = np.array([-1.0, 0.0, 0.0])       # w prawo OPERATORA
    forward = np.cross(up, right)            # w strone kamery
    outward = right if side == "Right" else -right

    elevation = math.radians(elevation_deg)
    azimuth = math.radians(azimuth_deg)
    upper_dir = (
        math.sin(elevation) * up
        + math.cos(elevation) * math.cos(azimuth) * outward
        + math.cos(elevation) * math.sin(azimuth) * forward
    )

    # Przedramie zgiete o `elbow_deg` - obracamy je wokol osi prostopadlej
    # do ramienia, wybranej tak, zeby zgiecie szlo "do przodu" jak u czlowieka.
    axis = np.cross(upper_dir, forward)
    if np.linalg.norm(axis) < 1e-6:
        axis = np.cross(upper_dir, up)
    axis = axis / np.linalg.norm(axis)
    angle = math.radians(elbow_deg)
    fore_dir = (
        upper_dir * math.cos(angle)
        + np.cross(axis, upper_dir) * math.sin(angle)
        + axis * np.dot(axis, upper_dir) * (1 - math.cos(angle))
    )

    shoulder_i = RIGHT_SHOULDER if side == "Right" else LEFT_SHOULDER
    elbow_i = RIGHT_ELBOW if side == "Right" else LEFT_ELBOW
    wrist_i = RIGHT_WRIST if side == "Right" else LEFT_WRIST
    world[elbow_i] = world[shoulder_i] + upper_dir * upper_arm
    world[wrist_i] = world[elbow_i] + fore_dir * forearm

    # Punkty obrazu sa tu tylko po to, zeby HUD mial co rysowac.
    image = np.zeros((NUM_POSE_LANDMARKS, 3), dtype=np.float32)
    image[:, 0] = 0.5 + world[:, 0] * 0.4
    image[:, 1] = 0.5 + world[:, 1] * 0.4

    vis = np.full(NUM_POSE_LANDMARKS, visibility, dtype=np.float32)
    return PoseSample(image, world, vis)
