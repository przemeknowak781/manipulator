"""Katy ramienia operatora liczone w ukladzie jego tulowia.

Trzy liczby wystarczaja, zeby opisac ulozenie ramienia i przelozyc je wprost
na trzy pierwsze stawy robota:

* ``elevation`` - jak wysoko podniesiona jest reka   -> `shoulder_lift`
* ``azimuth``   - w ktora strone jest wyciagnieta    -> `shoulder_pan`
* ``elbow``     - jak mocno zgiety jest lokiec       -> `elbow_flex`

Katy sa liczone w ukladzie TULOWIA, a nie kamery. To istotna roznica: gdyby
liczyc je w ukladzie obrazu, przechylenie sie na krzesle albo obrot bokiem do
kamery zmienialoby zadana poze robota, chociaz reka nie drgnela.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .pose import (
    LEFT_ELBOW,
    LEFT_HIP,
    LEFT_SHOULDER,
    LEFT_WRIST,
    RIGHT_ELBOW,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
    PoseSample,
)

#: Indeksy stawow dla obu stron: (bark, lokiec, nadgarstek).
SIDE_POINTS = {
    "Left": (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST),
    "Right": (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST),
}

#: Ponizej tej dlugosci ogniwa uznajemy pomiar za smieciowy.
MIN_SEGMENT_M = 0.05


@dataclass
class ArmFeatures:
    """Ulozenie ramienia operatora."""

    present: bool = False
    #: Strona OPERATORA: "Left" albo "Right".
    side: str = "Right"
    #: Kat nad poziomem [stopnie]: -90 reka opuszczona, 0 poziomo, +90 w gore.
    elevation: float = 0.0
    #: Kierunek w plaszczyznie poziomej [stopnie]: 0 w bok od ciala,
    #: +90 do przodu (w strone kamery), -90 do tylu.
    azimuth: float = 0.0
    #: Zgiecie lokcia [stopnie]: 0 reka wyprostowana, ~150 zgieta maksymalnie.
    elbow: float = 0.0
    #: Najslabiej widoczny punkt ramienia (0..1).
    visibility: float = 0.0
    #: Piksele do rysowania: bark, lokiec, nadgarstek.
    points_px: tuple[tuple[int, int], ...] = ()

    @classmethod
    def absent(cls) -> "ArmFeatures":
        return cls(present=False)


def torso_frame(world: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Uklad wspolrzednych tulowia: (w gore, w prawo operatora, do przodu).

    Osie sa ortonormalne i wyznaczone z samej sylwetki, wiec nie zaleza od
    tego, jak operator jest ustawiony wzgledem kamery.
    """
    shoulder_mid = (world[LEFT_SHOULDER] + world[RIGHT_SHOULDER]) / 2.0
    hip_mid = (world[LEFT_HIP] + world[RIGHT_HIP]) / 2.0

    up = shoulder_mid - hip_mid
    norm = np.linalg.norm(up)
    if norm < MIN_SEGMENT_M:
        return None
    up = up / norm

    # Wektor miedzy barkami wskazuje w prawo OPERATORA; prostujemy go
    # wzgledem osi pionowej, zeby uklad byl ortonormalny.
    right = world[RIGHT_SHOULDER] - world[LEFT_SHOULDER]
    right = right - np.dot(right, up) * up
    norm = np.linalg.norm(right)
    if norm < MIN_SEGMENT_M:
        return None
    right = right / norm

    # W ukladzie MediaPipe X rosnie w prawo obrazu, Y w dol, a Z w glab sceny.
    # Przy tej konwencji `up x right` wychodzi w strone kamery, czyli tam,
    # gdzie patrzy operator. Sprawdza to `test_torso_frame_faces_the_camera`.
    forward = np.cross(up, right)
    return up, right, forward


def extract_arm_features(
    pose: PoseSample,
    side: str,
    frame_size: tuple[int, int],
    min_visibility: float = 0.6,
) -> ArmFeatures:
    """Wylicza katy ramienia; zwraca `absent`, gdy pomiar jest niewiarygodny."""
    if side not in SIDE_POINTS:
        raise ValueError(f"Nieznana strona ramienia: {side!r} (Left|Right)")

    shoulder_i, elbow_i, wrist_i = SIDE_POINTS[side]
    visibility = pose.arm_visibility(side)
    if visibility < min_visibility or not pose.has_world:
        return ArmFeatures.absent()

    world = pose.world_landmarks
    frame = torso_frame(world)
    if frame is None:
        return ArmFeatures.absent()
    up, right, forward = frame

    upper = world[elbow_i] - world[shoulder_i]
    fore = world[wrist_i] - world[elbow_i]
    if np.linalg.norm(upper) < MIN_SEGMENT_M or np.linalg.norm(fore) < MIN_SEGMENT_M:
        return ArmFeatures.absent()

    upper_dir = upper / np.linalg.norm(upper)
    fore_dir = fore / np.linalg.norm(fore)

    # "Na zewnatrz od ciala" znaczy co innego dla lewej i prawej reki -
    # po tym odbiciu obie daja te same znaki dla tego samego gestu.
    outward = right if side == "Right" else -right

    component_up = float(np.dot(upper_dir, up))
    component_out = float(np.dot(upper_dir, outward))
    component_fwd = float(np.dot(upper_dir, forward))

    elevation = math.degrees(math.asin(max(-1.0, min(1.0, component_up))))
    azimuth = math.degrees(math.atan2(component_fwd, component_out))
    elbow = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(upper_dir, fore_dir))))))

    width, height = frame_size
    points_px = tuple(
        (int(pose.landmarks[i][0] * width), int(pose.landmarks[i][1] * height))
        for i in (shoulder_i, elbow_i, wrist_i)
    )

    return ArmFeatures(
        present=True,
        side=side,
        elevation=elevation,
        azimuth=azimuth,
        elbow=elbow,
        visibility=visibility,
        points_px=points_px,
    )


def pick_arm(pose: PoseSample, preferred: str = "auto", min_visibility: float = 0.6) -> str | None:
    """Wybiera ramie do sterowania: wskazane albo lepiej widoczne."""
    if preferred in SIDE_POINTS:
        return preferred if pose.arm_visibility(preferred) >= min_visibility else None

    scores = {side: pose.arm_visibility(side) for side in SIDE_POINTS}
    best = max(scores, key=lambda s: scores[s])
    return best if scores[best] >= min_visibility else None
