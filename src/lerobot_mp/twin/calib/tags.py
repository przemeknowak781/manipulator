"""AprilTag (36h11) detection and single-tag pose, via OpenCV's aruco module.

Przeniesione z galaxeo-manipulators `sim/calib/tags.py` (commit 02641c4) bez
zmian w dzialaniu; doszlo tylko `render`, ktore rysuje tag z bialym marginesem
do tekstury w symulacji i do arkusza do druku - wczesniej byly to gotowe PNG-i.

    from lerobot_mp.twin.calib.tags import detect, tag_pose, corners_in_tag
    seen = detect(rgb)                       # {id: (4, 2) pixel corners}
    T_tag2cam = tag_pose(seen[0], size, K, dist)

Tag frame: origin at the centre, x right, y up, z out of the face toward the
viewer (OpenCV's marker convention). Corners are ordered top-left, top-right,
bottom-right, bottom-left in the tag's own image, which is what `detect`
returns and what `corners_in_tag` assumes.

"Rozmiar taga" to wszedzie bok CZARNEGO kwadratu (8 z 10 komorek), bez bialego
marginesu - tak liczy OpenCV i tak trzeba zmierzyc wydruk linijka.
"""

from __future__ import annotations

import cv2
import numpy as np

DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
#: Bok czarnego kwadratu / bok calej plytki z jednokomorkowym bialym marginesem.
MARKER_FRACTION = 8 / 10
#: Doszlifowanie rogow `CORNER_REFINE_APRILTAG` zwraca je w konwencji NAROZNIKOW
#: pikseli (lewy gorny rog obrazu = 0), a model otworkowy OpenCV - K,
#: `projectPoints`, `solvePnP` - liczy w konwencji SRODKOW pikseli. Zmierzone na
#: tagu o znanych krawedziach (139,5 i 459,5): APRILTAG oddaje 140,0 i 460,0,
#: rowno w obu osiach i takze na rozmytym obrazie; SUBPIX trafia w 139,56 i 459,44.
#: Bez tej poprawki solver dostawal rogi przesuniete o pol piksela i "naprawial"
#: je obrotem kamery. Dotyczy tez galaxeo, z ktorego ten modul pochodzi.
APRILTAG_OFFSET = 0.5
_DET = None


def detector():
    global _DET
    if _DET is None:
        p = cv2.aruco.DetectorParameters()
        p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
        p.minMarkerPerimeterRate = 0.01
        _DET = cv2.aruco.ArucoDetector(DICT, p)
    return _DET


def detect(rgb):
    """{tag id: (4, 2) float32 pixel corners} for every tag found in an RGB image."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if rgb.ndim == 3 else rgb
    corners, ids, _ = detector().detectMarkers(gray)
    if ids is None:
        return {}
    return {int(i): c.reshape(4, 2) - APRILTAG_OFFSET for c, i in zip(corners, ids.ravel())}


def corners_in_tag(size):
    h = size / 2
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], float)


def tag_pose(corners, size, K, dist=None):
    """4x4 transform of the tag frame in the camera frame from its four corners."""
    dist = np.zeros(5) if dist is None else np.asarray(dist, float)
    ok, rvec, tvec = cv2.solvePnP(corners_in_tag(size), np.asarray(corners, float), K, dist,
                                  flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        raise RuntimeError("solvePnP failed")
    T = np.eye(4)
    T[:3, :3], _ = cv2.Rodrigues(rvec)
    T[:3, 3] = tvec.ravel()
    return T


def project(points_cam, K, dist=None):
    """Pixel coordinates of (N, 3) points already in the camera frame."""
    dist = np.zeros(5) if dist is None else np.asarray(dist, float)
    px, _ = cv2.projectPoints(np.asarray(points_cam, float), np.zeros(3), np.zeros(3), K, dist)
    return px.reshape(-1, 2)


def render(tag_id: int, px: int = 400) -> np.ndarray:
    """Tag `tag_id` jako obraz (px, px) uint8 z jednokomorkowym bialym marginesem.

    Czarny kwadrat zajmuje srodkowe `MARKER_FRACTION` obrazu. Bez marginesu
    detektor nie odroznia krawedzi taga od ciemnego tla za karta.
    """
    cell = px // 10
    marker = cv2.aruco.generateImageMarker(DICT, int(tag_id), cell * 8, borderBits=1)
    out = np.full((cell * 10, cell * 10), 255, np.uint8)
    out[cell : cell * 9, cell : cell * 9] = marker
    if out.shape[0] != px:
        out = cv2.resize(out, (px, px), interpolation=cv2.INTER_NEAREST)
    return out
