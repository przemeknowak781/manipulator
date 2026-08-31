"""Programowy renderer 3D podgladu ramienia (numpy + OpenCV, bez OpenGL).

Ramie ma po uproszczeniu ~29 tys. trojkatow. Rysowanie ich algorytmem malarza
(sortowanie od najdalszego) kosztuje ok. 1 us na trojkat, czyli kilkanascie
milisekund na klatke - dosc, zeby odswiezac podglad kilkanascie razy na
sekunde obok petli sterowania, i bez zadnej dodatkowej zaleznosci graficznej.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .model import ArmModel


@dataclass
class Camera:
    """Kamera orbitujaca wokol punktu obserwacji."""

    azimuth_deg: float = 35.0
    elevation_deg: float = 22.0
    distance: float = 0.62
    target: tuple[float, float, float] = (0.05, 0.0, 0.12)
    fov_deg: float = 42.0

    def orbit(self, d_azimuth: float, d_elevation: float) -> None:
        self.azimuth_deg = (self.azimuth_deg + d_azimuth) % 360.0
        self.elevation_deg = float(min(max(self.elevation_deg + d_elevation, -85.0), 85.0))

    def zoom(self, factor: float) -> None:
        self.distance = float(min(max(self.distance * factor, 0.18), 2.0))

    def view_matrix(self) -> np.ndarray:
        """Macierz swiat -> kamera (kamera patrzy wzdluz -Z, os Y w gore)."""
        az = math.radians(self.azimuth_deg)
        el = math.radians(self.elevation_deg)
        target = np.array(self.target, dtype=np.float64)

        eye = target + self.distance * np.array(
            [math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)]
        )
        world_up = np.array([0.0, 0.0, 1.0])

        forward = target - eye
        forward /= np.linalg.norm(forward) or 1.0
        right = np.cross(forward, world_up)
        norm = np.linalg.norm(right)
        if norm < 1e-9:  # kamera dokladnie nad ukladem - dowolne "prawo"
            right = np.array([1.0, 0.0, 0.0])
        else:
            right /= norm
        up = np.cross(right, forward)

        view = np.eye(4)
        view[0, :3], view[1, :3], view[2, :3] = right, up, -forward
        view[:3, 3] = -view[:3, :3] @ eye
        return view


class Renderer3D:
    """Rysuje ramie w zadanej pozie na obrazie BGR."""

    #: Kierunek swiatla w ukladzie kamery (lekko z gory i z lewej).
    LIGHT = np.array([-0.35, 0.55, 0.75])

    #: Minimalne pole trojkata w pikselach kwadratowych (podwojone pole ze
    #: znakiem). Ponizej tego progu trojkat nie zmienilby ani jednego piksela.
    MIN_AREA_PX = 1.0

    def __init__(
        self,
        model: ArmModel,
        size: tuple[int, int] = (360, 460),
        background: tuple[int, int, int] = (44, 40, 38),
    ):
        self.model = model
        self.width, self.height = size
        self.background = background
        self.camera = Camera()
        self.fit()
        self._light = self.LIGHT / np.linalg.norm(self.LIGHT)
        # Kolory czlonow trzymamy w BGR, bo OpenCV rysuje w BGR.
        self._colors_bgr = model.link_colors[:, ::-1].astype(np.float32)

    def fit(self, margin: float = 0.88) -> None:
        """Dobiera odleglosc i punkt obserwacji tak, by ramie miescilo sie w kadrze.

        Kadr ustalamy RAZ, na podstawie skrajnych poz - gdyby dopasowywac go
        do biezacej pozy, podglad "oddychalby" przy kazdym ruchu.
        """
        # Skrajne pozy w plaszczyznie pionowej. Obrotu podstawy tu nie ma
        # celowo: ramie obracajace sie o +-110 stopni zakresla dysk, ktorego
        # pudelko jest duzo wieksze niz sam mechanizm - kadr dopasowany do
        # niego pokazywalby robota jako punkcik.
        extremes = [
            {},  # wyciagniete do przodu
            {"shoulder_lift": -100.0, "elbow_flex": 96.8, "wrist_flex": -95.0},  # zlozone
            {"shoulder_lift": -60.0, "elbow_flex": 60.0, "wrist_flex": -30.0},   # robocza
            {"shoulder_lift": 45.0, "elbow_flex": -45.0},                        # opuszczone
        ]
        points = np.concatenate([self.model.posed_vertices(pose) for pose in extremes])
        low, high = points.min(axis=0), points.max(axis=0)
        center = (low + high) * 0.5
        # Promien kuli otaczajacej, a nie polowa przekatnej pudelka - dla
        # bryly plaskiej przekatna zawyza rozmiar o kilkadziesiat procent.
        radius = float(np.linalg.norm(points - center, axis=1).max())

        self.camera.target = (float(center[0]), float(center[1]), float(center[2]))
        half_fov = math.radians(self.camera.fov_deg) * 0.5
        # Kadr jest wezszy niz wyzszy, wiec o dopasowaniu decyduje szerokosc.
        aspect = self.width / self.height
        effective = half_fov if aspect >= 1.0 else math.atan(math.tan(half_fov) * aspect)
        self.camera.distance = radius / max(math.tan(effective), 1e-6) * margin

    # --------------------------------------------------------------- render
    def render(
        self,
        joints_deg: dict[str, float],
        ee_target: tuple[float, float, float] | None = None,
        highlight: tuple[int, int, int] | None = None,
    ) -> np.ndarray:
        image = np.full((self.height, self.width, 3), self.background, dtype=np.uint8)

        view = self.camera.view_matrix()
        world = self.model.posed_vertices(joints_deg).astype(np.float64)
        camera_space = world @ view[:3, :3].T + view[:3, 3]

        self._draw_ground(image, view)

        faces = self.model.faces
        tri = camera_space[faces]                       # (M, 3, 3)
        depth = -tri[:, :, 2]                           # dodatnie = przed kamera

        # Odrzucamy trojkaty za kamera lub przecinajace plaszczyzne obrazu.
        visible = depth.min(axis=1) > 1e-3
        if not visible.any():
            return image
        faces_v, tri, depth = faces[visible], tri[visible], depth[visible]

        screen = self._project(tri.reshape(-1, 3)).reshape(-1, 3, 2)

        # Eliminacja scianek tylnych: znak pola w ukladzie ekranu.
        edge1 = screen[:, 1] - screen[:, 0]
        edge2 = screen[:, 2] - screen[:, 0]
        area = edge1[:, 0] * edge2[:, 1] - edge1[:, 1] * edge2[:, 0]
        # Odrzucamy scianki tylne (pole dodatnie) oraz trojkaty mniejsze niz
        # pol piksela - te drugie i tak nie zapalilyby zadnego piksela, a przy
        # tej gestosci siatki stanowia spora czesc calosci.
        front = area < -self.MIN_AREA_PX
        if not front.any():
            return image
        screen, tri, depth, faces_v = screen[front], tri[front], depth[front], faces_v[front]
        links = self.model.face_link[visible][front]

        shade = self._shading(tri)
        base = self._colors_bgr[links]
        if highlight is not None:
            base = np.full_like(base, np.array(highlight, dtype=np.float32)[::-1])
        colors = np.clip(base * shade[:, None], 0, 255).astype(np.uint8)

        # Algorytm malarza: od najdalszych do najblizszych.
        order = np.argsort(-depth.mean(axis=1))
        points = screen[order].astype(np.int32)
        # `tolist()` na kolorach i iteracja po wierszach tablicy sa tu istotne:
        # zamiana kolorow na liczby Pythona przy kazdym trojkacie kosztowala
        # 60% wiecej czasu na cala klatke.
        colors_list = colors[order].tolist()

        for triangle, color in zip(points, colors_list):
            cv2.fillConvexPoly(image, triangle, color, cv2.LINE_8)

        if ee_target is not None:
            self._draw_target(image, view, ee_target)
        return image

    # ------------------------------------------------------------- pomocnicze
    def _project(self, points_camera: np.ndarray) -> np.ndarray:
        """Rzut perspektywiczny punktow z ukladu kamery na piksele."""
        focal = 0.5 * self.height / math.tan(math.radians(self.camera.fov_deg) * 0.5)
        z = np.maximum(-points_camera[:, 2], 1e-4)
        x = points_camera[:, 0] / z * focal + self.width * 0.5
        y = -points_camera[:, 1] / z * focal + self.height * 0.5
        return np.stack([x, y], axis=1)

    def _shading(self, tri_camera: np.ndarray) -> np.ndarray:
        """Cieniowanie plaskie Lamberta z niewielkim swiatlem otoczenia."""
        normals = np.cross(
            tri_camera[:, 1] - tri_camera[:, 0], tri_camera[:, 2] - tri_camera[:, 0]
        )
        lengths = np.linalg.norm(normals, axis=1)
        lengths[lengths < 1e-12] = 1.0
        normals /= lengths[:, None]

        lambert = np.abs(normals @ self._light)
        return 0.32 + 0.68 * lambert

    def _draw_ground(self, image: np.ndarray, view: np.ndarray) -> None:
        """Siatka podloza - daje oku odniesienie do wysokosci i obrotu."""
        step, count = 0.05, 6
        limit = step * count
        lines: list[np.ndarray] = []
        for i in range(-count, count + 1):
            lines.append(np.array([[i * step, -limit, 0.0], [i * step, limit, 0.0]]))
            lines.append(np.array([[-limit, i * step, 0.0], [limit, i * step, 0.0]]))

        points = np.concatenate(lines)
        camera_space = points @ view[:3, :3].T + view[:3, 3]
        if (-camera_space[:, 2] <= 1e-3).any():
            return  # czesc siatki za kamera - pomijamy, zeby nie rysowac smieci
        screen = self._project(camera_space).astype(np.int32).reshape(-1, 2, 2)
        for a, b in screen:
            cv2.line(image, tuple(a), tuple(b), (62, 58, 55), 1, cv2.LINE_AA)

    def _draw_target(
        self, image: np.ndarray, view: np.ndarray, target: tuple[float, float, float]
    ) -> None:
        point = np.array(target, dtype=np.float64) @ view[:3, :3].T + view[:3, 3]
        if -point[2] <= 1e-3:
            return
        pixel = self._project(point[None])[0].astype(int)
        cv2.drawMarker(image, tuple(pixel), (60, 190, 245), cv2.MARKER_TILTED_CROSS, 14, 2)
