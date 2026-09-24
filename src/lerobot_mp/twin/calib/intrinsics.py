"""Intrynsyki kamery z tablicy ChArUco: K i dystorsja, zanim kamera zobaczy karte w chwytaku.

Bez tego kamera ma nominalne K z 65 st. pola widzenia, a kalibracja polozenia
(karta w chwytaku) dopasowuje poze do zlej ogniskowej: wychodzi pewna siebie
i przesunieta. Dlatego intrynsyki sa pierwszym krokiem kazdej nowej kamery.

    board = Board()
    cv2.imwrite("tablica.png", board.image())      # druk A4, skala 100%, ZMIERZ kwadrat
    col = Collector(board, (640, 480))
    for frame in frames:
        ok, why = col.add(frame)                   # tylko kadry, ktore wnosza nowe ujecie
    res = col.solve()                              # K, dist, residuum, werdykt

Tablica to ArUco 5x5 - inny slownik niz AprilTagi karty (36h11), wiec detektor
jednej nigdy nie pomyli sie z druga. Rogi ChArUco sa doszlifowane subpikselowo
w konwencji srodkow pikseli, czyli tej samej, w ktorej liczy `projectPoints`
- bez polpikselowej poprawki, ktorej wymagaja rogi AprilTagow.

Kadr jest przyjmowany, gdy wnosi NOWE ujecie: inne miejsce w kadrze, inna
odleglosc albo inne pochylenie tablicy niz wszystkie dotychczasowe. Dwadziescia
kadrow z tej samej pozycji daje pewne siebie i zle K - z tego samego powodu
fala kalibracyjna pilnuje rozrzutu obrotow.

Samo przesuwanie tablicy po kadrze to za malo: tablica zawsze rownolegla do
matrycy nie odroznia ogniskowej od odleglosci. Dlatego `solve` wymaga rozrzutu
POCHYLEN tablicy, a gdy kadr jest juz pokryty, `add` przyjmuje tylko kadry,
ktore ten rozrzut poszerzaja.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

DICT = cv2.aruco.DICT_5X5_100


@dataclass(frozen=True)
class Board:
    #: Kwadraty (kolumny, wiersze) i ich bok [m]; znacznik zajmuje `marker` boku kwadratu.
    squares: tuple[int, int] = (9, 6)
    square: float = 0.028
    marker: float = 0.021

    def cv(self) -> cv2.aruco.CharucoBoard:
        return cv2.aruco.CharucoBoard(self.squares, self.square, self.marker, cv2.aruco.getPredefinedDictionary(DICT))

    @property
    def size(self) -> tuple[float, float]:
        return self.squares[0] * self.square, self.squares[1] * self.square

    def board_image(self, px_per_m: float) -> np.ndarray:
        w, h = (int(round(s * px_per_m)) for s in self.size)
        return self.cv().generateImage((w, h), marginSize=0, borderBits=1)

    def image(self, dpi: int = 300) -> np.ndarray:
        """Arkusz A4 w poziomie z tablica na srodku i opisem, do druku w skali 100%."""
        px_per_m = dpi / 0.0254
        W, H = int(0.297 * px_per_m), int(0.210 * px_per_m)
        sheet = np.full((H, W), 255, np.uint8)
        b = self.board_image(px_per_m)
        y0, x0 = (H - b.shape[0]) // 2, (W - b.shape[1]) // 2
        sheet[y0:y0 + b.shape[0], x0:x0 + b.shape[1]] = b
        text = (f"ChArUco {self.squares[0]}x{self.squares[1]}, kwadrat {self.square * 1000:.1f} mm - "
                f"drukuj w skali 100% i ZMIERZ kwadrat linijka")
        cv2.putText(sheet, text, (x0, max(40, y0 - 25)), cv2.FONT_HERSHEY_SIMPLEX, dpi / 300.0, 0, 2)
        return sheet


@dataclass
class View:
    obj: np.ndarray          # (n, 3) rogi w ukladzie tablicy
    img: np.ndarray          # (n, 2) ich piksele
    #: Odcisk ujecia: srodek i wielkosc w kadrze, pochylenie tablicy (do pilnowania roznorodnosci).
    centre: np.ndarray = field(default_factory=lambda: np.zeros(2))
    scale: float = 0.0
    normal: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))


@dataclass
class Result:
    K: np.ndarray
    dist: np.ndarray
    rms_px: float
    per_view_px: list[float]
    n_views: int
    coverage: float
    size: tuple[int, int]
    trusted: bool
    reason: str = ""


class Collector:
    def __init__(self, board: Board, size: tuple[int, int], *, min_corners: int = 12, min_move: float = 0.08,
                 min_tilt_deg: float = 8.0, grid: tuple[int, int] = (8, 6), min_tilt_spread: float = 35.0,
                 prefer_tilt_after: int = 6, prefer_tilt_coverage: float = 0.55):
        self.board = board
        self.cv_board = board.cv()
        self.detector = cv2.aruco.CharucoDetector(self.cv_board)
        self.size = (int(size[0]), int(size[1]))
        self.min_corners, self.min_move, self.min_tilt = min_corners, min_move, np.radians(min_tilt_deg)
        #: Najmniejszy laczny rozrzut pochylen tablicy [st.] (hypot rozrzutow w obu osiach kadru),
        #: przy ktorym K jest zaufane.
        self.min_tilt_spread = float(min_tilt_spread)
        #: Po tylu kadrach i takim pokryciu kadru, dopoki brakuje pochylen, przyjmujemy tylko
        #: kadry, ktore je poszerzaja - samo przesuwanie tablicy nic juz nie wnosi.
        self.prefer_tilt_after, self.prefer_tilt_coverage = prefer_tilt_after, prefer_tilt_coverage
        self.grid = grid
        self.views: list[View] = []
        self.hits = np.zeros(grid[::-1], bool)
        W, H = self.size
        f = (W / 2) / np.tan(np.radians(32.5))            # nominalne K tylko do odcisku pochylenia
        self._K0 = np.array([[f, 0, (W - 1) / 2], [0, f, (H - 1) / 2], [0, 0, 1.0]])

    def detect(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        """(obj (n, 3), img (n, 2)) rogow tablicy albo None."""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
        corners, ids, _, _ = self.detector.detectBoard(gray)
        if ids is None or len(ids) < self.min_corners:
            return None
        obj, pts = self.cv_board.matchImagePoints(corners, ids)
        if obj is None or len(obj) < self.min_corners:
            return None
        return obj.reshape(-1, 3).astype(np.float64), pts.reshape(-1, 2).astype(np.float64)

    def _view(self, obj: np.ndarray, pts: np.ndarray) -> View:
        W = self.size[0]
        centre = pts.mean(0) / np.array(self.size, float)
        scale = float(np.sqrt(cv2.contourArea(cv2.convexHull(pts.astype(np.float32))))) / W
        ok, rvec, _ = cv2.solvePnP(obj, pts, self._K0, None)
        normal = cv2.Rodrigues(rvec)[0][:, 2] if ok else np.array([0.0, 0.0, 1.0])
        return View(obj, pts, centre, scale, normal)

    def novelty(self, v: View) -> str | None:
        """None, gdy ujecie jest nowe; inaczej powod, dla ktorego nie."""
        for old in self.views:
            moved = np.linalg.norm(v.centre - old.centre) > self.min_move
            rescaled = abs(v.scale - old.scale) > self.min_move
            tilted = np.arccos(np.clip(abs(float(v.normal @ old.normal)), -1, 1)) > self.min_tilt
            if not (moved or rescaled or tilted):
                return "to samo ujecie co wczesniej - przesun, przybliz albo pochyl tablice"
        if len(self.views) >= self.prefer_tilt_after and self.coverage >= self.prefer_tilt_coverage:
            sx, sy = self.tilt_spread()
            now = float(np.hypot(sx, sy))
            if now < self.min_tilt_spread and float(np.hypot(*self.tilt_spread(v))) < now + 1.0:
                axis = "w gore albo w dol" if sx <= sy else "w lewo albo w prawo"
                return f"kadr jest pokryty - teraz pochyl tablice mocniej {axis} (20-30 st.)"
        return None

    def add(self, img: np.ndarray, force: bool = False) -> tuple[bool, str]:
        det = self.detect(img)
        if det is None:
            return False, f"tablica niewidoczna albo mniej niz {self.min_corners} rogow"
        v = self._view(*det)
        why = None if force else self.novelty(v)
        if why:
            return False, why
        self.views.append(v)
        gx = np.clip((v.img[:, 0] / self.size[0] * self.grid[0]).astype(int), 0, self.grid[0] - 1)
        gy = np.clip((v.img[:, 1] / self.size[1] * self.grid[1]).astype(int), 0, self.grid[1] - 1)
        self.hits[gy, gx] = True
        return True, f"kadr {len(self.views)}: {len(v.obj)} rogow"

    @property
    def coverage(self) -> float:
        return float(self.hits.mean())

    def tilt_spread(self, extra: View | None = None) -> tuple[float, float]:
        """Rozrzut pochylen tablicy [st.]: wokol osi poziomej i pionowej kadru (max - min)."""
        views = self.views + ([extra] if extra is not None else [])
        if not views:
            return 0.0, 0.0
        n = np.array([v.normal for v in views], float)
        n *= np.where(n[:, 2] < 0, -1.0, 1.0)[:, None]
        about_x = np.degrees(np.arctan2(n[:, 1], n[:, 2]))
        about_y = np.degrees(np.arctan2(n[:, 0], n[:, 2]))
        return float(np.ptp(about_x)), float(np.ptp(about_y))

    def solve(self, *, max_rms: float = 0.6, min_views: int = 8, min_coverage: float = 0.55,
              min_tilt_spread: float | None = None) -> Result:
        if len(self.views) < 4:
            raise RuntimeError(f"za malo kadrow ({len(self.views)}) - potrzeba co najmniej 4, zalecane {min_views}+")
        obj = [v.obj.astype(np.float32) for v in self.views]
        pts = [v.img.astype(np.float32) for v in self.views]
        # Przy malej liczbie kadrow trzeci wspolczynnik radialny tylko dopasowuje szum.
        flags = cv2.CALIB_FIX_K3 if len(self.views) < 15 else 0
        out = cv2.calibrateCameraExtended(obj, pts, self.size, None, None, flags=flags)
        rms, K, dist = out[0], out[1], out[2]
        per_view = [float(e) for e in np.asarray(out[7]).ravel()]
        reasons = []
        if rms > max_rms:
            reasons.append(f"residuum {rms:.2f} px > {max_rms} px")
        if len(self.views) < min_views:
            reasons.append(f"tylko {len(self.views)} kadrow, zalecane {min_views}")
        if self.coverage < min_coverage:
            reasons.append(f"tablica pokryla {self.coverage:.0%} kadru, potrzeba {min_coverage:.0%} - pokaz ja w rogach")
        # Plaska tablica zawsze rownolegla do matrycy to przypadek zdegenerowany: ogniskowa
        # i odleglosc wymieniaja sie jedna za druga, a residuum zostaje ~0,2 px. Zmierzone na
        # syntetycznych rogach (fx 610, szum 0,15 px, 14 kadrow): pochylenia do 2 st. dawaly
        # fx od 424 do 27340, do 5 st. 500-720 - wszystko "zaufane". Liczy sie laczny rozrzut
        # (wystarczy pochylanie wokol jednej osi): ~21 st. - fx do 3-5% obok, ~32 st. - do 2%,
        # >= 40 st. - ponizej 1%. Stad bramka na lacznym rozrzucie.
        min_spread = self.min_tilt_spread if min_tilt_spread is None else min_tilt_spread
        sx, sy = self.tilt_spread()
        if np.hypot(sx, sy) < min_spread:
            reasons.append(f"tablica prawie zawsze rownolegla do kamery (rozrzut pochylen {sx:.0f} st. "
                           f"gora-dol, {sy:.0f} st. lewo-prawo, potrzeba lacznie {min_spread:.0f}) - "
                           f"pochylaj ja o 20-30 st. w gore i w dol albo w lewo i w prawo")
        # Druga siatka bezpieczenstwa: ogniskowa spoza rozsadnego pola widzenia kamery
        # internetowej (ok. 36-104 st.) albo wyraznie niekwadratowe piksele to zle K,
        # nawet przy malym residuum.
        f0 = self._K0[0, 0]
        fx, fy = float(K[0, 0]), float(K[1, 1])
        if not (0.5 * f0 <= fx <= 2.0 * f0 and 0.5 * f0 <= fy <= 2.0 * f0) or abs(fx / fy - 1.0) > 0.1:
            reasons.append(f"nieprawdopodobna ogniskowa fx {fx:.0f}, fy {fy:.0f} px "
                           f"(oczekiwane okolo {f0:.0f} px) - zbierz kadry od nowa")
        return Result(np.asarray(K, float), np.asarray(dist, float).ravel(), float(rms), per_view, len(self.views),
                      self.coverage, self.size, not reasons, "; ".join(reasons))


# ------------------------------------------------------------ symulacja
def simulate(K: np.ndarray, size: tuple[int, int] = (640, 480), n: int = 14, seed: int = 0,
             board: Board | None = None) -> tuple[Result, Collector]:
    """Sesja w blizniaku: tablica na blacie, kamera o znanym K oglada ja z roznych miejsc.

    Tablica jest renderowana tym samym rendererem i ta sama sciezka kamery co
    scena blizniaka - wiec to sprawdza cala konwencje pikseli od K do rogow.
    """
    import mujoco

    from .. import scene as sc
    from ..kinematics import pose
    from ..robots import SO101

    board = board or Board()
    rng = np.random.default_rng(seed)
    W, H = size
    img = board.board_image(4000.0)
    # Tablica lezy plasko na blacie, przed ramieniem; uklad tablicy OpenCV: (0,0) w lewym gornym
    # rogu, x w prawo, y w dol obrazu. Panel ma srodek w srodku i +y do gory obrazu.
    bw, bh = board.size
    centre = np.array([0.25, 0.0, 0.001])
    T_panel = pose(np.eye(3), centre)
    T_board2base = T_panel @ pose(np.diag([1.0, -1.0, -1.0]), np.array([-bw / 2, bh / 2, 0.0]))

    def look(eye, target):
        z = target - eye
        z /= np.linalg.norm(z)
        x = np.cross(z, [0.0, 0.0, 1.0])
        x /= np.linalg.norm(x)
        return pose(np.column_stack([x, np.cross(z, x), z]), eye)

    view = sc.CameraView("intr", np.asarray(K, float), W, H, look(np.array([0.25, -0.3, 0.4]), centre))
    cfg = sc.SceneConfig(SO101, cameras=[view], panels=[sc.Panel("tablica", img, board.size, T_panel)])
    col = Collector(board, size)
    with sc.build(cfg) as scene:
        scene.set_joints({"shoulder_pan": 110.0, "shoulder_lift": -95.0, "elbow_flex": 90.0})
        cam = scene.model.camera("intr").id
        tries = 0
        while len(col.views) < n and tries < 20 * n:
            tries += 1
            # Kamera 25-50 cm od tablicy, z roznych stron, cel przesuniety po tablicy -
            # tablica trafia w rozne miejsca kadru i pod roznym katem.
            az, el = rng.uniform(-np.pi, np.pi), rng.uniform(0.6, 1.35)
            dist = rng.uniform(0.25, 0.5)
            eye = centre + dist * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
            target = centre + np.r_[rng.uniform(-0.07, 0.07, 2), 0.0]
            T = scene.T_base2world @ look(eye, target)
            m = scene.model
            m.cam_pos[cam] = T[:3, 3]
            q = np.zeros(4)
            mujoco.mju_mat2Quat(q, (T[:3, :3] @ sc.CV_TO_MJ).ravel())
            m.cam_quat[cam] = q
            mujoco.mj_forward(m, scene.data)
            col.add(scene.render("intr"))
    return col.solve(), col
