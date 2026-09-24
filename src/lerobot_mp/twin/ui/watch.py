"""Czy kamera sie przestawila od kalibracji - kadr teraz kontra kadr zapamietany po niej.

Kalibracja jest prawdziwa tylko dopoty, dopoki kamera stoi tam, gdzie stala.
Potracenie statywu o 1 st. to przy f = 600 px ok. 10 px przesuniecia kadru -
mapa stolu rozjezdza sie o centymetry, a polityka widzi co innego niz w
symulacji, i nikt tego nie zauwaza.

Porownanie to dopasowanie obrotu i przesuniecia (ECC, plus skala z dopasowania
afinicznego) kadru teraz do kadru odniesienia, startujace z przesuniecia
z korelacji fazowej, na obrazie
w odcieniach szarosci, z RAMIENIEM WYCIETYM: ramie rusza sie miedzy kadrami,
a jego piksele przesuwalyby wynik. Maske ramienia daje blizniak - segmentacja
geomow robota wyrenderowana z tej samej kamery w jej skalibrowanej pozie.

Miara jest NAJWIEKSZE przesuniecie naroznika kadru, nie przesuniecie calosci:
sama korelacja fazowa nie widziala obrotu wokol osi optycznej ani ruchu wzdluz
niej (srodek kadru stoi, brzegi jada). Zmierzone na kadrze 640x480: obrot
o 3 st. przesuwal narozniki o 21 px, a korelacja zglaszala 1,6 px; zblizenie
o 4% - 16 px naroznikow i 0,55 px korelacji. Prog 3 px nie ruszal.
"""

from __future__ import annotations

import cv2
import mujoco
import numpy as np

from ..scene import PREFIX


def arm_mask(scene, camera: str, dilate: int = 9) -> np.ndarray:
    """Piksele ramienia w kadrze kamery sceny (bool, poszerzone o `dilate` px).

    Render sceny to kamera otworkowa bez dystorsji - do surowego kadru prawdziwej
    kamery maske trzeba jeszcze przepuscic przez `distort_mask`.
    """
    view = scene.camera(camera)
    key = ("seg", view.width, view.height)
    r = scene._renderers.get(key)
    if r is None:
        r = mujoco.Renderer(scene.model, height=view.height, width=view.width)
        r.enable_segmentation_rendering()
        scene._renderers[key] = r
    r.update_scene(scene.data, camera=camera)
    seg = r.render()
    m = scene.model
    robot = np.array([m.body(m.geom_bodyid[g]).name.startswith(PREFIX) for g in range(m.ngeom)])
    ids, types = seg[..., 0], seg[..., 1]
    is_geom = (types == int(mujoco.mjtObj.mjOBJ_GEOM)) & (ids >= 0)
    mask = np.zeros(ids.shape, bool)
    mask[is_geom] = robot[ids[is_geom]]
    if dilate:
        mask = cv2.dilate(mask.astype(np.uint8), np.ones((dilate, dilate), np.uint8)) > 0
    return mask


_DISTORT_MAPS: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}


def distortion_maps(K: np.ndarray, dist: np.ndarray | None, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray] | None:
    """Mapy `cv2.remap`: piksel SUROWEGO kadru -> piksel renderu otworkowego z tym samym K.

    None, gdy kamera nie ma dystorsji (render i kadr maja wtedy te sama geometrie).
    """
    if dist is None or not np.any(np.asarray(dist, float)):
        return None
    K = np.asarray(K, float)
    d = np.asarray(dist, float).ravel()
    w, h = size
    key = (w, h, K.tobytes(), d.tobytes())
    maps = _DISTORT_MAPS.get(key)
    if maps is None:
        u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        pts = np.stack([u.ravel(), v.ravel()], axis=1)[:, None, :]
        und = cv2.undistortPoints(pts, K, d, P=K).reshape(h, w, 2)
        maps = (np.ascontiguousarray(und[..., 0], np.float32), np.ascontiguousarray(und[..., 1], np.float32))
        if len(_DISTORT_MAPS) > 16:
            _DISTORT_MAPS.clear()
        _DISTORT_MAPS[key] = maps
    return maps


def distort_mask(mask: np.ndarray, K: np.ndarray, dist: np.ndarray | None) -> np.ndarray:
    """Maska z renderu (otworkowego) w geometrii surowego kadru kamery z dystorsja `dist`.

    Dla kamerki z k1 ok. -0,3 maska renderu mijala ramie w kadrze o 6 px przy
    (480, 360) i ~20 px przy (560, 420) - piksele ramienia glosowaly wtedy jako
    "tu nie ma kostki", a percepcja przy chwytaniu gubila kostke.
    """
    maps = distortion_maps(K, dist, (mask.shape[1], mask.shape[0]))
    if maps is None:
        return mask
    out = cv2.remap(mask.astype(np.uint8), maps[0], maps[1], cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    return out > 0


def _gray(img: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    return g.astype(np.float32)


def _prep(img: np.ndarray, mask: np.ndarray | None, scale: float) -> np.ndarray:
    g = _gray(img)
    if mask is not None and mask.any():
        # Wyciete piksele wypelniamy rozmyciem otoczenia, zeby krawedz maski nie
        # stala sie "krawedzia", ktora korelacja chetnie by dopasowala.
        blur = cv2.GaussianBlur(g, (0, 0), 15)
        g = np.where(mask, blur, g)
    return cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def _small_mask(mask: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray | None:
    if mask is None or not mask.any():
        return None
    return cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0


class CameraWatch:
    #: Ponizej tej korelacji (ECC) dopasowanie nie jest wiarygodne - kadr jest
    #: "nie wiadomo jaki", a to traktujemy jak przestawiony, nie jak zero ruchu.
    #: Zmierzone: ten sam kadr po obrocie/przesunieciu 0,99+, zupelnie inny kadr 0,23.
    min_cc = 0.6
    #: Zmiana skali ponizej tej nie liczy sie jako ruch kamery (szum dopasowania).
    zoom_deadband = 0.015

    def __init__(self, threshold_px: float = 3.0, scale: float = 0.5):
        self.threshold = threshold_px
        self.scale = scale
        self.refs: dict[str, np.ndarray] = {}
        self._ref_masks: dict[str, np.ndarray | None] = {}
        self.shift: dict[str, float] = {}
        #: Szczegoly ostatniego sprawdzenia: przesuniecie calosci, obrot, skala, pewnosc.
        self.detail: dict[str, dict[str, float]] = {}

    def remember(self, name: str, img: np.ndarray, mask: np.ndarray | None = None) -> None:
        self.refs[name] = _prep(img, mask, self.scale)
        self._ref_masks[name] = _small_mask(mask, self.refs[name].shape)
        self.shift.pop(name, None)
        self.detail.pop(name, None)

    def forget(self, name: str) -> None:
        self.refs.pop(name, None)
        self._ref_masks.pop(name, None)
        self.shift.pop(name, None)
        self.detail.pop(name, None)

    def check(self, name: str, img: np.ndarray, mask: np.ndarray | None = None) -> float | None:
        """Najwieksze przesuniecie naroznika kadru wzgledem odniesienia [px pelnej rozdzielczosci] albo None.

        `inf`, gdy kadru nie da sie dopasowac do odniesienia (inna scena, zaslonieta
        kamera) - nie wiadomo, gdzie kamera stoi, wiec kalibracji nie ufamy.
        """
        ref = self.refs.get(name)
        if ref is None or img is None:
            return None
        cur = _prep(img, mask, self.scale)
        if cur.shape != ref.shape:
            return None
        win = cv2.createHanningWindow(cur.shape[::-1], cv2.CV_32F)
        # KOPIE: `phaseCorrelate` z oknem mnozy wejscia przez okno W MIEJSCU (OpenCV 5.0) -
        # zapamietany kadr odniesienia wygaszal sie co sprawdzenie (co 2 s) do samego
        # srodka, a brzegi, na ktorych widac obrot, znikaly z porownania.
        (dx, dy), response = cv2.phaseCorrelate(ref.copy(), cur.copy(), win)
        valid = np.ones(ref.shape, bool)
        for m in (self._ref_masks.get(name), _small_mask(mask, ref.shape)):
            if m is not None:
                valid &= ~m
        valid8 = valid.astype(np.uint8)
        crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-6)
        cc = float("nan")
        # Obrot + przesuniecie (ECC, MOTION_EUCLIDEAN). Pelne afiniczne dopasowanie
        # na ubogim w teksture kadrze (blat, ramie) "znajdowalo" obrot 0,4 st. i skale
        # 1,006 po samym ruchu ramienia (cienie) - 4,7 px naroznika, prawie prog.
        # gaussFiltSize=1: domyslne rozmycie 5 px na kadrze w polowie rozdzielczosci
        # zanizalo dokladnosc (przesuniecie 5 px mierzone jako 5,9 px).
        warp = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], np.float32)
        try:
            cc, warp = cv2.findTransformECC(ref, cur, warp, cv2.MOTION_EUCLIDEAN, crit, valid8, 1)
            cc = float(cc)
        except cv2.error:
            warp = None                                   # nie zbieglo sie - kadr zbyt inny albo bez tekstury
        h, w = ref.shape
        corners = np.array([[0, 0, 1], [w - 1, 0, 1], [0, h - 1, 1], [w - 1, h - 1, 1]], float)
        if warp is not None and cc >= self.min_cc:
            disp = corners @ warp.astype(float).T - corners[:, :2]
            shift = float(np.linalg.norm(disp, axis=1).max()) / self.scale
            rot = float(np.degrees(np.arctan2(warp[1, 0], warp[0, 0])))
            # Ruch wzdluz osi optycznej = skala. Z afinicznego dopasowania, ale z martwa
            # strefa `zoom_deadband` (patrz wyzej: 0,6% skali z samego ruchu ramienia).
            scale = 1.0
            try:
                _, aff = cv2.findTransformECC(ref, cur, warp.copy(), cv2.MOTION_AFFINE, crit, valid8, 1)
                scale = float(np.sqrt(abs(np.linalg.det(aff[:, :2].astype(float)))))
            except cv2.error:
                pass
            half_diag = 0.5 * float(np.hypot(w - 1, h - 1)) / self.scale
            shift += max(0.0, abs(scale - 1.0) - self.zoom_deadband) * half_diag
        elif warp is None and response >= 0.3:
            # Kadr bez tekstury dla ECC, ale korelacja fazowa pewna - zostaje jej przesuniecie.
            shift, rot, scale = float(np.hypot(dx, dy)) / self.scale, float("nan"), float("nan")
        else:
            shift, rot, scale = float("inf"), float("nan"), float("nan")
        self.shift[name] = shift
        self.detail[name] = dict(dx=float(dx) / self.scale, dy=float(dy) / self.scale, rot_deg=rot, scale=scale,
                                 cc=cc, response=float(response))
        return shift

    def moved(self, name: str) -> bool:
        return self.shift.get(name, 0.0) > self.threshold
