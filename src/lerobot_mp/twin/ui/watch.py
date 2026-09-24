"""Czy kamera sie przestawila od kalibracji - kadr teraz kontra kadr zapamietany po niej.

Kalibracja jest prawdziwa tylko dopoty, dopoki kamera stoi tam, gdzie stala.
Potracenie statywu o 1 st. to przy f = 600 px ok. 10 px przesuniecia kadru -
mapa stolu rozjezdza sie o centymetry, a polityka widzi co innego niz w
symulacji, i nikt tego nie zauwaza.

Porownanie to korelacja fazowa (przesuniecie calego kadru) na obrazie w
odcieniach szarosci, z RAMIENIEM WYCIETYM: ramie rusza sie miedzy kadrami,
a jego piksele przesuwalyby wynik. Maske ramienia daje blizniak - segmentacja
geomow robota wyrenderowana z tej samej kamery w jej skalibrowanej pozie.
"""

from __future__ import annotations

import cv2
import mujoco
import numpy as np

from ..scene import PREFIX


def arm_mask(scene, camera: str, dilate: int = 9) -> np.ndarray:
    """Piksele ramienia w kadrze kamery sceny (bool, poszerzone o `dilate` px)."""
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


def _prep(img: np.ndarray, mask: np.ndarray | None, scale: float) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    g = g.astype(np.float32)
    if mask is not None and mask.any():
        # Wyciete piksele wypelniamy rozmyciem otoczenia, zeby krawedz maski nie
        # stala sie "krawedzia", ktora korelacja chetnie by dopasowala.
        blur = cv2.GaussianBlur(g, (0, 0), 15)
        g = np.where(mask, blur, g)
    return cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


class CameraWatch:
    def __init__(self, threshold_px: float = 3.0, scale: float = 0.5):
        self.threshold = threshold_px
        self.scale = scale
        self.refs: dict[str, np.ndarray] = {}
        self.shift: dict[str, float] = {}

    def remember(self, name: str, img: np.ndarray, mask: np.ndarray | None = None) -> None:
        self.refs[name] = _prep(img, mask, self.scale)
        self.shift.pop(name, None)

    def forget(self, name: str) -> None:
        self.refs.pop(name, None)
        self.shift.pop(name, None)

    def check(self, name: str, img: np.ndarray, mask: np.ndarray | None = None) -> float | None:
        """Przesuniecie kadru wzgledem odniesienia [px pelnej rozdzielczosci] albo None."""
        ref = self.refs.get(name)
        if ref is None or img is None:
            return None
        cur = _prep(img, mask, self.scale)
        if cur.shape != ref.shape:
            return None
        win = cv2.createHanningWindow(cur.shape[::-1], cv2.CV_32F)
        (dx, dy), response = cv2.phaseCorrelate(ref, cur, win)
        shift = float(np.hypot(dx, dy)) / self.scale
        self.shift[name] = shift
        return shift

    def moved(self, name: str) -> bool:
        return self.shift.get(name, 0.0) > self.threshold
