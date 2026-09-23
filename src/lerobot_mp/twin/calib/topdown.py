"""Kadr przerysowany na plaszczyzne stolu; mapa stolu sklejona z wielu kamer.

Pomysl z galaxeo-manipulators `sim/planner/topdown.py` (commit 02641c4): skoro
poza kamery jest znana, kazdy kadr mozna przerysowac na plaszczyzne blatu.
Przedmiot stojacy na stole trafia wtedy na mapie w swoje prawdziwe (x, y),
niezaleznie od tego, skad patrzy kamera - polityka nie musi uczyc sie
lokalizacji 3D z kazdego punktu widzenia osobno. Galaxeo zmierzylo, ze to
wlasnie ta zmiana dala pierwsze udane nalania w zamknietej petli.

Dwie roznice wzgledem oryginalu:

* **Rzut zamiast homografii.** Kazdy piksel mapy to punkt na blacie, rzutowany
  do kadru przez `projectPoints` - wiec dystorsja obiektywu jest uwzgledniona,
  a homografia jej nie widzi. Siatka rzutu liczy sie raz na poze kamery.
* **Wiele kamer.** Mapy z kamer sa zszywane z wagami: kamera patrzaca na blat
  z gory widzi go wyrazniej niz ta, ktora patrzy po skosie, wiec dostaje
  wieksza wage. Piksele poza kadrem i za kamera nie licza sie wcale.

Mapa obejmuje `side` metrow w kwadracie wokol `centre` (uklad podstawy ramienia,
blat na z = 0), +y w gore, +x w prawo. Poprawnie lezy tylko to, co jest NA
blacie; to, co wystaje, rozmazuje sie od kamery - zgodnie dla danej pozy.
"""

from __future__ import annotations

from collections.abc import Mapping

import cv2
import numpy as np

SIDE = 0.7      # metry blatu na bok mapy
N = 256         # piksele mapy na bok


def grid(centre_xy, side: float = SIDE, n: int = N, table_z: float = 0.0) -> np.ndarray:
    """Punkty blatu odpowiadajace pikselom mapy, (n, n, 3), uklad podstawy."""
    s = side / n
    x0, y0 = centre_xy[0] - side / 2, centre_xy[1] + side / 2
    u, v = np.meshgrid(np.arange(n) + 0.5, np.arange(n) + 0.5)
    return np.stack([x0 + s * u, y0 - s * v, np.full_like(u, table_z)], axis=-1)


def warp_maps(K, dist, T_cam2base, points: np.ndarray):
    """(map_x, map_y, waga) do `cv2.remap` - gdzie w kadrze lezy kazdy piksel mapy."""
    T = np.linalg.inv(np.asarray(T_cam2base, float))
    P = points.reshape(-1, 3)
    pc = P @ T[:3, :3].T + T[:3, 3]
    front = pc[:, 2] > 1e-3
    px, _ = cv2.projectPoints(P, cv2.Rodrigues(T[:3, :3])[0], T[:3, 3], np.asarray(K, float),
                              None if dist is None else np.asarray(dist, float))
    px = px.reshape(-1, 2)
    # Waga: jak bardzo kamera patrzy na blat z gory - kosinus miedzy promieniem
    # od kamery do punktu a kierunkiem "w dol" blatu, oba w ukladzie kamery.
    ray = pc / np.maximum(np.linalg.norm(pc, axis=1, keepdims=True), 1e-9)
    down = -(T[:3, :3] @ np.array([0.0, 0.0, 1.0]))
    weight = np.clip(ray @ down, 0.0, 1.0) * front
    n = points.shape[0]
    return (px[:, 0].reshape(n, n).astype(np.float32), px[:, 1].reshape(n, n).astype(np.float32),
            weight.reshape(n, n).astype(np.float32))


def topdown(img, K, T_cam2base, centre_xy, dist=None, side: float = SIDE, n: int = N,
            table_z: float = 0.0) -> np.ndarray:
    """Jeden kadr jako mapa blatu (n, n, 3) uint8."""
    mx, my, _ = warp_maps(K, dist, T_cam2base, grid(centre_xy, side, n, table_z))
    return cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def fused(frames: Mapping[str, np.ndarray], cameras: Mapping[str, tuple], centre_xy, side: float = SIDE,
          n: int = N, table_z: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Mapa blatu zszyta z wielu kamer i mapa pokrycia (ile wagi mial kazdy piksel).

    `cameras[nazwa] = (K, dist, T_cam2base)`; kamery bez kadru sa pomijane.
    """
    pts = grid(centre_xy, side, n, table_z)
    acc = np.zeros((n, n, 3), np.float32)
    wsum = np.zeros((n, n), np.float32)
    for name, img in frames.items():
        if name not in cameras or img is None:
            continue
        K, dist, T = cameras[name]
        mx, my, w = warp_maps(K, dist, T, pts)
        h, wid = img.shape[:2]
        inside = (mx >= 0) & (mx <= wid - 1) & (my >= 0) & (my <= h - 1)
        w = w * inside
        warped = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT).astype(np.float32)
        acc += warped * w[..., None]
        wsum += w
    out = np.where(wsum[..., None] > 1e-6, acc / np.maximum(wsum[..., None], 1e-6), 0.0)
    return out.clip(0, 255).astype(np.uint8), wsum


def map_of(p_world, centre_xy, side: float = SIDE, n: int = N) -> np.ndarray:
    """Piksel mapy (u, v) punktu w ukladzie podstawy (jego z jest pomijane)."""
    s = side / n
    x0, y0 = centre_xy[0] - side / 2, centre_xy[1] + side / 2
    p = np.asarray(p_world, float)
    return np.array([(p[0] - x0) / s - 0.5, (y0 - p[1]) / s - 0.5])
