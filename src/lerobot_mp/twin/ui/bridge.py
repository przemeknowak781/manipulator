"""Scena MuJoCo w przegladarce: wezel visera na CIALO, geomy jako jego dzieci.

Siatki (348 tys. trojkatow ramienia) ida do przegladarki RAZ, przy
polaczeniu. Potem leca tylko pozy cial, ktore sie ruszyly: geom siedzi
w swoim ciele na stalej pozie lokalnej, wiec zamiast ~50 geomow co takt
wystarczy ~9 cial - i zero komunikatow, gdy ramie stoi. Wczesniejsza wersja
wysylala poze kazdego geomu 30 razy na sekunde (~3000 komunikatow/s) i
przegladarka nie nadazala: polaczenie zrywalo sie po kilkudziesieciu
sekundach.

Wszystko w ukladzie SWIATA sceny MuJoCo (z w gore), wiec pozy kamer ze
stanowiska przechodza przez `T_base2world` tak samo jak w scenie.
"""

from __future__ import annotations

import mujoco
import numpy as np

#: Grupy geomow, ktore rysujemy: 0-2 to wizualne (MJCF ramienia, stol, karta, obiekty).
VISUAL_GROUPS = (0, 1, 2)


def mat_to_wxyz(R: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).ravel())
    return q


def _rgb(rgba) -> tuple[int, int, int]:
    return tuple(int(np.clip(c, 0, 1) * 255) for c in rgba[:3])


def decimate(vertices: np.ndarray, faces: np.ndarray, cell: float) -> tuple[np.ndarray, np.ndarray]:
    """Uproszczenie siatki przez sklejenie wierzcholkow w szescianach `cell` [m] - tylko do WIDOKU.

    Siatki SO-101 z CAD maja 348 tys. trojkatow; przegladarka przetwarzala je
    ~10 s po kazdym polaczeniu, zanim cokolwiek pokazala. Symulacja i kolizje
    dalej jada na pelnych siatkach - to dotyczy wylacznie tego, co rysuje panel.
    """
    q = np.floor(vertices / cell).astype(np.int64)
    _, first, inv = np.unique(q, axis=0, return_index=True, return_inverse=True)
    inv = inv.ravel()
    counts = np.bincount(inv).astype(float)
    merged = np.zeros((len(first), 3))
    np.add.at(merged, inv, vertices)
    merged /= counts[:, None]
    f = inv[faces]
    keep = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])
    f = f[keep]
    _, uniq = np.unique(np.sort(f, axis=1), axis=0, return_index=True)
    return merged.astype(np.float32), f[np.sort(uniq)].astype(np.int32)


class SceneMirror:
    def __init__(self, server, model: mujoco.MjModel, root: str = "/scena", mesh_cell: float | None = 0.0012):
        self.server, self.model, self.root = server, model, root
        self.triangles = 0
        self.bodies: dict[int, object] = {}
        self.geoms: list[object] = []
        self._sent: dict[int, np.ndarray] = {}
        m = model
        sc = server.scene
        for g in range(m.ngeom):
            if m.geom_group[g] not in VISUAL_GROUPS:
                continue
            rgba = m.mat_rgba[m.geom_matid[g]] if m.geom_matid[g] >= 0 else m.geom_rgba[g]
            if rgba[3] < 0.05:
                continue
            b = int(m.geom_bodyid[g])
            if b not in self.bodies:
                self.bodies[b] = sc.add_frame(f"{root}/b{b}", show_axes=False)
            name = f"{root}/b{b}/g{g}"
            t, size, color = m.geom_type[g], m.geom_size[g], _rgb(rgba)
            opacity = float(rgba[3]) if rgba[3] < 0.99 else None
            local = dict(position=m.geom_pos[g].copy(), wxyz=m.geom_quat[g].copy())
            if t == mujoco.mjtGeom.mjGEOM_MESH:
                mid = m.geom_dataid[g]
                va, vn = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
                fa, fn = m.mesh_faceadr[mid], m.mesh_facenum[mid]
                verts, faces = m.mesh_vert[va:va + vn].copy(), m.mesh_face[fa:fa + fn].copy()
                if mesh_cell and fn > 500:
                    verts, faces = decimate(verts, faces, mesh_cell)
                self.triangles += len(faces)
                h = sc.add_mesh_simple(name, verts, faces, color=color, opacity=opacity, **local)
            elif t == mujoco.mjtGeom.mjGEOM_BOX:
                h = sc.add_box(name, color=color, dimensions=tuple(2 * size), opacity=opacity, **local)
            elif t == mujoco.mjtGeom.mjGEOM_SPHERE:
                h = sc.add_icosphere(name, radius=float(size[0]), color=color, opacity=opacity, **local)
            elif t in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE):
                h = sc.add_cylinder(name, radius=float(size[0]), height=float(2 * size[1]), color=color,
                                    opacity=opacity, **local)
            elif t == mujoco.mjtGeom.mjGEOM_PLANE:
                h = sc.add_grid(name, width=4.0, height=4.0, cell_size=0.1, section_size=0.5,
                                plane_color=(60, 64, 70), plane_opacity=1.0, cell_color=(90, 95, 100),
                                section_color=(120, 125, 130), **local)
            else:
                continue
            self.geoms.append(h)

    def update(self, data: mujoco.MjData, tol: float = 1e-4) -> int:
        """Pozy cial, ktore sie ruszyly od ostatniej wysylki. Zwraca, ile ich poszlo."""
        changed = []
        for b in self.bodies:
            state = np.concatenate([data.xpos[b], data.xquat[b]])
            last = self._sent.get(b)
            if last is None or np.abs(state - last).max() > tol:
                changed.append((b, state))
        if changed:
            with self.server.atomic():
                for b, state in changed:
                    h = self.bodies[b]
                    h.position, h.wxyz = state[:3], state[3:]
                    self._sent[b] = state
        return len(changed)

    def remove(self) -> None:
        for h in self.geoms:
            h.remove()
        for h in self.bodies.values():
            h.remove()
        self.geoms.clear()
        self.bodies.clear()
        self._sent.clear()
