"""Czy ramie w danej pozie - i po drodze do niej - nie wchodzi w stol, obiekty ani w siebie.

Sprawdza to MuJoCo na scenie blizniaka (ramie + stol + to, co o stanowisku
wiadomo), na wlasnym `MjData`, wiec nie rusza stanu symulacji ani prawdziwego
ramienia. Uzywa tego fala kalibracyjna i teleoperacja z UI.

Dwie rzeczy, ktorych nauczyl galaxeo-manipulators:

* **Sprawdzac trzeba droge, nie tylko cel.** Poza docelowa moze byc czysta,
  a przejazd do niej przez stol. Probkujemy odcinek w przestrzeni stawow co
  0,04 rad najwiekszego ruchu - w galaxeo dziesiec probek na dwusekundowy ruch
  przepuszczalo palec przez butelke.
* **Kontakty montazowe sa dozwolone.** Pary cial stykajace sie juz w pozie
  spoczynkowej (podstawa na blacie) nie sa kolizja; kazda nowa para - jest.
  Dzieki temu sprawdzacz dziala dla dowolnego ramienia bez recznej listy wyjatkow.
"""

from __future__ import annotations

from collections.abc import Mapping

import mujoco
import numpy as np

from .kinematics import RobotKinematics
from .scene import PREFIX, Scene


class CollisionChecker:
    def __init__(self, scene: Scene):
        self.model = scene.model
        self.data = mujoco.MjData(scene.model)
        self.kin = RobotKinematics(scene.cfg.robot, scene.model, PREFIX)
        m = self.model
        base = m.body(PREFIX + scene.cfg.robot.base_body).id
        in_robot = np.zeros(m.nbody, bool)
        for b in range(m.nbody):                        # poddrzewo podstawy ramienia
            p = b
            while p > 0 and p != base:
                p = m.body_parentid[p]
            in_robot[b] = p == base
        self.robot_geom = in_robot[m.geom_bodyid]
        self.allowed = self._contacts(scene.cfg.robot.home)

    def _contacts(self, joints: Mapping[str, float]) -> set[tuple[int, int]]:
        d = self.data
        d.qpos[self.kin.qadr] = self.kin.to_q(joints)
        mujoco.mj_fwdPosition(self.model, d)
        out = set()
        for c in d.contact[: d.ncon]:
            g1, g2 = int(c.geom1), int(c.geom2)
            if self.robot_geom[g1] or self.robot_geom[g2]:
                out.add((min(g1, g2), max(g1, g2)))
        return out

    def _describe(self, pair: tuple[int, int]) -> str:
        m = self.model
        names = []
        for g in pair:
            n = m.geom(g).name or m.body(m.geom_bodyid[g]).name
            names.append(n.removeprefix(PREFIX))
        return " z ".join(names)

    def config_clear(self, joints: Mapping[str, float]) -> str | None:
        """None, gdy poza jest czysta; inaczej opis pierwszej kolizji."""
        bad = self._contacts(joints) - self.allowed
        return f"kolizja: {self._describe(min(bad))}" if bad else None

    def path_clear(self, start: Mapping[str, float], end: Mapping[str, float],
                   step_rad: float = 0.04) -> str | None:
        """None, gdy caly liniowy przejazd w przestrzeni stawow jest czysty."""
        q0, q1 = self.kin.to_q(start), self.kin.to_q(end)
        travel = float(np.max(np.abs(q1 - q0)))
        n = max(1, int(np.ceil(travel / step_rad)))
        for s in np.linspace(0.0, 1.0, n + 1)[1:]:
            why = self.config_clear(self.kin.from_q(q0 + s * (q1 - q0)))
            if why:
                return f"{why} w {s:.0%} drogi"
        return None
