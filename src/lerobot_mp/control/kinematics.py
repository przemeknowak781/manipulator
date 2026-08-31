"""Kinematyka ramienia SO-101 (LeRobot 101) - model plaski, rozwiazany analitycznie.

SO-101 to lancuch: obrot podstawy + trzy ogniwa w jednej plaszczyznie pionowej
(ramie, przedramie, nadgarstek z chwytakiem) + obrot nadgarstka + chwytak.
Taka struktura ma *analityczne* rozwiazanie odwrotnej kinematyki - nie potrzeba
solvera numerycznego, URDF-a ani zadnej dodatkowej biblioteki.

Wszystkie stale geometryczne (`ArmGeometryConfig`) sa wyliczone z prawdziwego
zlozenia SO-101, a nie oszacowane: patrz `scripts/derive_geometry.py`, ktory
je wyprowadza z modelu Articulusa i od razu sprawdza, jak bardzo ten uproszczony
model rozjezdza sie z pelna kinematyka (blad rzedu dziesiatych milimetra).

Uklad odniesienia (podstawa robota):
    X - do przodu, Y - w lewo, Z - w gore.

Katy stawow sa w stopniach, w konwencji LeRobot (`use_degrees=True`): zero to
srodek zakresu ustalony podczas kalibracji.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..config import ArmGeometryConfig


@dataclass
class IKResult:
    """Wynik odwrotnej kinematyki."""

    shoulder_pan: float
    shoulder_lift: float
    elbow_flex: float
    wrist_flex: float
    #: True, gdy zadany punkt lezal poza zasiegiem i zostal przyciety.
    clamped: bool = False
    #: Faktycznie osiagalny punkt (po przycieciu), we wspolrzednych podstawy.
    reached: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def as_dict(self) -> dict[str, float]:
        return {
            "shoulder_pan": self.shoulder_pan,
            "shoulder_lift": self.shoulder_lift,
            "elbow_flex": self.elbow_flex,
            "wrist_flex": self.wrist_flex,
        }


class ArmKinematics:
    """Kinematyka prosta i odwrotna ramienia SO-101."""

    def __init__(self, geometry: ArmGeometryConfig):
        self.geo = geometry
        if geometry.upper_arm <= 0 or geometry.forearm <= 0:
            raise ValueError("Dlugosci ogniw musza byc dodatnie")

    # ------------------------------------------------------- katy ogniw
    def _link_angles(
        self, shoulder_lift: float, elbow_flex: float, wrist_flex: float
    ) -> tuple[float, float, float]:
        """Bezwzgledne katy trzech ogniw w plaszczyznie pionowej [rad]."""
        g = self.geo
        a1 = math.radians(g.lift_sign * shoulder_lift + g.lift_offset_deg)
        a2 = a1 + math.radians(g.elbow_sign * elbow_flex + g.elbow_offset_deg)
        a3 = a2 + math.radians(g.wrist_sign * wrist_flex + g.wrist_offset_deg)
        return a1, a2, a3

    # ------------------------------------------------------------------ FK
    def forward(
        self,
        shoulder_pan: float,
        shoulder_lift: float,
        elbow_flex: float,
        wrist_flex: float,
    ) -> tuple[float, float, float]:
        """Pozycja koncowki chwytaka [m] dla zadanych katow stawow [stopnie]."""
        g = self.geo
        a1, a2, a3 = self._link_angles(shoulder_lift, elbow_flex, wrist_flex)

        r = (
            g.shoulder_offset
            + g.upper_arm * math.cos(a1)
            + g.forearm * math.cos(a2)
            + g.wrist_to_tip * math.cos(a3)
        )
        z = (
            g.base_height
            + g.upper_arm * math.sin(a1)
            + g.forearm * math.sin(a2)
            + g.wrist_to_tip * math.sin(a3)
        )

        # Plaszczyzna ramienia jest przesunieta w bok o `lateral_offset`, wiec
        # obrot podstawy nie jest zwyklym (r, 0) -> (r cos, r sin).
        theta = math.radians(g.pan_sign * shoulder_pan)
        d = g.lateral_offset
        x = g.pan_axis_x + r * math.cos(theta) - d * math.sin(theta)
        y = r * math.sin(theta) + d * math.cos(theta)
        return (x, y, z)

    def chain_points(
        self,
        shoulder_lift: float,
        elbow_flex: float,
        wrist_flex: float,
    ) -> list[tuple[float, float]]:
        """Punkty lancucha w plaszczyznie pionowej (r, z) [m] - do wizualizacji.

        Zwraca kolejno: podstawe, bark, lokiec, nadgarstek i koncowke chwytaka.
        """
        g = self.geo
        a1, a2, a3 = self._link_angles(shoulder_lift, elbow_flex, wrist_flex)

        base = (g.pan_axis_x, 0.0)
        shoulder = (g.pan_axis_x + g.shoulder_offset, g.base_height)
        elbow = (shoulder[0] + g.upper_arm * math.cos(a1), shoulder[1] + g.upper_arm * math.sin(a1))
        wrist = (elbow[0] + g.forearm * math.cos(a2), elbow[1] + g.forearm * math.sin(a2))
        tip = (
            wrist[0] + g.wrist_to_tip * math.cos(a3),
            wrist[1] + g.wrist_to_tip * math.sin(a3),
        )
        return [base, shoulder, elbow, wrist, tip]

    def tool_pitch(self, shoulder_lift: float, elbow_flex: float, wrist_flex: float) -> float:
        """Kat bezwzgledny narzedzia w plaszczyznie pionowej [stopnie]."""
        return math.degrees(self._link_angles(shoulder_lift, elbow_flex, wrist_flex)[2])

    @property
    def max_reach(self) -> float:
        """Maksymalny zasieg nadgarstka od barku (bez chwytaka)."""
        return self.geo.upper_arm + self.geo.forearm

    @property
    def min_reach(self) -> float:
        return abs(self.geo.upper_arm - self.geo.forearm)

    # ------------------------------------------------------------------ IK
    def inverse(
        self,
        x: float,
        y: float,
        z: float,
        tool_pitch_deg: float,
    ) -> IKResult:
        """Katy stawow dla zadanej pozycji koncowki i kata narzedzia.

        Args:
            x, y, z: pozycja koncowki chwytaka [m] w ukladzie podstawy.
            tool_pitch_deg: zadany kat narzedzia wzgledem poziomu [stopnie];
                0 = chwytak poziomo, -90 = skierowany w dol.

        Returns:
            `IKResult`; jesli punkt byl poza zasiegiem, `clamped=True`, a katy
            odpowiadaja najblizszemu osiagalnemu punktowi (kierunek zachowany).
        """
        g = self.geo
        clamped = False

        # 1. Obrot podstawy. Liczymy wzgledem osi obrotu, ktora jest wysunieta
        #    do przodu od poczatku ukladu, i wzgledem bocznego przesuniecia
        #    koncowki.
        xr = x - g.pan_axis_x
        planar = math.hypot(xr, y)
        d = g.lateral_offset
        if planar <= abs(d):
            # Cel praktycznie na osi obrotu - ramie go nie siega.
            clamped = True
            r = 1e-4
            theta = math.atan2(y, xr)
        else:
            r = math.sqrt(planar * planar - d * d)
            theta = math.atan2(y, xr) - math.atan2(d, r)
        pan = math.degrees(theta) / g.pan_sign

        # 2. Cofniecie od koncowki do przegubu nadgarstka wzdluz osi narzedzia.
        phi = math.radians(tool_pitch_deg)
        rw = r - g.shoulder_offset - g.wrist_to_tip * math.cos(phi)
        zw = z - g.base_height - g.wrist_to_tip * math.sin(phi)

        # 3. Klasyczne IK dwoch ogniw w plaszczyznie (rw, zw).
        dist = math.hypot(rw, zw)
        d_max = self.max_reach * 0.999  # margines, zeby uniknac osobliwosci
        d_min = max(self.min_reach * 1.001, 1e-4)
        if dist > d_max:
            clamped = True
            scale = d_max / dist if dist > 1e-9 else 0.0
            rw, zw, dist = rw * scale, zw * scale, d_max
        elif dist < d_min:
            clamped = True
            if dist < 1e-9:
                rw, zw, dist = d_min, 0.0, d_min
            else:
                scale = d_min / dist
                rw, zw, dist = rw * scale, zw * scale, d_min

        cos_elbow = (dist * dist - g.upper_arm**2 - g.forearm**2) / (
            2.0 * g.upper_arm * g.forearm
        )
        cos_elbow = max(-1.0, min(1.0, cos_elbow))
        elbow = math.acos(cos_elbow)
        if not g.elbow_up:
            elbow = -elbow

        a1 = math.atan2(zw, rw) - math.atan2(
            g.forearm * math.sin(elbow), g.upper_arm + g.forearm * math.cos(elbow)
        )
        a2 = a1 + elbow

        # 4. Katy ogniw -> katy stawow (odwrocenie znakow i offsetow).
        shoulder_lift = (math.degrees(a1) - g.lift_offset_deg) / g.lift_sign
        elbow_flex = (math.degrees(elbow) - g.elbow_offset_deg) / g.elbow_sign
        wrist_flex = (tool_pitch_deg - math.degrees(a2) - g.wrist_offset_deg) / g.wrist_sign

        reached = self.forward(pan, shoulder_lift, elbow_flex, wrist_flex)
        return IKResult(
            shoulder_pan=pan,
            shoulder_lift=shoulder_lift,
            elbow_flex=elbow_flex,
            wrist_flex=wrist_flex,
            clamped=clamped,
            reached=reached,
        )
