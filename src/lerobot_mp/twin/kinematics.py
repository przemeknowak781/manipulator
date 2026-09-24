"""Kinematyka ramienia liczona przez MuJoCo na opisie z `robots`.

Nie ma tu wlasnego lancucha kinematycznego do utrzymywania: kinematyka prosta
to `mj_kinematics` na tym samym MJCF, ktory symuluje i rysuje scene, wiec
blizniak, kalibracja i srodowisko RL nie moga sie rozjechac w geometrii.

Odwrotna kinematyka to tlumione najmniejsze kwadraty na jakobianie site'u TCP,
jak w `a1x_control.Arm.ik` z galaxeo-manipulators, z dwiema zmianami:

* **Pozycja ma pierwszenstwo przed orientacja.** A1X ma szesc stawow ramienia,
  SO-101 piec: dowolnej pelnej orientacji chwytaka nie osiagnie. Wazenie obu
  bledow w jednej sumie kwadratow tu nie dziala - radiany i metry sie nie
  porownuja, i przy celu obroconym o 90 stopni obrot odciagal ramie o 20 cm od
  zadanego punktu. Zamiast tego orientacja jest poprawiana wylacznie w
  przestrzeni zerowej jakobianu pozycji, czyli tylko ruchami, ktore pozycji nie
  zmieniaja (priorytet zadan). Punkt trafia zawsze, gdy jest osiagalny, a obrot
  jest "tak blisko, jak sie da" przy tym punkcie.
* **Wiele startow.** Pojedynczy start DLS utyka w minimum lokalnym - galaxeo
  zmierzylo 30 cm bledu na wiekszosci przestrzeni roboczej A1X. Startujemy
  z podanej pozy, z pozy spoczynkowej i z kilku losowych, i bierzemy najlepszy.

Wszystkie pozy sa w ukladzie PODSTAWY ramienia, nie swiata sceny: kamera
skalibrowana wzgledem podstawy zostaje poprawna, gdziekolwiek ramie postawimy.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache

import mujoco
import numpy as np

from .robots import RobotSpec

#: Tiki serwa STS3215 na obrot - te same, co w backendzie `feetech`.
TICKS_PER_REV = 4096


@lru_cache(maxsize=1)
def backend_gripper_ticks() -> tuple[float, float, float]:
    """(zamkniety, otwarty, zero) chwytaka w tikach serwa - z tej samej konfiguracji, co backend."""
    from ..config import load_config

    rc = load_config().robot
    return float(rc.gripper_closed_ticks), float(rc.gripper_open_ticks), float(rc.center_ticks)


def pose(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Macierz 4x4 z obrotu i przesuniecia."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def inverse(T: np.ndarray) -> np.ndarray:
    """Odwrotnosc transformacji sztywnej bez ogolnego odwracania macierzy."""
    R, t = T[:3, :3], T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


@dataclass
class IKSolution:
    """Wynik odwrotnej kinematyki w jednostkach aplikacji."""

    joints: dict[str, float]
    #: Blad pozycji TCP [m] i orientacji [rad] w osiagnietej pozie.
    pos_err: float
    rot_err: float
    #: Pozycja trafiona w tolerancji. Orientacja jest miekka - patrz modul.
    ok: bool


class RobotKinematics:
    """Kinematyka jednego ramienia na skompilowanym modelu MuJoCo.

    `model` moze byc samym ramieniem albo cala scena, w ktora ramie zostalo
    wstawione pod prefiksem nazw (`prefix`) - obliczenia i tak ida na osobnym
    `MjData`, wiec nie ruszaja stanu symulacji.
    """

    def __init__(self, spec: RobotSpec, model: mujoco.MjModel | None = None, prefix: str = "",
                 gripper_ticks: tuple[float, float, float] | None = None):
        self.spec = spec
        #: (zamkniety, otwarty, zero) chwytaka w tikach serwa - patrz `to_q`.
        self.gripper_ticks = gripper_ticks if gripper_ticks is not None else backend_gripper_ticks()
        self.prefix = prefix
        self.model = model if model is not None else mujoco.MjModel.from_xml_path(str(spec.mjcf_path))
        self.data = mujoco.MjData(self.model)

        m = self.model
        ids = [m.joint(prefix + name).id for name in spec.joints]
        self.qadr = np.array([m.jnt_qposadr[i] for i in ids])
        self.dadr = np.array([m.jnt_dofadr[i] for i in ids])
        self.lo = m.jnt_range[ids, 0].copy()
        self.hi = m.jnt_range[ids, 1].copy()
        #: Maska stawow ramienia (bez chwytaka) - tylko nimi rusza IK.
        self.arm = np.array([name != spec.gripper for name in spec.joints])
        self.site_id = m.site(prefix + spec.tcp_site).id
        self.base_id = m.body(prefix + spec.base_body).id
        self._approach = np.asarray(spec.tcp_approach, float)
        self._closing = np.asarray(spec.tcp_closing, float)

    # ------------------------------------------------------------ jednostki
    def to_q(self, joints: Mapping[str, float]) -> np.ndarray:
        """Jednostki aplikacji -> wektor kata stawow MJCF [rad], w kolejnosci opisu.

        Stawy ramienia to stopnie tej samej kalibracji, wiec tylko zmiana jednostki.
        Chwytak ma 0..100, a MJCF kat szczeki. Przeliczenie idzie przez TE SAME
        tiki, co w backendzie `feetech` (`gripper_closed_ticks..gripper_open_ticks`,
        zero w `center_ticks`), przyciete do zakresu stawu z MJCF. Wczesniej 0..100
        szlo liniowo na caly zakres MJCF (-10..100 st.), a serwo przejezdza na
        0..100 tylko 60 st. (-5..55): ta sama szczeka byla w blizniaku 1,8 raza
        szerzej otwarta niz na ramieniu, polityka i sledzenie kostki widzialy
        inna liczbe niz w treningu. Brakujace stawy biora wartosc z pozy spoczynkowej.
        """
        q = np.empty(len(self.spec.joints))
        for k, name in enumerate(self.spec.joints):
            value = float(joints.get(name, self.spec.home.get(name, 0.0)))
            if name == self.spec.gripper:
                q[k] = self._grip_to_q(k, value)
            else:
                q[k] = np.radians(value)
        return q

    def _grip_to_q(self, k: int, value: float) -> float:
        closed, opened, center = self.gripper_ticks
        if abs(opened - closed) < 1e-9:                    # konfiguracja bez zakresu - liniowo na MJCF
            fraction = min(max(value / 100.0, 0.0), 1.0)
            return float(self.lo[k] + fraction * (self.hi[k] - self.lo[k]))
        ticks = closed + value / 100.0 * (opened - closed)
        q = np.radians((ticks - center) * 360.0 / TICKS_PER_REV)
        return float(min(max(q, self.lo[k]), self.hi[k]))

    def _grip_from_q(self, k: int, q: float) -> float:
        closed, opened, center = self.gripper_ticks
        if abs(opened - closed) < 1e-9:
            span = self.hi[k] - self.lo[k]
            return float((q - self.lo[k]) / span * 100.0) if span > 0 else 0.0
        ticks = center + np.degrees(q) * TICKS_PER_REV / 360.0
        return float((ticks - closed) * 100.0 / (opened - closed))

    def from_q(self, q: np.ndarray) -> dict[str, float]:
        """Wektor MJCF [rad] -> jednostki aplikacji."""
        out: dict[str, float] = {}
        for k, name in enumerate(self.spec.joints):
            if name == self.spec.gripper:
                out[name] = self._grip_from_q(k, float(q[k]))
            else:
                out[name] = float(np.degrees(q[k]))
        return out

    def clip(self, joints: Mapping[str, float]) -> dict[str, float]:
        """Te same stawy, przyciete do zakresow z MJCF."""
        return self.from_q(np.clip(self.to_q(joints), self.lo, self.hi))

    # ------------------------------------------------------------ kinematyka
    def _apply(self, q: np.ndarray) -> None:
        self.data.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.model, self.data)

    def _base_inv(self) -> np.ndarray:
        d = self.data
        return inverse(pose(d.xmat[self.base_id].reshape(3, 3), d.xpos[self.base_id]))

    def tcp(self, joints: Mapping[str, float]) -> np.ndarray:
        """Poza TCP w ukladzie podstawy (4x4)."""
        self._apply(self.to_q(joints))
        d = self.data
        return self._base_inv() @ pose(d.site_xmat[self.site_id].reshape(3, 3), d.site_xpos[self.site_id])

    def body(self, name: str, joints: Mapping[str, float]) -> np.ndarray:
        """Poza dowolnego ciala ramienia w ukladzie podstawy (4x4)."""
        self._apply(self.to_q(joints))
        i = self.model.body(self.prefix + name).id
        d = self.data
        return self._base_inv() @ pose(d.xmat[i].reshape(3, 3), d.xpos[i])

    def site(self, name: str, joints: Mapping[str, float]) -> np.ndarray:
        """Poza dowolnego site'u ramienia w ukladzie podstawy (4x4)."""
        self._apply(self.to_q(joints))
        i = self.model.site(self.prefix + name).id
        d = self.data
        return self._base_inv() @ pose(d.site_xmat[i].reshape(3, 3), d.site_xpos[i])

    def tool_axes(self, joints: Mapping[str, float]) -> tuple[np.ndarray, np.ndarray]:
        """(kierunek podejscia, os zamykania szczek) w ukladzie podstawy."""
        R = self.tcp(joints)[:3, :3]
        return R @ self._approach, R @ self._closing

    # -------------------------------------------------------------------- IK
    def ik(
        self,
        position: np.ndarray,
        rotation: np.ndarray | None = None,
        seed: Mapping[str, float] | None = None,
        *,
        rot_gain: float = 0.5,
        tol: float = 1e-3,
        iters: int = 300,
        damping: float = 0.02,
        restarts: int = 6,
        rng: np.random.Generator | None = None,
    ) -> IKSolution:
        """Stawy, przy ktorych TCP trafia w `position`, obrocony jak najblizej `rotation`.

        Obie wielkosci w ukladzie podstawy. Chwytak zostaje taki, jak w `seed`.
        Sposrod startow, ktore trafily w punkt, wygrywa ten z najmniejszym
        bledem obrotu; jesli zaden nie trafil - ten, ktory byl najblizej.
        """
        seed = dict(seed) if seed is not None else dict(self.spec.home)
        rng = rng if rng is not None else np.random.default_rng(0)
        q_seed = self.to_q(seed)

        starts = [q_seed, self.to_q(self.spec.home)]
        for _ in range(restarts):
            q = q_seed.copy()
            q[self.arm] = rng.uniform(self.lo[self.arm], self.hi[self.arm])
            starts.append(q)

        best: tuple[tuple[bool, float], np.ndarray, float, float] | None = None
        target = np.asarray(position, float)
        for q0 in starts:
            q, e_pos, e_rot = self._solve(q0, target, rotation, rot_gain, iters, damping, tol)
            hit = e_pos < tol
            # Najpierw "trafil w punkt", potem mniejszy blad tego, co jeszcze do poprawienia.
            key = (not hit, e_rot if hit else e_pos)
            if best is None or key < best[0]:
                best = (key, q, e_pos, e_rot)
            if hit and (rotation is None or e_rot < 0.02):
                break  # lepiej sie nie da - szkoda liczyc dalsze starty

        assert best is not None
        _, q, e_pos, e_rot = best
        return IKSolution(self.from_q(q), float(e_pos), float(e_rot), bool(e_pos < tol))

    def _solve(
        self,
        q: np.ndarray,
        position: np.ndarray,
        rotation: np.ndarray | None,
        rot_gain: float,
        iters: int,
        damping: float,
        tol: float,
    ) -> tuple[np.ndarray, float, float]:
        """Jeden start: pozycja pseudo-odwrotnoscia, obrot gradientem w przestrzeni zerowej."""
        m, d = self.model, self.data
        q = np.clip(q.copy(), self.lo, self.hi)
        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))
        cols = self.dadr[self.arm]
        n = int(self.arm.sum())
        want_quat = np.zeros(4)
        cur_quat = np.zeros(4)
        err_quat = np.zeros(4)
        w = np.zeros(3)
        e_pos = e_rot = float("inf")

        self._apply(q)
        # Podstawa sie nie rusza, wiec cel przeliczamy do ukladu swiata raz.
        base = pose(d.xmat[self.base_id].reshape(3, 3), d.xpos[self.base_id])
        target = base[:3, :3] @ position + base[:3, 3]
        if rotation is not None:
            mujoco.mju_mat2Quat(want_quat, (base[:3, :3] @ np.asarray(rotation, float)).ravel())

        for _ in range(iters):
            self._apply(q)
            e_p = target - d.site_xpos[self.site_id]
            e_pos = float(np.linalg.norm(e_p))
            if rotation is not None:
                mujoco.mju_mat2Quat(cur_quat, d.site_xmat[self.site_id])
                mujoco.mju_negQuat(cur_quat, cur_quat)
                mujoco.mju_mulQuat(err_quat, want_quat, cur_quat)
                mujoco.mju_quat2Vel(w, err_quat, 1.0)
                e_rot = float(np.linalg.norm(w))
            else:
                e_rot = 0.0

            mujoco.mj_comPos(m, d)
            mujoco.mj_jacSite(m, d, jacp, jacr, self.site_id)
            Jp = jacp[:, cols]
            Jp_pinv = Jp.T @ np.linalg.inv(Jp @ Jp.T + damping**2 * np.eye(3))
            dq = Jp_pinv @ e_p
            if rotation is not None:
                # Zadanie drugorzedne rzutowane na przestrzen zerowa pozycji:
                # ruch, ktory poprawia obrot, a punktu TCP nie przesuwa.
                null = np.eye(n) - Jp_pinv @ Jp
                dq_rot = null @ (rot_gain * jacr[:, cols].T @ w)
                dq = dq + dq_rot
                if e_pos < tol * 0.5 and float(np.linalg.norm(dq_rot)) < 1e-4:
                    break  # punkt trafiony, a obrotu da sie juz poprawic tylko kosztem punktu
            elif e_pos < tol * 0.5:
                break

            # Ograniczenie kroku jak w galaxeo: duzy skok przy zlym uwarunkowaniu
            # przerzuca rozwiazanie na inna galaz i iteracja juz z niej nie wraca.
            step = min(1.0, 0.3 / (float(np.linalg.norm(dq)) + 1e-9))
            q[self.arm] = np.clip(q[self.arm] + step * dq, self.lo[self.arm], self.hi[self.arm])

        self._apply(q)
        e_pos = float(np.linalg.norm(target - d.site_xpos[self.site_id]))
        return q, e_pos, e_rot
