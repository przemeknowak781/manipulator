"""Identyfikacja dynamiki serw: nagranie prawdziwego ruchu i dopasowanie do niego symulacji.

    rec = record(twin, excitation(spec.home))            # ~20 s ruchu pobudzajacego przez nadzor
    dyn, err = fit(rec)                                    # mnozniki modelu + opoznienie
    ws.dynamics = dyn.to_dict(); ws.save()                 # srodek randomizacji w treningu

Model Menagerie ma wzmocnienia serw policzone z karty katalogowej, nie
zmierzone na TWOIM ramieniu: inne zasilanie, inny egzemplarz STS3215, inne
tarcie w przekladni. Polityka uczona na zlym modelu dojezdza "prawie" -
i to prawie na biurku wyglada jak drgania albo niedojazd. Identyfikacja
mierzy, jak ramie naprawde odpowiada na rozkaz, i stawia randomizacje
wokol tego, a nie wokol katalogu.

Dopasowanie odtwarza w MuJoCo DOKLADNIE ten ciag rozkazow, ktory poszedl do
serw (wyjscie `SafetySupervisor`, z czasem), z opoznieniem `delay`, i
porownuje katy w chwilach, w ktorych je zmierzono. Minimalizowany jest blad
sredniokwadratowy stawow ramienia; optymalizator to Nelder-Mead po logarytmach
mnoznikow (sa dodatnie i dzialaja multiplikatywnie).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from .. import scene as sc
from ..robots import SO101, RobotSpec
from .randomize import Dynamics

JOINTS = SO101.joints


@dataclass
class Recording:
    t: np.ndarray                 # (T,) [s] od poczatku
    command: np.ndarray           # (T, 6) rozkaz, ktory POSZEDL do serw [jednostki aplikacji]
    measured: np.ndarray          # (T, 6) zmierzone katy; NaN, gdy w tym takcie nie czytano
    meta: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, t=self.t, command=self.command, measured=self.measured,
                            meta=json.dumps(self.meta))
        return path

    @staticmethod
    def load(path: str | Path) -> Recording:
        z = np.load(path, allow_pickle=False)
        return Recording(z["t"], z["command"], z["measured"], json.loads(str(z["meta"])))


# ------------------------------------------------------------ pobudzenie
def excitation(home: dict[str, float], amplitude: float = 12.0, hold: float = 0.9,
               joints: tuple[str, ...] = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"),
               ) -> list[tuple[float, dict[str, float]]]:
    """Plan (czas [s], cel) - skoki po kolei na kazdym stawie, potem ruch wszystkimi naraz.

    Skok pokazuje wzmocnienie i opoznienie, powrot z przeciwnej strony -
    tarcie suche (histereza), ruch wszystkimi naraz - sprzezenia miedzy
    stawami. Amplituda mala (12 st.), zeby ramie nie wyjechalo nad stol.
    """
    plan: list[tuple[float, dict[str, float]]] = [(0.0, dict(home))]
    t = hold
    for j in joints:
        for s in (+1.0, -1.0, 0.0):
            plan.append((t, dict(home, **{j: home[j] + s * amplitude})))
            t += hold
    for k, s in enumerate((+1.0, -1.0, +0.5, 0.0)):
        target = {j: home[j] + s * amplitude * (1 if i % 2 == 0 else -1) for i, j in enumerate(joints)}
        plan.append((t, dict(home, **target)))
        t += hold * (1.5 if k < 3 else 1.0)
    plan.append((t, dict(home)))
    return plan


def plan_target(plan: list[tuple[float, dict[str, float]]], t: float) -> dict[str, float]:
    target = plan[0][1]
    for t_k, j in plan:
        if t_k > t:
            break
        target = j
    return target


def record(twin, plan: list[tuple[float, dict[str, float]]], settle: float = 1.0, on_tick=None) -> Recording:
    """Wykonuje plan na blizniaku (sim albo prawdziwe ramie) i nagrywa rozkaz i odpowiedz.

    Rozkazy ida przez `Twin.set_target`, czyli przez nadzor bezpieczenstwa -
    nagrywamy jego WYJSCIE (`status.command`), bo to ono naprawde poszlo do serw.
    """
    if not twin.connected:
        raise RuntimeError("ramie nie jest polaczone")
    duration = plan[-1][0] + settle
    twin.set_engaged(True)
    ts, cmds, meas = [], [], []
    last_meas = None
    t0 = time.monotonic()
    period = 1.0 / twin.loop_hz
    try:
        while True:
            t = time.monotonic() - t0
            if t > duration:
                break
            if twin.safety_state is not None and twin.safety_state.value == "ESTOP":
                raise RuntimeError("stop awaryjny w trakcie identyfikacji")
            twin.set_target(plan_target(plan, t))
            st = twin.status
            m = [st.measured.get(j, np.nan) for j in JOINTS]
            # Prawdziwe serwa czytamy rzadziej niz petla - powtorzony odczyt to NIE nowy pomiar.
            fresh = last_meas is None or m != last_meas
            last_meas = m
            ts.append(t)
            cmds.append([st.command.get(j, np.nan) for j in JOINTS])
            meas.append(m if fresh else [np.nan] * len(JOINTS))
            if on_tick:
                on_tick(t / duration)
            time.sleep(period)
    finally:
        twin.set_engaged(False)
    return Recording(np.array(ts), np.array(cmds, float), np.array(meas, float),
                     {"backend": twin.status.backend, "loop_hz": twin.loop_hz,
                      "time": time.strftime("%Y-%m-%dT%H:%M:%S")})


# ------------------------------------------------------------ symulacja
class Replayer:
    """Odtwarza nagrany ciag rozkazow w MuJoCo z zadana dynamika."""

    def __init__(self, spec: RobotSpec = SO101):
        self.scene = sc.build(sc.SceneConfig(spec))
        m = self.scene.model
        self.act, self.dof = self.scene.act_ids, self.scene.kin.dadr
        self.kp = m.actuator_gainprm[self.act, 0].copy()
        self.bias = m.actuator_biasprm[self.act, 1].copy()
        self.nom = {k: getattr(m, f"dof_{k}")[self.dof].copy() for k in ("damping", "armature", "frictionloss")}

    def apply(self, dyn: Dynamics) -> None:
        m = self.scene.model
        m.actuator_gainprm[self.act, 0] = self.kp * dyn.kp
        m.actuator_biasprm[self.act, 1] = self.bias * dyn.kp
        for k in ("damping", "armature", "frictionloss"):
            getattr(m, f"dof_{k}")[self.dof] = self.nom[k] * getattr(dyn, k)
        mujoco.mj_setConst(m, self.scene.data)

    def run(self, rec: Recording, dyn: Dynamics, delay_s: float) -> np.ndarray:
        """Katy (T, 6) w jednostkach aplikacji w chwilach `rec.t`."""
        self.apply(dyn)
        s = self.scene
        kin = s.kin
        m, d = s.model, s.data
        mujoco.mj_resetData(m, d)
        first = np.where(~np.isnan(rec.measured[:, 0]))[0]
        start = rec.measured[first[0]] if len(first) else rec.command[0]
        s.set_joints(dict(zip(JOINTS, start)))
        # Jednostki przeliczone raz dla calego nagrania - optymalizator woła to setki razy.
        cmd_q = np.array([kin.to_q(dict(zip(JOINTS, c))) for c in rec.command])
        start_q = kin.to_q(dict(zip(JOINTS, start)))
        dt = m.opt.timestep
        # Dla kazdego kroku fizyki: indeks rozkazu obowiazujacego w chwili (t - delay).
        n_steps = int(np.floor(rec.t[-1] / dt + 1e-9))
        t_steps = np.arange(n_steps) * dt - delay_s
        k_cmd = np.searchsorted(rec.t, t_steps, side="right") - 1
        # Probki: po ilu krokach fizyki przypada kazda chwila pomiaru.
        after = np.floor(rec.t / dt + 1e-9).astype(int)
        q = np.zeros((len(rec.t), len(JOINTS)))
        qadr, act = kin.qadr, self.act
        k = 0
        for i, n_i in enumerate(after):
            while k < min(n_i, n_steps):
                d.ctrl[act] = cmd_q[k_cmd[k]] if k_cmd[k] >= 0 else start_q
                mujoco.mj_step(m, d)
                k += 1
            q[i] = d.qpos[qadr]
        return np.array([[kin.from_q(row)[j] for j in JOINTS] for row in q])


def error_deg(rec: Recording, pred: np.ndarray, joints=range(5)) -> float:
    idx = list(joints)
    m = rec.measured[:, idx]
    ok = ~np.isnan(m)
    return float(np.sqrt(np.mean((pred[:, idx][ok] - m[ok]) ** 2)))


def _nelder_mead(f, x0: np.ndarray, step: float = 0.25, iters: int = 120, tol: float = 1e-4):
    n = len(x0)
    pts = [x0] + [x0 + step * np.eye(n)[i] for i in range(n)]
    vals = [f(p) for p in pts]
    for _ in range(iters):
        order = np.argsort(vals)
        pts, vals = [pts[i] for i in order], [vals[i] for i in order]
        if abs(vals[-1] - vals[0]) < tol:
            break
        c = np.mean(pts[:-1], axis=0)
        xr = c + (c - pts[-1])
        fr = f(xr)
        if fr < vals[0]:
            xe = c + 2 * (c - pts[-1])
            fe = f(xe)
            pts[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = xr, fr
        else:
            xc = c + 0.5 * (pts[-1] - c)
            fc = f(xc)
            if fc < vals[-1]:
                pts[-1], vals[-1] = xc, fc
            else:
                pts = [pts[0]] + [pts[0] + 0.5 * (p - pts[0]) for p in pts[1:]]
                vals = [vals[0]] + [f(p) for p in pts[1:]]
    i = int(np.argmin(vals))
    return pts[i], vals[i]


def fit(rec: Recording, spec: RobotSpec = SO101, control_hz: float = 20.0, on_progress=None) -> tuple[Dynamics, float]:
    """Mnozniki kp, tlumienia, armatury, tarcia i opoznienie, przy ktorych symulacja powtarza nagranie."""
    rp = Replayer(spec)
    names = ("kp", "damping", "armature", "frictionloss")
    calls = [0]

    def unpack(x):
        dyn = Dynamics(**{n: float(np.exp(v)) for n, v in zip(names, x[:4])})
        return dyn, float(np.clip(x[4], 0.0, 0.2))

    def cost(x):
        calls[0] += 1
        dyn, delay = unpack(x)
        c = error_deg(rec, rp.run(rec, dyn, delay))
        if on_progress:
            on_progress(calls[0], c)
        return c

    base = error_deg(rec, rp.run(rec, Dynamics(), 0.0))
    # Wzmocnienie, bezwladnosc i opoznienie czesciowo sie kompensuja - jeden start
    # Neldera-Meada utykal w dolinie (kp 1,17 przy prawdzie 0,80). Zgrubna siatka
    # po kp i opoznieniu wybiera kilka startow, potem kazdy dopracowujemy, a na
    # koniec najlepszy jeszcze raz z mniejszym simpleksem.
    grid = [(np.log(k), d) for k in (0.7, 0.85, 1.0, 1.2, 1.45) for d in (0.0, 0.02, 0.04, 0.07, 0.1)]
    scored = sorted(grid, key=lambda g: cost(np.array([g[0], 0, 0, 0, g[1]]) ))
    best_x, best_c = None, np.inf
    for lk, d in scored[:3]:
        x, c = _nelder_mead(cost, np.array([lk, 0.0, 0.0, 0.0, d]), step=0.25, iters=150)
        if c < best_c:
            best_x, best_c = x, c
    x, c = _nelder_mead(cost, best_x, step=0.08, iters=150, tol=1e-5)
    dyn, delay = unpack(x)
    dyn.delay = delay * control_hz
    dyn.source = f"identyfikacja {rec.meta.get('time', '')} ({rec.meta.get('backend', '?')})".strip()
    dyn.fit_deg = c
    rp.scene.close()
    return dyn, float(base)
