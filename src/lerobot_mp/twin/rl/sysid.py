"""Identyfikacja dynamiki serw: nagranie prawdziwego ruchu i dopasowanie do niego symulacji.

    rec = record(twin, excitation(spec.home))            # ~20 s ruchu pobudzajacego przez nadzor
    dyn, err = fit(rec)                                    # tlumienie, armatura, opoznienie
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
sredniokwadratowy stawow ramienia - skanem po wspolrzednych na coraz
drobniejszej siatce (logarytmy mnoznikow i opoznienie), bo sam blad jest
postrzepiony i metody gradientowe stawaly w pierwszym dolku.

Co z tego naprawde wychodzi - zmierzone na PELNYM pobudzeniu panelu
(`excitation(home)`, 5 stawow), 13 nagran syntetycznych o znanej dynamice
spoza siatek startowych (kp 0,7-1,3, tlumienie 0,7-1,4, armatura 0,8-1,3,
tarcie 0,55-1,8, opoznienie 0-92 ms), szum 0,05 st.: tlumienie do 10,5 %,
armatura do 8,2 %, opoznienie do 1,9 ms, blad dopasowania 0,05-0,10 st.
kp i tarcie suche z tego ruchu NIE sa wyznaczalne (`IDENTIFIED`) - zostaja
z modelu, `Dynamics.fitted` i opis wyniku (`Dynamics.source`) mowia to wprost,
a randomizacja treningu daje im szerokie zakresy (`UNFITTED_RANGES`).
Niepewnosc kazdego dopasowanego parametru (`FitResult.band`) liczy sie
z tym, ze kp i tarcie sa nieznane - w tych 13 przypadkach blad zawsze miescil
sie w pasmie (tlumienie 5-37 %, armatura 6-15 %, opoznienie 2-5 ms).
Wczesniejsze wersje: dopasowywaly wszystko i oddawaly kp 0,72 przy prawdzie
1,0; potem na pelnym pobudzeniu stawaly w lokalnym minimum (armatura -13 %,
opoznienie +5 ms) przy "pasmie" +-5 % z samej krzywizny.
"""

from __future__ import annotations

import copy
import json
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from .. import scene as sc
from ..robots import SO101, RobotSpec
from .randomize import UNFITTED_RANGES, Dynamics

JOINTS = SO101.joints
#: Nazwa wlasciciela ramienia (`Twin.claim`) na czas nagrania.
OWNER = "identyfikacja"


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


def check_recording(rec: Recording, max_gap: float = 0.5, joints=range(5)) -> None:
    """ValueError, gdy nagranie nie pokrywa calego pobudzenia.

    Zmierzone: petla ramienia padla po 4 s z 20 s nagrania, `record` oddal
    nagranie z 193 z 970 wierszy wazonych, a dopasowanie na samym poczatku
    wygladalo tak samo dobrze jak na pelnym (tarcie x1,73 zamiast x0,80) -
    i dalo sie je zapisac jako dynamike stanowiska.
    """
    if len(rec.t) < 10:
        raise ValueError("nagranie za krotkie")
    if not np.isfinite(rec.command[:, list(joints)]).all():
        bad = rec.t[~np.isfinite(rec.command[:, list(joints)]).all(axis=1)]
        raise ValueError(f"nagranie bez rozkazu od {bad[0]:.2f} s - petla ramienia przerwana?")
    idx = list(joints)
    for k in idx:
        ok = np.isfinite(rec.measured[:, k])
        if not ok.any():
            raise ValueError(f"brak pomiarow stawu {JOINTS[k]}")
        t_ok = np.r_[rec.t[0], rec.t[ok], rec.t[-1]]
        gap = float(np.diff(t_ok).max())
        if gap > max_gap:
            at = float(t_ok[int(np.argmax(np.diff(t_ok)))])
            raise ValueError(f"{JOINTS[k]}: {gap:.2f} s bez pomiaru od {at:.2f} s - nagranie niepelne")


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


def record(twin, plan: list[tuple[float, dict[str, float]]], settle: float = 1.0, on_tick=None,
           should_stop: Callable[[], bool] | None = None, approach_speed: float = 30.0,
           stale_after: float = 0.5, take: bool = True, ramp_timeout: float = 10.0) -> Recording:
    """Wykonuje plan na blizniaku (sim albo prawdziwe ramie) i nagrywa rozkaz i odpowiedz.

    Rozkazy ida przez `Twin.set_target`, czyli przez nadzor bezpieczenstwa -
    nagrywamy jego WYJSCIE (`status.command`), bo to ono naprawde poszlo do serw.

    Przed nagraniem ramie dojezdza do poczatku planu plynna rampa
    (`approach_speed` st./s): pierwszy cel planu to dom, a ramie stalo tam,
    gdzie zostawila je polityka albo suwaki - skok 70 st. z pelna predkoscia
    serw nagrywal sie jako pierwsza sekunda "pobudzenia".

    Ramie bierze RAZ, na samym poczatku (`Twin.claim(OWNER)`): wolne - bierze,
    juz nalezace do `OWNER` - zostaje, cudze - RuntimeError "ramie zajete". Potem
    jezdzi tylko `move(..., take=False)` i `set_target(owner=OWNER)`, wiec odebranie
    (Dom, STOP, Polacz) konczy nagranie na dobre. Zmierzone przed ta zmiana: Dom
    wcisniety tuz po "Identyfikuj", zanim watek zadania doszedl do `record`, zwalnial
    ramie - a `record` bral je z powrotem, `claim` przerywal rampe do domu, a dojazd
    jechal na ramieniu, ktore operator wlasnie zatrzymal. Dlatego tez:
    - rampa do domu (Dom) w toku -> RuntimeError bez brania ramienia; rampa startowa
      po polaczeniu -> czekamy na jej koniec (najwyzej `ramp_timeout` s), nie przerywamy jej;
    - `should_stop()` sprawdzane przed wzieciem ramienia i przed dojazdem, nie dopiero w nagraniu;
    - `take=False`: nawet na poczatku nie bierze wolnego ramienia - musi juz nalezec do
      `OWNER` (panel wzial je przed startem zadania), inaczej RuntimeError.

    RuntimeError, gdy ramie odebrano (Dom, STOP, inny wlasciciel), `should_stop()`
    zwrocilo True, petla ramienia padla albo przez `stale_after` s nie przyszedl
    zaden nowy odczyt serw.
    """
    if not twin.connected:
        raise RuntimeError("ramie nie jest polaczone")
    stop = should_stop or (lambda: False)
    t_end = time.monotonic() + ramp_timeout
    while True:
        if stop():
            raise RuntimeError("identyfikacja przerwana")
        state = getattr(twin.safety_state, "value", None)
        if state == "HOMING":
            raise RuntimeError("identyfikacja przerwana: ramie jedzie do domu")
        if state != "STARTING":
            break
        if time.monotonic() > t_end:
            raise RuntimeError("ramie wciaz jedzie do pozycji startowej - identyfikacja nie ruszyla")
        time.sleep(0.02)
    preempted = threading.Event()
    if take:
        twin.claim(OWNER, preempt=preempted.set)
    elif twin.owner != OWNER:
        # Bez `claim`: panel wzial ramie i ktos mu je odebral - nie bierzemy go z powrotem.
        reason = getattr(twin, "preempt_reason", "") or f"ramie ma: {twin.owner or 'nikt'}"
        raise RuntimeError(f"identyfikacja przerwana - ramie odebrane ({reason})")
    ts, cmds, meas = [], [], []
    try:
        if stop():
            raise RuntimeError("identyfikacja przerwana")
        start = plan[0][1]
        cur = dict(twin.status.command or twin.joints())
        dist = max([abs(float(v) - float(cur.get(k, v))) for k, v in start.items()] or [0.0])
        twin.move(start, duration=max(1.0, dist / approach_speed), settle=0.5, owner=OWNER, take=False)

        duration = plan[-1][0] + settle
        # Sprzeglo zostawil wlaczone `move` (razem z wlasnoscia) - osobne `set_engaged(True)`
        # tutaj wlaczaloby je z powrotem, gdyby STOP odebral ramie miedzy dojazdem a nagraniem.
        last_t, last_meas = None, None
        t0 = time.monotonic()
        last_fresh = t0
        period = 1.0 / twin.loop_hz
        while True:
            now = time.monotonic()
            t = now - t0
            if t > duration:
                break
            if preempted.is_set():
                raise RuntimeError(f"identyfikacja przerwana: {getattr(twin, 'preempt_reason', '') or 'ramie odebrane'}")
            if stop():
                raise RuntimeError("identyfikacja przerwana")
            st = twin.status
            if not twin.connected or not st.connected:
                raise RuntimeError(f"petla ramienia przerwana w trakcie identyfikacji: {st.error or 'rozlaczone'}")
            if twin.safety_state is not None and twin.safety_state.value == "ESTOP":
                raise RuntimeError("stop awaryjny w trakcie identyfikacji")
            twin.set_target(plan_target(plan, t), owner=OWNER)
            m = [st.measured.get(j, np.nan) for j in JOINTS]
            # Prawdziwe serwa czytamy rzadziej niz petla - powtorzony odczyt to NIE nowy pomiar.
            # Chwila odczytu z petli (`measured_t`), a nie porownanie list: lista z NaN
            # porownywala sie jako "rowna" i martwa petla dawala same NaN bez bledu.
            mt = getattr(st, "measured_t", None)
            fresh = (mt != last_t) if mt is not None else (last_meas is None or m != last_meas)
            fresh = fresh and bool(np.isfinite(m).all())
            last_t, last_meas = mt, m
            if fresh:
                last_fresh = now
            elif now - last_fresh > stale_after:
                raise RuntimeError(f"brak nowych odczytow serw od {1000 * (now - last_fresh):.0f} ms")
            ts.append(t)
            cmds.append([st.command.get(j, np.nan) for j in JOINTS])
            meas.append(m if fresh else [np.nan] * len(JOINTS))
            if on_tick:
                on_tick(t / duration)
            time.sleep(period)
    finally:
        if twin.owner == OWNER:
            twin.set_engaged(False)
            twin.release(OWNER)
    rec = Recording(np.array(ts), np.array(cmds, float), np.array(meas, float),
                    {"backend": twin.status.backend, "loop_hz": twin.loop_hz,
                     "time": time.strftime("%Y-%m-%dT%H:%M:%S")})
    check_recording(rec)
    return rec


# ------------------------------------------------------------ symulacja
class Replayer:
    """Odtwarza nagrany ciag rozkazow w MuJoCo z zadana dynamika.

    `run_many` puszcza wiele dynamik naraz przez `mujoco.rollout` (petla krokow
    w C, w kilku watkach, bez GIL). Petla krokow w Pythonie kosztowala 57 ms na
    20-sekundowe nagranie, 12 dynamik naraz - 0,11 s zamiast 0,63 s; dopiero
    to pozwolilo zmierzyc niepewnosc dopasowania (`identify`) w czasie, ktory
    operator przeczeka, a nie tylko jego czulosc.
    """

    #: Watki rolloutu: polowa rdzeni - reszta dla petli ramienia i panelu, ktore
    #: dzialaja w tym samym procesie, gdy dopasowanie idzie w tle.
    threads = max(1, (os.cpu_count() or 2) // 2)

    def __init__(self, spec: RobotSpec = SO101):
        self.scene = sc.build(sc.SceneConfig(spec))
        m = self.scene.model
        # Pobudzenie nie dotyka niczego (male ruchy wokol domu nad stolem) - bez
        # detekcji kolizji odtworzenie jest szybsze, a wynik ten sam.
        m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
        self.act, self.dof = self.scene.act_ids, self.scene.kin.dadr
        self.kp = m.actuator_gainprm[self.act, 0].copy()
        self.bias = m.actuator_biasprm[self.act, 1].copy()
        self.nom = {k: getattr(m, f"dof_{k}")[self.dof].copy() for k in ("damping", "armature", "frictionloss")}
        self._models: list[mujoco.MjModel] = []           # kopie modelu - po jednej na dynamike w partii
        self._datas: list[mujoco.MjData] = []             # po jednej na watek rolloutu
        self._prep: tuple[Recording, dict] | None = None

    def apply(self, dyn: Dynamics, model: mujoco.MjModel | None = None) -> None:
        m = model if model is not None else self.scene.model
        m.actuator_gainprm[self.act, 0] = self.kp * dyn.kp
        m.actuator_biasprm[self.act, 1] = self.bias * dyn.kp
        for k in ("damping", "armature", "frictionloss"):
            getattr(m, f"dof_{k}")[self.dof] = self.nom[k] * getattr(dyn, k)
        mujoco.mj_setConst(m, self.scene.data)

    def _prepare(self, rec: Recording) -> dict:
        """Wszystko, co nie zalezy od dynamiki - liczone raz na nagranie (optymalizator wola setki razy).

        Pamietane po TOZSAMOSCI nagrania: nagranie zmienione w miejscu miedzy wywolaniami
        tego samego `Replayer` dostaloby stare rozkazy - `identify` tworzy wlasny.
        """
        if self._prep is not None and self._prep[0] is rec:
            return self._prep[1]
        s = self.scene
        kin = s.kin
        m, d = s.model, s.data
        mujoco.mj_resetData(m, d)
        first = np.where(~np.isnan(rec.measured[:, 0]))[0]
        start = rec.measured[first[0]] if len(first) else rec.command[0]
        s.set_joints(dict(zip(JOINTS, start)))
        spec_state = mujoco.mjtState.mjSTATE_FULLPHYSICS
        init = np.zeros(mujoco.mj_stateSize(m, spec_state))
        mujoco.mj_getState(m, d, init, spec_state)
        cmd_q = self._to_q(rec.command)
        start_q = kin.to_q(dict(zip(JOINTS, start)))
        cmd_q = np.vstack([start_q[None], cmd_q])            # indeks 0 = zanim przyszedl pierwszy rozkaz
        dt = m.opt.timestep
        prep = {"init": init, "start_q": start_q, "cmd_q": cmd_q,
                "n_steps": int(np.floor(rec.t[-1] / dt + 1e-9)),
                # Probki: po ilu krokach fizyki przypada kazda chwila pomiaru.
                "after": np.floor(rec.t / dt + 1e-9).astype(int)}
        self._prep = (rec, prep)
        return prep

    def _controls(self, rec: Recording, prep: dict, delay_s: float) -> np.ndarray:
        # Dla kazdego kroku fizyki: rozkaz obowiazujacy w chwili (t - delay), a w kroku,
        # w ktorym rozkaz sie zmienia - mieszanka proporcjonalna do czasu po zmianie.
        # Bez tego blad byl schodkowy w opoznieniu (co 5 ms), a optymalizator stawal
        # na pierwszym schodku (37,5 ms przy prawdzie 40 ms).
        dt = self.scene.model.opt.timestep
        cmd_q = prep["cmd_q"]
        t_a = np.arange(prep["n_steps"]) * dt - delay_s
        k0 = np.searchsorted(rec.t, t_a, side="right")
        k1 = np.searchsorted(rec.t, t_a + dt, side="right")
        alpha = np.where(k1 > k0, (t_a + dt - rec.t[np.maximum(k1 - 1, 0)]) / dt, 0.0)
        return cmd_q[k0] + alpha[:, None] * (cmd_q[k1] - cmd_q[k0])

    def run(self, rec: Recording, dyn: Dynamics, delay_s: float) -> np.ndarray:
        """Katy (T, 6) w jednostkach aplikacji w chwilach `rec.t`."""
        return self.run_many(rec, [(dyn, delay_s)])[0]

    def run_many(self, rec: Recording, sets: list[tuple[Dynamics, float]]) -> np.ndarray:
        """Katy (N, T, 6) w jednostkach aplikacji dla N par (dynamika, opoznienie [s])."""
        return np.stack([self._from_q(q) for q in self.run_q(rec, sets)])

    def run_q(self, rec: Recording, sets: list[tuple[Dynamics, float]]) -> np.ndarray:
        """Katy stawow (N, T, 6) w radianach MuJoCo - jedno wywolanie rolloutu. Stawy ramienia
        to po prostu stopnie; przeliczenie chwytaka (w Pythonie, probka po probce) kosztowalo
        20 % dopasowania, ktore chwytaka nie uzywa."""
        from mujoco import rollout

        prep = self._prepare(rec)
        m = self.scene.model
        n, nu = len(sets), m.nu
        while len(self._models) < n:
            self._models.append(copy.copy(m))
        while len(self._datas) < min(n, self.threads):
            self._datas.append(mujoco.MjData(m))
        ctrl = np.zeros((n, prep["n_steps"], nu))
        for i, (dyn, delay_s) in enumerate(sets):
            self.apply(dyn, self._models[i])
            ctrl[i][:, self.act] = self._controls(rec, prep, delay_s)
        init = np.tile(prep["init"], (n, 1))
        state, _ = rollout.rollout(self._models[:n], self._datas[:min(n, self.threads)], init, ctrl)
        # Stan po k krokach: state[:, k - 1]; chwila pomiaru przed pierwszym krokiem - stan poczatkowy.
        kin = self.scene.kin
        qcol = 1 + np.asarray(kin.qadr)                       # stan FULLPHYSICS: [czas, qpos, qvel, ...]
        after = np.minimum(prep["after"], prep["n_steps"])
        return np.where((after > 0)[None, :, None], state[:, np.maximum(after - 1, 0)][:, :, qcol],
                        prep["init"][qcol][None, None, :])

    def _to_q(self, units: np.ndarray) -> np.ndarray:
        """`kin.to_q` dla calej tablicy (T, 6) naraz - wiersz po wierszu to polowa czasu dopasowania."""
        kin = self.scene.kin
        g = JOINTS.index(kin.spec.gripper)
        q = np.radians(np.asarray(units, float))
        q[:, g] = [kin._grip_to_q(g, float(v)) for v in units[:, g]]
        return q

    def _from_q(self, q: np.ndarray) -> np.ndarray:
        kin = self.scene.kin
        g = JOINTS.index(kin.spec.gripper)
        out = np.degrees(q)
        out[:, g] = [kin._grip_from_q(g, float(v)) for v in q[:, g]]
        return out


def error_deg(rec: Recording, pred: np.ndarray, joints=range(5)) -> float:
    idx = list(joints)
    m = rec.measured[:, idx]
    ok = ~np.isnan(m)
    return float(np.sqrt(np.mean((pred[:, idx][ok] - m[ok]) ** 2)))


PARAMS = ("kp", "damping", "armature", "frictionloss", "delay")
#: Co identyfikacja dopasowuje domyslnie. kp i tarcie suche NIE: na nagraniu
#: syntetycznym o znanej dynamice (szum 0,05 st., jak enkoder) ich szacunki byly
#: szumem wokol ~0,8 niezaleznie od prawdy (kp 0,63 / 0,75 / 0,85 / 0,87 przy
#: prawdzie 1,0 / 0,8 / 1,3 / 0,75; tarcie 0,7..2,6 przy 0,6..1,4) - blad prawie
#: sie nie zmienia, gdy tlumienie i armatura je kompensuja. Przesuniecie srodka
#: randomizacji na taki "pomiar" (kp 0,72 przy prawdzie 1,0) wyrzucalo prawdziwe
#: ramie poza wszystkie swiaty treningu - lepiej zostawic model i jego rozrzut.
IDENTIFIED = ("damping", "armature", "delay")


@dataclass
class FitResult:
    dyn: Dynamics
    #: Blad symulacji nominalnej (Menagerie, bez opoznienia) wzgledem nagrania [st.].
    base: float
    #: Niepewnosc dopasowanych parametrow (mnozniki - wzglednie, 0,1 = 10 %; opoznienie -
    #: w sekundach); niedopasowane - inf. Wieksza z dwoch miar: czulosci (o ile mozna ruszyc
    #: parametr, zanim blad urosnie o 10 %) i przesuniecia minimum, gdy kp i tarcie suche
    #: przebiegaja swoj nieznany zakres (`UNFITTED_RANGES`). Sama czulosc klamala: na
    #: pelnym pobudzeniu przy prawdzie kp 0,8 wynik mial armature -13 % i opoznienie
    #: +5 ms, a "pasmo" mowilo +-5 % i +-5 ms - blad szedl z kp przyjetego z modelu.
    band: dict[str, float] = field(default_factory=dict)
    calls: int = 0
    #: Dopasowania przy kp / tarciu na krancach zakresu: (nazwa, wartosc, x, blad [st.]).
    profiles: list[tuple[str, float, np.ndarray, float]] = field(default_factory=list)


def _coordinate_scan(costs, x0s: list[np.ndarray], spans: np.ndarray, cycles: int = 6, points: int = 7,
                     order=None) -> tuple[list[np.ndarray], list[float]]:
    """Kilka niezaleznych skanow naraz, wspolrzedne po kolei: na kazdej osi siatka `points`
    punktow w +-span wokol biezacego minimum, po kazdym cyklu siatka o polowe wezsza.
    `costs(lista x)` liczy cala partie (wszystkie punkty osi wszystkich skanow) jednym
    rolloutem. Zwraca (x, f(x)) kazdego skanu.

    Bez pochodnych: blad dopasowania jest postrzepiony w skali ulamkow procenta
    (drgania modelu serwa ~25 Hz probkowane 50 Hz - faza zalezy od kp i armatury),
    wiec Levenberg-Marquardt i Nelder-Mead stawaly w pierwszym lokalnym dolku
    (kp 0,72 przy prawdzie 1,0; kp 1,67 przy prawdzie 0,8). Skan na coraz
    drobniejszej siatce widzi ksztalt doliny, a nie jej zeby.
    """
    xs = [np.asarray(x, float).copy() for x in x0s]
    fx = [float(c) for c in costs(xs)]
    span = np.asarray(spans, float).copy()
    for _ in range(cycles):
        for i in (order if order is not None else range(len(xs[0]))):
            grid = np.linspace(-span[i], span[i], points)
            grid = grid[grid != 0.0]
            cand = []
            for x in xs:
                for g in grid:
                    y = x.copy()
                    y[i] += g
                    cand.append(y)
            fc = costs(cand)
            for s in range(len(xs)):
                chunk = fc[s * len(grid):(s + 1) * len(grid)]
                j = int(np.argmin(chunk))
                if chunk[j] < fx[s]:
                    xs[s], fx[s] = cand[s * len(grid) + j], float(chunk[j])
        span *= 0.5
    return xs, fx


def identify(rec: Recording, spec: RobotSpec = SO101, control_hz: float = 20.0, on_progress=None,
             free: tuple[str, ...] = IDENTIFIED) -> FitResult:
    """Parametry `free` (reszta = model Menagerie), przy ktorych symulacja powtarza nagranie,
    i ich niepewnosc (`FitResult.band`) przy tym, ze reszta jest NIEZNANA, a nie rowna modelowi."""
    check_recording(rec)
    unknown = set(free) - set(PARAMS)
    if unknown:
        raise ValueError(f"nieznane parametry: {sorted(unknown)}")
    rp = Replayer(spec)
    idx = list(range(5))
    meas = rec.measured[:, idx]
    ok = ~np.isnan(meas)
    target = meas[ok]
    calls = [0]
    best = [np.inf]
    axes = [PARAMS.index(n) for n in PARAMS if n in free]

    def unpack(x):
        dyn = Dynamics(**{n: float(np.exp(v)) for n, v in zip(PARAMS[:4], x[:4])})
        return dyn, float(np.clip(x[4], 0.0, 0.2))

    def costs(xs) -> np.ndarray:
        pred = np.degrees(rp.run_q(rec, [unpack(x) for x in xs])[:, :, idx])[:, ok]
        r = pred - target[None]
        c = np.sqrt(np.mean(r * r, axis=1))
        calls[0] += len(xs)
        best[0] = min(best[0], float(c.min()))
        if on_progress:
            on_progress(calls[0], best[0])
        return c

    try:
        base = float(costs([np.zeros(5)])[0])
        x = np.zeros(5)
        if 4 in axes:
            # Najpierw samo opoznienie - jego dolina jest ostra i gladka.
            delays = np.linspace(0.0, 0.12, 13)
            x[4] = delays[int(np.argmin(costs([np.array([0, 0, 0, 0, dl]) for dl in delays])))]
        spans = np.array([0.6, 0.6, 0.4, 0.8, 0.03])
        # Najostrzej wyznaczone osie najpierw (opoznienie, armatura, tlumienie) - z trzech
        # startow armatury (model i +-25 %), bo armatura i opoznienie czesciowo sie kompensuja.
        # Same dwa starty +-25 % konczyly na pelnym pobudzeniu w lokalnym minimum (armatura
        # 1,13 i opoznienie 60 ms przy prawdzie 1,3 i 55 ms, blad 0,099 st. zamiast 0,064).
        order = [a for a in (4, 2, 1, 0, 3) if a in axes]
        starts = [x.copy()]
        if 2 in axes:
            starts += [np.r_[x[:2], la, x[3:]] for la in (np.log(0.8), np.log(1.25))]
        xs, cs = _coordinate_scan(costs, starts, spans, cycles=5, order=order)
        k = int(np.argmin(cs))
        x, best_c = xs[k], float(cs[k])
        band = {name: (0.0 if i in axes else np.inf) for i, name in enumerate(PARAMS)}
        # Czulosc: najmniejszy krok, po ktorym blad rosnie o 10 % (jedna partia na wszystkie osie).
        steps = {i: ((0.05, 0.1, 0.2, 0.35, 0.6) if i < 4 else (0.002, 0.005, 0.01, 0.02, 0.04)) for i in axes}
        probe = [x + sgn * d * np.eye(5)[i] for i in axes for d in steps[i] for sgn in (+1, -1)]
        if probe:
            c = iter(costs(probe))
            for i in axes:
                width = np.inf
                for d in steps[i]:
                    if min(next(c), next(c)) > 1.1 * best_c and np.isinf(width):
                        width = d
                band[PARAMS[i]] = float(width)
        # Profil: kp i tarcie suche na krancach swojego zakresu (osobno), reszta dopasowana
        # od nowa. Przesuniecie minimum to niepewnosc wyniku, ktorej sama czulosc nie widzi.
        prof_starts, prof_names = [], []
        for name, (lo, hi) in UNFITTED_RANGES.items():
            i = PARAMS.index(name)
            if i in axes or not axes:
                continue
            for v in (lo, hi):
                y = x.copy()
                y[i] = np.log(v)
                prof_starts.append(y)
                prof_names.append((name, v))
        profiles = []
        if prof_starts:
            pxs, pcs = _coordinate_scan(costs, prof_starts, np.array([0.3, 0.3, 0.3, 0.3, 0.02]), cycles=4,
                                        order=order)
            for (name, v), px, pc in zip(prof_names, pxs, pcs):
                profiles.append((name, v, px, float(pc)))
                for i in axes:
                    shift = abs(px[i] - x[i])
                    band[PARAMS[i]] = max(band[PARAMS[i]], float(np.expm1(shift) if i < 4 else shift))
    finally:
        rp.scene.close()
    dyn, delay = unpack(x)
    dyn.delay = delay * control_hz
    dyn.fitted = tuple(n for n in PARAMS if n in free)
    parts = [f"{k} +-{100 * band[k]:.0f}%" for k in PARAMS[:4] if k in free]
    if "delay" in free:
        parts.append(f"opoznienie +-{1000 * band['delay']:.0f} ms")
    fixed = [k for k in PARAMS if k not in free]
    ranges = ", ".join(f"{k} x{lo:g}-{hi:g}" for k, (lo, hi) in UNFITTED_RANGES.items() if k in fixed)
    dyn.source = (f"identyfikacja {rec.meta.get('time', '')} ({rec.meta.get('backend', '?')}); "
                  f"niepewnosc: {', '.join(parts)}"
                  + (f"; z modelu (niewyznaczalne z tego ruchu): {', '.join(fixed)}"
                     + (f", w treningu szeroko ({ranges})" if ranges else "") if fixed else "")).strip()
    dyn.fit_deg = best_c
    return FitResult(dyn, base, band, calls[0], profiles)


def fit(rec: Recording, spec: RobotSpec = SO101, control_hz: float = 20.0, on_progress=None) -> tuple[Dynamics, float]:
    """`identify` w starym ksztalcie: (dynamika, blad symulacji nominalnej)."""
    out = identify(rec, spec, control_hz, on_progress)
    return out.dyn, out.base


