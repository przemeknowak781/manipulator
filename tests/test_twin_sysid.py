"""Identyfikacja dynamiki serw: nagranie z symulacji o ZNANEJ dynamice ma dac ja z powrotem.

Prawdy sa celowo "krzywe": poza siatka startowa dopasowania i z kp oraz tarciem
innymi niz w modelu. Dopasowanie ma oddac to, co z tego ruchu da sie wyznaczyc
(tlumienie, armatura, opoznienie), a kp i tarcia NIE udawac - stary test
sprawdzal tylko tlumienie i opoznienie na prawdzie z siatki startowej i nie
zauwazyl kp 0,72 przy prawdzie 1,0. Potem test jechal na skroconym pobudzeniu
(2 stawy), a na PELNYM - tym, ktore puszcza panel - pierwsza z jego prawd
dawala armature -13 % i opoznienie +5 ms przy "pasmie" +-5 % i +-5 ms.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("mujoco")

from lerobot_mp.twin.rl.randomize import Dynamics, Randomization  # noqa: E402
from lerobot_mp.twin.rl.sysid import (  # noqa: E402
    IDENTIFIED, JOINTS, OWNER, Recording, Replayer, check_recording, excitation, fit, identify, plan_target, record,
)
from lerobot_mp.twin.robots import SO101  # noqa: E402


def synthetic_recording(truth: Dynamics, delay: float, joints=None, hold=0.7):
    """Nagranie jak z `record`: `joints=None` - pelne pobudzenie panelu (`excitation(home)`)."""
    plan = excitation(SO101.home) if joints is None else excitation(SO101.home, joints=joints, hold=hold)
    hz = 50.0
    t = np.arange(0, plan[-1][0] + 0.5, 1 / hz)
    cur = np.array([SO101.home[j] for j in JOINTS])
    cmd = []
    for tk in t:                                   # jak nadzor: cel z planu, najwyzej 120 st/s
        want = np.array([plan_target(plan, tk)[j] for j in JOINTS])
        cur = cur + np.clip(want - cur, -120 / hz, 120 / hz)
        cmd.append(cur.copy())
    cmd = np.array(cmd)
    q = Replayer().run(Recording(t, cmd, np.full_like(cmd, np.nan)), truth, delay)
    meas = q + np.random.default_rng(0).normal(0, 0.05, q.shape)
    meas[1::2] = np.nan                             # serwa czytane co drugi takt
    return Recording(t, cmd, meas, {"backend": "test", "time": "test"})


@pytest.mark.parametrize("truth, delay", [
    (Dynamics(kp=0.8, damping=1.4, armature=1.3, frictionloss=0.7), 0.055),   # z przegladu: armatura -13 %
    (Dynamics(kp=0.7, damping=1.07, armature=0.93, frictionloss=1.0), 0.043),
    (Dynamics(kp=1.3, damping=0.72, armature=1.12, frictionloss=1.4), 0.067),
])
def test_identification_on_the_panel_excitation_is_as_good_as_it_says(truth, delay):
    rec = synthetic_recording(truth, delay)
    out = identify(rec)
    dyn = out.dyn
    assert dyn.fit_deg < 0.1 * out.base              # symulacja po dopasowaniu wyraznie blizej nagrania
    assert dyn.fit_deg < 0.15                        # blisko szumu (0,05 st.) mimo kp i tarcia z modelu
    # Dokladnosc zmierzona na 13 prawdach (kp 0,7-1,3, tarcie 0,55-1,8): tlumienie do 10,5 %,
    # armatura do 8,2 %, opoznienie do 1,9 ms. Tu z zapasem, ale ciasniej niz stary wynik (13 %, 5 ms).
    assert dyn.damping == pytest.approx(truth.damping, rel=0.12)
    assert dyn.armature == pytest.approx(truth.armature, rel=0.1)
    assert dyn.delay / 20 == pytest.approx(delay, abs=0.003)
    # Niepewnosc mowi prawde: blad kazdego dopasowanego parametru miesci sie w jego pasmie,
    # a pasmo nie jest "nieskonczone na wszelki wypadek".
    for name in ("damping", "armature"):
        err = abs(getattr(dyn, name) / getattr(truth, name) - 1)
        assert err <= out.band[name] <= 0.4, (name, err, out.band[name])
    assert abs(dyn.delay / 20 - delay) <= out.band["delay"] <= 0.01
    # Pasmo idzie z dynamika do workspace'u (opoznienie w taktach polityki, jak `dyn.delay`).
    assert dyn.band == {"damping": out.band["damping"], "armature": out.band["armature"],
                        "delay": pytest.approx(20 * out.band["delay"])}
    print("idealne", {k: round(v, 3) for k, v in out.band.items()}, out.reasons)
    # kp i tarcie nie sa wyznaczalne z tego ruchu - zostaja z modelu, wynik i opis to mowia.
    assert dyn.kp == 1.0 and dyn.frictionloss == 1.0
    assert dyn.fitted == IDENTIFIED == ("damping", "armature", "delay")
    assert "z modelu" in dyn.source and "kp" in dyn.source and "frictionloss" in dyn.source
    assert np.isinf(out.band["kp"]) and np.isinf(out.band["frictionloss"])
    # Prawdziwe ramie lezy w swiatach treningu wokol wyniku - WSZYSTKIE parametry, takze kp
    # i tarcie (stare kp 0,85-1,15 wokol 1,0 nie zawieralo kp 0,7 / 0,8 / 1,3).
    rand = Randomization.around(dyn)
    for name in ("kp", "damping", "armature", "frictionloss"):
        lo, hi = getattr(rand, name)
        assert lo * getattr(dyn, name) <= getattr(truth, name) <= hi * getattr(dyn, name), name
    assert rand.min_delay <= delay * 20 <= rand.max_delay


def test_dynamics_remember_what_was_fitted():
    """`fitted` przezywa zapis w workspace (JSON) i odczyt; stara dynamika bez pola - nic nie dopasowano."""
    import json

    dyn = Dynamics(damping=1.2, armature=0.9, delay=1.1, source="identyfikacja", fitted=IDENTIFIED)
    back = Dynamics.from_dict(json.loads(json.dumps(dyn.to_dict())))
    assert back == dyn and back.fitted == IDENTIFIED and back.is_fitted("armature") and not back.is_fitted("kp")
    old = dyn.to_dict()
    old.pop("fitted")
    assert Dynamics.from_dict(old).fitted == () and Dynamics.from_dict(None).fitted == ()


def test_a_fit_that_explains_nothing_is_not_useful():
    """s6 weryfikatora: blad 0,55 -> 0,51 st., armatura x0,45 z pasmem +-inf - panel proponowal
    zapis, a trening stawal wokol 0,45 przy prawdzie 1,0. Taki wynik ma sie przyznac, ze nic nie mowi."""
    from lerobot_mp.twin.rl.sysid import FitResult, assess

    inf = float("inf")
    bad = Dynamics(damping=1.25, armature=0.45, delay=0.02, fitted=IDENTIFIED, fit_deg=0.51,
                   band={"damping": 0.6, "armature": inf, "delay": inf})
    res = FitResult(bad, 0.55, {"damping": 0.6, "armature": inf, "delay": inf})
    assert not res.useful
    assert any("armature" in r and "nieskonczona" in r for r in res.reasons)
    assert any("nic nie wyjasnia" in r for r in res.reasons)
    good = Dynamics(damping=1.1, armature=0.9, delay=0.6, fitted=IDENTIFIED, fit_deg=0.06,
                    band={"damping": 0.12, "armature": 0.08, "delay": 0.04})
    assert assess(good, 0.85) == [] and FitResult(good, 0.85).useful
    assert assess(good, 0.07) != []                             # 0,07 -> 0,06: nic nie poprawilo
    wide = Dynamics(**{**good.__dict__, "band": dict(good.band, damping=0.35)})
    assert any("damping" in r for r in assess(wide, 0.85))      # pasmo 35 % (s17: blad 62 %)
    assert assess(Dynamics(**{**good.__dict__, "band": dict(good.band, delay=0.5)}), 0.85)
    assert assess(Dynamics(**{**good.__dict__, "band": {}}), 0.85)     # brak pasma = nie wiadomo
    assert assess(Dynamics(), None)                                     # nic nie mierzono


def test_fit_keeps_the_old_shape():
    truth = Dynamics(kp=1.0, damping=1.4, armature=1.0, frictionloss=1.0)
    dyn, base = fit(synthetic_recording(truth, 0.04, joints=("shoulder_lift", "elbow_flex")))
    assert isinstance(dyn, Dynamics) and base > dyn.fit_deg


def test_truncated_recording_is_rejected():
    """Petla ramienia padla w trakcie: reszta nagrania to NaN - nie wolno z tego dopasowac dynamiki."""
    rec = synthetic_recording(Dynamics(), 0.04, joints=("shoulder_lift", "elbow_flex"))
    cut = len(rec.t) // 4
    rec.measured[cut:] = np.nan
    with pytest.raises(ValueError, match="bez pomiaru"):
        check_recording(rec)
    rec.command[cut:] = np.nan
    with pytest.raises(ValueError, match="bez rozkazu"):
        identify(rec)


def test_batched_replay_matches_a_plain_step_loop():
    """`run_many` (rollout, wiele dynamik naraz, w watkach) = zwykla petla `mj_step` w Pythonie."""
    import mujoco

    rec = synthetic_recording(Dynamics(), 0.0, joints=("elbow_flex",), hold=0.3)
    rp = Replayer()
    try:
        sets = [(Dynamics(kp=0.8, armature=1.2), 0.05), (Dynamics(damping=1.3, frictionloss=0.6), 0.013)]
        many = rp.run_many(rec, sets)
        prep = rp._prepare(rec)
        m, d = rp.scene.model, rp.scene.data
        for (dyn, dl), got in zip(sets, many):
            rp.apply(dyn)
            mujoco.mj_setState(m, d, prep["init"], mujoco.mjtState.mjSTATE_FULLPHYSICS)
            ctrl = rp._controls(rec, prep, dl)
            q, k = np.zeros((len(rec.t), 6)), 0
            for i, n_i in enumerate(prep["after"]):
                while k < min(n_i, prep["n_steps"]):
                    d.ctrl[rp.act] = ctrl[k]
                    mujoco.mj_step(m, d)
                    k += 1
                q[i] = d.qpos[rp.scene.kin.qadr]
            assert np.abs(rp._from_q(q) - got).max() < 1e-9
        assert np.abs(many[0] - many[1]).max() > 0.1
    finally:
        rp.scene.close()


def test_rows_from_ticks_pair_each_reading_with_the_command_of_its_tick():
    """Wiersz = takt petli: pomiar z taktu przed jego wysylka, takt bez wysylki trzyma rozkaz,
    odczyt z `measured_t` sprzed taktu dostaje wlasny wiersz z rozkazem sprzed taktu."""
    from lerobot_mp.twin.rl.sysid import _rows_from_ticks

    home = dict(SO101.home)
    m1 = dict(home, shoulder_pan=1.0)
    m2 = dict(home, shoulder_pan=2.0)
    samples = [
        (10.00, dict(home, shoulder_pan=5.0), [m1[j] for j in JOINTS], 10.00),
        (10.03, {}, None, 10.00),                                   # lacze wstrzymane - nic nie poszlo
        (10.07, {"shoulder_pan": 7.0}, [m2[j] for j in JOINTS], 10.065),
    ]
    t, cmd, meas = _rows_from_ticks(samples, home)
    assert np.allclose(t, [0.0, 0.03, 0.065, 0.07])
    assert list(cmd[:, 0]) == [5.0, 5.0, 5.0, 7.0]
    assert meas[0, 0] == 1.0 and np.isnan(meas[1]).all() and meas[2, 0] == 2.0 and np.isnan(meas[3]).all()


def test_recording_roundtrip(tmp_path):
    rec = Recording(np.arange(3.0), np.ones((3, 6)), np.full((3, 6), np.nan), {"a": 1})
    back = Recording.load(rec.save(tmp_path / "r.npz"))
    assert np.allclose(back.t, rec.t) and back.meta == {"a": 1}


# ------------------------------------------------------------ nagranie na blizniaku
@pytest.fixture
def twin():
    from lerobot_mp.twin.runtime import Twin
    from lerobot_mp.twin.workspace import Workspace

    tw = Twin(Workspace())
    try:
        yield tw
    finally:
        tw.close()


def short_plan():
    home = dict(SO101.home)
    return [(0.0, home), (0.3, dict(home, shoulder_pan=5.0)), (0.6, home)]


def test_record_approaches_the_start_smoothly_before_recording(twin):
    """Ramie stoi 40 st. od domu: nagranie zaczyna sie w domu, a dojazd idzie rampa, nie skokiem."""
    with twin.lock:
        twin.scene.set_joints(dict(SO101.home, shoulder_pan=40.0))
    twin.connect("sim")
    seen = []
    orig = twin.set_target

    def spy(joints, owner=None):
        seen.append(float(joints.get("shoulder_pan", np.nan)))
        orig(joints, owner=owner)
    twin.set_target = spy
    rec = record(twin, short_plan(), settle=0.2)
    assert abs(rec.command[0, 0] - SO101.home["shoulder_pan"]) < 1.0      # dojazd PRZED nagraniem
    assert twin.owner is None and not twin._engaged                       # ramie oddane
    # Nagrywane rozkazy nie skacza: najwyzej ~5 st. (skok planu) miedzy taktami.
    assert np.abs(np.diff(rec.command[:, 0])).max() < 6.0


def test_record_stops_when_the_arm_loop_dies(twin):
    """Kabel wypadl w trakcie: `record` ma rzucic, a nie oddac nagrania z samymi NaN."""
    twin.connect("sim")
    b = twin._backend
    calls = [0]
    real = b.read_joints

    def failing():
        calls[0] += 1
        if calls[0] > 150:
            raise OSError("kabel wypadl")
        return real()
    b.read_joints = failing
    with pytest.raises(RuntimeError, match="petla ramienia|odebrane|przerwana"):
        record(twin, short_plan() + [(3.0, dict(SO101.home))], settle=0.2)


def test_record_is_preempted_by_home(twin):
    import threading

    twin.connect("sim")
    threading.Timer(1.8, twin.home).start()
    with pytest.raises(RuntimeError, match="przerwan"):
        record(twin, short_plan() + [(3.0, dict(SO101.home))], settle=0.2)
    assert twin.owner is None


def test_record_refuses_while_another_owner_drives(twin):
    twin.connect("sim")
    twin.claim("polityka")
    with pytest.raises(RuntimeError, match="ramie zajete: polityka"):
        record(twin, short_plan())


def test_record_claims_once_and_only_drives_an_arm_it_still_has(twin):
    """Kontrakt K3: `record` bierze ramie RAZ, potem juz tylko `move(take=False)`."""
    twin.connect("sim")
    claims, takes = [], []
    claim, move = twin.claim, twin.move

    def spy_claim(owner, preempt=None):
        claims.append((owner, twin.safety_state.value))
        claim(owner, preempt)

    def spy_move(*a, **kw):
        takes.append(kw.get("take", True))
        move(*a, **kw)
    twin.claim, twin.move = spy_claim, spy_move
    record(twin, short_plan(), settle=0.2)
    assert claims == [(OWNER, "IDLE")] and takes == [False]
    assert twin.owner is None


def test_record_does_not_take_back_an_arm_freed_by_home(twin):
    """Dom wcisniety tuz po "Identyfikuj", zanim watek zadania doszedl do `record`.

    Zmierzone przed poprawka: `record` bral zwolnione ramie z powrotem, `claim`
    przerywal rampe do domu, a dojazd jechal jako "identyfikacja". Flage przerwania
    zadania czyscil jeszcze `Job.start` - wiec bez niej tez nie wolno ruszyc.
    """
    with twin.lock:
        twin.scene.set_joints(dict(SO101.home, shoulder_pan=40.0))
    twin.connect("sim")
    cancel = threading.Event()
    twin.claim(OWNER, preempt=cancel.set)            # panel bierze ramie dla identyfikacji
    twin.home()                                      # operator: Dom
    assert cancel.is_set() and twin.owner is None
    with pytest.raises(RuntimeError, match="przerwana"):
        record(twin, short_plan(), settle=0.2, should_stop=cancel.is_set)
    cancel.clear()                                   # jak po `Job.start`
    for kw in ({}, {"take": False}):
        with pytest.raises(RuntimeError, match="przerwana"):
            record(twin, short_plan(), settle=0.2, should_stop=cancel.is_set, **kw)
        assert twin.owner is None
    assert twin.safety_state.value == "HOMING"       # rampy do domu nikt nie przerwal


def test_record_with_take_false_needs_the_arm_already_taken(twin):
    twin.connect("sim")
    with pytest.raises(RuntimeError, match="odebrane"):
        record(twin, short_plan(), settle=0.2, take=False)
    assert twin.owner is None and not twin._engaged
    twin.claim(OWNER)                                # panel wzial ramie przed startem zadania
    rec = record(twin, short_plan(), settle=0.2, take=False)
    assert len(rec.t) > 10 and twin.owner is None and not twin._engaged


def test_record_waits_for_the_start_ramp_instead_of_cutting_it(twin):
    """Polaczenie z `go_home`: nadzor jedzie rampa startowa - `record` jej nie przerywa `claim`-em."""
    with twin.lock:
        twin.scene.set_joints(dict(SO101.home, shoulder_pan=25.0))
    twin.connect("sim", go_home=True)
    assert twin.safety_state.value == "STARTING"
    states = []
    claim = twin.claim

    def spy_claim(owner, preempt=None):
        states.append(twin.safety_state.value)
        claim(owner, preempt)
    twin.claim = spy_claim
    record(twin, short_plan(), settle=0.2)
    assert states and states[0] != "STARTING"


# ------------------------------------------------------------ nagranie z czasem prawdziwego ramienia
class JitterArm:
    """Petla ramienia z czasem jak na prawdziwym ramieniu (s17 weryfikatora), w zegarze wirtualnym.

    30 Hz z jitterem 0-8 ms i 3 % przestojow 30 ms, serwa czytane co drugi takt, odczyt
    odpowiada pozycji 0-4 ms po poczatku taktu, status widac dopiero po koncu taktu.
    Rozkazy ida za planem pobudzenia jak nadzor (najwyzej 120 st./s) - odpowiedz liczy
    `Replayer` ze znana dynamika. `record` jedzie na tym przez podmieniony `sysid.time`:
    `sleep` przesuwa zegar (z narzutem rejestratora 0,5-2 ms) i puszcza takty, ktore minely.
    """

    def __init__(self, truth: Dynamics, delay: float, listeners: bool, hz: float = 30.0, jit: float = 0.008,
                 settle: float = 0.5, seed: int = 3):
        rng = np.random.default_rng(seed)
        self.rng = rng
        plan = excitation(SO101.home)
        span = plan[-1][0] + settle + 1.0
        tau = [0.0]
        while tau[-1] < span:
            tau.append(tau[-1] + 1 / hz + rng.uniform(0, jit) + (0.03 if rng.random() < 0.03 else 0.0))
        self.tau = np.array(tau)
        cur, prev, cmd = np.array([SO101.home[j] for j in JOINTS]), 0.0, []
        for tk in self.tau:
            want = np.array([plan_target(plan, tk)[j] for j in JOINTS])
            step = 120 * max(tk - prev, 1e-3)
            prev = tk
            cur = cur + np.clip(want - cur, -step, step)
            cmd.append(cur.copy())
        cmd = np.array(cmd)
        tf = np.arange(0, span, 0.001)
        kf = np.searchsorted(self.tau, tf, side="right") - 1
        q = Replayer().run(Recording(tf, cmd[kf], np.full((len(tf), 6), np.nan)), truth, delay)
        lag = rng.uniform(0.0, 0.004, len(self.tau))           # serwo podaje pozycje chwile po starcie taktu
        meas = np.array([np.interp(self.tau + lag, tf, q[:, j]) for j in range(6)]).T
        meas += rng.normal(0, 0.05, meas.shape)
        self.shown = self.tau + lag + 0.001                     # status po koncu taktu
        self.samples, self.statuses = [], []
        m_last, mt_last = {}, 0.0
        for k, tk in enumerate(self.tau):
            fresh = k % 2 == 0
            if fresh:
                m_last, mt_last = dict(zip(JOINTS, map(float, meas[k]))), float(tk)
            sent = dict(zip(JOINTS, map(float, cmd[k])))
            self.samples.append(SimpleNamespace(t=float(tk), sent=sent, measured=m_last, measured_t=mt_last,
                                                fresh=fresh))
            self.statuses.append(SimpleNamespace(connected=True, error="", backend="jitter", measured=m_last,
                                                 measured_t=mt_last, command=sent))
        self.now, self.fired = -0.2, 0
        self.listeners: dict[int, object] = {}
        if listeners:
            self.add_tick_listener = self._add
            self.remove_tick_listener = self.listeners.pop
        self.connected, self.loop_hz, self.owner, self.preempt_reason = True, hz, None, ""
        self.safety_state = SimpleNamespace(value="IDLE")
        self.time = SimpleNamespace(monotonic=lambda: self.now, sleep=self._sleep, strftime=lambda f: "test")

    def _add(self, fn):
        self.listeners[id(fn)] = fn
        return id(fn)

    def _sleep(self, d: float) -> None:
        self.now += d + self.rng.uniform(0.0005, 0.002)
        while self.fired < len(self.tau) and self.shown[self.fired] <= self.now:
            for fn in list(self.listeners.values()):
                fn(self.samples[self.fired])
            self.fired += 1

    @property
    def status(self):
        k = int(np.searchsorted(self.shown, self.now, side="right")) - 1
        return self.statuses[k] if k >= 0 else SimpleNamespace(
            connected=True, error="", backend="jitter", measured={}, measured_t=0.0, command=dict(SO101.home))

    def joints(self):
        return dict(SO101.home)

    def claim(self, owner, preempt=None):
        self.owner = owner

    def release(self, owner):
        self.owner = None

    def move(self, *a, **kw):
        self._sleep(0.2)

    def set_target(self, joints, owner=None):
        pass

    def set_engaged(self, on, owner=None):
        pass


@pytest.mark.parametrize("listeners", [True, False], ids=["tick_listener", "status_fallback"])
def test_record_under_real_arm_timing_keeps_the_documented_accuracy(listeners, monkeypatch):
    """Petla z jitterem i rzadszym odczytem serw: nagranie przez `record` ma dac te sama dynamike.

    Zmierzone przed poprawka (s17 weryfikatora, ta sama prawda): `record` stemplowal wiersze
    zegarem rejestratora - tlumienie +62 % (pasmo 35 %), armatura -11 %, opoznienie -7,5 ms,
    blad 0,35 st. Nagranie w taktach petli: tlumienie i armatura w kilku procentach.
    """
    from lerobot_mp.twin.rl import sysid

    truth, delay = Dynamics(kp=0.85, damping=1.1, armature=0.9, frictionloss=1.2), 0.03
    arm = JitterArm(truth, delay, listeners)
    monkeypatch.setattr(sysid, "time", arm.time)
    rec = record(arm, excitation(SO101.home), settle=0.5)
    assert rec.meta["stamps"] == ("tick" if listeners else "measured_t")
    assert not arm.listeners                                   # sluchacz zdjety po nagraniu
    out = identify(rec)
    dyn = out.dyn
    errs = {"damping": dyn.damping / truth.damping - 1, "armature": dyn.armature / truth.armature - 1,
            "delay_ms": 1000 * (dyn.delay / 20 - delay)}
    print("jitter", "listener" if listeners else "fallback", {k: round(v, 3) for k, v in errs.items()},
          {k: round(v, 3) for k, v in out.band.items()}, round(dyn.fit_deg, 3), round(out.base, 3))
    assert dyn.fit_deg < 0.2 * out.base and dyn.fit_deg < 0.2
    assert abs(errs["damping"]) <= 0.15 and abs(errs["armature"]) <= 0.12 and abs(errs["delay_ms"]) <= 3.0
    for name in ("damping", "armature"):
        assert abs(errs[name]) <= out.band[name], (name, errs[name], out.band[name])
    assert abs(errs["delay_ms"]) / 1000 <= out.band["delay"]
    assert out.useful, out.reasons
