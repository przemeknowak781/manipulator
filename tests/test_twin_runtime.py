"""Petla blizniaka na prawdziwym (tu: udawanym) ramieniu - wlasnosc ramienia, STOP, bledy serw.

Wiekszosc testow idzie bez watku petli (`connect(..., threaded=False)` + `step`),
w czasie symulowanym: wynik nie zalezy od obciazenia procesora.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

pytest.importorskip("mujoco")

from lerobot_mp.robot.base import RobotBackend, RobotInfo  # noqa: E402
from lerobot_mp.twin import runtime as rt  # noqa: E402
from lerobot_mp.twin import scene as sc  # noqa: E402
from lerobot_mp.twin.robots import SO101  # noqa: E402
from lerobot_mp.twin.runtime import RenderWorker, Twin  # noqa: E402
from lerobot_mp.twin.workspace import Workspace  # noqa: E402

DT = 0.02


class FakeArm(RobotBackend):
    """Ramie "prawdziwe" (simulated=False): stawy ida za rozkazem, chyba ze sa zablokowane.

    `vmax` - serwo jedzie do celu najwyzej tyle st./s (w `step`), zamiast byc tam od razu.
    `silent` - ile kolejnych odczytow bez odpowiedzi: jak `FeetechArm`, oddaje wtedy stare
    pozycje, ustawia `_link_silent` i nic nie wysyla. `glitch` - nadpisania kolejnych
    odczytow (przeklamana ramka z poprawna suma kontrolna).

    `delay` - odczyt pokazuje poze sprzed tylu sekund (wiek pomiaru wzgledem najnowszego rozkazu:
    okres odczytu + most; pesymistycznie, bo prawdziwa probka jest mlodsza niz chwila zapytania
    w petli), `amax` - serwo
    rozpedza sie i hamuje najwyzej tyle st./s^2 (jak STS3215: P do celu, limit predkosci
    i przyspieszenia, kroki 1 ms). `vmax_of` - inna predkosc dla wybranych stawow.
    `late` - nastepny swiezy odczyt zwraca te poze (spozniona odpowiedz sprzed przestoju).
    """

    def __init__(self, start: dict[str, float] | None = None, vmax: float | None = None,
                 delay: float = 0.0, amax: float | None = None, vmax_of: dict[str, float] | None = None):
        self.info = RobotInfo(name="udawane", simulated=False)
        self.pos = dict(SO101.home, **(start or {}))
        self.goal = dict(self.pos)
        self.vmax = vmax
        self.vmax_of = dict(vmax_of or {})
        self.delay, self.amax = delay, amax
        self.vel = {k: 0.0 for k in self.pos}
        self.t = 0.0
        self.history: list[tuple[float, dict[str, float]]] = [(0.0, dict(self.pos))]
        self.late: dict[str, float] | None = None
        self.blocked: dict[str, float] = {}
        #: Twarde oparcie stawu (lo, hi): serwo dalej nie pojedzie, ale do niego tak.
        self.wall: dict[str, tuple[float, float]] = {}
        self.sent: list[dict[str, float]] = []
        self.fault_list: list[str] = []
        self.limits: dict[str, tuple[float, float]] = {}
        self.clamp: dict[str, tuple[float, float]] = {}
        #: Jak `FeetechArm`: przycina rozkaz do limitow swojej konfiguracji (`cfg` z `create_backend`).
        self.clamp_to_cfg = False
        self.cfg = None
        self.silent = 0
        self._link_silent = False
        self._last_read = dict(self.pos)
        self.glitch: list[dict[str, float]] = []
        self.grip_ticks: tuple[float, float, float] | None = None
        self._on = False

    def connect(self):
        self._on = True

    def disconnect(self):
        self._on = False

    @property
    def is_connected(self):
        return self._on

    def read_joints(self):
        if self.silent > 0:
            self.silent -= 1
            self._link_silent = True
            return dict(self._last_read)
        self._link_silent = False
        out = dict(self.pos)
        if self.delay > 0:
            old = [p for t, p in self.history if t <= self.t - self.delay]
            out = dict(old[-1] if old else self.history[0][1])
        if self.late is not None:
            out, self.late = dict(self.late), None
        if self.glitch:
            out.update(self.glitch.pop(0))
        self._last_read = out
        return dict(out)

    def send_joints(self, targets):
        if self._link_silent:
            return {}
        out = {}
        for k, v in targets.items():
            lo, hi = self.clamp.get(k, (-1e9, 1e9))
            if self.clamp_to_cfg:
                lo, hi = max(lo, self.cfg.joint(k).min), min(hi, self.cfg.joint(k).max)
            out[k] = min(max(v, lo), hi)
        self.sent.append(out)
        for k, v in out.items():
            self.goal[k] = v
            if self.vmax is None:
                self.pos[k] = self._walled(k, self.blocked.get(k, v))
        return out

    def step(self, dt):
        self.t += dt
        if self.vmax is None:
            self._record()
            return
        if self.amax is None:
            for k, g in self.goal.items():
                vm = self.vmax_of.get(k, self.vmax)
                p = self.pos[k] + float(np.clip(g - self.pos[k], -vm * dt, vm * dt))
                self.pos[k] = self._walled(k, self.blocked.get(k, p))
            self._record()
            return
        n = max(1, int(round(dt / 0.001)))
        h = dt / n
        da = self.amax * h
        for _ in range(n):
            for k, g in self.goal.items():
                vm = self.vmax_of.get(k, self.vmax)
                vd = min(max(25.0 * (g - self.pos[k]), -vm), vm)
                self.vel[k] += min(max(vd - self.vel[k], -da), da)
                p = self.pos[k] + self.vel[k] * h
                if k in self.blocked:
                    p, self.vel[k] = self.blocked[k], 0.0
                if self._walled(k, p) != p:
                    p, self.vel[k] = self._walled(k, p), 0.0
                self.pos[k] = p
        self._record()

    def _walled(self, k, v):
        lo, hi = self.wall.get(k, (-1e9, 1e9))
        return min(max(v, lo), hi)

    def _record(self):
        if self.delay > 0:
            self.history.append((self.t, dict(self.pos)))
            self.history = [h for h in self.history if h[0] >= self.t - self.delay - 0.1] or self.history[-1:]

    def faults(self):
        return list(self.fault_list)

    def joint_limits(self):
        return dict(self.limits)

    def gripper_ticks(self):
        return self.grip_ticks


@pytest.fixture
def twin():
    tw = Twin(Workspace())
    try:
        yield tw
    finally:
        tw.close()


def connect_fake(twin, monkeypatch, arm: FakeArm, **kw):
    def make(cfg):
        arm.cfg = cfg
        return arm
    monkeypatch.setattr(rt, "create_backend", make)
    twin.connect("feetech", port="COM_TEST", threaded=False, **kw)


def run(twin, seconds):
    for _ in range(int(round(seconds / DT))):
        twin.step(DT)


# ------------------------------------------------------------ wlasnosc ramienia
def test_claim_is_exclusive_and_release_only_by_the_owner(twin):
    twin.claim("kalibracja")
    with pytest.raises(RuntimeError, match="ramie zajete: kalibracja"):
        twin.claim("polityka")
    twin.claim("kalibracja")                             # ten sam wlasciciel - wolno
    twin.release("polityka")
    assert twin.owner == "kalibracja"
    twin.release("kalibracja")
    assert twin.owner is None


def test_home_preempts_the_owner_and_the_arm_does_not_snap_back(twin, monkeypatch):
    """Pozycja domowa w trakcie polityki: po rampie ramie zostaje w domu, a nie wraca do celu polityki.

    Zmierzone przed poprawka: 57,8 -> 1,3 (rampa), potem 6 -> 45 -> 60 st. w pol sekundy.
    """
    arm = FakeArm()
    connect_fake(twin, monkeypatch, arm)
    called = []
    twin.claim("polityka", preempt=lambda: called.append(True))
    twin.set_engaged(True)
    twin.set_target({"shoulder_pan": 60.0}, owner="polityka")
    run(twin, 1.0)
    assert arm.pos["shoulder_pan"] > 55
    twin.home()
    assert called == [True] and twin.owner is None
    with pytest.raises(RuntimeError, match="odebrane"):
        twin.set_target({"shoulder_pan": 60.0}, owner="polityka")   # spozniony takt starej polityki
    twin.set_target({"shoulder_pan": 60.0})                          # albo ktos bez wlasnosci
    run(twin, 3.5)
    tail = []
    for _ in range(50):
        twin.step(DT)
        tail.append(arm.pos["shoulder_pan"])
    assert max(abs(v - SO101.home["shoulder_pan"]) for v in tail) < 0.5


def test_move_raises_when_the_arm_is_taken_away():
    tw = Twin(Workspace())
    try:
        tw.connect("sim")
        err = []

        def mover():
            try:
                tw.move({"shoulder_pan": 60.0}, duration=2.0)
            except RuntimeError as exc:
                err.append(str(exc))
        th = threading.Thread(target=mover)
        th.start()
        time.sleep(0.5)
        tw.home()
        th.join(5.0)
        assert err and "pozycja domowa" in err[0]
    finally:
        tw.close()


def test_move_belongs_to_the_calibration_and_refuses_a_foreign_owner():
    """Fala (`move`) i polityka nie moga pisac celu na zmiane: fala bierze ramie, polityka
    wtedy nie rusza, a fala nie ruszy, gdy ramie ma polityka."""
    tw = Twin(Workspace())
    try:
        tw.connect("sim")
        tw.move({"shoulder_pan": 5.0}, duration=0.2, settle=0.05)
        assert tw.owner == "kalibracja"
        with pytest.raises(RuntimeError, match="ramie zajete: kalibracja"):
            tw.claim("polityka")
        tw.home()                                         # koniec fali oddaje ramie
        assert tw.owner is None
        tw.claim("polityka")
        with pytest.raises(RuntimeError, match="ramie zajete: polityka"):
            tw.move({"shoulder_pan": 10.0}, duration=0.2)
        tw.release("polityka")
        # Fala, ktorej ramie odebrano miedzy przejazdami, nie bierze wolnego ramienia na nowo.
        with pytest.raises(RuntimeError, match="ramie odebrane"):
            tw.move({"shoulder_pan": 10.0}, duration=0.2, take=False)
        assert tw.owner is None
    finally:
        tw.close()


def test_claim_during_homing_takes_the_arm_where_it_stands(twin, monkeypatch):
    arm = FakeArm({"shoulder_pan": 50.0})
    connect_fake(twin, monkeypatch, arm, go_home=True)
    run(twin, 0.8)
    mid = arm.pos["shoulder_pan"]
    assert 5 < mid < 50
    twin.claim("polityka")
    run(twin, 3.0)                                        # rampa przerwana - ramie stoi
    # W zmierzonej pozie, czyli z odczytu sprzed najwyzej jednego okresu odczytu (40 ms).
    assert arm.pos["shoulder_pan"] == pytest.approx(mid, abs=2.0)


# ------------------------------------------------------------ STOP i trzymanie zmierzonej pozy
def test_hold_measured_releases_a_joint_pressing_into_an_obstacle(twin, monkeypatch):
    """Staw zablokowany na 10 st., rozkaz uciekl do 35: po `hold_measured` rozkaz = poza zmierzona."""
    arm = FakeArm()
    connect_fake(twin, monkeypatch, arm)
    arm.blocked["shoulder_pan"] = 10.0
    twin.set_engaged(True)
    twin.set_target({"shoulder_pan": 35.0})
    run(twin, 1.0)
    assert twin.status.command["shoulder_pan"] == pytest.approx(35.0, abs=0.5)
    twin.hold_measured()
    twin.set_engaged(False)
    run(twin, 0.5)
    assert arm.sent[-1]["shoulder_pan"] == pytest.approx(10.0, abs=0.5)


def test_estop_holds_the_measured_pose_not_the_last_command(twin, monkeypatch):
    arm = FakeArm()
    connect_fake(twin, monkeypatch, arm)
    arm.blocked["elbow_flex"] = 30.0
    twin.set_engaged(True)
    twin.set_target({"elbow_flex": 70.0})
    run(twin, 1.0)
    twin.estop()
    run(twin, 0.3)
    assert twin.safety_state.value == "ESTOP"
    assert arm.sent[-1]["elbow_flex"] == pytest.approx(30.0, abs=0.5)


@pytest.mark.parametrize("clamp_to_cfg", [False, True])
def test_connect_without_moving_sends_nothing_to_a_joint_outside_the_limits(twin, monkeypatch, clamp_to_cfg):
    """Ramie nr 2 spoczywa z shoulder_lift -101 st. przy limicie -95: polaczenie bez ruchu nie moze
    skoczyc do limitu, a pierwszy ruch wraca w zakres powoli - takze przez backend, ktory
    przycina rozkaz do limitow konfiguracji (jak `FeetechArm.send_joints`)."""
    arm = FakeArm({"shoulder_lift": -101.0})
    arm.clamp_to_cfg = clamp_to_cfg
    connect_fake(twin, monkeypatch, arm)
    run(twin, 1.0)
    assert arm.sent == []
    twin.set_engaged(True)
    twin.set_target({"shoulder_pan": 5.0})
    twin.step(DT)
    assert arm.sent[0]["shoulder_lift"] == pytest.approx(-101.0, abs=0.5)
    run(twin, 0.2)
    steps = np.diff([s["shoulder_lift"] for s in arm.sent])
    assert np.abs(steps).max() <= 15.0 * DT * 1.5 + 1e-6
    run(twin, 1.0)
    assert arm.sent[-1]["shoulder_lift"] >= -95.0 - 1e-6
    # Poszerzenie limitow backendu bylo chwilowe - staw w zakresie, limity znow z konfiguracji.
    assert arm.cfg.joint("shoulder_lift").min == pytest.approx(-95.0)
    assert twin.joint_limits()["shoulder_lift"][0] == pytest.approx(-95.0)


# ------------------------------------------------------------ bledy serw, przerwy, limity
def test_servo_fault_triggers_estop_and_preempts_the_owner(twin, monkeypatch):
    arm = FakeArm()
    connect_fake(twin, monkeypatch, arm)
    called = []
    twin.claim("identyfikacja", preempt=lambda: called.append(True))
    twin.set_engaged(True)
    run(twin, 0.2)
    arm.fault_list = ["shoulder_lift: przeciazenie"]
    twin.step(DT)
    assert twin.safety_state.value == "ESTOP"
    assert "przeciazenie" in twin.status.error
    assert twin.owner is None
    time.sleep(0.1)                                        # wywolanie zwrotne z watku petli idzie osobno
    assert called == [True]


def test_link_loss_stops_sending(twin, monkeypatch):
    arm = FakeArm()
    connect_fake(twin, monkeypatch, arm)
    twin.set_engaged(True)
    twin.set_target({"shoulder_pan": 20.0})
    run(twin, 0.2)
    n = len(arm.sent)
    arm.fault_list = ["brak odpowiedzi serw od 300 ms"]
    run(twin, 0.5)
    assert len(arm.sent) == n                              # nic nie czeka w buforze gniazda


def test_a_long_loop_gap_is_not_turned_into_a_command_jump(twin, monkeypatch):
    """Po przestoju petli (render, odczyt z zerwanego lacza) rozkaz nie skacze o max_vel * 0,2 s."""
    arm = FakeArm()
    connect_fake(twin, monkeypatch, arm)
    twin.claim("polityka")
    twin.set_engaged(True)
    twin.set_target({"shoulder_pan": 80.0}, owner="polityka")
    run(twin, 0.1)
    before = arm.sent[-1]["shoulder_pan"]
    twin.step(0.9)                                         # 0,9 s bez taktu
    after = arm.sent[-1]["shoulder_pan"]
    assert after - before <= 140.0 * 1.5 * DT + 1e-6
    assert twin.owner is None and "przerwa" in twin.status.error


def test_backend_limits_narrow_the_config_limits(twin, monkeypatch):
    arm = FakeArm()
    arm.limits = {"wrist_flex": (-200.0, 88.0)}
    connect_fake(twin, monkeypatch, arm)
    assert twin.joint_limits()["wrist_flex"] == (-95.0, 88.0)
    twin.set_engaged(True)
    twin.set_target({"wrist_flex": 95.0})
    run(twin, 2.0)
    assert arm.sent[-1]["wrist_flex"] == pytest.approx(88.0)


def test_status_command_is_what_the_backend_really_sent(twin, monkeypatch):
    arm = FakeArm()
    arm.clamp = {"shoulder_pan": (-50.0, 50.0)}            # serwo przycina po swojemu
    connect_fake(twin, monkeypatch, arm)
    twin.set_engaged(True)
    twin.set_target({"shoulder_pan": 70.0})
    run(twin, 1.0)
    assert twin.status.command["shoulder_pan"] == pytest.approx(50.0)
    twin.set_target({"shoulder_pan": 0.0})
    twin.step(DT)
    # Nadzor liczy dalej od tego, co poszlo (50), a nie od 70 - pierwszy krok w dol jest od razu widoczny.
    assert arm.sent[-1]["shoulder_pan"] < 50.0


# ------------------------------------------------------------ scena i render
def test_render_does_not_hold_the_loop_lock(twin):
    """W trakcie renderu petla ramienia bierze `Twin.lock` od razu (zmierzone: tworzenie renderera
    pod blokada trzymalo ja do 0,7 s, a nastepny rozkaz skakal o 0,2 s x max_vel)."""
    got = []

    def grab():                                            # jak takt petli: wez i oddaj w tym samym watku
        ok = twin.lock.acquire(timeout=0.5)
        got.append(ok)
        if ok:
            twin.lock.release()

    def fn(scene):
        th = threading.Thread(target=grab)
        th.start()
        th.join()
        return scene.data.qpos.copy()
    q = twin.render_with(fn)
    assert got == [True]
    assert np.allclose(q, twin.scene.data.qpos)


def test_rebuild_keeps_the_cube_where_it_is(twin):
    twin.configure(objects=[sc.Box("cube", (0.015,) * 3, (0.2, 0.0))])
    with twin.lock:
        m, d = twin.scene.model, twin.scene.data
        a = m.jnt_qposadr[m.body_jntadr[m.body("cube").id]]
        d.qpos[a:a + 3] += [0.06, -0.08, 0.09]
        want = d.qpos[a:a + 7].copy()
    twin.rebuild()
    with twin.lock:
        m, d = twin.scene.model, twin.scene.data
        a = m.jnt_qposadr[m.body_jntadr[m.body("cube").id]]
        assert np.allclose(d.qpos[a:a + 7], want)


def test_render_worker_call_after_stop_raises_instead_of_hanging():
    rw = RenderWorker()
    rw.stop()
    t0 = time.monotonic()
    with pytest.raises(RuntimeError, match="zatrzymany"):
        rw.call(lambda: 1)
    assert time.monotonic() - t0 < 2.0


def test_loop_death_preempts_the_owner(twin, monkeypatch):
    arm = FakeArm()
    connect_fake(twin, monkeypatch, arm)
    called = []
    twin.claim("identyfikacja", preempt=lambda: called.append(True))

    def boom():
        raise OSError("kabel wypadl")
    arm.read_joints = boom
    twin.step(DT)
    assert not twin.connected and "kabel" in twin.status.error
    assert called == [True] and twin.owner is None


# ------------------------------------------------------------ krotkie milczenie lacza
def stall(twin, arm, cycles: int, block: float = 0.27):
    """`cycles` odczytow bez odpowiedzi; kazdy trzyma petle `block` s (timeout odczytu przez most)."""
    arm.silent = cycles
    for _ in range(cycles):
        twin.step(block)


def test_a_short_link_stall_does_not_send_a_jump_when_the_link_returns(twin, monkeypatch):
    """Przestoj ponizej progu utraty lacza: nadzor stoi, a pierwszy rozkaz po powrocie nie skacze.

    Zmierzone przed poprawka (most, przestoj 0,9 s): rozkaz 10,5 -> 27,2 st. w jednej paczce.
    """
    arm = FakeArm()
    connect_fake(twin, monkeypatch, arm)
    twin.claim("kalibracja")
    twin.set_engaged(True)
    twin.set_target({"shoulder_pan": 80.0}, owner="kalibracja")
    run(twin, 0.1)
    n, last = len(arm.sent), arm.sent[-1]["shoulder_pan"]
    t_meas = twin.status.measured_t
    stall(twin, arm, 3)
    assert len(arm.sent) == n                              # nic nie poszlo w trakcie
    # Panel i identyfikacja widza to, co naprawde poszlo, i ze pomiaru nie bylo.
    assert twin.status.command["shoulder_pan"] == pytest.approx(last)
    assert twin.status.measured_t == t_meas
    run(twin, 0.5)
    goals = [last] + [s["shoulder_pan"] for s in arm.sent[n:]]
    assert np.abs(np.diff(goals)).max() <= 140.0 * 1.5 * DT + 1e-6
    assert goals[-1] > last + 20                           # ruch jedzie dalej
    assert twin.safety_state.value == "ACTIVE" and twin.owner == "kalibracja"


def test_after_a_stall_the_command_restarts_from_the_measured_pose(twin, monkeypatch):
    """Staw, ktory w trakcie przestoju nie dojechal (przeszkoda), nie dostaje po nim docisku."""
    arm = FakeArm()
    connect_fake(twin, monkeypatch, arm)
    arm.blocked["elbow_flex"] = arm.pos["elbow_flex"] + 2.0
    twin.set_engaged(True)
    twin.set_target({"elbow_flex": 90.0})
    run(twin, 0.2)                                         # rozkaz ~20 st. za przeszkoda
    assert arm.sent[-1]["elbow_flex"] > arm.blocked["elbow_flex"] + 15
    n = len(arm.sent)
    stall(twin, arm, 2)
    run(twin, 0.1)
    assert arm.sent[n]["elbow_flex"] <= arm.blocked["elbow_flex"] + 120.0 * 1.5 * DT + 1e-6


# ------------------------------------------------------------ straznik rozjazdu
def test_a_limp_servo_without_an_error_bit_triggers_estop(twin, monkeypatch):
    """Serwo bez momentu i bez bitu bledu: rozkaz jedzie, ramie stoi - STOP z nazwa stawu.

    Przed poprawka `Twin.move` konczyl "normalnie" 63 st. od celu, a nastepne ruchy szly dalej.
    """
    arm = FakeArm()
    connect_fake(twin, monkeypatch, arm)
    arm.blocked["shoulder_lift"] = arm.pos["shoulder_lift"]
    twin.claim("kalibracja")
    twin.set_engaged(True)
    twin.set_target({"shoulder_lift": 40.0}, owner="kalibracja")
    run(twin, 1.5)
    assert twin.safety_state.value == "ESTOP"
    assert "shoulder_lift" in twin.status.error
    assert twin.owner is None
    assert arm.sent[-1]["shoulder_lift"] == pytest.approx(arm.blocked["shoulder_lift"], abs=0.5)


def test_twin_move_on_a_limp_servo_raises_instead_of_finishing(monkeypatch):
    arm = FakeArm()
    arm.blocked["shoulder_lift"] = arm.pos["shoulder_lift"]
    monkeypatch.setattr(rt, "create_backend", lambda cfg: arm)
    tw = Twin(Workspace())
    try:
        tw.connect("feetech", port="COM_TEST")
        with pytest.raises(RuntimeError, match="stop awaryjny"):
            tw.move({"shoulder_lift": 40.0}, duration=1.0, settle=0.5)
        assert "shoulder_lift" in tw.status.error
    finally:
        tw.close()


def test_normal_motion_does_not_trip_the_tracking_guard(twin, monkeypatch):
    """Serwo 250 st./s (szybsze niz max_vel nadzoru), odczyt co 40 ms: rampa do domu, pelne
    skoki suwakow, powrot stawu spoza limitow i chwytak sciskajacy kostke - bez STOP-u."""
    arm = FakeArm({"shoulder_pan": 90.0, "elbow_flex": -80.0, "shoulder_lift": -101.0}, vmax=250.0)
    arm.clamp_to_cfg = True
    connect_fake(twin, monkeypatch, arm, go_home=True)
    run(twin, 4.0)
    assert twin.safety_state.value == "IDLE"
    arm.blocked["gripper"] = 30.0
    twin.claim("panel")
    twin.set_engaged(True)
    for target in ({"shoulder_pan": 100.0, "wrist_roll": 150.0, "gripper": 0.0},
                   {"shoulder_pan": -100.0, "wrist_roll": -150.0, "elbow_flex": 90.0},
                   {"shoulder_lift": 90.0, "wrist_flex": -90.0}):
        twin.set_target(target, owner="panel")
        run(twin, 2.0)
        assert twin.safety_state.value == "ACTIVE", twin.status.error
    twin.home()
    run(twin, 3.0)
    assert twin.safety_state.value == "IDLE" and not twin.status.error


# ------------------------------------------------------------ chwytak: trzymanie i bledy
def squeeze(twin, monkeypatch, arm, owner="polityka"):
    """Chwytak zablokowany na kostce (30), rozkaz 5 - sciska."""
    connect_fake(twin, monkeypatch, arm)
    arm.blocked["gripper"] = 30.0
    twin.claim(owner)
    twin.set_engaged(True)
    twin.set_target({"gripper": 5.0}, owner=owner)
    run(twin, 1.0)
    assert arm.sent[-1]["gripper"] == pytest.approx(5.0)


@pytest.mark.parametrize("action", ["estop", "home", "preempt"])
def test_stop_home_and_preempt_keep_the_gripper_squeeze(twin, monkeypatch, action):
    """STOP/Dom/odebranie trzymaly zmierzony kat zablokowanej szczeki (30) = zerowa sila:
    kostka niesiona nad blatem wypadala. Stawy ramienia dalej trzymaja pomiar."""
    arm = FakeArm()
    squeeze(twin, monkeypatch, arm)
    n = len(arm.sent)
    getattr(twin, action)(*(("test",) if action == "preempt" else ()))
    run(twin, 3.5)
    assert max(s["gripper"] for s in arm.sent[n:]) == pytest.approx(5.0)
    assert twin.status.command["gripper"] == pytest.approx(5.0)


def test_gripper_overload_eases_the_squeeze_instead_of_estop(twin, monkeypatch):
    """Przeciazenie TYLKO chwytaka przy mocnym chwycie: bez STOP-u, szczeki luzniej, ostrzezenie."""
    arm = FakeArm()
    squeeze(twin, monkeypatch, arm)
    arm.fault_list = ["gripper (serwo 6): przeciazenie"]
    run(twin, 1.0)
    assert twin.safety_state.value == "ACTIVE" and twin.owner == "polityka"
    assert not twin.status.error
    ease = 30.0 - Twin.grip_ease
    assert arm.sent[-1]["gripper"] == pytest.approx(ease)  # wlasciciel dalej chce 5
    assert 30.0 - arm.sent[-1]["gripper"] > Twin.grip_squeeze_margin   # chwyt zostaje
    assert any("chwytak" in w and "przeciazenie" in w for w in twin.status.warnings)
    twin.set_target({"shoulder_pan": 10.0}, owner="polityka")   # reszta ramienia jedzie dalej
    run(twin, 0.5)
    assert arm.sent[-1]["shoulder_pan"] == pytest.approx(10.0)
    # Blad trwa mimo odciazenia - STOP, a chwytak staje na zmierzonym rozwarciu: to docisk
    # podtrzymywal przeciazenie (patrz test_gripper_fault_stop_releases_the_squeeze_...).
    run(twin, 1.0)
    assert twin.safety_state.value == "ESTOP" and "gripper" in twin.status.error
    run(twin, 0.5)
    assert arm.sent[-1]["gripper"] == pytest.approx(30.0)


def test_gripper_fault_that_clears_does_not_stop(twin, monkeypatch):
    arm = FakeArm()
    squeeze(twin, monkeypatch, arm)
    arm.fault_list = ["gripper (serwo 6): przeciazenie"]
    run(twin, 0.5)
    arm.fault_list = []
    run(twin, 3.0)
    assert twin.safety_state.value == "ACTIVE"
    assert arm.sent[-1]["gripper"] == pytest.approx(30.0 - Twin.grip_ease)   # odciazenie zostaje


def test_arm_fault_with_a_gripper_fault_stops_at_once(twin, monkeypatch):
    arm = FakeArm()
    squeeze(twin, monkeypatch, arm)
    arm.fault_list = ["gripper (serwo 6): przeciazenie", "elbow_flex (serwo 3): przegrzanie"]
    twin.step(DT)
    assert twin.safety_state.value == "ESTOP" and "elbow_flex" in twin.status.error
    run(twin, 0.3)
    assert arm.sent[-1]["gripper"] == pytest.approx(30.0 - Twin.grip_ease)


# ------------------------------------------------------------ STOP w szybkim ruchu
@pytest.mark.parametrize("extra", [0, 1])
def test_stop_during_fast_motion_barely_moves_backwards(twin, monkeypatch, extra):
    """STOP w trakcie szybkiego ruchu: trzymana poza nie lezy za ramieniem.

    Przed poprawka trzymany byl odczyt sprzed do 40 ms - ramie cofalo sie o 3,7-6,4 st.
    (serwo 250 st./s). Teraz odczyt przesuniety wzdluz predkosci, najwyzej do rozkazu.
    `extra`: STOP w takcie z odczytem albo takt po nim.
    """
    arm = FakeArm(vmax=250.0)
    connect_fake(twin, monkeypatch, arm)
    twin.set_engaged(True)
    twin.set_target({"shoulder_pan": 100.0})
    run(twin, 0.3 + extra * DT)
    at_stop = arm.pos["shoulder_pan"]
    twin.estop()
    run(twin, 0.5)
    assert at_stop - arm.pos["shoulder_pan"] < 1.5
    assert arm.pos["shoulder_pan"] <= twin.status.command["shoulder_pan"] + 1e-6


# ------------------------------------------------------------ polaczenie
def test_connect_ignores_a_single_corrupt_read(twin, monkeypatch):
    """Jedna przeklamana ramka (0 tikow = -180 st.) tuz po polaczeniu nie zostaje poza startowa."""
    arm = FakeArm()
    arm.glitch = [{"shoulder_pan": -180.0}]
    connect_fake(twin, monkeypatch, arm)
    assert twin.status.command["shoulder_pan"] == pytest.approx(SO101.home["shoulder_pan"])
    twin.set_engaged(True)
    run(twin, 0.2)
    assert all(abs(s["shoulder_pan"] - SO101.home["shoulder_pan"]) < 1.0 for s in arm.sent)


def test_connect_refuses_reads_that_never_agree(twin, monkeypatch):
    arm = FakeArm()
    arm.glitch = [{"shoulder_pan": v} for v in (-180.0, 0.0) * 5]
    with pytest.raises(RuntimeError, match="nie zgadzaja"):
        connect_fake(twin, monkeypatch, arm)
    assert not arm.is_connected and not twin.connected


@pytest.mark.parametrize("ticks,warn", [(None, False), ((1986.0, 2670.0, 2048.0), False),
                                        ((1990.0, 2660.0, 2048.0), False), ((1850.0, 2670.0, 2048.0), True)])
def test_gripper_tick_mismatch_is_a_warning(twin, monkeypatch, ticks, warn):
    """Backend mapuje chwytak 0..100 na inne tiki niz blizniak - kat szczek sie nie zgadza (C1/C3)."""
    arm = FakeArm()
    arm.grip_ticks = ticks
    connect_fake(twin, monkeypatch, arm)
    run(twin, 0.1)
    assert any("chwytak" in w for w in twin.status.warnings) == warn
    assert twin.safety_state.value != "ESTOP"


def test_backend_calibration_warnings_reach_the_status(twin, monkeypatch):
    """Backend `lerobot` liczy katy od srodka zakresu kalibracji - panel ma o tym wiedziec."""
    arm = FakeArm()
    arm.calibration_warnings = lambda: ["zero stawow w kalibracji LeRobota nie jest zerem blizniaka"]
    connect_fake(twin, monkeypatch, arm)
    run(twin, 0.1)
    assert any("zero stawow" in w for w in twin.status.warnings)
    arm2 = FakeArm()
    arm2.calibration_warnings = lambda: 1 / 0                  # diagnostyka nie blokuje polaczenia
    connect_fake(twin, monkeypatch, arm2)
    assert twin.status.connected


# ------------------------------------------------------------ wlasnosc: sprzeglo i odmowy
def test_set_engaged_with_an_owner_only_touches_its_own_clutch(twin, monkeypatch):
    connect_fake(twin, monkeypatch, FakeArm())
    twin.claim("panel")
    assert twin.set_engaged(True, owner="panel")
    assert not twin.set_engaged(False, owner="polityka")   # spozniony runner
    twin.step(DT)
    assert twin.status.engaged


def test_refusals_name_the_current_owner_not_an_old_preempt(twin, monkeypatch):
    """Po odebraniu ramienia panelowi (ponowne laczenie) i oddaniu go polityce spozniony suwak
    slyszal "ponowne laczenie" sprzed minut zamiast tego, ze ramie ma polityka."""
    connect_fake(twin, monkeypatch, FakeArm())
    twin.claim("panel")
    twin.preempt("ponowne laczenie")
    twin.claim("panel")
    twin.release("panel")
    twin.claim("polityka")
    assert twin.preempt_reason == ""
    with pytest.raises(RuntimeError, match="ramie ma: polityka"):
        twin.set_target({"shoulder_pan": 5.0}, owner="panel")
    twin.home()
    with pytest.raises(RuntimeError, match="pozycja domowa"):
        twin.set_target({"shoulder_pan": 5.0}, owner="polityka")


# ------------------------------------------------------------ scena po fizyce
def test_scene_step_leaves_body_poses_matching_the_joint_angles(twin):
    """K7 w zywej scenie: po `Scene.step` xpos/site_xpos pasuja do qpos (kostka, HUD, uchwyt TCP)."""
    import mujoco

    s = twin.scene
    s.command({"shoulder_pan": 60.0, "elbow_flex": -40.0})
    for _ in range(5):
        s.step(0.02)
    xpos, site = s.data.xpos.copy(), s.data.site_xpos.copy()
    mujoco.mj_kinematics(s.model, s.data)
    assert np.allclose(xpos, s.data.xpos, atol=1e-9)
    assert np.allclose(site, s.data.site_xpos, atol=1e-9)


# ------------------------------------------------------------ runda 3: spozniona odpowiedz po przestoju
def pan_trace(twin, arm, seconds):
    """Takty petli; (pozycje serwa pan po kazdym takcie, rozkazy pan wyslane w tym czasie)."""
    n = len(arm.sent)
    pos = []
    for _ in range(int(round(seconds / DT))):
        twin.step(DT)
        pos.append(arm.pos["shoulder_pan"])
    return pos, [s["shoulder_pan"] for s in arm.sent[n:]]


@pytest.mark.parametrize("case", ["stall", "stop", "link_lost"])
def test_a_late_reply_after_a_stall_does_not_become_the_reseed_pose(twin, monkeypatch, case):
    """Przestoj lacza w szybkim ruchu; pierwsza "swieza" odpowiedz po nim to spozniona ramka sprzed
    przestoju (TCP oddaje ja po powrocie, SYNC READ dopasowuje tylko po ID serwa).

    Zmierzone (emulator STS3215 za mostem, pan 140 st./s): reseed do niej cofal rozkaz o 6-7 st.
    i serwo zawracalo z ~90 st./s - takze po STOP-ie wcisnietym w przestoju i po utracie lacza.
    """
    arm = FakeArm(vmax=140.0)
    connect_fake(twin, monkeypatch, arm)
    twin.claim("panel")
    twin.set_engaged(True, owner="panel")
    twin.set_target({"shoulder_pan": 90.0}, owner="panel")
    run(twin, 0.2)
    late = dict(arm.pos)                                   # poza, o ktora pytal odczyt sprzed przestoju
    run(twin, 0.12)
    arm.silent, arm.late = 8, late
    if case == "link_lost":
        arm.fault_list = ["brak odpowiedzi serw od 200 ms"]
    for k in range(16):                                    # 8 odczytow bez odpowiedzi (co 40 ms)
        if case == "stop" and k == 6:
            twin.estop()
        twin.step(DT)
    arm.fault_list = []
    at_release = arm.pos["shoulder_pan"]
    assert at_release - late["shoulder_pan"] > 10           # serwo dojechalo do ostatniego rozkazu
    pos, sent = pan_trace(twin, arm, 0.6)
    assert sent, "po powrocie lacza nic nie poszlo"
    assert min(sent) >= at_release - 1.0
    assert min(pos) >= at_release - 1.0


def test_a_long_blocking_read_does_not_hold_a_stale_reply(twin, monkeypatch):
    """Jeden odczyt trzymal petle 0,8 s (bez flagi milczenia) i oddal spozniona odpowiedz sprzed
    przestoju, nastepny druga, identyczna. Stop na przerwie petli trzymal te poze (predkosc zero =
    "zablokowany") i serwo cofalo sie o 7 st. z 85 st./s (emulator, STOP w przestoju)."""
    arm = FakeArm(vmax=140.0)
    connect_fake(twin, monkeypatch, arm)
    twin.claim("panel")
    twin.set_engaged(True, owner="panel")
    twin.set_target({"shoulder_pan": 90.0}, owner="panel")
    run(twin, 0.2)
    late = arm.pos["shoulder_pan"]
    run(twin, 0.12)
    at = arm.pos["shoulder_pan"]
    arm.glitch = [{"shoulder_pan": late}] * 2              # dwie zalegle odpowiedzi sprzed przestoju
    twin.step(0.9)                                         # odczyt trzymal petle 0,9 s
    pos, sent = pan_trace(twin, arm, 0.6)
    assert twin.owner is None and "przerwa" in twin.status.error
    assert sent and min(sent) >= at - 1.0
    assert min(pos) >= at - 1.0


@pytest.mark.parametrize("case", ["stale_replies", "starting_to_move"])
def test_a_command_delivered_after_the_stall_is_not_undone(twin, monkeypatch, case):
    """Rozkaz wyslany tuz przed przestojem dochodzi do serwa PO nim (TCP). Dwa odczyty po przestoju
    zgadzaja sie ze soba (zalegle odpowiedzi sprzed jego dojscia albo serwo dopiero rusza), a reseed
    do nich cofal serwo. Zmierzone (emulator za mostem): +4,2 st. do spoznionego rozkazu, zaraz
    potem -5,5 st. z powrotem, do ~90 st./s."""
    arm = FakeArm(vmax=140.0, amax=2232.0 if case == "starting_to_move" else None)
    connect_fake(twin, monkeypatch, arm)
    twin.claim("panel")
    twin.set_engaged(True, owner="panel")
    twin.set_target({"shoulder_pan": 90.0}, owner="panel")
    run(twin, 0.3)
    lost = arm.sent[-1]["shoulder_pan"]
    prev = lost - (4.0 if case == "stale_replies" else 8.0)    # serwo tyle za ostatnim rozkazem
    arm.goal["shoulder_pan"] = arm.pos["shoulder_pan"] = prev   # ostatni rozkaz utknal w gniezdzie
    arm.vel["shoulder_pan"] = 0.0
    arm.silent = 8
    for _ in range(16):
        twin.step(DT)
    arm.goal["shoulder_pan"] = lost                             # ...i dochodzi po przestoju
    if case == "stale_replies":
        arm.glitch = [{"shoulder_pan": prev}] * 2               # zalegle odpowiedzi sprzed jego dojscia
    pos, sent = pan_trace(twin, arm, 0.6)
    assert min(sent) >= lost - 0.5
    assert all(b >= a - 0.5 for a, b in zip(pos, pos[1:]))      # serwo nie zawraca


def test_reads_that_never_agree_after_a_stall_stop_the_arm(twin, monkeypatch):
    """Po przestoju odczyty co takt inne (np. przeklamane ramki) - petla dalej stoi, a po
    `confirm_timeout_s` STOP z powodem zamiast cichego "ACTIVE", w ktorym nic sie nie rusza."""
    arm = FakeArm()
    connect_fake(twin, monkeypatch, arm)
    twin.claim("panel")
    twin.set_engaged(True, owner="panel")
    run(twin, 0.1)
    n = len(arm.sent)
    arm.silent = 2
    arm.glitch = [{"shoulder_pan": v} for v in (10.0, -10.0) * 30]
    run(twin, 1.5)
    assert len(arm.sent) == n
    assert twin.safety_state.value == "ESTOP" and "nie zgadzaja" in twin.status.error


# ------------------------------------------------------------ runda 3: STOP od bledu chwytaka
def test_gripper_fault_stop_releases_the_squeeze_and_can_be_cleared_to_open(twin, monkeypatch):
    """Przeciazenie chwytaka nie mija mimo odciazenia (albo serwo trzyma bit) - STOP po 2 s.

    Zmierzone przed poprawka (emulator): STOP trzymal szczeke przy progu odciazenia (docisk
    dalej karmil przeciazenie), a skasowanie STOP-u wracalo do STOP-u w tym samym takcie
    (licznik bledu nie ruszal od nowa) - szczek nie dalo sie otworzyc z panelu.
    """
    arm = FakeArm()
    squeeze(twin, monkeypatch, arm, owner="panel")
    arm.fault_list = ["gripper (serwo 6): przeciazenie"]
    run(twin, 2.5)
    assert twin.safety_state.value == "ESTOP" and "gripper" in twin.status.error
    run(twin, 0.3)
    assert arm.sent[-1]["gripper"] == pytest.approx(30.0)   # bez docisku: zmierzone rozwarcie
    twin.clear_estop()
    twin.claim("panel")
    twin.set_engaged(True, owner="panel")
    twin.set_target({"gripper": 100.0}, owner="panel")     # otwarcie przy wciaz zglaszanym bledzie
    arm.blocked.pop("gripper")
    run(twin, 1.0)
    assert twin.safety_state.value == "ACTIVE", twin.status.error
    assert arm.pos["gripper"] == pytest.approx(100.0)


def test_gripper_easing_ends_once_the_jaws_let_go(twin, monkeypatch):
    """Odciazenie po chwilowym przeciazeniu zostawalo na cala sesje: pusty chwytak nie zamykal
    sie ponizej starego progu (22 zamiast 0), a ostrzezenie wisialo w panelu."""
    arm = FakeArm()
    squeeze(twin, monkeypatch, arm)
    arm.fault_list = ["gripper (serwo 6): przeciazenie"]
    run(twin, 0.5)
    arm.fault_list = []
    run(twin, 1.5)
    assert arm.sent[-1]["gripper"] == pytest.approx(30.0 - Twin.grip_ease)   # wciaz trzyma - odciazenie zostaje
    twin.set_target({"gripper": 100.0}, owner="polityka")
    arm.blocked.pop("gripper")
    run(twin, 1.0)
    twin.set_target({"gripper": 0.0}, owner="polityka")
    run(twin, 1.0)
    assert arm.pos["gripper"] == pytest.approx(0.0)
    assert not any("docisk" in w for w in twin.status.warnings)


@pytest.mark.parametrize("action", ["estop", "home"])
def test_stop_or_home_while_the_gripper_closes_does_not_keep_closing(twin, monkeypatch, action):
    """STOP w trakcie zamykania szczek (palec miedzy nimi): regula "sciska" brala zamykajaca sie
    szczeke za sciskajaca i trzymala pelny rozkaz zamkniecia - szczeki zamykaly sie dalej
    o 15-27 jednostek. Dom zostawial chwytak w przypadkowym rozwarciu zamiast w domu."""
    arm = FakeArm(vmax=300.0, delay=0.03, vmax_of={"gripper": 150.0})
    connect_fake(twin, monkeypatch, arm)
    twin.claim("panel")
    twin.set_engaged(True, owner="panel")
    twin.set_target({"gripper": 100.0}, owner="panel")
    run(twin, 1.0)
    twin.set_target({"gripper": 0.0}, owner="panel")
    run(twin, 0.12)
    at = arm.pos["gripper"]
    getattr(twin, action)()
    low = at
    for _ in range(int(3.5 / DT)):
        twin.step(DT)
        low = min(low, arm.pos["gripper"])
    if action == "estop":
        assert at - low < 3.0                              # najwyzej droga hamowania
        assert arm.pos["gripper"] >= at - 1.0
    else:
        assert arm.pos["gripper"] == pytest.approx(twin._supervisor.home["gripper"], abs=0.5)


# ------------------------------------------------------------ runda 3: straznik rozjazdu
SWEEPS = (("shoulder_pan", -100.0, 100.0), ("wrist_roll", -145.0, 145.0), ("wrist_flex", -90.0, 90.0),
          ("elbow_flex", -85.0, 85.0), ("shoulder_lift", -60.0, 60.0))


@pytest.mark.parametrize("vmax,delay", [(120.0, 0.04), (150.0, 0.07), (120.0, 0.07)])
def test_tracking_guard_tolerates_a_slower_servo_and_stale_reads(twin, monkeypatch, vmax, delay):
    """Suwak od konca do konca: nadzor jedzie max_vel (wrist_roll 220 st./s), serwo wolniej
    (120-150 st./s, shoulder_lift pod obciazeniem 60), odczyt ma 40-70 ms. Zmierzone przed
    poprawka (emulator za mostem): STOP "serwo nie nadaza" przy serwie <= 185 st./s."""
    arm = FakeArm(vmax=vmax, delay=delay, amax=2232.0, vmax_of={"shoulder_lift": 60.0})
    connect_fake(twin, monkeypatch, arm)
    twin.claim("panel")
    twin.set_engaged(True, owner="panel")
    for joint, lo, hi in SWEEPS:
        for tgt in (lo, hi):
            twin.set_target({joint: tgt}, owner="panel")
            run(twin, 3.0 if joint == "shoulder_lift" else 2.6)
            assert twin.safety_state.value == "ACTIVE", twin.status.error
            assert arm.pos[joint] == pytest.approx(tgt, abs=1.0)


@pytest.mark.parametrize("case", ["collision", "limp"])
def test_tracking_guard_still_stops_a_blocked_joint_with_stale_reads(twin, monkeypatch, case):
    """Ten sam wolny serwo i stary odczyt: staw zatrzymany przeszkoda w pelnym biegu albo bez
    momentu od poczatku - STOP w ciagu ~1 s, ramie trzyma zmierzona poze (bez docisku)."""
    arm = FakeArm(vmax=150.0, delay=0.07, amax=2232.0)
    connect_fake(twin, monkeypatch, arm)
    twin.claim("panel")
    twin.set_engaged(True, owner="panel")
    if case == "limp":
        arm.blocked["shoulder_pan"] = arm.pos["shoulder_pan"]
    twin.set_target({"shoulder_pan": 100.0}, owner="panel")
    if case == "collision":
        run(twin, 0.3)
        arm.blocked["shoulder_pan"] = arm.pos["shoulder_pan"]
    t = 0.0
    while twin.safety_state.value != "ESTOP" and t < 3.0:
        twin.step(DT)
        t += DT
    assert twin.safety_state.value == "ESTOP" and "shoulder_pan" in twin.status.error
    assert t <= 1.0
    run(twin, 0.3)
    assert arm.sent[-1]["shoulder_pan"] == pytest.approx(arm.blocked["shoulder_pan"], abs=1.0)


# ------------------------------------------------------------ runda 3: trzymanie pozy przy STOP/Dom
@pytest.mark.parametrize("extra", [0.0, 0.06, 0.12, 0.18])
def test_stop_at_speed_does_not_snap_back_after_the_stopping_distance(twin, monkeypatch, extra):
    """STOP w pelnym biegu wrist_roll (220 st./s): serwo hamuje najwyzej ~2230 st./s^2, czyli
    ~11 st. - trzymanie punktu przed ta droga hamowania cofalo je o 6-14 st. (emulator)."""
    arm = FakeArm(vmax=250.0, amax=2232.0)
    connect_fake(twin, monkeypatch, arm)
    twin.set_engaged(True)
    twin.set_target({"wrist_roll": -140.0})
    run(twin, 2.0)
    twin.set_target({"wrist_roll": 140.0})
    run(twin, 0.4 + extra)
    twin.estop()
    trace = []
    for _ in range(40):
        twin.step(DT)
        trace.append(arm.pos["wrist_roll"])
    # Zostaje tylko przeregulowanie wlasnego celu przez serwo (P hamuje od v/kp = 8,8 st., a droga
    # hamowania to 10,8 st.) - ~2,6 st., tyle samo na koncu kazdego szybkiego ruchu suwakiem.
    assert max(trace) - trace[-1] < 3.0
    assert trace[-1] <= twin.status.command["wrist_roll"] + 0.5


@pytest.mark.parametrize("vmax", [140.0, 90.0])
def test_home_mid_motion_does_not_snap_the_servo_back(twin, monkeypatch, vmax):
    """Dom w szybkim ruchu: trzymanie zmierzonej pozy (+ predkosc x wiek odczytu) cofalo rozkaz
    o 4-11 st. w jednym takcie, a serwo w biegu hamowalo za mocno i zawracalo o 2,3-3,6 st.
    Teraz rampa do domu startuje tam, gdzie serwo sie zatrzyma, i zawraca plynnie (w 0,3 s
    ~0,5 st. samej rampy). Serwo 90 st./s wolniejsze od nadzoru (120) - rozkaz odjechal mu
    daleko, wiec krok rozkazu bywa duzy, ale serwo i tak nie zawraca."""
    arm = FakeArm(vmax=vmax, amax=2232.0)
    connect_fake(twin, monkeypatch, arm)
    twin.claim("polityka")
    twin.set_engaged(True, owner="polityka")
    twin.set_target({"elbow_flex": 85.0}, owner="polityka")
    run(twin, 0.25)
    twin.home()
    trace = []
    for _ in range(15):
        twin.step(DT)
        trace.append(arm.pos["elbow_flex"])
    assert max(trace) - trace[-1] < 1.2
    run(twin, 3.5)
    assert twin.safety_state.value == "IDLE"
    assert arm.pos["elbow_flex"] == pytest.approx(SO101.home["elbow_flex"], abs=1.0)


# ------------------------------------------------------------ runda 3: move po przestoju
def test_move_does_not_finish_while_the_arm_catches_up_after_a_stall(monkeypatch):
    """Przestoj lacza 1,1 s w trakcie `move`: zegar rampy szedl dalej, a `move` wracal "normalnie"
    z ramieniem 17 st. od celu jadacym 130 st./s (fala brala wtedy kadry i FK)."""
    arm = FakeArm(vmax=250.0)
    monkeypatch.setattr(rt, "create_backend", lambda cfg: arm)
    tw = Twin(Workspace())
    try:
        tw.connect("feetech", port="COM_TEST")

        def stall():
            time.sleep(0.5)
            arm.silent = 28                                # ~1,1 s bez odpowiedzi (odczyt co 40 ms)
        threading.Thread(target=stall, daemon=True).start()
        tw.move({"shoulder_pan": 80.0}, duration=1.5, settle=0.4)
        assert arm.pos["shoulder_pan"] == pytest.approx(80.0, abs=2.0)
    finally:
        tw.close()


def test_move_raises_when_the_arm_stops_short_of_the_command(monkeypatch):
    """Staw oparty 15 st. przed celem (ponizej progu straznika 25 st.) - `move` nie konczy sie
    "normalnie", tylko mowi, ktory staw nie dojechal."""
    arm = FakeArm()
    arm.wall["shoulder_pan"] = (-1e9, 65.0)
    monkeypatch.setattr(rt, "create_backend", lambda cfg: arm)
    tw = Twin(Workspace())
    try:
        tw.connect("feetech", port="COM_TEST")
        with pytest.raises(RuntimeError, match="shoulder_pan"):
            tw.move({"shoulder_pan": 80.0}, duration=0.5, settle=0.2)
    finally:
        tw.close()


# ------------------------------------------------------------ runda 3: konfiguracja backendu (D3)
def test_twin_backend_has_no_relative_target_cap(twin, monkeypatch):
    """LeRobot przycinal cel do +-12 st. od pozycji: rozjazd nigdy nie przekraczal ~12 st., straznik
    (25 st.) nie mogl zadzialac, a serwo dociskalo do przeszkody bez konca. Blizniak ma wlasny
    ogranicznik predkosci; aplikacja dloni zostaje przy swojej konfiguracji."""
    from lerobot_mp.config import load_config

    arm = FakeArm()
    connect_fake(twin, monkeypatch, arm)
    assert arm.cfg.robot.max_relative_target is None
    assert load_config().robot.max_relative_target == pytest.approx(12.0)


# ------------------------------------------------------------ runda 3: sprzeglo z wlascicielem
def test_sysid_and_job_cleanup_pass_their_owner_to_the_clutch(twin, monkeypatch):
    """Sprzatanie po identyfikacji wylacza sprzeglo tylko z wlascicielem (sprawdzenie i zmiana pod
    jedna blokada) - `if owner == X: set_engaged(False)` wylaczalo sprzeglo panelu, ktory wzial
    ramie miedzy sprawdzeniem a wylaczeniem."""
    from lerobot_mp.twin.rl import sysid
    from lerobot_mp.twin.ui import jobs

    connect_fake(twin, monkeypatch, FakeArm())
    calls = []
    real = twin.set_engaged

    def spy(engaged, owner=None):
        calls.append((engaged, owner))
        return real(engaged, owner=owner)
    monkeypatch.setattr(twin, "set_engaged", spy)

    def boom(*a, **k):
        raise RuntimeError("przerwane w tescie")
    monkeypatch.setattr(twin, "move", boom)
    with pytest.raises(RuntimeError, match="w tescie"):
        sysid.record(twin, sysid.excitation(dict(SO101.home)))
    assert calls and all(owner == sysid.OWNER for _, owner in calls)
    assert twin.owner is None

    calls.clear()
    twin.claim(jobs.SYSID_OWNER)
    monkeypatch.setattr(sysid, "record", boom)
    with pytest.raises(RuntimeError, match="w tescie"):
        jobs.run_sysid(jobs.Job("identyfikacja"), twin)
    assert calls and all(owner == jobs.SYSID_OWNER for _, owner in calls)
    assert twin.owner is None


# ------------------------------------------------------------ runda 3b: sluchacze taktow (E1)
def test_tick_listener_sees_what_was_really_sent_and_a_failing_listener_is_dropped(twin, monkeypatch):
    """Identyfikacja stemplowala nagranie wlasnym zegarem (verify2: tlumienie do 62% obok) - petla
    podaje teraz swoj takt, rozkaz, ktory NAPRAWDE poszedl, i czy odczyt byl swiezy."""
    arm = FakeArm()
    arm.clamp["shoulder_pan"] = (-1e9, 3.0)               # serwo przycina rozkaz: w probce to, co poszlo
    connect_fake(twin, monkeypatch, arm)
    got, calls = [], []

    def bad(sample):
        calls.append(sample)
        raise ValueError("zly sluchacz")
    h = twin.add_tick_listener(got.append)
    twin.add_tick_listener(bad)
    twin.set_engaged(True)
    twin.set_target({"shoulder_pan": 10.0})
    run(twin, 0.5)
    assert len(calls) == 1                                  # usuniety po pierwszym wyjatku, petla dziala
    assert twin.connected and len(got) == 25
    ts = [s.t for s in got]
    assert ts == sorted(ts) and ts[-1] == pytest.approx(0.5)
    sent = [s for s in got if s.sent]
    assert sent and sent[-1].sent == arm.sent[-1] and sent[-1].sent["shoulder_pan"] == pytest.approx(3.0)
    assert any(s.fresh and s.measured_t == pytest.approx(s.t) for s in got)
    assert all(s.measured_t <= s.t + 1e-9 for s in got)
    arm.silent = 3                                          # lacze milczy: nic nie poszlo, odczyt nieswiezy
    n = len(got)
    run(twin, 0.2)
    # Prawdziwe serwa czytane co drugi takt (40 ms): takt bez odczytu tez nie jest swiezy, ale wysyla.
    stale = [s for s in got[n:] if not s.fresh]
    assert stale and all(s.measured_t < s.t for s in stale)
    assert any(s.sent == {} for s in got[n:])               # milczenie lacza: nic nie poszlo
    assert len(arm.sent) == len([s for s in got if s.sent])
    twin.remove_tick_listener(h)
    n = len(got)
    run(twin, 0.1)
    assert len(got) == n
    twin.remove_tick_listener(h)                            # drugi raz - bez bledu


# ------------------------------------------------------------ runda 3b: STOP i Dom w trakcie startu
def test_claim_and_clutch_are_refused_under_estop(twin, monkeypatch):
    """Polityka ladowana 0,1 s po Uruchom brala ramie POD STOP-em (verify2, s15): `claim` i sprzeglo
    tego nie sprawdzaly, a po Skasuj STOP ramie jechalo 83,7 st. pod zatrzymana polityka."""
    arm = FakeArm()
    connect_fake(twin, monkeypatch, arm)
    twin.estop("STOP z panelu")
    with pytest.raises(RuntimeError, match="aktywny STOP"):
        twin.claim("polityka")
    assert twin.owner is None
    assert not twin.set_engaged(True, owner="polityka")
    assert not twin.set_engaged(True)
    assert not twin._engaged
    with pytest.raises(RuntimeError, match="aktywny STOP"):
        twin.move({"shoulder_pan": 20.0}, duration=0.2)
    assert twin.owner is None and not twin._engaged
    twin.clear_estop()
    twin.claim("polityka")
    assert twin.set_engaged(True, owner="polityka")


def test_claim_after_a_preempt_since_the_mark_is_refused_and_home_keeps_ramping(twin, monkeypatch):
    """Dom wcisniety w trakcie startu polityki: jej `claim` przerywal rampe do domu (verify2, s15c)."""
    arm = FakeArm({"shoulder_pan": 50.0})
    connect_fake(twin, monkeypatch, arm)
    mark = twin.preempt_gen
    twin.home()
    with pytest.raises(RuntimeError, match="odebrane w trakcie startu.*pozycja domowa"):
        twin.claim("polityka", gen=mark)
    with pytest.raises(RuntimeError, match="w trakcie startu"):
        twin.claim("polityka", guard=lambda: False)
    assert twin.owner is None
    run(twin, 3.5)
    assert arm.pos["shoulder_pan"] == pytest.approx(SO101.home["shoulder_pan"], abs=0.5)
    twin.claim("polityka", gen=twin.preempt_gen, guard=lambda: True)
    assert twin.owner == "polityka"


# ------------------------------------------------------------ runda 3b: Polacz z kostka w szczekach
class GoalArm(FakeArm):
    """Backend, ktory zna cel serw (jak Goal_Position) - `Twin._squeeze_at_connect`."""

    def goal_positions(self):
        return dict(self.goal)


@pytest.mark.parametrize("go_home", [False, True])
def test_reconnect_keeps_a_squeezing_gripper_closed_when_the_backend_knows_the_goal(twin, monkeypatch, go_home):
    """Polacz w trakcie lift: nadzor startowal od zmierzonego kata szczeki (30), a pierwsze sprzeglo
    wysylalo go jako cel - zerowa sila, kostka zsuwala sie 3,8 -> 2,3 cm (verify2, s1)."""
    arm = GoalArm()
    squeeze(twin, monkeypatch, arm)
    connect_fake(twin, monkeypatch, arm, go_home=go_home)
    assert twin.status.command["gripper"] == pytest.approx(5.0)
    n = len(arm.sent)
    run(twin, 3.5 if go_home else 0.0)                     # rampa startowa do domu nie otwiera szczek
    twin.claim("panel")
    twin.set_engaged(True, owner="panel")
    twin.set_target({"shoulder_pan": 5.0}, owner="panel")
    run(twin, 1.0)
    assert arm.sent[n:] and max(s["gripper"] for s in arm.sent[n:]) == pytest.approx(5.0)


def test_reconnect_without_goal_positions_starts_from_the_measured_jaw(twin, monkeypatch):
    """Backend bez `goal_positions` (feetech przed jego dodaniem): jak dotad - od pomiaru (TWIN.md)."""
    arm = FakeArm()
    squeeze(twin, monkeypatch, arm)
    connect_fake(twin, monkeypatch, arm)
    assert twin.status.command["gripper"] == pytest.approx(30.0)


def test_scene_backend_reports_the_goal_its_actuators_hold(twin):
    twin.connect("sim", threaded=False)
    twin.set_engaged(True)
    twin.set_target({"gripper": 5.0})
    run(twin, 0.1)
    goal = twin._backend.goal_positions()
    assert goal["gripper"] == pytest.approx(twin.status.command["gripper"], abs=1e-6)
