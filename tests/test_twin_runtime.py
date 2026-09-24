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
    """

    def __init__(self, start: dict[str, float] | None = None, vmax: float | None = None):
        self.info = RobotInfo(name="udawane", simulated=False)
        self.pos = dict(SO101.home, **(start or {}))
        self.goal = dict(self.pos)
        self.vmax = vmax
        self.blocked: dict[str, float] = {}
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
                self.pos[k] = self.blocked.get(k, v)
        return out

    def step(self, dt):
        if self.vmax is None:
            return
        for k, g in self.goal.items():
            p = self.pos[k] + float(np.clip(g - self.pos[k], -self.vmax * dt, self.vmax * dt))
            self.pos[k] = self.blocked.get(k, p)

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
    # Blad trwa mimo odciazenia - STOP, a chwytak dalej trzyma (luzniej), nie puszcza.
    run(twin, 1.0)
    assert twin.safety_state.value == "ESTOP" and "gripper" in twin.status.error
    run(twin, 0.5)
    assert arm.sent[-1]["gripper"] == pytest.approx(ease)


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
