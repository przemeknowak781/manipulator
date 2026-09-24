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
    """Ramie "prawdziwe" (simulated=False): stawy ida za rozkazem, chyba ze sa zablokowane."""

    def __init__(self, start: dict[str, float] | None = None):
        self.info = RobotInfo(name="udawane", simulated=False)
        self.pos = dict(SO101.home, **(start or {}))
        self.blocked: dict[str, float] = {}
        self.sent: list[dict[str, float]] = []
        self.fault_list: list[str] = []
        self.limits: dict[str, tuple[float, float]] = {}
        self.clamp: dict[str, tuple[float, float]] = {}
        #: Jak `FeetechArm`: przycina rozkaz do limitow swojej konfiguracji (`cfg` z `create_backend`).
        self.clamp_to_cfg = False
        self.cfg = None
        self._on = False

    def connect(self):
        self._on = True

    def disconnect(self):
        self._on = False

    @property
    def is_connected(self):
        return self._on

    def read_joints(self):
        return dict(self.pos)

    def send_joints(self, targets):
        out = {}
        for k, v in targets.items():
            lo, hi = self.clamp.get(k, (-1e9, 1e9))
            if self.clamp_to_cfg:
                lo, hi = max(lo, self.cfg.joint(k).min), min(hi, self.cfg.joint(k).max)
            out[k] = min(max(v, lo), hi)
        self.sent.append(out)
        for k, v in out.items():
            self.pos[k] = self.blocked.get(k, v)
        return out

    def faults(self):
        return list(self.fault_list)

    def joint_limits(self):
        return dict(self.limits)


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
