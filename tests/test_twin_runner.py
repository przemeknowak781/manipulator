"""PolicyRunner na (udawanym) prawdziwym ramieniu: straznicy, ktorych trening nie ma.

Wszystko w czasie symulowanym: petla blizniaka bez watku (`connect(threaded=False)`
+ `Twin.step`), polityka bez watku (`start(threaded=False)` + `step_once`) - wynik
nie zalezy od obciazenia procesora.
"""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pytest

pytest.importorskip("mujoco")
torch = pytest.importorskip("torch")

from lerobot_mp.twin.rl import task as tk  # noqa: E402
from lerobot_mp.twin.rl.policy import Policy, PolicyMeta  # noqa: E402
from lerobot_mp.twin.rl.runner import PolicyRunner, project_goal  # noqa: E402
from lerobot_mp.twin.robots import SO101  # noqa: E402
from lerobot_mp.twin.runtime import Twin  # noqa: E402
from lerobot_mp.twin.workspace import Workspace  # noqa: E402

from test_twin_runtime import FakeArm, connect_fake  # noqa: E402


def constant_policy(task: tk.TaskConfig, action: dict[int, float] | None = None) -> Policy:
    """Polityka, ktora zawsze daje te sama akcje (zera + `action`)."""
    pol = Policy(PolicyMeta(task=asdict(task), hidden=[8], obs_dim=task.obs_dim, act_dim=6))
    with torch.no_grad():
        for p in pol.actor.parameters():
            p.zero_()
        for k, v in (action or {}).items():
            pol.actor[-1].bias[k] = v
    return pol


@pytest.fixture
def twin():
    tw = Twin(Workspace())
    try:
        yield tw
    finally:
        tw.close()


def drive(twin, runner, seconds: float, t0: float = 0.0) -> float:
    """Petla blizniaka 50 Hz i polityka 20 Hz na wspolnym zegarze symulowanym (krok 10 ms)."""
    n = int(round(seconds / 0.01))
    for k in range(1, n + 1):
        if k % 2 == 0:
            twin.step(0.02)
        if k % 5 == 0 and runner.status.running:
            runner.step_once(t0 + k * 0.01)
    return t0 + n * 0.01


# ------------------------------------------------------------ rozjazd = kolizja
def test_tracking_stop_holds_the_measured_pose_instead_of_pressing_on(twin, monkeypatch):
    """Staw zablokowany (kolizja): polityka staje, a ramie trzyma ZMIERZONA poze.

    Przed poprawka `_halt` robil tylko `set_engaged(False)`, a nadzor trzymal ostatni
    rozkaz - ~25 st. za przeszkoda, czyli serwo dociskalo do niej z pelnym momentem.
    """
    arm = FakeArm()
    connect_fake(twin, monkeypatch, arm)
    arm.blocked["shoulder_pan"] = 10.0
    runner = PolicyRunner(twin, constant_policy(tk.make_task("reach"), {0: 1.0}), episode_limit=False)
    runner.start(threaded=False)
    drive(twin, runner, 3.0)
    assert not runner.status.running
    assert "nie nadaza" in runner.status.stopped_because
    drive(twin, runner, 0.3)
    assert arm.sent[-1]["shoulder_pan"] == pytest.approx(10.0, abs=0.5)
    assert twin.status.command["shoulder_pan"] == pytest.approx(10.0, abs=0.5)
    assert twin.owner is None


# ------------------------------------------------------------ wlasnosc ramienia
def test_runner_does_not_start_while_another_owner_drives(twin, monkeypatch):
    connect_fake(twin, monkeypatch, FakeArm())
    twin.claim("kalibracja")
    runner = PolicyRunner(twin, constant_policy(tk.make_task("reach")))
    with pytest.raises(RuntimeError, match="ramie zajete: kalibracja"):
        runner.start(threaded=False)
    assert not runner.status.running and twin.owner == "kalibracja"


def test_home_stops_the_runner(twin, monkeypatch):
    arm = FakeArm()
    connect_fake(twin, monkeypatch, arm)
    runner = PolicyRunner(twin, constant_policy(tk.make_task("reach"), {0: 0.3}), episode_limit=False)
    runner.start(threaded=False)
    t = drive(twin, runner, 1.0)
    assert runner.status.running and twin.owner == "polityka"
    twin.home()
    assert not runner.status.running and "pozycja domowa" in runner.status.stopped_because
    drive(twin, runner, 4.0, t)
    assert arm.pos["shoulder_pan"] == pytest.approx(SO101.home["shoulder_pan"], abs=0.5)


# ------------------------------------------------------------ koniec po sukcesie (lift)
def lift_runner(twin, cube_z: float) -> PolicyRunner:
    task = tk.make_task("lift")
    cube = (np.array([0.2, 0.0, cube_z]), np.eye(3))
    return PolicyRunner(twin, constant_policy(task), cube_provider=lambda: cube)


def test_lift_ends_when_the_cube_is_held_above_the_table(twin, monkeypatch):
    """Po podniesieniu kostki polityka lift-v2 krecila ramieniem do limitow stawow przez reszte
    epizodu (8,5 s) - runner ma skonczyc po `end_on_success` taktach sukcesu z rzedu."""
    connect_fake(twin, monkeypatch, FakeArm())
    runner = lift_runner(twin, cube_z=0.015 + 0.08)
    assert runner.end_on_success == 10
    runner.start(threaded=False)
    drive(twin, runner, 2.0)
    assert runner.status.stopped_because == "zadanie wykonane"
    assert runner.status.step == 10
    assert twin.owner is None


def test_lift_does_not_end_while_the_cube_is_on_the_table(twin, monkeypatch):
    connect_fake(twin, monkeypatch, FakeArm())
    runner = lift_runner(twin, cube_z=0.015)
    runner.start(threaded=False)
    drive(twin, runner, 2.0)
    assert runner.status.running and not runner.status.success


# ------------------------------------------------------------ cel reach
@pytest.mark.parametrize("goal", [
    (0.25, 0.05, -0.05),        # pod blatem
    (-0.20, 0.00, 0.10),        # za podstawa
    (0.02, 0.01, 0.10),         # w podstawie
    (0.60, 0.20, 0.40),         # za daleko i za wysoko
])
def test_reach_goal_is_projected_into_the_training_region(twin, goal):
    task = tk.make_task("reach")
    runner = PolicyRunner(twin, constant_policy(task))
    runner.goal = goal
    g = runner.goal
    r = np.hypot(g[0], g[1])
    assert task.goal_radius[0] - 1e-9 <= r <= task.goal_radius[1] + 1e-9
    assert task.goal_height[0] - 1e-9 <= g[2] <= task.goal_height[1] + 1e-9
    assert g[0] > 0.05
    assert runner.status.goal_clamped


def test_reach_goal_inside_the_region_is_left_alone(twin):
    task = tk.make_task("reach")
    runner = PolicyRunner(twin, constant_policy(task))
    runner.goal = (0.22, -0.05, 0.12)
    assert np.allclose(runner.goal, (0.22, -0.05, 0.12)) and not runner.status.goal_clamped
    # Losowane w treningu cele leza w obszarze - rzutowanie ich nie rusza.
    for g in tk.sample_goals(runner.kin, task, np.random.default_rng(0), 200):
        assert np.allclose(project_goal(task, g), g, atol=1e-9)


# ------------------------------------------------------------ limity serw
def test_policy_target_stays_inside_the_servo_limits(twin, monkeypatch):
    """Ramie nr 2: wrist_flex w EEPROM do +88 st., w konfiguracji +95 - q_cmd nie wychodzi za serwo."""
    arm = FakeArm()
    arm.limits = {"wrist_flex": (-200.0, 88.0)}
    connect_fake(twin, monkeypatch, arm)
    runner = PolicyRunner(twin, constant_policy(tk.make_task("reach"), {3: 1.0}), episode_limit=False)
    runner.start(threaded=False)
    drive(twin, runner, 3.0)
    k = SO101.joints.index("wrist_flex")
    assert np.degrees(runner.q_cmd[k]) <= 88.0 + 1e-6
    assert max(s["wrist_flex"] for s in arm.sent) <= 88.0 + 1e-6
