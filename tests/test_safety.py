"""Nadzor bezpieczenstwa - limity, watchdog, stop awaryjny."""

from __future__ import annotations

import numpy as np
import pytest

from lerobot_mp.config import JOINT_NAMES, load_config
from lerobot_mp.control.safety import SafetyState, SafetySupervisor

DT = 1.0 / 30.0


@pytest.fixture
def supervisor(cfg):
    sup = SafetySupervisor(cfg)
    sup.start({name: 0.0 for name in JOINT_NAMES})
    return sup


def settle(sup, targets=None, seconds=6.0, hand=True, engaged=True):
    """Przewija symulacje o zadany czas i zwraca ostatni rozkaz."""
    command = sup.command
    for _ in range(int(seconds / DT)):
        command, _ = sup.step(targets, DT, hand_present=hand, engaged=engaged)
    return command


def test_step_before_start_is_an_error(cfg):
    with pytest.raises(RuntimeError):
        SafetySupervisor(cfg).step(None, DT, hand_present=False, engaged=False)


def test_starts_from_measured_position_not_from_home(cfg):
    """Pierwszy rozkaz nie moze byc skokiem - startujemy tam, gdzie stoi ramie."""
    sup = SafetySupervisor(cfg)
    sup.start({name: 40.0 for name in JOINT_NAMES})
    command, _ = sup.step(None, DT, hand_present=False, engaged=False)
    assert command["shoulder_pan"] == pytest.approx(40.0, abs=1.0)
    assert sup.state is SafetyState.STARTING


def test_reaches_home_after_startup_ramp(supervisor, cfg):
    command = settle(supervisor, None, seconds=4.0, hand=False, engaged=False)
    for name in JOINT_NAMES:
        assert command[name] == pytest.approx(supervisor.home[name], abs=1.0)
    assert supervisor.state is SafetyState.IDLE


def test_targets_are_clamped_to_joint_limits(supervisor, cfg):
    settle(supervisor, None, seconds=4.0, hand=False, engaged=False)
    command = settle(supervisor, {"shoulder_pan": 9000.0}, seconds=8.0)
    assert command["shoulder_pan"] == pytest.approx(cfg.joint("shoulder_pan").max)


def test_velocity_limit_is_enforced(supervisor, cfg):
    settle(supervisor, None, seconds=4.0, hand=False, engaged=False)
    before = supervisor.command["shoulder_pan"]
    command, report = supervisor.step(
        {"shoulder_pan": 100.0}, DT, hand_present=True, engaged=True
    )
    step = abs(command["shoulder_pan"] - before)
    assert step <= cfg.joint("shoulder_pan").max_vel * DT + 1e-9
    assert "shoulder_pan" in report.rate_limited


def test_velocity_scale_slows_motion():
    """Ten sam cel, dwa mnozniki predkosci - wolniejszy ma dojechac dalej w tyle."""

    def travelled(scale: float) -> float:
        cfg = load_config(overrides={"safety": {"velocity_scale": scale, "startup_ramp_s": 0.01}})
        sup = SafetySupervisor(cfg)
        sup.start({name: 0.0 for name in JOINT_NAMES})
        settle(sup, None, seconds=0.2, hand=False, engaged=False)
        start = sup.command["shoulder_pan"]
        command = settle(sup, {"shoulder_pan": 100.0}, seconds=0.3)
        return abs(command["shoulder_pan"] - start)

    assert travelled(0.25) < travelled(1.0) / 2.0


def test_holding_means_a_lost_hand_not_a_fresh_start(cfg):
    """Po starcie, zanim pojawi sie dlon, stan ma byc spokojny (IDLE)."""
    sup = SafetySupervisor(cfg)
    sup.start({name: 0.0 for name in JOINT_NAMES})
    settle(sup, None, seconds=4.0, hand=False, engaged=False)
    assert sup.state is SafetyState.IDLE

    settle(sup, {"shoulder_pan": 10.0}, seconds=0.5)          # sterowalismy...
    sup.step(None, DT, hand_present=False, engaged=False)      # ...i zgubilismy dlon
    assert sup.state is SafetyState.HOLDING


def test_losing_the_hand_freezes_motion(supervisor):
    settle(supervisor, None, seconds=4.0, hand=False, engaged=False)
    settle(supervisor, {"shoulder_pan": 30.0}, seconds=2.0)
    frozen = supervisor.command["shoulder_pan"]

    command, report = supervisor.step(
        {"shoulder_pan": 90.0}, DT, hand_present=False, engaged=False
    )
    assert command["shoulder_pan"] == pytest.approx(frozen)
    assert supervisor.state is SafetyState.HOLDING
    assert report.seconds_without_hand > 0


def test_long_absence_returns_home(supervisor, cfg):
    settle(supervisor, None, seconds=4.0, hand=False, engaged=False)
    settle(supervisor, {"shoulder_pan": 60.0}, seconds=3.0)
    assert supervisor.command["shoulder_pan"] > 10.0

    settle(supervisor, None, seconds=cfg.safety.return_home_timeout_s + 4.0, hand=False, engaged=False)
    assert supervisor.command["shoulder_pan"] == pytest.approx(
        supervisor.home["shoulder_pan"], abs=1.0
    )


def test_estop_freezes_and_ignores_targets(supervisor):
    settle(supervisor, None, seconds=4.0, hand=False, engaged=False)
    supervisor.trigger_estop()
    frozen = supervisor.command

    command, report = supervisor.step({"shoulder_pan": 90.0}, DT, hand_present=True, engaged=True)
    assert command == pytest.approx(frozen)
    assert report.state is SafetyState.ESTOP
    assert supervisor.estopped


def test_estop_can_be_cleared(supervisor):
    supervisor.trigger_estop()
    supervisor.clear_estop()
    assert not supervisor.estopped
    command = settle(supervisor, {"shoulder_pan": 30.0}, seconds=3.0)
    assert command["shoulder_pan"] > 5.0


def test_home_values_are_clamped_to_limits(cfg):
    cfg.safety.home["shoulder_pan"] = 9999.0
    sup = SafetySupervisor(cfg)
    assert sup.home["shoulder_pan"] == cfg.joint("shoulder_pan").max


def test_at_limit_is_reported(supervisor, cfg):
    settle(supervisor, None, seconds=4.0, hand=False, engaged=False)
    _, report = supervisor.step({"shoulder_pan": 9000.0}, DT, hand_present=True, engaged=True)
    assert "shoulder_pan" in report.at_limit


def test_disengaged_holds_position(supervisor):
    settle(supervisor, None, seconds=4.0, hand=False, engaged=False)
    held = supervisor.command["shoulder_pan"]
    command = settle(supervisor, {"shoulder_pan": 80.0}, seconds=1.0, hand=True, engaged=False)
    assert command["shoulder_pan"] == pytest.approx(held)


# ------------------------------------------------------------ tryb blizniaka (dodatki)
def test_hand_app_start_still_clamps_a_joint_outside_the_limits(cfg):
    """Bez `keep_outside` (aplikacja dloni) zachowanie jak dotad: start przyciety do limitu."""
    sup = SafetySupervisor(cfg)
    lo = cfg.joint("shoulder_lift").min
    sup.start({"shoulder_lift": lo - 6.0}, go_home=False)
    assert sup.command["shoulder_lift"] == pytest.approx(lo)
    assert sup.outside == {}


def test_twin_start_keeps_a_joint_outside_and_brings_it_back_slowly(cfg):
    """Ramie nr 2 spoczywa 6 st. za limitem: rozkaz startuje DOKLADNIE tam i wraca w zakres
    z `outside_vel`, dopiero gdy sterowanie jest wlaczone - nie skokiem do limitu."""
    sup = SafetySupervisor(cfg)
    lo = cfg.joint("shoulder_lift").min
    sup.start({"shoulder_lift": lo - 6.0}, go_home=False, keep_outside=True)
    command = settle(sup, None, seconds=1.0, hand=True, engaged=False)
    assert command["shoulder_lift"] == pytest.approx(lo - 6.0)
    prev = command["shoulder_lift"]
    for _ in range(int(0.3 / DT)):
        command, _ = sup.step({"shoulder_pan": 0.0}, DT, hand_present=True, engaged=True)
        assert abs(command["shoulder_lift"] - prev) <= sup.outside_vel * DT + 1e-6
        prev = command["shoulder_lift"]
    command = settle(sup, {"shoulder_pan": 0.0}, seconds=1.0)
    assert command["shoulder_lift"] == pytest.approx(lo)
    assert sup.outside == {}
    command = settle(sup, {"shoulder_lift": lo - 20.0}, seconds=1.0)     # z powrotem na zewnatrz - nie
    assert command["shoulder_lift"] == pytest.approx(lo)


def test_hold_stops_a_ramp_at_the_measured_pose(supervisor):
    """Po kolizji rozkaz lezy za przeszkoda - `hold` zaczyna od pozy zmierzonej, bez skoku."""
    settle(supervisor, {"shoulder_pan": 30.0}, seconds=2.0)
    supervisor.begin_homing()
    supervisor.hold({"shoulder_pan": 12.0})
    assert supervisor.state is SafetyState.IDLE
    command = settle(supervisor, None, seconds=1.0, engaged=False)
    assert command["shoulder_pan"] == pytest.approx(12.0)
    supervisor.trigger_estop()
    supervisor.hold({"shoulder_pan": 11.0})
    assert supervisor.state is SafetyState.ESTOP and supervisor.command["shoulder_pan"] == pytest.approx(11.0)


@pytest.mark.parametrize("start", [160.0, 157.0])
def test_homing_a_joint_outside_the_limits_has_no_speed_jump(cfg, start):
    """Dom z wrist_roll 160 (limit 150): staw pelzal 15 st./s, a po powrocie w zakres ogranicznik
    od razu puszczal pelne 220 st./s - rampa stawu zaczyna sie od nowa tam, gdzie wrocil."""
    sup = SafetySupervisor(cfg)
    sup.start({"wrist_roll": start}, go_home=False, keep_outside=True)
    sup.begin_homing()
    dt = 0.02
    cmds = []
    for _ in range(int(4.0 / dt)):
        command, _ = sup.step(None, dt, hand_present=True, engaged=False)
        cmds.append(command["wrist_roll"])
    vel = np.abs(np.diff(cmds)) / dt
    hi = cfg.joint("wrist_roll").max
    # Smoothstep od limitu do domu: szczyt 1,5 * droga / czas rampy.
    assert vel.max() <= 1.5 * (hi - sup.home["wrist_roll"]) / cfg.safety.startup_ramp_s + 1.0
    assert cmds[-1] == pytest.approx(sup.home["wrist_roll"], abs=0.5)
    assert sup.state is SafetyState.IDLE


def test_begin_homing_can_keep_a_joint_where_it_is(supervisor, cfg):
    """Blizniak: chwytak sciskajacy kostke nie otwiera sie w rampie do domu."""
    settle(supervisor, {"gripper": 5.0, "shoulder_pan": 30.0}, seconds=2.0)
    supervisor.begin_homing(keep={"gripper": 5.0})
    command = settle(supervisor, None, seconds=4.0, hand=False, engaged=False)
    assert command["gripper"] == pytest.approx(5.0)
    assert command["shoulder_pan"] == pytest.approx(supervisor.home["shoulder_pan"], abs=0.5)
    supervisor.begin_homing()                             # bez `keep` - jak dotad, wszystko do domu
    command = settle(supervisor, None, seconds=4.0, hand=False, engaged=False)
    assert command["gripper"] == pytest.approx(supervisor.home["gripper"], abs=0.5)


@pytest.mark.parametrize("mode", ["homing", "starting"])
def test_homing_is_not_done_while_a_joint_is_still_outside(cfg, mode):
    """wrist_roll 195 st. (45 st. za limitem 150): powrot 15 st./s trwa 3 s, rampa 2,5 s.
    Nadzor konczyl Dom (IDLE) z rozkazem 158,1 - staw dalej poza zakresem, a rozkaz
    zamarzal tam, bo poza ACTIVE limity poszerzone zostaja."""
    sup = SafetySupervisor(cfg)
    sup.start({"wrist_roll": 195.0}, go_home=mode == "starting", keep_outside=True)
    if mode == "homing":
        sup.begin_homing()
    dt = 0.02
    hi = cfg.joint("wrist_roll").max
    cmds = []
    for _ in range(int(9.0 / dt)):
        command, _ = sup.step(None, dt, hand_present=True, engaged=False)
        cmds.append(command["wrist_roll"])
        # Koniec rampy tylko ze stawem w zakresie.
        assert not sup.is_homing_done or command["wrist_roll"] <= hi + 1e-6
    assert sup.is_homing_done and sup.state is SafetyState.IDLE
    assert cmds[-1] == pytest.approx(sup.home["wrist_roll"], abs=0.5)
    # Po powrocie w zakres rampa stawu od nowa - bez skoku do pelnego max_vel.
    vel = np.abs(np.diff(cmds)) / dt
    assert vel.max() <= 1.5 * (hi - sup.home["wrist_roll"]) / cfg.safety.startup_ramp_s + 1.0


def test_hand_app_homing_is_unchanged_by_the_outside_rule(supervisor, cfg):
    """Aplikacja dloni (bez keep_outside) - Dom konczy sie po rampie jak dotad."""
    settle(supervisor, {"wrist_roll": 140.0}, seconds=3.0)
    supervisor.begin_homing()
    n = 0
    while not supervisor.is_homing_done:
        supervisor.step(None, DT, hand_present=True, engaged=False)
        n += 1
    assert n * DT == pytest.approx(cfg.safety.startup_ramp_s, abs=DT + 1e-9)
