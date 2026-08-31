"""Nadzor bezpieczenstwa - limity, watchdog, stop awaryjny."""

from __future__ import annotations

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
