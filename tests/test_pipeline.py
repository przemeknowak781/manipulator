"""Test calego lancucha: cechy dloni -> mapowanie -> nadzor -> robot.

Nie uruchamiamy kamery ani MediaPipe - podajemy zaplanowana trajektorie dloni
i sprawdzamy, ze ramie robi to, czego operator by oczekiwal, i ze zadne
ogniwo lancucha nie wypuszcza rozkazu poza limity.
"""

from __future__ import annotations

import pytest

from lerobot_mp.config import JOINT_NAMES, load_config
from lerobot_mp.control.mapping import HandToJointMapper
from lerobot_mp.control.safety import SafetyState, SafetySupervisor
from lerobot_mp.robot.sim import SimulatedArm
from lerobot_mp.vision.features import HandFeatures, extract_features

from conftest import make_hand

DT = 1.0 / 30.0
FRAME = (1280, 720)


class Rig:
    """Pelny lancuch sterowania spiety z symulatorem."""

    def __init__(self, **overrides):
        base = {"clutch": {"mode": "always", "engaged_on_start": True},
                "safety": {"startup_ramp_s": 0.2}}
        base.update(overrides)
        self.cfg = load_config(overrides=base)
        self.mapper = HandToJointMapper(self.cfg)
        self.supervisor = SafetySupervisor(self.cfg)
        self.arm = SimulatedArm(self.cfg)
        self.arm.connect()
        self.supervisor.start(self.arm.read_joints())
        self.report = None

    def run(self, features: HandFeatures, seconds: float = 1.0):
        for _ in range(int(seconds / DT)):
            output = self.mapper.update(features, self.supervisor.command, DT)
            command, self.report = self.supervisor.step(
                output.targets, DT, hand_present=features.present, engaged=output.engaged
            )
            self.arm.send_joints(command)
            self.arm.step(DT)
            self.check_invariants(command)
        return self.arm.read_joints()

    def check_invariants(self, command):
        for name in JOINT_NAMES:
            joint = self.cfg.joint(name)
            assert joint.min - 1e-6 <= command[name] <= joint.max + 1e-6, name


def hand(**kwargs) -> HandFeatures:
    return extract_features(make_hand(**kwargs), FRAME)


def test_arm_reaches_home_then_follows_the_hand():
    rig = Rig()
    rig.run(HandFeatures.absent(), seconds=0.6)
    for name in JOINT_NAMES:
        assert rig.arm.read_joints()[name] == pytest.approx(rig.supervisor.home[name], abs=2.0)

    rig.run(hand(center=(0.5, 0.5)), seconds=1.0)
    centred = rig.arm.read_joints()["shoulder_pan"]
    rig.run(hand(center=(0.78, 0.5)), seconds=2.0)
    assert rig.arm.read_joints()["shoulder_pan"] > centred + 5.0


def test_hand_up_and_down_moves_the_shoulder_both_ways():
    rig = Rig()
    rig.run(HandFeatures.absent(), seconds=0.6)
    rig.run(hand(center=(0.5, 0.5)), seconds=1.0)
    middle = rig.arm.read_joints()["shoulder_lift"]

    rig.run(hand(center=(0.5, 0.25)), seconds=2.0)
    up = rig.arm.read_joints()["shoulder_lift"]
    rig.run(hand(center=(0.5, 0.75)), seconds=3.0)
    down = rig.arm.read_joints()["shoulder_lift"]
    assert up > middle > down


def test_gripper_opens_and_closes_with_the_pinch():
    rig = Rig()
    rig.run(HandFeatures.absent(), seconds=0.6)
    rig.run(hand(pinch=1.0), seconds=1.5)
    assert rig.arm.read_joints()["gripper"] > 80.0
    rig.run(hand(pinch=0.05), seconds=1.5)
    assert rig.arm.read_joints()["gripper"] < 20.0


def test_losing_the_hand_stops_the_arm():
    rig = Rig()
    rig.run(HandFeatures.absent(), seconds=0.6)
    rig.run(hand(center=(0.7, 0.4)), seconds=1.0)
    frozen = rig.arm.read_joints()

    rig.run(HandFeatures.absent(), seconds=1.0)
    assert rig.supervisor.state is SafetyState.HOLDING
    for name in JOINT_NAMES:
        assert rig.arm.read_joints()[name] == pytest.approx(frozen[name], abs=1.5)


def test_curled_fingers_pause_and_let_you_reposition():
    """Sedno sprzegla: zwin palce, przeloz reke, wyprostuj - robot nie skacze."""
    rig = Rig(clutch={"mode": "gesture", "engaged_on_start": True})
    rig.run(HandFeatures.absent(), seconds=0.6)
    rig.run(hand(center=(0.5, 0.5), curl=0.9), seconds=1.0)
    before = rig.arm.read_joints()["shoulder_pan"]

    # Palce zwiniete - dlon jedzie na drugi kraniec kadru, ramie stoi.
    rig.run(hand(center=(0.5, 0.5), curl=0.2), seconds=0.5)
    rig.run(hand(center=(0.2, 0.5), curl=0.2), seconds=1.0)
    assert rig.arm.read_joints()["shoulder_pan"] == pytest.approx(before, abs=2.0)

    # Palce wyprostowane w nowym miejscu - ruch rusza stad, bez przeskoku.
    rig.run(hand(center=(0.2, 0.5), curl=0.9), seconds=0.5)
    assert rig.arm.read_joints()["shoulder_pan"] == pytest.approx(before, abs=6.0)


def test_estop_stops_everything():
    rig = Rig()
    rig.run(HandFeatures.absent(), seconds=0.6)
    rig.run(hand(center=(0.7, 0.4)), seconds=1.0)
    rig.supervisor.trigger_estop()
    frozen = rig.arm.read_joints()

    rig.run(hand(center=(0.2, 0.8)), seconds=2.0)
    assert rig.supervisor.state is SafetyState.ESTOP
    for name in JOINT_NAMES:
        assert rig.arm.read_joints()[name] == pytest.approx(frozen[name], abs=1.0)


def test_commands_never_exceed_the_velocity_limit():
    """Nagly skok dloni z rogu do rogu nie moze dac szarpniecia."""
    rig = Rig()
    rig.run(HandFeatures.absent(), seconds=0.6)
    rig.run(hand(center=(0.15, 0.15), scale=0.08), seconds=1.0)
    previous = rig.supervisor.command

    for _ in range(30):
        output = rig.mapper.update(hand(center=(0.9, 0.9), scale=0.22), rig.supervisor.command, DT)
        command, _ = rig.supervisor.step(output.targets, DT, hand_present=True, engaged=True)
        for name in JOINT_NAMES:
            limit = rig.cfg.joint(name).max_vel * DT + 1e-6
            assert abs(command[name] - previous[name]) <= limit, name
        previous = command


def test_ik_mode_runs_the_whole_chain():
    rig = Rig(mapping={"mode": "ik"})
    rig.run(HandFeatures.absent(), seconds=0.6)
    rig.run(hand(center=(0.5, 0.5)), seconds=1.0)
    centred = rig.arm.read_joints()["shoulder_pan"]
    rig.run(hand(center=(0.8, 0.5)), seconds=2.0)
    assert abs(rig.arm.read_joints()["shoulder_pan"] - centred) > 3.0
