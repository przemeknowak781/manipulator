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


def tip_height(rig) -> float:
    """Wysokosc koncowki chwytaka - mierzymy skutek, nie wartosc stawu."""
    from lerobot_mp.control.kinematics import ArmKinematics

    joints = rig.arm.read_joints()
    return ArmKinematics(rig.cfg.geometry).forward(
        joints["shoulder_pan"], joints["shoulder_lift"], joints["elbow_flex"], joints["wrist_flex"]
    )[2]


def test_hand_up_and_down_raises_and_lowers_the_tip():
    """Reka w gore ma PODNIESC koncowke - w tej kalibracji `shoulder_lift`
    rosnie przy opuszczaniu, wiec test na samej liczbie mierzylby znak, a nie ruch."""
    rig = Rig()
    rig.run(HandFeatures.absent(), seconds=0.6)
    rig.run(hand(center=(0.5, 0.5)), seconds=1.0)
    middle = tip_height(rig)

    rig.run(hand(center=(0.5, 0.25)), seconds=2.0)
    up = tip_height(rig)
    rig.run(hand(center=(0.5, 0.75)), seconds=3.0)
    down = tip_height(rig)
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


# --------------------------------------------------------------------------
# Tryb `arm` - caly lancuch z sylwetka operatora
# --------------------------------------------------------------------------

from conftest import make_pose  # noqa: E402
from lerobot_mp.vision.arm_features import ArmFeatures, extract_arm_features  # noqa: E402


def arm(**kwargs) -> ArmFeatures:
    return extract_arm_features(make_pose(**kwargs), kwargs.get("side", "Right"), FRAME)


class ArmRig(Rig):
    """Rig karmiony sylwetka zamiast sama dlonia."""

    def run(self, arm_features, hand_features=None, seconds: float = 1.0):
        hand_features = hand_features if hand_features is not None else HandFeatures.absent()
        for _ in range(int(seconds / DT)):
            output = self.mapper.update(
                hand_features, self.supervisor.command, DT, arm=arm_features
            )
            command, self.report = self.supervisor.step(
                output.targets, DT, hand_present=arm_features.present, engaged=output.engaged
            )
            self.arm.send_joints(command)
            self.arm.step(DT)
            self.check_invariants(command)
        return self.arm.read_joints()


def test_arm_mode_drives_the_robot_end_to_end():
    rig = ArmRig(mapping={"mode": "arm"})
    rig.run(ArmFeatures.absent(), seconds=0.6)
    rig.run(arm(elevation_deg=-60.0), seconds=1.5)
    low = tip_height(rig)
    rig.run(arm(elevation_deg=-10.0), seconds=2.5)
    assert tip_height(rig) > low + 0.02


def test_losing_the_arm_freezes_the_robot():
    rig = ArmRig(mapping={"mode": "arm"})
    rig.run(ArmFeatures.absent(), seconds=0.6)
    rig.run(arm(elevation_deg=-40.0, azimuth_deg=20.0), seconds=1.5)
    frozen = rig.arm.read_joints()

    rig.run(ArmFeatures.absent(), seconds=1.0)
    assert rig.supervisor.state is SafetyState.HOLDING
    for name in JOINT_NAMES:
        assert rig.arm.read_joints()[name] == pytest.approx(frozen[name], abs=1.5)


def test_arm_mode_keeps_the_gripper_on_the_pinch():
    rig = ArmRig(mapping={"mode": "arm"})
    rig.run(ArmFeatures.absent(), seconds=0.6)
    steady = arm(elevation_deg=-40.0)
    rig.run(steady, hand(pinch=1.0), seconds=1.5)
    assert rig.arm.read_joints()["gripper"] > 80.0
    rig.run(steady, hand(pinch=0.05), seconds=1.5)
    assert rig.arm.read_joints()["gripper"] < 20.0


def test_arm_mode_respects_velocity_limits_on_a_sudden_move():
    rig = ArmRig(mapping={"mode": "arm"})
    rig.run(ArmFeatures.absent(), seconds=0.6)
    rig.run(arm(elevation_deg=-80.0, azimuth_deg=-60.0), seconds=1.0)
    previous = rig.supervisor.command

    jump = arm(elevation_deg=60.0, azimuth_deg=80.0, elbow_deg=140.0)
    for _ in range(30):
        output = rig.mapper.update(HandFeatures.absent(), rig.supervisor.command, DT, arm=jump)
        command, _ = rig.supervisor.step(output.targets, DT, hand_present=True, engaged=True)
        for name in JOINT_NAMES:
            assert abs(command[name] - previous[name]) <= rig.cfg.joint(name).max_vel * DT + 1e-6
        previous = command
