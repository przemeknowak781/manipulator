"""Backend `lerobot` z panelu: nigdy nie zdejmuje momentu i nie pyta o nic w konsoli.

Odtworzone na tej maszynie: bez pliku kalibracji `SOFollower.connect()` wolal
`calibrate()`, ten zdejmowal moment ze wszystkich serw (ramie trzymane przez
sesje `feetech` spadalo na stol) i wisial na `input()` w konsoli serwera.
Testy chodza na atrapie klas LeRobota - bez biblioteki i bez sprzetu.
"""

from __future__ import annotations

import builtins
import dataclasses

import pytest

from lerobot_mp.config import load_config
from lerobot_mp.robot.lerobot_backend import LeRobotArm


@dataclasses.dataclass
class FakeConfig:
    port: str
    id: str | None = None
    use_degrees: bool = True
    max_relative_target: float | None = None
    disable_torque_on_disconnect: bool = True


class FakeBus:
    def __init__(self, calibrated: bool, torque: int):
        self.calibrated = calibrated
        self.torque = torque
        self.connected = False
        self.log: list[tuple] = []

    def connect(self) -> None:
        self.connected = True
        self.log.append(("connect",))

    def disconnect(self, disable_torque: bool = True) -> None:
        self.connected = False
        self.log.append(("disconnect", disable_torque))
        if disable_torque:
            self.torque = 0

    @property
    def is_calibrated(self) -> bool:
        return self.calibrated

    def sync_read(self, name: str) -> dict[str, int]:
        assert name == "Torque_Enable"
        return {"shoulder_pan": self.torque, "gripper": self.torque}

    def disable_torque(self) -> None:
        self.torque = 0
        self.log.append(("torque_off",))


class FakeRobot:
    """Tyle `SOFollower`, ile backend dotyka - z ta sama logika `connect(calibrate=True)`."""

    calibration_file = True
    servos_calibrated = True
    servos_torque = 1
    last: "FakeRobot | None" = None

    def __init__(self, config: FakeConfig):
        self.config = config
        self.calibration = {"shoulder_pan": object()} if self.calibration_file else {}
        self.calibration_fpath = "C:/cache/lerobot/calibration/robots/so_follower/so101_follower.json"
        self.bus = FakeBus(self.servos_calibrated, self.servos_torque)
        self.configured = False
        FakeRobot.last = self

    @property
    def is_connected(self) -> bool:
        return self.bus.connected

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def connect(self, calibrate: bool = True) -> None:
        self.bus.connect()
        if not self.is_calibrated and calibrate:
            self.calibrate()
        self.configure()

    def calibrate(self) -> None:
        self.bus.disable_torque()
        input("Move the arm to the middle ...")

    def configure(self) -> None:
        # Jak w LeRobot: zapisy pod `torque_disabled()`.
        self.bus.disable_torque()
        self.configured = True
        self.bus.torque = 1

    def disconnect(self) -> None:
        self.bus.disconnect(self.config.disable_torque_on_disconnect)

    @property
    def action_features(self) -> dict:
        return {"shoulder_pan.pos": float, "gripper.pos": float}


@pytest.fixture
def arm(monkeypatch):
    FakeRobot.calibration_file, FakeRobot.servos_calibrated, FakeRobot.servos_torque = True, True, 1
    monkeypatch.setattr(LeRobotArm, "_resolve_classes", staticmethod(lambda kind: (FakeRobot, FakeConfig, "atrapa")))

    def no_console(*_a, **_k):
        raise AssertionError("backend zapytal o cos w konsoli serwera")

    monkeypatch.setattr(builtins, "input", no_console)
    cfg = load_config(overrides={"robot": {"backend": "lerobot", "port": "COM_TEST"}})
    return LeRobotArm(cfg)


def test_missing_calibration_file_is_refused_before_touching_the_port(arm):
    FakeRobot.calibration_file = False
    with pytest.raises(RuntimeError, match="lerobot-calibrate"):
        arm.connect()
    robot = FakeRobot.last
    assert robot.bus.log == []                      # port nietkniety - moment zostaje tam, gdzie byl
    assert robot.bus.torque == 1
    assert not arm.is_connected


def test_calibration_that_differs_from_the_servos_is_refused_without_dropping_torque(arm):
    FakeRobot.servos_calibrated = False
    with pytest.raises(RuntimeError, match="lerobot-calibrate"):
        arm.connect()
    robot = FakeRobot.last
    assert ("torque_off",) not in robot.bus.log
    assert robot.bus.log[-1] == ("disconnect", False)
    assert robot.bus.torque == 1


def test_an_arm_holding_a_pose_is_connected_without_lerobot_configure(arm):
    """`configure()` zdejmuje moment na czas zapisow - ramie w powietrzu by opadlo."""
    arm.connect()
    robot = FakeRobot.last
    assert arm.is_connected and robot.bus.connected
    assert not robot.configured
    assert ("torque_off",) not in robot.bus.log


def test_a_resting_arm_gets_the_regular_lerobot_connect_without_calibration(arm):
    FakeRobot.servos_torque = 0
    arm.connect()
    robot = FakeRobot.last
    assert robot.configured and robot.bus.connected


def test_disconnect_keeps_torque_unless_configured_otherwise(arm):
    arm.connect()
    robot = FakeRobot.last
    arm.disconnect()
    assert robot.bus.log[-1] == ("disconnect", False)
    assert robot.bus.torque == 1
