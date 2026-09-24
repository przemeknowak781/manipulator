"""Backend `lerobot` z panelu: nigdy nie zdejmuje momentu i nie pyta o nic w konsoli.

Odtworzone na tej maszynie: bez pliku kalibracji `SOFollower.connect()` wolal
`calibrate()`, ten zdejmowal moment ze wszystkich serw (ramie trzymane przez
sesje `feetech` spadalo na stol) i wisial na `input()` w konsoli serwera.
Testy chodza na atrapie klas LeRobota - bez biblioteki i bez sprzetu.

Do tego to, czego ten backend nie mial, a `feetech` tak: bity ochrony serw
(`faults`), limity z kalibracji (`joint_limits`) i tiki chwytaka (`gripper_ticks`)
- przeliczenia sprawdzone takze na prawdziwych klasach zainstalowanego LeRobota.
"""

from __future__ import annotations

import builtins
import dataclasses
import enum
import logging

import pytest

from lerobot_mp.config import load_config
from lerobot_mp.robot import lerobot_backend
from lerobot_mp.robot.lerobot_backend import STATUS_EVERY_READS, STATUS_FAIL_LIMIT, LeRobotArm


@dataclasses.dataclass
class FakeConfig:
    port: str
    id: str | None = None
    use_degrees: bool = True
    max_relative_target: float | None = None
    disable_torque_on_disconnect: bool = True


class NormMode(enum.Enum):
    """Nazwy jak `lerobot.motors.MotorNormMode`."""

    DEGREES = "degrees"
    RANGE_M100_100 = "range_m100_100"
    RANGE_0_100 = "range_0_100"


@dataclasses.dataclass
class Cal:
    """Pola jak `lerobot.motors.MotorCalibration` - tak wyglada wpis w pliku kalibracji."""

    id: int
    drive_mode: int
    homing_offset: int
    range_min: int
    range_max: int


@dataclasses.dataclass
class Motor:
    id: int
    model: str
    norm_mode: NormMode


#: Typowa kalibracja SO-101 z `lerobot-calibrate` (zakresy nagrane reka, wrist_roll pelny obrot).
TYPICAL_CALIBRATION = {
    "shoulder_pan": (1, 0, -1470, 758, 3292),
    "shoulder_lift": (2, 0, 157, 815, 3264),
    "elbow_flex": (3, 0, 1270, 858, 3093),
    "wrist_flex": (4, 0, -1014, 857, 3220),
    "wrist_roll": (5, 0, 1453, 0, 4095),
    "gripper": (6, 0, 1041, 2031, 3524),
}


class FakeBus:
    def __init__(self, calibrated: bool, torque: int, calibration: dict | None = None):
        self.calibrated = calibrated
        self.torque = torque
        self.connected = False
        self.log: list[tuple] = []
        self.calibration = calibration or {}
        self.apply_drive_mode = True
        self.model_resolution_table = {"sts3215": 4096}
        self.motors = {
            name: Motor(fields[0], "sts3215", NormMode.RANGE_0_100 if name == "gripper" else NormMode.DEGREES)
            for name, fields in TYPICAL_CALIBRATION.items()
        }
        #: Rejestr Status kazdego serwa (0x20 = przeciazenie) i czy jego odczyt sie udaje.
        self.status = {name: 0 for name in self.motors}
        self.status_fails = False
        self.status_reads = 0

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

    def sync_read(self, name: str, motors=None, *, normalize: bool = True, num_retry: int = 0) -> dict[str, int]:
        if name == "Status":
            assert not normalize
            self.status_reads += 1
            if self.status_fails:
                raise ConnectionError("Failed to sync read 'Status' [TxRxResult] There is no status packet!")
            return dict(self.status)
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
        self.calibration = ({name: Cal(*fields) for name, fields in TYPICAL_CALIBRATION.items()}
                            if self.calibration_file else {})
        self.calibration_fpath = "C:/cache/lerobot/calibration/robots/so_follower/so101_follower.json"
        self.bus = FakeBus(self.servos_calibrated, self.servos_torque, self.calibration)
        self.configured = False
        self.positions = {name: 0.0 for name in TYPICAL_CALIBRATION}
        self.sent: list[dict] = []
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
        return {f"{name}.pos": float for name in TYPICAL_CALIBRATION}

    def get_observation(self) -> dict:
        return {f"{name}.pos": value for name, value in self.positions.items()}

    def send_action(self, action: dict) -> dict:
        self.sent.append(dict(action))
        return dict(action)


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


# ------------------------------------------------------------ bity ochrony serw
@pytest.fixture
def live(arm):
    arm.connect()
    return arm, FakeRobot.last


def cycle(arm, n: int = 1) -> None:
    """`n` okien po STATUS_EVERY_READS odczytow pozycji - w kazdym dokladnie jeden odczyt Status (na poczatku)."""
    for _ in range(n * STATUS_EVERY_READS):
        arm.read_joints()


def test_servo_protection_bits_are_reported_as_faults(live):
    """Przeciazony bark zwalnia moment, ale dalej odpowiada - bez Status wygladal na zdrowy,
    a fala jechala dalej reszta ramienia. Zdanie takie samo jak w `feetech`."""
    arm, robot = live
    cycle(arm)
    assert arm.faults() == []
    robot.bus.status["shoulder_lift"] = 0x20
    cycle(arm)
    assert arm.faults() == ["shoulder_lift (serwo 2): przeciazenie"]
    robot.bus.status["shoulder_lift"] = 0
    cycle(arm)
    assert arm.faults() == []


def test_status_is_read_once_per_few_position_reads(live):
    """Petla czyta pozycje 10 Hz, a `faults()` wola co takt - Status nie moze isc za kazdym razem."""
    arm, robot = live
    for _ in range(10):
        arm.faults()
    assert robot.bus.status_reads == 0                  # `faults()` samo nie dotyka magistrali
    cycle(arm, 4)
    assert robot.bus.status_reads == 4
    assert STATUS_EVERY_READS >= 2


def test_a_failed_status_read_keeps_the_last_bits_and_never_breaks_reading(live):
    arm, robot = live
    robot.bus.status["gripper"] = 0x04
    cycle(arm)
    robot.bus.status_fails = True
    assert arm.read_joints()["gripper"] == 0.0          # odczyt pozycji dziala dalej
    cycle(arm)
    # Zgubiona ramka nie kasuje przegrzania - dalej STOP.
    assert arm.faults() == ["gripper (serwo 6): przegrzanie"]


def test_status_that_stays_unreadable_is_a_fault(live):
    """Ochrona serw niewidoczna ~1,5 s to usterka - ale nie pojedyncza czkawka."""
    arm, robot = live
    robot.bus.status_fails = True
    cycle(arm, STATUS_FAIL_LIMIT - 1)
    assert arm.faults() == []
    cycle(arm)
    faults = arm.faults()
    assert len(faults) == 1 and "Status" in faults[0]
    robot.bus.status_fails = False
    cycle(arm)
    assert arm.faults() == []


def test_faults_never_raise(live):
    arm, _ = live
    arm._status = {"shoulder_pan": "smiec"}             # cokolwiek pojdzie nie tak w srodku
    faults = arm.faults()
    assert len(faults) == 1 and "stanu serw" in faults[0]


def test_disconnect_forgets_the_servo_status(live):
    arm, robot = live
    robot.bus.status["elbow_flex"] = 0x20
    cycle(arm)
    assert arm.faults()
    arm.disconnect()
    assert arm.faults() == []


# ------------------------------------------------------------ limity i chwytak
def test_joint_limits_are_the_calibration_range_in_app_units(live):
    """Zakres z kalibracji = Min/Max_Position_Limit w EEPROM (sprawdzone przez `is_calibrated`)."""
    arm, _ = live
    limits = arm.joint_limits()
    # Stopnie LeRobota: zero w srodku zakresu, 4095 tikow na 360 st.
    assert limits["shoulder_pan"] == pytest.approx((-(3292 - 758) / 2 * 360 / 4095, (3292 - 758) / 2 * 360 / 4095))
    assert limits["wrist_flex"] == pytest.approx((-103.868, 103.868), abs=1e-3)
    assert limits["gripper"] == pytest.approx((0.0, 100.0))
    assert "wrist_roll" not in limits                   # pelny obrot - serwo niczego nie ogranicza


def test_send_joints_returns_what_the_servo_really_accepts(live):
    """W stopniach LeRobot nie przycina celu - serwo robilo to po cichu do EEPROM."""
    arm, robot = live
    sent = arm.send_joints({"wrist_flex": 150.0, "shoulder_pan": -200.0, "elbow_flex": 10.0})
    assert sent["wrist_flex"] == pytest.approx(103.868, abs=1e-3)
    assert sent["shoulder_pan"] == pytest.approx(-111.385, abs=1e-3)
    assert sent["elbow_flex"] == 10.0
    assert robot.sent[-1]["wrist_flex.pos"] == pytest.approx(103.868, abs=1e-3)


def test_gripper_ticks_follow_the_lerobot_calibration(live):
    """LeRobot skaluje chwytak 0..100 po swoim zakresie z kalibracji, nie po tikach `feetech`."""
    arm, robot = live
    assert arm.gripper_ticks() == (2031.0, 3524.0, 2047.0)
    robot.bus.calibration["gripper"].drive_mode = 1     # odwrocona skala: 0 = range_max
    assert arm.gripper_ticks() == (3524.0, 2031.0, 2047.0)


def test_gripper_ticks_are_unknown_before_connecting(arm):
    assert arm.gripper_ticks() is None
    assert arm.joint_limits() == {}


def test_a_joint_zero_off_the_twin_zero_is_logged(arm, caplog):
    """LeRobot liczy stopnie od srodka zakresu, blizniak od tiku 2048 - kilka stopni roznicy."""
    with caplog.at_level(logging.WARNING, logger=lerobot_backend.__name__):
        arm.connect()
    offsets = arm.joint_zero_offsets()
    assert offsets["shoulder_pan"] == pytest.approx(((758 + 3292) / 2 - 2048) * 360 / 4095)
    assert set(offsets) == {"shoulder_pan", "elbow_flex"}   # wrist_roll 0..4095 -> srodek 2047.5, w tolerancji
    [warning] = arm.calibration_warnings()
    assert "shoulder_pan -2.0 st." in warning and "feetech" in warning
    assert warning in caplog.text


def test_a_calibration_centred_on_the_twin_zero_gives_no_warning(arm):
    arm.connect()
    for cal in FakeRobot.last.calibration.values():
        half = (cal.range_max - cal.range_min) // 2
        cal.range_min, cal.range_max = 2048 - half, 2048 + half
    assert arm.joint_zero_offsets() == {}
    assert arm.calibration_warnings() == []


# ------------------------------------------------ na prawdziwych klasach LeRobota
def _real_robot(tmp_path, gripper_drive_mode: int = 0):
    so = pytest.importorskip("lerobot.robots.so_follower")
    import json
    from pathlib import Path

    data = {name: dict(zip(("id", "drive_mode", "homing_offset", "range_min", "range_max"), fields))
            for name, fields in TYPICAL_CALIBRATION.items()}
    data["gripper"]["drive_mode"] = gripper_drive_mode
    (Path(tmp_path) / "arm.json").write_text(json.dumps(data), encoding="utf-8")
    # Sama konstrukcja - port nie jest otwierany.
    return so.SOFollower(so.SOFollowerRobotConfig(port="COM_TEST", id="arm", calibration_dir=Path(tmp_path)))


@pytest.mark.parametrize("drive_mode", [0, 1])
def test_conversions_match_the_installed_lerobot(tmp_path, drive_mode):
    """Wzory backendu = `MotorsBus._normalize/_unnormalize` zainstalowanego LeRobota."""
    robot = _real_robot(tmp_path, drive_mode)
    cfg = load_config(overrides={"robot": {"backend": "lerobot", "port": "COM_TEST"}})
    arm = LeRobotArm(cfg)
    arm._robot = robot
    bus = robot.bus
    for name, motor in bus.motors.items():
        cal = bus.calibration[name]
        for t in (cal.range_min, (cal.range_min + cal.range_max) // 2, cal.range_max):
            assert arm._ticks_to_units(name, t) == pytest.approx(bus._normalize({motor.id: t})[motor.id])

    closed, opened, zero = arm.gripper_ticks()
    assert bus._unnormalize({6: 0.0})[6] == closed
    assert bus._unnormalize({6: 100.0})[6] == opened
    # Kalibracja LeRobota stawia poze srodkowa dokladnie na `zero` (pol obrotu).
    assert 3000 - bus._get_half_turn_homings({"gripper": 3000})["gripper"] == zero

    for name, (lo, hi) in arm.joint_limits().items():
        cal, mid = bus.calibration[name], bus.motors[name].id
        ends = sorted(bus._normalize({mid: t})[mid] for t in (cal.range_min, cal.range_max))
        assert (lo, hi) == pytest.approx(tuple(ends))


def test_the_status_register_exists_in_the_installed_lerobot():
    """Odczyt po nazwie "Status" - adres 65, te same bity co bajt bledu (ERRBIT_* scservo_sdk)."""
    tables = pytest.importorskip("lerobot.motors.feetech.tables")
    assert tables.STS_SMS_SERIES_CONTROL_TABLE["Status"] == (65, 1)
    sdk = pytest.importorskip("scservo_sdk.protocol_packet_handler")
    from lerobot_mp.robot.feetech import SERVO_ERROR_BITS

    bits = {bit for bit, _ in SERVO_ERROR_BITS}
    assert {sdk.ERRBIT_VOLTAGE, sdk.ERRBIT_ANGLE, sdk.ERRBIT_OVERHEAT, sdk.ERRBIT_OVERELE, sdk.ERRBIT_OVERLOAD} == bits
