"""Backend `feetech`: protokol STS3215, przeliczenia jednostek i kolejnosc startu.

Testy chodza po *symulowanej magistrali* - odpowiada na pakiety tak, jak
prawdziwe serwo, i zapisuje kolejnosc operacji. Dzieki temu da sie sprawdzic
rzeczy, ktorych na sprzecie sprawdzic nie sposob bez ryzyka: na przyklad ze
moment zalacza sie DOPIERO po nadpisaniu rejestru celu.
"""

from __future__ import annotations

import pytest

from lerobot_mp.config import JOINT_NAMES, load_config
from lerobot_mp.robot.feetech import (
    ADDR_GOAL_POSITION,
    ADDR_MAX_ANGLE_LIMIT,
    ADDR_MIN_ANGLE_LIMIT,
    ADDR_PRESENT_POSITION,
    ADDR_TORQUE_ENABLE,
    ADDR_PRESENT_VOLTAGE,
    BROADCAST_ID,
    INST_PING,
    INST_READ,
    INST_SYNC_READ,
    INST_SYNC_WRITE,
    INST_WRITE,
    FeetechArm,
    FeetechBus,
    checksum,
)

REST_TICKS = 2048


class FakeBusLink:
    """Szesc serw na jednej magistrali - tyle protokolu, ile backend uzywa."""

    #: Starszy firmware nie zna SYNC READ i po prostu milczy.
    knows_sync_read = True

    def __init__(self, ids=(1, 2, 3, 4, 5, 6), start_ticks: int = REST_TICKS):
        self.registers = {
            dev_id: {
                ADDR_TORQUE_ENABLE: 0,
                ADDR_GOAL_POSITION: 3000,  # celowo INNY niz pozycja biezaca
                ADDR_PRESENT_POSITION: start_ticks,
                ADDR_PRESENT_VOLTAGE: 119,
                ADDR_MIN_ANGLE_LIMIT: 0,
                ADDR_MAX_ANGLE_LIMIT: 4095,
            }
            for dev_id in ids
        }
        #: Slad operacji: ("ping"|"read"|"write"|"sync", id, addr, wartosc).
        self.log: list[tuple] = []
        self._out = bytearray()
        self.closed = False

    # ------------------------------------------------------------- pyserial
    def reset_input_buffer(self) -> None:
        self._out.clear()

    def reset_output_buffer(self) -> None:
        pass

    def flush(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True

    def read(self, size: int = 1) -> bytes:
        out, self._out = bytes(self._out[:size]), self._out[size:]
        return out

    @property
    def in_waiting(self) -> int:
        return len(self._out)

    # --------------------------------------------------------------- serwo
    def write(self, data: bytes) -> int:
        assert data[:2] == b"\xff\xff", "brak naglowka pakietu"
        dev_id, length, instruction = data[2], data[3], data[4]
        params = data[5:-1]
        assert checksum(data[2:-1]) == data[-1], "zla suma kontrolna"
        assert length == len(params) + 2, "dlugosc nie zgadza sie z parametrami"

        # Serwo, ktorego nie ma na magistrali, po prostu milczy - i tak wlasnie
        # wyglada brakujacy staw dla backendu.
        if dev_id != BROADCAST_ID and dev_id not in self.registers:
            self.log.append(("cisza", dev_id, None, None))
            return len(data)

        if instruction == INST_PING:
            self.log.append(("ping", dev_id, None, None))
            self._reply(dev_id, b"")
        elif instruction == INST_READ:
            addr, size = params[0], params[1]
            self.log.append(("read", dev_id, addr, None))
            value = self.registers[dev_id].get(addr, 0)
            self._reply(dev_id, value.to_bytes(size, "little"))
        elif instruction == INST_WRITE:
            addr, value = params[0], int.from_bytes(params[1:], "little")
            self.registers[dev_id][addr] = value
            self.log.append(("write", dev_id, addr, value))
            self._reply(dev_id, b"")
        elif instruction == INST_SYNC_READ:
            assert dev_id == BROADCAST_ID, "sync read musi isc rozgloszeniowo"
            addr, size = params[0], params[1]
            self.log.append(("syncread", None, addr, tuple(params[2:])))
            if self.knows_sync_read:
                for target in params[2:]:
                    if target in self.registers:
                        self._reply(target, self.registers[target].get(addr, 0).to_bytes(size, "little"))
        elif instruction == INST_SYNC_WRITE:
            assert dev_id == BROADCAST_ID, "sync write musi isc rozgloszeniowo"
            addr, size = params[0], params[1]
            body = params[2:]
            stride = size + 1
            for offset in range(0, len(body), stride):
                target = body[offset]
                value = int.from_bytes(body[offset + 1 : offset + stride], "little")
                self.registers[target][addr] = value
                self.log.append(("sync", target, addr, value))
        else:  # pragma: no cover - backend nie uzywa innych instrukcji
            raise AssertionError(f"nieznana instrukcja 0x{instruction:02X}")
        return len(data)

    def _reply(self, dev_id: int, payload: bytes) -> None:
        body = bytes([dev_id, len(payload) + 2, 0x00]) + payload
        self._out += b"\xff\xff" + body + bytes([checksum(body)])

    # ------------------------------------------------------------- pomocnicze
    def ops(self, kind: str) -> list[tuple]:
        return [entry for entry in self.log if entry[0] == kind]


@pytest.fixture
def arm_cfg():
    return load_config(overrides={"robot": {"backend": "feetech", "port": "COM_TEST"}})


def connected(cfg, link: FakeBusLink | None = None) -> tuple[FeetechArm, FakeBusLink]:
    link = link or FakeBusLink()
    bus = FeetechBus("COM_TEST")
    bus.open(link=link)
    arm = FeetechArm(cfg, bus=bus)
    arm.connect()
    return arm, link


def test_checksum_matches_a_known_good_ping_packet():
    """Pakiet ping do serwa 1 - zgodny z protokolem Dynamixel 1.0."""
    assert checksum(bytes([0x01, 0x02, 0x01])) == 0xFB


def test_connect_pings_every_joint(arm_cfg):
    _, link = connected(arm_cfg)
    assert {entry[1] for entry in link.ops("ping")} == {1, 2, 3, 4, 5, 6}


def test_connect_writes_the_goal_before_enabling_torque(arm_cfg):
    """Rejestr celu pamieta poprzednia sesje - moment zalaczony przed jego
    nadpisaniem szarpnalby ramieniem do tamtej pozy."""
    _, link = connected(arm_cfg)
    first_torque = next(
        i for i, entry in enumerate(link.log)
        if entry[0] == "write" and entry[2] == ADDR_TORQUE_ENABLE
    )
    goal_writes = [
        i for i, entry in enumerate(link.log)
        if entry[0] == "sync" and entry[2] == ADDR_GOAL_POSITION
    ]
    assert goal_writes, "cel nie zostal w ogole ustawiony"
    assert max(goal_writes) < first_torque


def test_connect_sets_the_goal_to_where_the_arm_actually_stands(arm_cfg):
    _, link = connected(arm_cfg)
    for dev_id, registers in link.registers.items():
        assert registers[ADDR_GOAL_POSITION] == registers[ADDR_PRESENT_POSITION]


def test_connect_enables_torque_on_every_joint(arm_cfg):
    _, link = connected(arm_cfg)
    assert all(regs[ADDR_TORQUE_ENABLE] == 1 for regs in link.registers.values())


def test_connect_refuses_when_a_servo_is_silent(arm_cfg):
    link = FakeBusLink(ids=(1, 2, 3, 4, 5))  # brakuje chwytaka
    bus = FeetechBus("COM_TEST")
    bus.open(link=link)
    with pytest.raises(RuntimeError, match="gripper"):
        FeetechArm(arm_cfg, bus=bus).connect()


def test_middle_tick_is_zero_degrees(arm_cfg):
    arm, _ = connected(arm_cfg)
    positions = arm.read_joints()
    for name in JOINT_NAMES:
        if name != "gripper":
            assert positions[name] == pytest.approx(0.0, abs=1e-9)


def test_one_full_turn_is_the_whole_tick_range(arm_cfg):
    arm, _ = connected(arm_cfg)
    assert arm._to_ticks("wrist_roll", 90.0) - arm._to_ticks("wrist_roll", 0.0) == 1024


def test_degrees_and_ticks_round_trip(arm_cfg):
    arm, _ = connected(arm_cfg)
    for value in (-90.0, -30.0, 0.0, 45.0, 90.0):
        ticks = arm._to_ticks("shoulder_pan", value)
        assert arm._to_units("shoulder_pan", ticks) == pytest.approx(value, abs=0.1)


def test_gripper_spans_the_configured_tick_range(arm_cfg):
    arm, _ = connected(arm_cfg)
    assert arm._to_ticks("gripper", 0.0) == arm_cfg.robot.gripper_closed_ticks
    assert arm._to_ticks("gripper", 100.0) == arm_cfg.robot.gripper_open_ticks


def test_send_joints_clamps_to_the_configured_limits(arm_cfg):
    arm, link = connected(arm_cfg)
    limit = arm_cfg.joint("shoulder_pan").max
    sent = arm.send_joints({"shoulder_pan": limit + 500.0})
    assert sent["shoulder_pan"] == pytest.approx(limit)
    assert link.registers[1][ADDR_GOAL_POSITION] == arm._to_ticks("shoulder_pan", limit)


def test_targets_never_leave_the_servo_tick_range(arm_cfg):
    """Nawet absurdalny cel nie moze przepelnic dwubajtowego rejestru."""
    arm, link = connected(arm_cfg)
    arm.send_joints({name: 1e6 for name in JOINT_NAMES})
    for registers in link.registers.values():
        assert 0 <= registers[ADDR_GOAL_POSITION] <= 4095


def test_all_six_targets_go_out_in_a_single_packet(arm_cfg):
    """Sync write jest powodem, dla ktorego petla nie placi za liczbe stawow."""
    arm, link = connected(arm_cfg)
    link.log.clear()
    arm.send_joints({name: 0.0 for name in JOINT_NAMES})
    assert len(link.ops("sync")) == 6
    assert link.ops("write") == []


def test_a_lost_reply_keeps_the_previous_position(arm_cfg):
    """Zgubiona ramka nie ma prawa udawac skoku stawu."""
    arm, link = connected(arm_cfg)
    link.registers[1][ADDR_PRESENT_POSITION] = 2048 + 512
    before = arm.read_joints()["shoulder_pan"]

    original_read = link.read

    def drop_first(size: int = 1) -> bytes:
        link.read = original_read
        return b""

    link.read = drop_first
    assert arm.read_joints()["shoulder_pan"] == pytest.approx(before)


def test_joints_are_read_with_one_sync_read_packet(arm_cfg):
    """Przez most sieciowy kazda transakcja to przebieg tam i z powrotem - jeden zamiast szesciu."""
    arm, link = connected(arm_cfg)
    link.registers[3][ADDR_PRESENT_POSITION] = 2048 + 256
    link.log.clear()
    positions = arm.read_joints()
    assert len(link.ops("syncread")) == 1
    assert link.ops("read") == []
    assert positions["elbow_flex"] == pytest.approx(22.5)


def test_old_firmware_without_sync_read_falls_back_to_single_reads(arm_cfg):
    link = FakeBusLink()
    link.knows_sync_read = False
    arm, _ = connected(arm_cfg, link)
    link.registers[2][ADDR_PRESENT_POSITION] = 2048 - 512
    for _ in range(4):
        positions = arm.read_joints()
        assert positions["shoulder_lift"] == pytest.approx(-45.0)
    link.log.clear()
    arm.read_joints()
    assert link.ops("syncread") == []                   # po kilku ciszach juz nie probuje
    assert len(link.ops("read")) == 6


def test_socket_url_opens_through_pyserial_url_handler(monkeypatch):
    """`socket://adres:port` (most lerobot-mp-bridge) - bez sterownika wirtualnego COM."""
    serial = pytest.importorskip("serial")
    seen = {}

    class Link:
        def reset_input_buffer(self): ...
        def reset_output_buffer(self): ...

    def fake_for_url(url, **kw):
        seen["url"], seen["timeout"] = url, kw["timeout"]
        return Link()

    monkeypatch.setattr(serial, "serial_for_url", fake_for_url)
    bus = FeetechBus("socket://10.0.0.5:5555")
    bus.open()
    assert seen["url"] == "socket://10.0.0.5:5555"
    assert seen["timeout"] >= 0.25


def test_disconnect_leaves_the_arm_holding_by_default(arm_cfg):
    """Wylaczony moment znaczy wiotkie ramie, ktore opada pod wlasnym ciezarem."""
    arm, link = connected(arm_cfg)
    arm.disconnect()
    assert all(regs[ADDR_TORQUE_ENABLE] == 1 for regs in link.registers.values())
    assert link.closed


def test_disconnect_can_release_torque_when_asked(arm_cfg):
    arm_cfg.robot.torque_off_on_exit = True
    arm, link = connected(arm_cfg)
    arm.disconnect()
    assert all(regs[ADDR_TORQUE_ENABLE] == 0 for regs in link.registers.values())


def test_voltage_below_the_sane_threshold_is_reported(arm_cfg, caplog):
    """Serwo na samym USB odpowiada, ale nie ruszy - to musi byc powiedziane."""
    link = FakeBusLink()
    for registers in link.registers.values():
        registers[ADDR_PRESENT_VOLTAGE] = 50  # 5,0 V
    with caplog.at_level("WARNING"):
        connected(arm_cfg, link)
    assert "zasilacz" in caplog.text.lower()


def test_servo_limits_are_read_and_reported(arm_cfg, caplog):
    """Serwo przycina rozkaz do wlasnego zakresu po cichu - to musi wyjsc na jaw."""
    link = FakeBusLink()
    link.registers[2][ADDR_MIN_ANGLE_LIMIT] = 2025  # shoulder_lift: tylko -2 stopnie w dol
    link.registers[2][ADDR_MAX_ANGLE_LIMIT] = 3006
    with caplog.at_level("WARNING"):
        arm, _ = connected(arm_cfg, link)
    assert arm.servo_limits["shoulder_lift"] == (2025, 3006)
    assert "shoulder_lift" in caplog.text


def test_targets_below_the_servo_limit_are_reported_as_clipped(arm_cfg):
    """Aplikacja ma dostac pozycje, ktora POJECHALA, a nie te, o ktora prosila."""
    link = FakeBusLink()
    link.registers[2][ADDR_MIN_ANGLE_LIMIT] = 2025
    link.registers[2][ADDR_MAX_ANGLE_LIMIT] = 3006
    arm, link = connected(arm_cfg, link)

    sent = arm.send_joints({"shoulder_lift": -23.3})
    assert link.registers[2][ADDR_GOAL_POSITION] == 2025
    assert sent["shoulder_lift"] == pytest.approx(arm._to_units("shoulder_lift", 2025), abs=0.1)
    assert sent["shoulder_lift"] > -23.3


def test_a_servo_with_the_full_range_is_not_treated_as_limited(arm_cfg):
    arm, _ = connected(arm_cfg)
    assert arm.servo_limits == {}


def test_backend_needs_a_port(arm_cfg):
    arm_cfg.robot.port = None
    with pytest.raises(ValueError, match="portu"):
        FeetechArm(arm_cfg)
