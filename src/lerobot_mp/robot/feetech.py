"""Bezposrednie sterowanie serwami Feetech STS3215 - bez LeRobot i bez torcha.

Po co drugi backend do tego samego ramienia
-------------------------------------------
`lerobot_backend` daje pelna zgodnosc z ekosystemem LeRobot (kalibracja,
nagrywanie zbiorow, uczenie), ale ciagnie za soba `torch` i `torchvision` -
kilka gigabajtow na to, zeby wpisac szesc liczb do rejestrow serw. Do samej
teleoperacji to nie jest potrzebne: serwa gadaja protokolem, ktory miesci sie
na jednym ekranie, a caly ruch to jeden pakiet `SYNC WRITE` na cykl petli.

Protokol jest zgodny z Dynamixel 1.0:
    FF FF <ID> <LEN> <INST> [parametry...] <SUMA>
gdzie ``LEN`` to liczba parametrow + 2, a suma kontrolna to negacja sumy
wszystkiego po naglowku. Wartosci dwubajtowe ida najmlodszym bajtem naprzod.

Czego ten backend NIE robi
--------------------------
Nie ma tu kalibracji LeRobota, wiec przeliczenie tikow na stopnie zaklada, ze
mechaniczne zero stawu pokrywa sie ze srodkiem zakresu serwa (tik 2048) - tak
wychodzi przy standardowym montazu SO-101, w ktorym serwa skladasz wysrodkowane.
Jesli ktorys staw ma zero przesuniete, popraw go przez `joints.<nazwa>.offset`,
a zwrot przez `joints.<nazwa>.invert`.
"""

from __future__ import annotations

import logging
from typing import Any

from ..config import GRIPPER, JOINT_NAMES, AppConfig
from .base import RobotBackend, RobotInfo

logger = logging.getLogger(__name__)

INST_PING = 0x01
INST_READ = 0x02
INST_WRITE = 0x03
INST_SYNC_WRITE = 0x83
BROADCAST_ID = 0xFE

#: Wlasne limity kata serwa, zapisane w jego EEPROM-ie. Serwo przycina do nich
#: KAZDY rozkaz po cichu, wiec aplikacja, ktora ich nie zna, zadaje pozy
#: niewykonalne i nie ma jak sie o tym dowiedziec.
ADDR_MIN_ANGLE_LIMIT = 9
ADDR_MAX_ANGLE_LIMIT = 11
ADDR_TORQUE_ENABLE = 40
ADDR_GOAL_POSITION = 42
ADDR_PRESENT_POSITION = 56
ADDR_PRESENT_VOLTAGE = 62
ADDR_PRESENT_TEMPERATURE = 63

TICKS_PER_REV = 4096
TICK_MIN = 0
TICK_MAX = 4095

#: SO-101 numeruje serwa od podstawy do chwytaka, czyli tak samo jak JOINT_NAMES.
DEFAULT_IDS: dict[str, int] = {name: index for index, name in enumerate(JOINT_NAMES, start=1)}

#: Ponizej tego napiecia serwo odpowiada (plytka zasila sie z USB), ale nie ma
#: z czego ruszyc. Warto powiedziec to wprost, zamiast pozwolic uzytkownikowi
#: szukac bledu w konfiguracji.
MIN_SANE_VOLTAGE = 9.0


def checksum(payload: bytes) -> int:
    """Suma kontrolna pakietu: negacja sumy wszystkiego po naglowku."""
    return (~sum(payload)) & 0xFF


class FeetechBus:
    """Warstwa protokolu na porcie szeregowym. Oddzielona, zeby dala sie testowac."""

    def __init__(self, port: str, baudrate: int = 1_000_000, timeout: float = 0.02):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self._link: Any = None

    # ------------------------------------------------------------------ port
    def open(self, link: Any = None) -> None:
        """Otwiera port. `link` podstawia gotowy obiekt (testy) zamiast pyserial."""
        if link is not None:
            self._link = link
            return
        try:
            import serial
        except ImportError as exc:  # pragma: no cover - zalezne od srodowiska
            raise RuntimeError(
                "Backend `feetech` wymaga biblioteki pyserial. "
                'Zainstaluj:  pip install -e ".[feetech]"'
            ) from exc
        self._link = serial.Serial(self.port, self.baudrate, timeout=self.timeout, write_timeout=1.0)
        self._link.reset_input_buffer()
        self._link.reset_output_buffer()

    def close(self) -> None:
        if self._link is not None:
            self._link.close()
            self._link = None

    @property
    def is_open(self) -> bool:
        return self._link is not None

    # -------------------------------------------------------------- transakcje
    def _send(self, dev_id: int, instruction: int, params: bytes = b"") -> None:
        body = bytes([dev_id, len(params) + 2, instruction]) + params
        self._link.reset_input_buffer()
        self._link.write(b"\xff\xff" + body + bytes([checksum(body)]))
        self._link.flush()

    def _receive(self, dev_id: int) -> bytes | None:
        """Odbiera odpowiedz i zwraca same dane (bez naglowka, dlugosci i sumy)."""
        head = self._link.read(4)
        if len(head) < 4 or head[0] != 0xFF or head[1] != 0xFF or head[2] != dev_id:
            return None
        rest = self._link.read(head[3])
        if len(rest) < head[3]:
            return None
        error = rest[0]
        if error:
            logger.debug("Serwo %d zglosilo blad 0x%02X", dev_id, error)
        return rest[1:-1]

    def ping(self, dev_id: int) -> bool:
        self._send(dev_id, INST_PING)
        return self._receive(dev_id) is not None

    def read(self, dev_id: int, addr: int, length: int) -> int | None:
        self._send(dev_id, INST_READ, bytes([addr, length]))
        data = self._receive(dev_id)
        if data is None or len(data) < length:
            return None
        return int.from_bytes(data[:length], "little")

    def write(self, dev_id: int, addr: int, value: int, length: int) -> None:
        self._send(dev_id, INST_WRITE, bytes([addr]) + value.to_bytes(length, "little"))
        self._receive(dev_id)

    def sync_write(self, addr: int, length: int, values: dict[int, int]) -> None:
        """Jeden pakiet dla wszystkich serw naraz - bez odpowiedzi, wiec bez czekania.

        To jest powod, dla ktorego petla sterowania nie placi za liczbe stawow:
        szesc celow kosztuje tyle, co jeden.
        """
        if not values:
            return
        params = bytearray([addr, length])
        for dev_id, value in sorted(values.items()):
            params.append(dev_id)
            params += value.to_bytes(length, "little")
        body = bytes([BROADCAST_ID, len(params) + 2, INST_SYNC_WRITE]) + bytes(params)
        self._link.write(b"\xff\xff" + body + bytes([checksum(body)]))
        self._link.flush()


class FeetechArm(RobotBackend):
    """Ramie SO-101 sterowane wprost przez port szeregowy."""

    def __init__(self, cfg: AppConfig, bus: FeetechBus | None = None):
        self.cfg = cfg
        rc = cfg.robot
        if not rc.port:
            raise ValueError(
                "Nie podano portu szeregowego ramienia. Uzyj --port COM11 (Windows) "
                "albo --port /dev/ttyACM0 (Linux)."
            )
        self.bus = bus or FeetechBus(rc.port, rc.baudrate)
        self.ids = dict(DEFAULT_IDS)
        self._connected = False
        self._positions: dict[str, float] = {}
        #: Limity odczytane z serw [tiki]. Pusty wpis = serwo nie ogranicza.
        self.servo_limits: dict[str, tuple[int, int]] = {}
        self.info = RobotInfo(
            name="SO-101 (feetech)",
            description=f"port {rc.port} @ {rc.baudrate} bd",
            simulated=False,
        )

    # ------------------------------------------------------------- przeliczenia
    def _to_ticks(self, name: str, value: float) -> int:
        rc = self.cfg.robot
        if name == GRIPPER:
            # Chwytak jedzie w 0..100 (0 = zwarty), a nie w stopniach.
            span = rc.gripper_open_ticks - rc.gripper_closed_ticks
            ticks = rc.gripper_closed_ticks + span * (value / 100.0)
        else:
            ticks = rc.center_ticks + value * TICKS_PER_REV / 360.0
        return int(round(min(max(ticks, TICK_MIN), TICK_MAX)))

    def _to_units(self, name: str, ticks: int) -> float:
        rc = self.cfg.robot
        if name == GRIPPER:
            span = rc.gripper_open_ticks - rc.gripper_closed_ticks
            if abs(span) < 1e-6:  # pragma: no cover - konfiguracja bez zakresu
                return 0.0
            return (ticks - rc.gripper_closed_ticks) * 100.0 / span
        return (ticks - rc.center_ticks) * 360.0 / TICKS_PER_REV

    # ---------------------------------------------------------------- polaczenie
    def connect(self) -> None:
        # Magistrala moze byc juz otwarta - tak wchodzi podstawiona w testach
        # albo wspoldzielona z diagnostyka. Powtorne otwarcie zerwaloby ja.
        if not self.bus.is_open:
            self.bus.open()

        missing = [name for name, dev_id in self.ids.items() if not self.bus.ping(dev_id)]
        if missing:
            self.bus.close()
            raise RuntimeError(
                f"Serwa nie odpowiadaja: {', '.join(missing)}. "
                "Sprawdz zasilanie ramienia i port szeregowy."
            )

        self._check_power()
        self._read_servo_limits()

        ticks = {name: self.bus.read(dev_id, ADDR_PRESENT_POSITION, 2) for name, dev_id in self.ids.items()}
        if any(value is None for value in ticks.values()):
            self.bus.close()
            raise RuntimeError("Nie udalo sie odczytac pozycji wszystkich stawow.")

        # KOLEJNOSC MA ZNACZENIE. Rejestr celu pamieta wartosc z poprzedniej
        # sesji, wiec samo zalaczenie momentu szarpneloby ramie do tamtej pozy.
        # Najpierw wpisujemy cel = tam, gdzie ramie faktycznie stoi.
        self.bus.sync_write(
            ADDR_GOAL_POSITION, 2, {self.ids[name]: int(value) for name, value in ticks.items()}
        )
        for dev_id in self.ids.values():
            self.bus.write(dev_id, ADDR_TORQUE_ENABLE, 1, 1)

        self._positions = {name: self._to_units(name, int(value)) for name, value in ticks.items()}
        self._connected = True
        logger.info(
            "Polaczono z ramieniem na %s. Pozycja startowa: %s",
            self.cfg.robot.port,
            ", ".join(f"{name}={value:.1f}" for name, value in self._positions.items()),
        )

    def _read_servo_limits(self) -> None:
        """Odczytuje wlasne limity serw i mowi glosno, gdy sa ciasniejsze niz konfiguracja.

        Bez tego aplikacja zadaje pozy, ktorych serwo nie wykona: przycina je
        po cichu do swojego zakresu i wraca w zupelnie inne miejsce. Widac to
        wtedy jako "staw nie slucha rozkazu", chociaz rozkaz doszedl poprawnie.
        """
        self.servo_limits = {}
        conflicts: list[str] = []
        for name, dev_id in self.ids.items():
            low = self.bus.read(dev_id, ADDR_MIN_ANGLE_LIMIT, 2)
            high = self.bus.read(dev_id, ADDR_MAX_ANGLE_LIMIT, 2)
            if low is None or high is None or low >= high:
                continue
            if low <= TICK_MIN and high >= TICK_MAX:
                continue  # pelny zakres = serwo niczego nie ogranicza
            self.servo_limits[name] = (low, high)

            jc = self.cfg.joint(name)
            servo_lo, servo_hi = self._to_units(name, low), self._to_units(name, high)
            if servo_lo > jc.min + 1.0 or servo_hi < jc.max - 1.0:
                conflicts.append(
                    f"{name}: serwo {servo_lo:.0f}..{servo_hi:.0f}, "
                    f"konfiguracja {jc.min:.0f}..{jc.max:.0f}"
                )

        if conflicts:
            logger.warning(
                "Serwa maja WLASNE limity kata, ciasniejsze niz `joints` w konfiguracji. "
                "Rozkazy poza nimi zostana przyciete przez serwo, a staw zatrzyma sie "
                "wczesniej, niz aplikacja zaklada:\n  %s",
                "\n  ".join(conflicts),
            )

    def _check_power(self) -> None:
        """Ostrzega, gdy ramie odpowiada, ale nie ma z czego ruszyc."""
        raw = self.bus.read(self.ids[JOINT_NAMES[0]], ADDR_PRESENT_VOLTAGE, 1)
        if raw is None:
            return
        volts = raw / 10.0
        if volts < MIN_SANE_VOLTAGE:
            logger.warning(
                "Napiecie zasilania %.1f V - serwa odpowiadaja, ale przy tym napieciu "
                "nie ruszy. Podlacz zasilacz ramienia.",
                volts,
            )
        else:
            logger.info("Zasilanie ramienia: %.1f V", volts)

    def disconnect(self) -> None:
        if not self._connected:
            self.bus.close()
            return
        # Momentu NIE wylaczamy: ramie zostaloby wtedy wiotkie i opadlo pod
        # wlasnym ciezarem. Zostaje tam, gdzie stanelo.
        if self.cfg.robot.torque_off_on_exit:
            for dev_id in self.ids.values():
                try:
                    self.bus.write(dev_id, ADDR_TORQUE_ENABLE, 0, 1)
                except Exception:  # pragma: no cover - rozlaczenie nie moze wysypac aplikacji
                    logger.exception("Nie udalo sie wylaczyc momentu serwa %d", dev_id)
        self._connected = False
        self.bus.close()

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------ wymiana
    def read_joints(self) -> dict[str, float]:
        if not self._connected:
            raise RuntimeError("Ramie nie jest polaczone")
        for name, dev_id in self.ids.items():
            ticks = self.bus.read(dev_id, ADDR_PRESENT_POSITION, 2)
            if ticks is not None:
                # Odczyt, ktory sie nie udal, zostawia poprzednia wartosc -
                # jedna zgubiona ramka nie ma prawa udawac skoku stawu.
                self._positions[name] = self._to_units(name, ticks)
        return dict(self._positions)

    def send_joints(self, targets: dict[str, float]) -> dict[str, float]:
        if not self._connected:
            raise RuntimeError("Ramie nie jest polaczone")
        sent: dict[str, float] = {}
        ticks: dict[int, int] = {}
        for name in JOINT_NAMES:
            if name not in targets:
                continue
            jc = self.cfg.joint(name)
            value = min(max(float(targets[name]), jc.min), jc.max)
            tick = self._to_ticks(name, value)

            # Serwo i tak przytnie rozkaz do swojego zakresu. Robiac to tutaj,
            # zwracamy wyzej wartosc, ktora NAPRAWDE pojechala - inaczej reszta
            # aplikacji (limity predkosci, HUD, podglad 3D) liczylaby na pozycji,
            # ktorej ramie nigdy nie osiagnelo.
            limits = self.servo_limits.get(name)
            if limits is not None:
                tick = min(max(tick, limits[0]), limits[1])
                value = self._to_units(name, tick)

            ticks[self.ids[name]] = tick
            sent[name] = value
        self.bus.sync_write(ADDR_GOAL_POSITION, 2, ticks)
        return sent
