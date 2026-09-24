"""Sterowanie prawdziwym ramieniem SO-101 przez biblioteke LeRobot.

Uklad modulow w LeRobot zmienial sie miedzy wydaniami (0.4 scalilo SO-100 i
SO-101 w `lerobot.robots.so_follower`, wczesniej byl osobny `so101_follower`,
a jeszcze wczesniej caly pakiet siedzial pod `lerobot.common`). Zamiast zakladac
jedna wersje, szukamy klas po kolei we wszystkich znanych lokalizacjach.
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import logging
from typing import Any

from ..config import AppConfig, GRIPPER, JOINT_NAMES
from .base import RobotBackend, RobotInfo
from .feetech import describe_servo_error

logger = logging.getLogger(__name__)

#: Co tyle odczytow pozycji odczyt rejestru Status serw (bity ochrony: przeciazenie,
#: przegrzanie, napiecie). `sync_read` LeRobota nie oddaje bajtu bledu odpowiedzi,
#: wiec bez tego serwo w ochronie (zwolniony moment, dalej grzecznie odpowiada)
#: wygladalo na zdrowe, a fala i identyfikacja jechaly dalej reszta ramienia.
#: Przy odczycie pozycji 10 Hz to jeden krotki SYNC READ co 0,3 s - petla tego nie czuje.
#: Liczone w odczytach, nie w sekundach: blizniak bez watku (testy) ma wlasny zegar,
#: a zegar sciany stal wtedy w miejscu i Status nie byl czytany wcale.
STATUS_EVERY_READS = 3

#: Tyle KOLEJNYCH nieudanych odczytow Status (~1,5 s przy 10 Hz) = ochrona serw niewidoczna,
#: co jest usterka sama w sobie. Pojedyncza czkawka magistrali nie zatrzymuje ramienia.
STATUS_FAIL_LIMIT = 5

#: Tiki na obrot STS3215, gdy magistrala LeRobota nie poda wlasnej tabeli rozdzielczosci.
DEFAULT_RESOLUTION = 4096

#: O tyle tikow (~1,8 st.) zero stawu LeRobota (srodek zakresu z kalibracji) moze
#: odbiegac od zera, ktore zaklada blizniak (`robot.center_ticks`), zanim ostrzezemy.
ZERO_TOLERANCE_TICKS = 20

#: Kolejne lokalizacje klas SO-follower - od najnowszej do najstarszej.
_MODULE_CANDIDATES: tuple[str, ...] = (
    "lerobot.robots.so_follower",
    "lerobot.robots.so101_follower",
    "lerobot.robots.so100_follower",
    "lerobot.common.robots.so101_follower",
    "lerobot.common.robots.so100_follower",
)


class LeRobotArm(RobotBackend):
    """Adapter na `SO101Follower` / `SO100Follower` z LeRobot."""

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self._robot: Any = None
        self._connected = False
        kind = cfg.robot.kind.lower()
        if kind not in ("so101", "so100"):
            raise ValueError(f"Nieznany typ ramienia: {cfg.robot.kind!r} (so101|so100)")
        self._kind = kind
        self._reset_status()
        self.info = RobotInfo(
            name=f"{kind.upper()} (LeRobot)",
            description=f"port {cfg.robot.port}",
            simulated=False,
        )

    # ------------------------------------------------------------- importy
    @staticmethod
    def _resolve_classes(kind: str) -> tuple[Any, Any, str]:
        """Znajduje (klasa_robota, klasa_konfiguracji, nazwa_modulu) w LeRobot."""
        robot_names = [f"{kind.upper()}Follower", "SOFollower"]
        config_names = [f"{kind.upper()}FollowerConfig", "SOFollowerRobotConfig", "SOFollowerConfig"]

        errors: list[str] = []
        for module_name in _MODULE_CANDIDATES:
            try:
                module = importlib.import_module(module_name)
            except ImportError as exc:
                errors.append(f"{module_name}: {exc}")
                continue

            robot_cls = next((getattr(module, n) for n in robot_names if hasattr(module, n)), None)
            config_cls = next((getattr(module, n) for n in config_names if hasattr(module, n)), None)
            if robot_cls is not None and config_cls is not None:
                return robot_cls, config_cls, module_name
            errors.append(f"{module_name}: brak klas {robot_names} / {config_names}")

        raise ImportError(
            "Nie znalazlem klas SO-follower w zainstalowanej wersji LeRobot.\n"
            "Zainstaluj biblioteke:  pip install 'lerobot[feetech]'\n"
            "Szczegoly prob importu:\n  " + "\n  ".join(errors)
        )

    def _build_config(self, config_cls: Any) -> Any:
        """Sklada obiekt konfiguracji, pomijajac pola nieznane danej wersji."""
        rc = self.cfg.robot
        if not rc.port:
            raise ValueError(
                "Nie podano portu szeregowego ramienia. Uzyj --port /dev/ttyACM0 "
                "(Linux), /dev/tty.usbmodem* (macOS) albo COM5 (Windows)."
            )

        wanted: dict[str, Any] = {
            "port": rc.port,
            "id": rc.robot_id,
            "use_degrees": rc.use_degrees,
            "max_relative_target": rc.max_relative_target,
            # LeRobot domyslnie zdejmuje moment przy rozlaczeniu - ramie opada pod
            # wlasnym ciezarem. U nas decyduje o tym ta sama opcja co w `feetech`.
            "disable_torque_on_disconnect": rc.torque_off_on_exit,
        }
        if rc.calibration_dir:
            from pathlib import Path

            wanted["calibration_dir"] = Path(rc.calibration_dir)

        available = {f.name for f in dataclasses.fields(config_cls)}
        kwargs = {k: v for k, v in wanted.items() if k in available}
        skipped = sorted(set(wanted) - set(kwargs))
        if skipped:
            logger.warning(
                "Ta wersja LeRobot nie obsluguje pol %s - pomijam. "
                "Sprawdz, czy jednostki stawow zgadzaja sie z konfiguracja.",
                skipped,
            )
        return config_cls(**kwargs)

    # ------------------------------------------------------------ polaczenie
    def connect(self) -> None:
        robot_cls, config_cls, module_name = self._resolve_classes(self._kind)
        logger.info("Uzywam klas LeRobot z modulu %s", module_name)

        config = self._build_config(config_cls)
        robot = robot_cls(config)
        self._require_calibration_file(robot)
        logger.info("Lacze z ramieniem na porcie %s ...", self.cfg.robot.port)
        self._connect_without_calibration(robot)
        self._robot = robot
        self._connected = True
        self._reset_status()
        self._warn_about_joint_zero()

        motors = self.motor_names()
        missing = [n for n in JOINT_NAMES if n not in motors]
        if missing:
            logger.warning(
                "Robot nie raportuje stawow %s - beda pomijane przy wysylaniu.", missing
            )
        logger.info("Polaczono. Stawy: %s", ", ".join(motors))

    def _calibrate_hint(self) -> str:
        rc = self.cfg.robot
        return (f"Skalibruj ramie w konsoli:  lerobot-calibrate --robot.type={self._kind}_follower "
                f"--robot.port={rc.port} --robot.id={rc.robot_id}  - albo uzyj backendu `feetech`.")

    def _require_calibration_file(self, robot: Any) -> None:
        """Odmawia polaczenia, zanim cokolwiek dotknie portu, gdy LeRobot nie ma pliku kalibracji.

        Bez pliku `SOFollower.connect()` wola `calibrate()`: zdejmuje moment ze
        wszystkich serw (ramie trzymane przez poprzednia sesje opada na stol)
        i czeka na `input()` w konsoli serwera - w panelu nic nie widac, a watek
        wisi. Dokonczona kalibracja nadpisalaby jeszcze Homing_Offset i limity
        w EEPROM, czyli zero i zakresy, na ktorych stoi backend `feetech`.
        """
        if getattr(robot, "calibration", None):
            return
        where = getattr(robot, "calibration_fpath", None) or "katalog kalibracji LeRobota"
        raise RuntimeError(
            f"Brak kalibracji LeRobota dla ramienia '{self.cfg.robot.robot_id}' ({where}). "
            "Bez niej LeRobot przy laczeniu zdejmuje moment ze wszystkich serw i czeka na Enter "
            "w konsoli. " + self._calibrate_hint()
        )

    def _connect_without_calibration(self, robot: Any) -> None:
        """Laczy tak, zeby LeRobot nigdy nie zdjal momentu ani nie zapytal o nic w konsoli."""
        bus = getattr(robot, "bus", None)
        if bus is not None and hasattr(bus, "connect") and hasattr(bus, "is_calibrated"):
            bus.connect()
            try:
                calibrated = bool(bus.is_calibrated)
                holding = calibrated and self._torque_on(bus)
            except Exception:
                self._close_bus_keeping_torque(bus)
                raise
            if not calibrated:
                self._close_bus_keeping_torque(bus)
                raise RuntimeError(
                    "Kalibracja LeRobota w pliku nie zgadza sie z zapisana w serwach (inne ramie "
                    "albo kalibracja zmieniona gdzie indziej). " + self._calibrate_hint()
                )
            if holding:
                # `configure()` LeRobota robi swoje zapisy pod `torque_disabled()` - przy
                # ramieniu trzymajacym poze (np. zaraz po sesji `feetech`) to zdjety moment
                # na czas kilkudziesieciu transakcji, a przez most sekunda swobodnego spadku.
                # Ustawienia z poprzednich polaczen LeRobota i tak siedza w EEPROM serw.
                logger.warning("Serwa trzymaja pozycje - pomijam konfiguracje LeRobota, "
                               "zeby nie zdejmowac momentu (ustawienia zostaja z EEPROM).")
                return
            self._close_bus_keeping_torque(bus)

        if "calibrate" not in inspect.signature(robot.connect).parameters:
            raise RuntimeError(
                "Ta wersja LeRobota nie pozwala polaczyc bez interaktywnej kalibracji - "
                "zaktualizuj ja albo uzyj backendu `feetech`."
            )
        robot.connect(calibrate=False)
        if not robot.is_calibrated:
            try:
                robot.disconnect()
            except Exception:  # pragma: no cover - rozlaczenie nie moze przykryc wlasciwego bledu
                logger.exception("Blad przy rozlaczaniu robota")
            raise RuntimeError(
                "Kalibracja LeRobota w pliku nie zgadza sie z zapisana w serwach. " + self._calibrate_hint()
            )

    @staticmethod
    def _torque_on(bus: Any) -> bool:
        """Czy ktores serwo trzyma moment. W razie watpliwosci - tak (bezpieczniej)."""
        try:
            return any(int(v) for v in bus.sync_read("Torque_Enable").values())
        except Exception:
            logger.debug("Nie udalo sie odczytac Torque_Enable - zakladam, ze serwa trzymaja", exc_info=True)
            return True

    @staticmethod
    def _close_bus_keeping_torque(bus: Any) -> None:
        try:
            bus.disconnect(disable_torque=False)
        except Exception:  # pragma: no cover - rozlaczenie nie moze przykryc wlasciwego bledu
            logger.exception("Blad przy zamykaniu magistrali LeRobota")

    def motor_names(self) -> list[str]:
        """Nazwy stawow zgloszone przez LeRobot (kolejnosc jak w sterowniku)."""
        if self._robot is None:
            return list(JOINT_NAMES)
        try:
            return [key.removesuffix(".pos") for key in self._robot.action_features]
        except Exception:  # pragma: no cover - zalezne od wersji LeRobot
            return list(JOINT_NAMES)

    def disconnect(self) -> None:
        if self._robot is not None and self._connected:
            try:
                self._robot.disconnect()
            except Exception:  # pragma: no cover - rozlaczenie nie moze wysypac aplikacji
                logger.exception("Blad przy rozlaczaniu robota")
        self._connected = False
        self._robot = None
        self._reset_status()

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------- wymiana IO
    def read_joints(self) -> dict[str, float]:
        if self._robot is None:
            raise RuntimeError("Robot nie jest polaczony")
        observation = self._robot.get_observation()
        self._poll_status()
        return {
            key.removesuffix(".pos"): float(value)
            for key, value in observation.items()
            if key.endswith(".pos")
        }

    def send_joints(self, targets: dict[str, float]) -> dict[str, float]:
        if self._robot is None:
            raise RuntimeError("Robot nie jest polaczony")
        known = set(self.motor_names())
        # W stopniach LeRobot nie przycina celu (przycina tylko skale -100..100 i 0..100),
        # a serwo robi to po cichu do Min/Max_Position_Limit z EEPROM. Przycinamy tutaj,
        # zeby wyzej wrocilo to, co NAPRAWDE pojechalo - inaczej nadzor, polityka
        # i identyfikacja liczyly na pozycji, ktorej ramie nigdy nie osiagnelo (jak w `feetech`).
        limits = self.joint_limits()
        action: dict[str, float] = {}
        for name, value in targets.items():
            if name not in known:
                continue
            value = float(value)
            if name in limits:
                value = min(max(value, limits[name][0]), limits[name][1])
            action[f"{name}.pos"] = value
        sent = self._robot.send_action(action)
        return {
            key.removesuffix(".pos"): float(value)
            for key, value in (sent or action).items()
            if key.endswith(".pos")
        }

    # ------------------------------------------------------------ stan serw
    def _reset_status(self) -> None:
        #: Ostatnio odczytany rejestr Status kazdego serwa {staw: bity}.
        self._status: dict[str, int] = {}
        #: Odczyty pozycji do nastepnego odczytu Status (0 = przy najblizszym).
        self._reads_to_status = 0
        #: KOLEJNE nieudane odczyty Status i ostatni blad (do komunikatu).
        self._status_failures = 0
        self._status_error = ""

    def _poll_status(self) -> None:
        """Czyta rejestr Status wszystkich serw co `STATUS_EVERY_READS` odczytow pozycji. Nigdy nie rzuca.

        Rejestr ma te same bity, co bajt bledu odpowiedzi (0x20 przeciazenie, 0x04
        przegrzanie...), a SYNC READ odpowiada nawet serwo w ochronie. Nieudany odczyt
        zostawia poprzedni stan - bity przeciazenia nie znikaja przez jedna zgubiona ramke.
        """
        if self._reads_to_status > 0:
            self._reads_to_status -= 1
            return
        self._reads_to_status = STATUS_EVERY_READS - 1
        try:
            raw = self._robot.bus.sync_read("Status", normalize=False)
            self._status = {str(name): int(bits) for name, bits in raw.items()}
            self._status_failures, self._status_error = 0, ""
        except Exception as exc:
            self._status_failures += 1
            self._status_error = f"{type(exc).__name__}: {exc}"
            if self._status_failures == STATUS_FAIL_LIMIT:
                logger.warning("Rejestr Status serw nie odpowiada od %d prob: %s",
                               self._status_failures, self._status_error)
            else:
                logger.debug("Nieudany odczyt Status serw: %s", self._status_error)

    def faults(self) -> list[str]:
        """Bity ochrony z rejestru Status serw i utrata ich odczytu - zdania dla operatora.

        Te same zdania co w `feetech` ("staw (serwo N): przeciazenie"), wiec nadzor
        blizniaka zatrzymuje ramie tak samo na obu backendach.
        """
        try:
            out: list[str] = []
            ids = self._motor_ids()
            for name, bits in self._status.items():
                if bits:
                    out.append(describe_servo_error(name, ids.get(name, 0), bits))
            if self._connected and self._status_failures >= STATUS_FAIL_LIMIT:
                out.append(f"stan serw nieznany - {self._status_failures} nieudanych odczytow rejestru "
                           f"Status z rzedu, ochrona serw niewidoczna ({self._status_error})")
            return out
        except Exception as exc:  # pragma: no cover - wolane co cykl petli, nie moze jej wysypac
            logger.exception("Nie udalo sie sprawdzic stanu serw")
            return [f"nie udalo sie sprawdzic stanu serw: {exc}"]

    # ------------------------------------------------------------ kalibracja
    def _bus(self) -> Any:
        return getattr(self._robot, "bus", None) if self._robot is not None else None

    def _calibration(self) -> dict[str, Any]:
        """Kalibracja, ktora LeRobot naprawde stosuje (ta z magistrali), inaczej ta z pliku.

        Przy laczeniu `is_calibrated` sprawdzilo, ze zakresy z pliku sa rowne
        Min/Max_Position_Limit w EEPROM serw - wiec to jest tez to, do czego serwo przycina.
        """
        cal = getattr(self._bus(), "calibration", None) or getattr(self._robot, "calibration", None)
        return dict(cal) if cal else {}

    def _motors(self) -> dict[str, Any]:
        return dict(getattr(self._bus(), "motors", None) or {})

    def _motor_ids(self) -> dict[str, int]:
        return {str(name): int(getattr(m, "id", 0)) for name, m in self._motors().items()}

    def _norm_mode(self, name: str) -> str:
        """Tryb normalizacji stawu w LeRobot: DEGREES, RANGE_M100_100 albo RANGE_0_100."""
        mode = getattr(self._motors().get(name), "norm_mode", None)
        return str(getattr(mode, "name", mode) or "").upper()

    def _resolution(self, name: str) -> int:
        table = getattr(self._bus(), "model_resolution_table", None) or {}
        return int(table.get(getattr(self._motors().get(name), "model", None), DEFAULT_RESOLUTION))

    def _drive_inverted(self, cal: Any) -> bool:
        return bool(getattr(self._bus(), "apply_drive_mode", True)) and bool(getattr(cal, "drive_mode", 0))

    def _ticks_to_units(self, name: str, ticks: float) -> float | None:
        """Tiki serwa -> jednostki LeRobota, tym samym wzorem co `MotorsBus._normalize`
        (bez przycinania do zakresu). None = staw bez kalibracji albo w nieznanym trybie."""
        cal = self._calibration().get(name)
        if cal is None:
            return None
        lo, hi = float(cal.range_min), float(cal.range_max)
        if hi == lo:
            return None
        mode = self._norm_mode(name)
        if mode == "DEGREES":
            # Zero stawu w stopniach to SRODEK zakresu z kalibracji; drive_mode tu nie dziala.
            return (ticks - (lo + hi) / 2.0) * 360.0 / (self._resolution(name) - 1)
        if mode == "RANGE_M100_100":
            norm = (ticks - lo) / (hi - lo) * 200.0 - 100.0
            return -norm if self._drive_inverted(cal) else norm
        if mode == "RANGE_0_100":
            norm = (ticks - lo) / (hi - lo) * 100.0
            return 100.0 - norm if self._drive_inverted(cal) else norm
        return None

    def joint_limits(self) -> dict[str, tuple[float, float]]:
        """Zakres z kalibracji LeRobota (= Min/Max_Position_Limit w EEPROM) w jednostkach aplikacji.

        Tylko stawy, ktore serwo naprawde ogranicza - pelny obrot (0..4095, zwykle
        wrist_roll) jest pomijany jak w `feetech`. Bez tego nadzor liczyl na cele
        spoza zakresu, ktore serwo po cichu przycinalo.
        """
        out: dict[str, tuple[float, float]] = {}
        for name, cal in self._calibration().items():
            lo, hi = int(cal.range_min), int(cal.range_max)
            if lo >= hi or (lo <= 0 and hi >= self._resolution(name) - 1):
                continue
            a, b = self._ticks_to_units(name, lo), self._ticks_to_units(name, hi)
            if a is None or b is None:
                continue
            out[str(name)] = (min(a, b), max(a, b))
        return out

    def gripper_ticks(self) -> tuple[float, float, float] | None:
        """(zamkniety, otwarty, zero) chwytaka w tikach wedlug kalibracji LeRobota.

        LeRobot skaluje chwytak 0..100 po `range_min..range_max` z kalibracji
        (`drive_mode` odwraca skale), a nie po `gripper_closed/open_ticks` z konfiguracji,
        z ktorych blizniak liczy kat szczeki. Zero = pol obrotu (2047): tam kalibracja
        LeRobota (`set_half_turn_homings`) stawia poze srodkowa ramienia. Rejestr pozycji
        jest ten sam co w `feetech` (Homing_Offset stosuje serwo), wiec tiki sa porownywalne.
        """
        cal = self._calibration().get(GRIPPER)
        if cal is None or self._norm_mode(GRIPPER) != "RANGE_0_100":
            return None
        lo, hi = float(cal.range_min), float(cal.range_max)
        closed, opened = (hi, lo) if self._drive_inverted(cal) else (lo, hi)
        return closed, opened, float((self._resolution(GRIPPER) - 1) // 2)

    def joint_zero_offsets(self) -> dict[str, float]:
        """O ile stopni zero stawu LeRobota lezy od zera blizniaka: {staw: stopnie} (ponad tolerancje).

        W stopniach LeRobot liczy kat od SRODKA zakresu z kalibracji, a blizniak
        (kinematyka, polityki) od `robot.center_ticks` - tak jak backend `feetech`.
        Dla typowej kalibracji SO-101 to kilka stopni na stawie, dla zakresu nagranego
        tylko w jedna strone (limity ramienia nr 1: wrist_flex 2025..2858) ~35 st.
        Nie przeliczamy tego po cichu - zmienilyby sie jednostki zapisow zgodnych z LeRobotem.
        """
        center = float(self.cfg.robot.center_ticks)
        out: dict[str, float] = {}
        for name, cal in self._calibration().items():
            if self._norm_mode(name) != "DEGREES":
                continue
            mid = (float(cal.range_min) + float(cal.range_max)) / 2.0
            if abs(mid - center) > ZERO_TOLERANCE_TICKS:
                out[str(name)] = (mid - center) * 360.0 / (self._resolution(name) - 1)
        return out

    def calibration_warnings(self) -> list[str]:
        """Niefatalne zastrzezenia do kalibracji tego ramienia - zdania dla panelu. Nigdy nie rzuca."""
        try:
            off = self.joint_zero_offsets()
            if not off:
                return []
            shifts = ", ".join(f"{name} {deg:+.1f} st." for name, deg in off.items())
            return [f"zero stawow w kalibracji LeRobota (srodek zakresu) nie jest zerem blizniaka "
                    f"(tik {int(self.cfg.robot.center_ticks)}) - katy z backendu `lerobot` sa przesuniete "
                    f"wzgledem modelu ({shifts}) - do pracy z blizniakiem uzyj backendu `feetech`."]
        except Exception as exc:  # pragma: no cover - sama diagnostyka nie moze zablokowac polaczenia
            logger.debug("Nie udalo sie porownac zera stawow z kalibracja LeRobota", exc_info=True)
            return [f"nie udalo sie sprawdzic kalibracji LeRobota: {exc}"]

    def _warn_about_joint_zero(self) -> None:
        for text in self.calibration_warnings():
            logger.warning(text)

