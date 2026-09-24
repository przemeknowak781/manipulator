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

from ..config import AppConfig, JOINT_NAMES
from .base import RobotBackend, RobotInfo

logger = logging.getLogger(__name__)

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

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------- wymiana IO
    def read_joints(self) -> dict[str, float]:
        if self._robot is None:
            raise RuntimeError("Robot nie jest polaczony")
        observation = self._robot.get_observation()
        return {
            key.removesuffix(".pos"): float(value)
            for key, value in observation.items()
            if key.endswith(".pos")
        }

    def send_joints(self, targets: dict[str, float]) -> dict[str, float]:
        if self._robot is None:
            raise RuntimeError("Robot nie jest polaczony")
        known = set(self.motor_names())
        action = {f"{name}.pos": float(value) for name, value in targets.items() if name in known}
        sent = self._robot.send_action(action)
        return {
            key.removesuffix(".pos"): float(value)
            for key, value in (sent or action).items()
            if key.endswith(".pos")
        }
