"""Nadzor bezpieczenstwa - ostatnie ogniwo przed wyslaniem rozkazu do serw.

Zasada: *nic* nie trafia do robota z pominieciem tej klasy. Nawet gdyby
mapowanie albo detekcja dloni zwarialy, tutaj cel jest przycinany do limitow
stawow i ograniczany co do predkosci, a brak dloni zamraza ruch.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

from ..config import AppConfig, JOINT_NAMES
from .filters import RateLimiter

logger = logging.getLogger(__name__)


class SafetyState(str, Enum):
    """Stan maszyny bezpieczenstwa."""

    STARTING = "STARTING"      # plynne dojscie do pozycji domowej po starcie
    IDLE = "IDLE"              # gotowy, sterowanie nieaktywne
    ACTIVE = "ACTIVE"          # robot podaza za dlonia
    HOLDING = "HOLDING"        # dlon zgubiona / sprzeglo rozlaczone - stoimy
    HOMING = "HOMING"          # powrot do pozycji domowej
    ESTOP = "ESTOP"            # zatrzymanie awaryjne


@dataclass
class SafetyReport:
    """Co nadzor zrobil w tym kroku - do wyswietlenia na HUD."""

    state: SafetyState = SafetyState.IDLE
    #: Stawy, ktore uderzyly w limit pozycji.
    at_limit: list[str] = field(default_factory=list)
    #: Stawy, ktorych zadany ruch byl ograniczony limitem predkosci.
    rate_limited: list[str] = field(default_factory=list)
    seconds_without_hand: float = 0.0


def _smoothstep(t: float) -> float:
    """Gladkie przejscie 0->1 z zerowa predkoscia na koncach."""
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


class SafetySupervisor:
    """Limity pozycji i predkosci, watchdog dloni, zatrzymanie awaryjne."""

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.state = SafetyState.STARTING
        self._limiters: dict[str, RateLimiter] = {}
        self._command: dict[str, float] = {}
        self._ramp_from: dict[str, float] = {}
        self._ramp_to: dict[str, float] = {}
        self._ramp_t = 0.0
        self._ramp_duration = max(cfg.safety.startup_ramp_s, 1e-3)
        self._no_hand_for = 0.0
        self._estop = False
        self._was_active = False
        self.home = self._clamped_home()

    # ------------------------------------------------------------- pomocnicze
    def _clamped_home(self) -> dict[str, float]:
        home = {}
        for name in JOINT_NAMES:
            jc = self.cfg.joint(name)
            value = self.cfg.safety.home.get(name, 0.0)
            home[name] = min(max(value, jc.min), jc.max)
        return home

    def _max_vel(self, name: str) -> float:
        return max(self.cfg.joint(name).max_vel * self.cfg.safety.velocity_scale, 1e-3)

    # ------------------------------------------------------------------ start
    def start(self, measured: dict[str, float], go_home: bool = True) -> None:
        """Inicjalizuje nadzor aktualna, *zmierzona* pozycja robota.

        Startujemy dokladnie tam, gdzie ramie faktycznie stoi, i dopiero stamtad
        plynnie jedziemy do pozycji domowej - inaczej pierwszy rozkaz bylby
        skokiem o kilkadziesiat stopni. `go_home=False` zostawia ramie tam,
        gdzie stoi - panel blizniaka laczy sie bez ruchu i ma osobny przycisk.
        """
        self._command = {}
        self._limiters = {}
        for name in JOINT_NAMES:
            jc = self.cfg.joint(name)
            value = min(max(float(measured.get(name, self.home[name])), jc.min), jc.max)
            self._command[name] = value
            self._limiters[name] = RateLimiter(self._max_vel(name), value)

        self._ramp_from = dict(self._command)
        self._ramp_to = dict(self.home)
        self._ramp_t = 0.0
        self._no_hand_for = 0.0
        self._was_active = False
        if not go_home:
            self.state = SafetyState.IDLE
            logger.info("Nadzor wystartowal - ramie zostaje tam, gdzie stoi.")
            return
        self.state = SafetyState.STARTING
        logger.info("Nadzor wystartowal, plynne dojscie do pozycji domowej (%.1f s).", self._ramp_duration)

    # ------------------------------------------------------------------- krok
    def step(
        self,
        desired: dict[str, float] | None,
        dt: float,
        *,
        hand_present: bool,
        engaged: bool,
    ) -> tuple[dict[str, float], SafetyReport]:
        """Zwraca bezpieczny rozkaz dla robota oraz raport dla HUD."""
        if not self._command:
            raise RuntimeError("Wywolaj SafetySupervisor.start() przed step()")

        dt = max(dt, 0.0)
        self._no_hand_for = 0.0 if hand_present else self._no_hand_for + dt

        target = self._resolve_target(desired, dt, hand_present=hand_present, engaged=engaged)

        report = SafetyReport(state=self.state, seconds_without_hand=self._no_hand_for)
        command: dict[str, float] = {}
        for name in JOINT_NAMES:
            jc = self.cfg.joint(name)
            raw = float(target.get(name, self._command[name]))

            clamped = min(max(raw, jc.min), jc.max)
            if abs(clamped - raw) > 1e-6:
                report.at_limit.append(name)

            limiter = self._limiters[name]
            limiter.max_rate = self._max_vel(name)
            value = limiter(clamped, dt)
            if abs(value - clamped) > 1e-6:
                report.rate_limited.append(name)
            command[name] = value

        self._command = command
        return dict(command), report

    def _resolve_target(
        self,
        desired: dict[str, float] | None,
        dt: float,
        *,
        hand_present: bool,
        engaged: bool,
    ) -> dict[str, float]:
        """Maszyna stanow: decyduje, *dokad* w ogole chcemy jechac."""
        if self._estop:
            self.state = SafetyState.ESTOP
            return dict(self._command)  # stoimy w miejscu

        if self.state in (SafetyState.STARTING, SafetyState.HOMING):
            self._ramp_t += dt
            alpha = _smoothstep(self._ramp_t / self._ramp_duration)
            target = {
                name: self._ramp_from[name] + alpha * (self._ramp_to[name] - self._ramp_from[name])
                for name in JOINT_NAMES
            }
            if self._ramp_t >= self._ramp_duration:
                self.state = SafetyState.IDLE
                self._was_active = False
            return target

        if engaged and desired is not None and hand_present:
            self.state = SafetyState.ACTIVE
            self._was_active = True
            return desired

        # Sterowanie nieaktywne: trzymamy pozycje...
        if self._no_hand_for >= self.cfg.safety.return_home_timeout_s > 0:
            # ...a po dluzszej nieobecnosci dloni wracamy do pozycji domowej.
            self.begin_homing()
            return dict(self._command)

        # HOLDING znaczy "sterowalismy i zgubilismy dlon", a nie "czekamy na
        # pierwsza dlon" - inaczej aplikacja startowalaby w stanie alarmowym.
        lost_hand = self._no_hand_for > 0 and self._was_active
        self.state = SafetyState.HOLDING if lost_hand else SafetyState.IDLE
        return dict(self._command)

    # ------------------------------------------------------------- sterowanie
    def begin_homing(self, duration: float | None = None) -> None:
        """Rozpoczyna plynny powrot do pozycji domowej."""
        self._ramp_from = dict(self._command)
        self._ramp_to = dict(self.home)
        self._ramp_duration = max(duration or self.cfg.safety.startup_ramp_s, 1e-3)
        self._ramp_t = 0.0
        self._no_hand_for = 0.0
        self._was_active = False
        self.state = SafetyState.HOMING

    def trigger_estop(self) -> None:
        self._estop = True
        self.state = SafetyState.ESTOP
        logger.warning("STOP AWARYJNY - ruch zatrzymany.")

    def clear_estop(self) -> None:
        if not self._estop:
            return
        self._estop = False
        self.state = SafetyState.IDLE
        logger.info("Stop awaryjny skasowany.")

    @property
    def estopped(self) -> bool:
        return self._estop

    @property
    def command(self) -> dict[str, float]:
        return dict(self._command)

    @property
    def is_homing_done(self) -> bool:
        return self.state not in (SafetyState.STARTING, SafetyState.HOMING)
