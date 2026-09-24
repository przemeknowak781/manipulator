"""Polityka na zywym blizniaku - na symulacji albo na prawdziwym ramieniu.

    runner = PolicyRunner(twin, Policy.load(path))
    runner.goal = np.array([0.25, 0.05, 0.10])      # reach: cel w ukladzie podstawy
    runner.start()                                  # watek z czestotliwoscia polityki
    ...
    runner.stop()                                   # ramie zostaje tam, gdzie stoi

Petla robi dokladnie to, co srodowisko w treningu - obserwacja z `task.observe`,
akcja przez `task.apply_action` - tylko katy nie pochodza z fizyki, a z serw
(`Twin.status.measured`). Cel stawow NIE idzie do serw wprost: trafia do
`Twin.set_target`, a stamtad przez `SafetySupervisor` (limity pozycji
i predkosci, STOP), tak samo jak kazdy inny rozkaz w tej aplikacji.

Straznicy, ktorych srodowisko treningowe nie ma, a prawdziwe ramie potrzebuje:

* **stop przy rozjezdzie** - gdy zmierzona poza odjezdza od celu o wiecej niz
  `max_tracking_deg` (staw zablokowany, kolizja, serwo nie nadaza), polityka
  sie zatrzymuje, zamiast dociskac dalej;
* **limit czasu** - epizod na ramieniu trwa tyle, ile w treningu.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from . import task as tk
from .policy import Policy

logger = logging.getLogger(__name__)


class NoCube(RuntimeError):
    """Zadanie z kostka, a percepcja nie wie, gdzie ona jest."""


@dataclass
class RunnerStatus:
    running: bool = False
    step: int = 0
    hz: float = 0.0
    distance: float = float("nan")
    success: bool = False
    stopped_because: str = ""
    last_action: list[float] = field(default_factory=list)


class PolicyRunner:
    def __init__(self, twin, policy: Policy, *, max_tracking_deg: float = 25.0,
                 cube_provider: Callable[[], tuple[np.ndarray, np.ndarray] | None] | None = None,
                 episode_limit: bool = True):
        from ..kinematics import RobotKinematics

        self.twin = twin
        self.policy = policy
        self.task = policy.task
        # Wlasna kinematyka (wlasne MjData): ta ze sceny liczy tez panel (IK uchwytu)
        # z innego watku, a `tcp()` pisze do swojego MjData.
        self.kin = RobotKinematics(twin.workspace.spec())
        self.limits = tk.Limits.of(self.kin)
        self.goal = np.array([0.22, 0.0, 0.12])
        #: lift: (polozenie (3,), obrot (3, 3)) kostki w ukladzie podstawy - z wizji albo z symulacji.
        self.cube_provider = cube_provider
        self.max_tracking = np.radians(max_tracking_deg)
        self.episode_limit = episode_limit
        #: Ile sekund ramie stoi, czekajac na pierwsza pozycje kostki z kamer.
        self.wait_for_cube = 3.0
        self.status = RunnerStatus()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.q_cmd = self.limits.home.copy()
        self.prev_action = np.zeros(6)

    # ------------------------------------------------------------ obserwacja
    def observation(self, measured: dict[str, float]) -> np.ndarray:
        q = self.kin.to_q(measured)
        tcp = self.kin.tcp(measured)[:3, 3]
        cube_pos, cube_rot = np.zeros(3), np.eye(3)
        if self.task.name == "lift":
            got = self.cube_provider() if self.cube_provider else None
            if got is None:
                raise NoCube("zadanie lift potrzebuje polozenia kostki - kamery jej nie widza")
            cube_pos, cube_rot = got
            if self.policy.meta.randomization.get("fold_yaw", False):
                # W treningu polityka widziala obrot zlozony do +-45 st. (jak z detektora);
                # poza "w dloni" i prawda z symulacji tego nie maja - skladamy tak samo.
                cube_rot = tk.fold_yaw(np, np.asarray(cube_rot, float))
        obs = tk.observe(np, self.task, self.limits, q[None], self.q_cmd[None], tcp[None], self.prev_action[None],
                         goal=self.goal[None], cube_pos=cube_pos[None], cube_rot=cube_rot[None])
        return obs[0].astype(np.float32)

    # ------------------------------------------------------------------ petla
    def start(self) -> None:
        if not self.twin.connected:
            raise RuntimeError("ramie blizniaka nie jest polaczone")
        self.stop()
        measured = self.twin.joints() if self.twin.status.simulated else dict(self.twin.status.measured)
        self.q_cmd = np.clip(self.kin.to_q(measured), self.limits.lo, self.limits.hi)
        self.prev_action = np.zeros(6)
        self.status = RunnerStatus(running=True)
        self._stop.clear()
        self.twin.set_engaged(True)
        self._thread = threading.Thread(target=self._loop, name="polityka", daemon=True)
        self._thread.start()

    def stop(self, reason: str = "") -> None:
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=2.0)
        self._thread = None
        if self.status.running:
            self.status.running = False
            self.status.stopped_because = reason or "zatrzymana"
            self.twin.set_engaged(False)          # ramie trzyma pozycje

    def _loop(self) -> None:
        period = 1.0 / self.task.control_hz
        t_hz, ticks = time.monotonic(), 0
        t_start = time.monotonic()
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                measured = dict(self.twin.status.measured) or self.twin.joints()
                q = self.kin.to_q(measured)
                lag = np.abs(q[:5] - self.q_cmd[:5]).max()
                if lag > self.max_tracking:
                    self._halt(f"ramie nie nadaza za celem ({np.degrees(lag):.0f} st.) - kolizja albo blokada")
                    return
                obs = self.observation(measured)
                action = self.policy.act(obs)
                self.q_cmd = tk.apply_action(np, self.task, self.limits, self.q_cmd[None], action[None])[0]
                self.twin.set_target(self.kin.from_q(self.q_cmd))
                self.prev_action = action
                st = self.status
                st.step += 1
                st.stopped_because = ""
                st.last_action = action.tolist()
                if self.task.name == "reach":
                    st.distance = float(np.linalg.norm(self.goal - self.kin.tcp(measured)[:3, 3]))
                    st.success = st.distance < self.task.success_dist
                ticks += 1
                now = time.monotonic()
                if now - t_hz >= 1.0:
                    st.hz, ticks, t_hz = ticks / (now - t_hz), 0, now
                if self.episode_limit and st.step >= self.task.episode_steps:
                    self._halt("koniec epizodu")
                    return
            except NoCube as exc:
                # Na starcie percepcja moze jeszcze nie miec pierwszej detekcji - ramie
                # stoi i czekamy; zgubiona kostka W TRAKCIE epizodu zatrzymuje polityke.
                if self.status.step == 0 and time.monotonic() - t_start < self.wait_for_cube:
                    self.status.stopped_because = "czekam na kostke z kamer"
                    time.sleep(period)
                    continue
                self._halt(str(exc))
                return
            except Exception as exc:                        # ramie sie rozlaczylo, blad percepcji...
                logger.exception("Petla polityki przerwana")
                self._halt(f"{type(exc).__name__}: {exc}")
                return
            time.sleep(max(0.0, period - (time.monotonic() - t0)))

    def _halt(self, reason: str) -> None:
        self.status.running = False
        self.status.stopped_because = reason
        self.twin.set_engaged(False)
        self._stop.set()
        logger.info("Polityka zatrzymana: %s", reason)
