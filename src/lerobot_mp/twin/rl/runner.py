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

Polityka bierze ramie na wlasnosc (`Twin.claim("polityka")`): nie ruszy, gdy
jedzie fala albo identyfikacja, a Dom/STOP/ponowne polaczenie ja zatrzymuja.

Straznicy, ktorych srodowisko treningowe nie ma, a prawdziwe ramie potrzebuje:

* **stop przy rozjezdzie** - gdy zmierzona poza odjezdza od celu o wiecej niz
  `max_tracking_deg` (staw zablokowany, kolizja, serwo nie nadaza), polityka
  sie zatrzymuje, a ramie staje w ZMIERZONEJ pozie - nie dociska dalej;
* **limit czasu** - epizod na ramieniu trwa tyle, ile w treningu;
* **koniec po sukcesie** - `end_on_success` taktow sukcesu z rzedu konczy
  epizod (lift: kostka nad blatem), zamiast pozwalac polityce krecic ramieniem
  z kostka do limitow stawow przez reszte epizodu; sukces lift liczy sie tylko
  z pozy swiezej (kamery albo "w dloni"), nigdy z "ostatnie widziane", i tylko
  gdy szczeka trzyma (stoi, ciasniej rozkazana, nie domyka sie w trakcie serii);
* **cel reach w obszarze treningu** - cel spoza niego (pod blatem, za
  podstawa) jest rzutowany na obszar, z ktorego losowano cele w treningu.
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

#: Nazwa wlasciciela ramienia (`Twin.claim`).
OWNER = "polityka"

#: Zrodla pozy kostki (`CubeTracker.source`), z ktorych wolno liczyc sukces lift: swieza
#: detekcja z kamer, kostka niesiona w szczekach albo prawda symulacji.
FRESH_CUBE_SOURCES = ("kamery", "w dloni", "symulacja")


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
    #: Ile taktow z rzedu zadanie jest wykonane.
    success_streak: int = 0
    #: reach: cel przyciety do obszaru treningu (panel pokazuje, ze kula stoi gdzie indziej).
    goal_clamped: bool = False


def project_goal(task: tk.TaskConfig, goal: np.ndarray) -> np.ndarray:
    """Najblizszy punkt obszaru, z ktorego `sample_goals` losowal cele reach.

    Promien od osi podstawy w `goal_radius`, wysokosc w `goal_height`, x > 0,05 m.
    Polityka poza nim nigdy nie byla - cel pod blatem prowadzil szczeki w blat.
    """
    g = np.asarray(goal, float).copy()
    g[2] = np.clip(g[2], *task.goal_height)
    r = float(np.hypot(g[0], g[1]))
    r_new = float(np.clip(r, *task.goal_radius))
    bearing = float(np.arctan2(g[1], g[0])) if r > 1e-9 else 0.0
    # x > 0,05 przy promieniu r to kat od osi x najwyzej arccos(0,05 / r); margines 1 mm.
    b_max = float(np.arccos(min(1.0, 0.051 / r_new)))
    bearing = float(np.clip(bearing, -b_max, b_max))
    g[0], g[1] = r_new * np.cos(bearing), r_new * np.sin(bearing)
    return g


class PolicyRunner:
    def __init__(self, twin, policy: Policy, *, max_tracking_deg: float = 25.0,
                 cube_provider: Callable[[], tuple | None] | None = None,
                 episode_limit: bool = True, cube_source: Callable[[], str] | None = None):
        from ..kinematics import RobotKinematics

        self.twin = twin
        self.policy = policy
        self.task = policy.task
        # Wlasna kinematyka (wlasne MjData): ta ze sceny liczy tez panel (IK uchwytu)
        # z innego watku, a `tcp()` pisze do swojego MjData.
        self.kin = RobotKinematics(twin.workspace.spec())
        self.limits = tk.Limits.of(self.kin)
        self.status = RunnerStatus()
        self._goal = np.array([0.22, 0.0, 0.12])
        #: lift: (polozenie (3,), obrot (3, 3)) kostki w ukladzie podstawy - z wizji albo z symulacji.
        #: Moze zwrocic trzeci element: zrodlo pozy (`CubeTracker.source`), patrz `FRESH_CUBE_SOURCES`.
        self.cube_provider = cube_provider
        #: Zrodlo pozy kostki, gdy dostawca zwraca tylko (polozenie, obrot) - np.
        #: `lambda: tracker.source`. Bez obu na prawdziwym ramieniu sukces lift sie nie liczy.
        self.cube_source = cube_source
        self._cube_src: str | None = None
        self.max_tracking = np.radians(max_tracking_deg)
        self.episode_limit = episode_limit
        #: Ile sekund ramie stoi, czekajac na pierwsza pozycje kostki z kamer.
        self.wait_for_cube = 3.0
        end = getattr(self.task, "end_on_success", None)
        #: Po ilu taktach sukcesu z rzedu epizod sie konczy (0 = nigdy). Z zadania (K8);
        #: polityki zapisane przed tym polem - lift 10, reach 0 (cel przeciagany na zywo).
        self.end_on_success = int(end) if end is not None else (10 if self.task.name == "lift" else 0)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.q_cmd = self.limits.home.copy()
        self.prev_action = np.zeros(6)
        self._hw_lo, self._hw_hi = self.limits.lo, self.limits.hi
        self._t_start: float | None = None
        self._t_hz, self._ticks = 0.0, 0
        self._cube: np.ndarray | None = None
        #: Szczeka "trzyma" (sukces lift): jak w `CubeTracker` - stoi (< `grip_still` rad miedzy
        #: taktami), szerzej niz rozkaz o `grip_block_margin` i niz zamknieta o `grip_open_margin`.
        self.grip_still, self.grip_block_margin, self.grip_open_margin = 0.02, 0.05, 0.05
        #: W trakcie serii sukcesu szczeka nie domyka sie o wiecej niz tyle [rad] (~3 jednostki).
        self.grip_creep = 0.03
        self._grip_prev: float | None = None
        self._streak_grip = 0.0

    # ------------------------------------------------------------------ cel
    @property
    def goal(self) -> np.ndarray:
        return self._goal

    @goal.setter
    def goal(self, value) -> None:
        g = np.asarray(value, float)
        if self.task.name == "reach":
            p = project_goal(self.task, g)
            self.status.goal_clamped = bool(np.linalg.norm(p - g) > 1e-4)
            g = p
        self._goal = g

    # ------------------------------------------------------------ obserwacja
    def observation(self, measured: dict[str, float]) -> np.ndarray:
        q = self.kin.to_q(measured)
        tcp = self.kin.tcp(measured)[:3, 3]
        cube_pos, cube_rot = np.zeros(3), np.eye(3)
        if self.task.name == "lift":
            got = self.cube_provider() if self.cube_provider else None
            if got is None:
                raise NoCube("zadanie lift potrzebuje polozenia kostki - kamery jej nie widza")
            cube_pos, cube_rot = got[0], got[1]
            self._cube_src = str(got[2]) if len(got) > 2 else (self.cube_source() if self.cube_source else None)
            self._cube = np.asarray(cube_pos, float).copy()
            if self.policy.meta.randomization.get("fold_yaw", False):
                # W treningu polityka widziala obrot zlozony do +-45 st. (jak z detektora);
                # poza "w dloni" i prawda z symulacji tego nie maja - skladamy tak samo.
                cube_rot = tk.fold_yaw(np, np.asarray(cube_rot, float))
        obs = tk.observe(np, self.task, self.limits, q[None], self.q_cmd[None], tcp[None], self.prev_action[None],
                         goal=self.goal[None], cube_pos=cube_pos[None], cube_rot=cube_rot[None])
        return obs[0].astype(np.float32)

    # ------------------------------------------------------------------ petla
    def start(self, *, threaded: bool = True) -> None:
        """Bierze ramie i rusza. `threaded=False`: bez watku - takty robi `step_once`."""
        if not self.twin.connected:
            raise RuntimeError("ramie blizniaka nie jest polaczone")
        self.stop()
        # Najpierw wlasnosc: fala albo identyfikacja w toku -> RuntimeError, nic nie rusza.
        self.twin.claim(OWNER, preempt=self._preempted)
        measured = self.twin.joints() if self.twin.status.simulated else dict(self.twin.status.measured)
        self._hw_lo, self._hw_hi = self._hardware_limits()
        self.q_cmd = np.clip(self.kin.to_q(measured), self._hw_lo, self._hw_hi)
        self.prev_action = np.zeros(6)
        self.status = RunnerStatus(running=True, goal_clamped=self.status.goal_clamped)
        self._cube, self._cube_src = None, None
        self._grip_prev = None
        self._stop.clear()
        self._t_start, self._ticks = None, 0            # zegar startuje z pierwszym taktem
        # Sprzeglo tylko, jesli ramie wciaz nasze - sprawdzenie i wlaczenie razem w `Twin`.
        # Wczesniej STOP miedzy claim a sprzeglem, a potem panel bioracy ramie: runner
        # wylaczal sprzeglo panelu, a pole wyboru w panelu dalej pokazywalo "wlaczone".
        if not self.twin.set_engaged(True, owner=OWNER):
            reason = getattr(self.twin, "preempt_reason", "") or f"ramie ma: {self.twin.owner or 'nikt'}"
            self.status.running = False
            self.status.stopped_because = f"przerwana: {reason}"
            raise RuntimeError(f"ramie odebrane ({reason})")
        if threaded:
            # Watek w zmiennej lokalnej: odebranie ramienia miedzy utworzeniem a startem
            # watku (Dom z panelu) zeruje `self._thread` w `stop()`.
            th = threading.Thread(target=self._loop, name="polityka", daemon=True)
            self._thread = th
            th.start()

    def _hardware_limits(self) -> tuple[np.ndarray, np.ndarray]:
        """Limity treningu przeciete z limitami, ktore naprawde obowiazuja na ramieniu (`Twin.joint_limits`).

        Normalizacja obserwacji zostaje na limitach treningu - zmiana skali
        bylaby dla polityki inna obserwacja. Przycinamy tylko cel: inaczej
        q_cmd stal do 7 st. za limitem serwa (ramie nr 2, wrist_flex +88).
        """
        lo, hi = self.limits.lo.copy(), self.limits.hi.copy()
        fn = getattr(self.twin, "joint_limits", None)
        hw = fn() if fn is not None else {}
        for k, name in enumerate(self.kin.spec.joints):
            if name in hw:
                a, b = self.kin.to_q({name: hw[name][0]})[k], self.kin.to_q({name: hw[name][1]})[k]
                lo[k], hi[k] = max(lo[k], min(a, b)), min(hi[k], max(a, b))
        bad = lo > hi
        lo[bad], hi[bad] = self.limits.lo[bad], self.limits.hi[bad]
        return lo, hi

    def _preempted(self) -> None:
        """Twin odebral ramie (Dom, STOP, polaczenie, inny wlasciciel) - tylko zatrzymanie watku."""
        self.stop(f"przerwana: {getattr(self.twin, 'preempt_reason', '') or 'ramie odebrane'}")

    def stop(self, reason: str = "") -> None:
        self._stop.set()
        th = self._thread
        # Watek jeszcze nie wystartowal (odebranie tuz po `Thread()`): join rzucal RuntimeError,
        # `status.running` zostawal True i panel widzial "polityka" do recznego Stop.
        # Taki watek po starcie zobaczy `_stop` i od razu wyjdzie.
        if th is not None and th is not threading.current_thread() and th.is_alive():
            th.join(timeout=2.0)
        self._thread = None
        if self.status.running:
            self.status.running = False
            self.status.stopped_because = reason or "zatrzymana"
            self._let_go(hold_measured=False)             # ramie trzyma ostatni cel

    def _loop(self) -> None:
        period = 1.0 / self.task.control_hz
        while not self._stop.is_set():
            t0 = time.monotonic()
            if not self.step_once(t0):
                return
            time.sleep(max(0.0, period - (time.monotonic() - t0)))

    def step_once(self, now: float | None = None) -> bool:
        """Jeden takt polityki. False = polityka skonczyla (powod w `status.stopped_because`).

        `now` - czas taktu [s]; watek podaje zegar, test - czas symulowany
        (wtedy `wait_for_cube` i czestotliwosc licza sie w tym samym czasie).
        """
        if not self.status.running:
            return False
        now = time.monotonic() if now is None else now
        if self._t_start is None:
            self._t_start = self._t_hz = now
        try:
            if not self.twin.connected:
                self._halt("ramie rozlaczone")
                return False
            measured = dict(self.twin.status.measured) or self.twin.joints()
            q = self.kin.to_q(measured)
            lag = np.abs(q[:5] - self.q_cmd[:5]).max()
            if lag > self.max_tracking:
                self._halt(f"ramie nie nadaza za celem ({np.degrees(lag):.0f} st.) - kolizja albo blokada",
                           hold_measured=True)
                return False
            holding = self._jaw_holding(q[5])
            obs = self.observation(measured)
            action = self.policy.act(obs)
            q_cmd = tk.apply_action(np, self.task, self.limits, self.q_cmd[None], action[None])[0]
            self.q_cmd = np.clip(q_cmd, self._hw_lo, self._hw_hi)
            self.twin.set_target(self.kin.from_q(self.q_cmd), owner=OWNER)
            self.prev_action = action
            st = self.status
            st.step += 1
            st.stopped_because = ""
            st.last_action = action.tolist()
            if self.task.name == "reach":
                st.distance = float(np.linalg.norm(self.goal - self.kin.tcp(measured)[:3, 3]))
                st.success = st.distance < self.task.success_dist
            else:
                # Na ramieniu nie ma czujnikow kontaktu szczek - sukcesem jest kostka
                # (z kamer albo "w dloni") wyzej nad blatem niz `lift_height`, i to tylko,
                # gdy szczeka ja TRZYMA (stoi, ciasniej rozkazana niz jest). Zmierzone
                # (blizniak, 1 z 18 sukcesow): seria 1 s przy 10,9 cm, a kostka obracala
                # sie w szczekach, szczeka domykala sie 31,8 -> 0 przez 3,5 s i kostka spadla
                # 4 s po "zadanie wykonane".
                # Szczeka "stoi" z taktu na takt nawet przy powolnym domykaniu (9 jednostek/s
                # to 0,005 rad na takt) - dlatego tez: w trakcie serii nie domyka sie o wiecej
                # niz `grip_creep` od jej poczatku.
                if holding and st.success_streak > 0 and q[5] < self._streak_grip - self.grip_creep:
                    holding = False
                st.success = (self._cube is not None and self._cube_fresh() and holding
                              and self._cube[2] - self.task.cube_half > self.task.lift_height)
                if st.success and st.success_streak == 0:
                    self._streak_grip = float(q[5])
            st.success_streak = st.success_streak + 1 if st.success else 0
            self._ticks += 1
            if now - self._t_hz >= 1.0:
                st.hz, self._ticks, self._t_hz = self._ticks / (now - self._t_hz), 0, now
            if self.end_on_success > 0 and st.success_streak >= self.end_on_success:
                self._halt("zadanie wykonane")
                return False
            if self.episode_limit and st.step >= self.task.episode_steps:
                self._halt("koniec epizodu")
                return False
        except NoCube as exc:
            # Na starcie percepcja moze jeszcze nie miec pierwszej detekcji - ramie
            # stoi i czekamy; zgubiona kostka W TRAKCIE epizodu zatrzymuje polityke.
            if self.status.step == 0 and now - self._t_start < self.wait_for_cube:
                self.status.stopped_because = "czekam na kostke z kamer"
                return True
            self._halt(str(exc))
            return False
        except Exception as exc:                        # ramie sie rozlaczylo, ramie odebrane, blad percepcji...
            if self.twin.owner != OWNER and self._stop.is_set():
                return False                            # zatrzymana przez `stop` w trakcie taktu
            logger.exception("Petla polityki przerwana")
            self._halt(f"{type(exc).__name__}: {exc}")
            return False
        return True

    def _jaw_holding(self, grip_q: float) -> bool:
        """Czy szczeka trzyma: stoi od poprzedniego taktu, jest szerzej niz rozkaz i niz pusta, zamknieta.

        Te same progi, co `CubeTracker` ("w dloni"). Rozkaz z POPRZEDNIEGO taktu - za nim
        serwo wlasnie jedzie. Szczeka, ktora sie domyka (kostka wypada, obraca sie), nie stoi.
        """
        prev, self._grip_prev = self._grip_prev, float(grip_q)
        still = prev is not None and abs(grip_q - prev) < self.grip_still
        squeezing = (grip_q - self.q_cmd[5] > self.grip_block_margin
                     and grip_q > self.limits.lo[5] + self.grip_open_margin)
        return bool(still and squeezing)

    def _cube_fresh(self) -> bool:
        """Czy poza kostki jest swieza - tylko z takiej wolno liczyc sukces lift.

        Tracker trzyma ostatnia poze do 6 s jako "ostatnie widziane". Zmierzone w
        blizniaku (kamera a): kostka wypadla ze szczek po 0,5 s niesienia, a runner
        liczyl sukces ze starej pozy w powietrzu i konczyl "zadanie wykonane" z kostka
        na blacie. Bez zrodla: symulacja (dostawca = prawda sceny) tak, prawdziwe ramie nie.
        """
        if self._cube_src is None:
            return bool(getattr(self.twin.status, "simulated", False))
        return self._cube_src in FRESH_CUBE_SOURCES

    def _halt(self, reason: str, hold_measured: bool = False) -> None:
        self.status.running = False
        self.status.stopped_because = reason
        self._let_go(hold_measured)
        self._stop.set()
        logger.info("Polityka zatrzymana: %s", reason)

    def _let_go(self, hold_measured: bool) -> None:
        """Oddaje ramie - tylko jesli wciaz je mamy: po odebraniu nalezy juz do kogos innego.

        `hold_measured` (rozjazd, kolizja): ramie staje tam, gdzie JEST. Ostatni cel
        lezy wtedy ~25 st. za przeszkoda i trzymanie go dociskalo serwa do niej
        z pelnym momentem. Zwykly koniec trzyma cel - szczeki dalej sciskaja kostke.
        """
        if self.twin.owner != OWNER:
            return
        if hold_measured:
            self.twin.hold_measured()
        self.twin.set_engaged(False, owner=OWNER)
        self.twin.release(OWNER)
