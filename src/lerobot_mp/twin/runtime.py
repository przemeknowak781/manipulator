"""Dzialajacy blizniak: scena, petla sterowania ramieniem i kamery - to, co napedza UI.

Jedna petla w osobnym watku, stala czestotliwosc, niezalezna od UI:

    zmierz ramie -> cel (z UI albo z kalibracji) -> nadzor -> rozkaz -> scena

Ramie jest albo w symulacji (MuJoCo z fizyka, ramie w scenie JEST robotem),
albo prawdziwe (backend `feetech`/`lerobot`, a scena tylko je odzwierciedla -
kinematycznie, katami zmierzonymi na serwach). Reszta aplikacji nie widzi
roznicy: `joints()`, `set_target()`, `move()` dzialaja tak samo.

Kazdy rozkaz przechodzi przez `control.safety.SafetySupervisor` - te same
limity pozycji i predkosci, stop awaryjny i plynny powrot do domu, co w
aplikacji sterowania dlonia.

Ramieniem steruje naraz JEDEN wlasciciel (`claim`/`release`): panel (suwaki
ze sprzeglem), polityka, fala kalibracyjna albo identyfikacja. Bez tego
trzy watki pisaly ten sam cel na zmiane i ramie skakalo miedzy nimi.
Dom, STOP, polaczenie i rozlaczenie odbieraja ramie wlascicielowi
(`preempt`) i zatrzymuja je w ZMIERZONEJ pozie.

MuJoCo nie jest bezpieczne watkowo, a scene czytaja: petla (fizyka), UI (poza
do rysowania) i kamery symulowane (render). Wszystko przez `self.lock` - ale
render (GL) juz poza nia, na kopii stanu: tworzenie renderera pod blokada
trzymalo petle prawdziwego ramienia do 0,7 s.
"""

from __future__ import annotations

import copy
import logging
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import mujoco
import numpy as np

from ..config import JOINT_NAMES, load_config
from ..control.safety import SafetyState, SafetySupervisor
from ..robot import create_backend
from ..robot.base import RobotBackend, RobotInfo
from . import scene as sc
from .cameras import CameraHub
from .workspace import Workspace

logger = logging.getLogger(__name__)


class SceneBackend(RobotBackend):
    """Ramie blizniaka jako backend robota - z fizyka MuJoCo, nie przyblizeniem."""

    def __init__(self, twin: Twin):
        self.twin = twin
        self.info = RobotInfo(name="blizniak (MuJoCo)", description="symulacja z fizyka", simulated=True)
        self._connected = False

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def read_joints(self) -> dict[str, float]:
        with self.twin.lock:
            return self.twin.scene.joints()

    def send_joints(self, targets: dict[str, float]) -> dict[str, float]:
        with self.twin.lock:
            self.twin.scene.command(targets)
        return dict(targets)

    def step(self, dt: float) -> None:
        with self.twin.lock:
            self.twin.scene.step(dt)

    def faults(self) -> list[str]:
        return []

    def joint_limits(self) -> dict[str, tuple[float, float]]:
        return {}


@dataclass
class RobotStatus:
    connected: bool = False
    backend: str = "brak"
    simulated: bool = True
    state: str = "rozlaczony"
    engaged: bool = False
    measured: dict[str, float] = field(default_factory=dict)
    command: dict[str, float] = field(default_factory=dict)
    at_limit: list[str] = field(default_factory=list)
    loop_hz: float = 0.0
    error: str = ""
    #: Kiedy (czas petli, monotoniczny) przyszedl ostatni odczyt `measured` - po tym
    #: identyfikacja odroznia nowy pomiar od powtorzonego.
    measured_t: float = 0.0
    #: Kto teraz steruje ramieniem (`Twin.owner`), "" = nikt.
    owner: str = ""
    #: Bledy zgloszone przez serwa (`RobotBackend.faults`) w tym takcie.
    faults: list[str] = field(default_factory=list)


def smoothstep(s: float) -> float:
    s = min(max(s, 0.0), 1.0)
    return s * s * (3 - 2 * s)


class RenderWorker:
    """Jeden watek, ktory robi wszystkie rendery OpenGL blizniaka.

    Kontekst OpenGL moze byc biezacy tylko w jednym watku naraz, a render
    zamawiaja: panel (podglady kamer), fala kalibracyjna (kamery symulowane),
    mapa stolu i porownanie sim-real - kazde z innego watku. Zamiast pilnowac
    `make_current` w kazdym z nich, wszystko idzie przez kolejke do tego watku.
    """

    def __init__(self):
        import queue

        self._q: queue.Queue = queue.Queue()
        self._stopped = threading.Event()
        self._thread = threading.Thread(target=self._run, name="render", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while True:
            fn, done, box = self._q.get()
            if fn is None:
                return
            try:
                box["out"] = fn()
            except BaseException as exc:                   # przekazane do zamawiajacego
                box["err"] = exc
            done.set()

    def call(self, fn):
        if threading.current_thread() is self._thread:
            return fn()
        if self._stopped.is_set() or not self._thread.is_alive():
            raise RuntimeError("watek renderujacy zatrzymany")
        done, box = threading.Event(), {}
        self._q.put((fn, done, box))
        # Zlecenie wrzucone za znacznik konca nigdy sie nie wykona - czekanie bez
        # sprawdzania, czy watek zyje, wieszalo watki viser'a i wyjscie z programu.
        while not done.wait(0.25):
            if not self._thread.is_alive():
                raise RuntimeError("watek renderujacy zatrzymany")
        if "err" in box:
            raise box["err"]
        return box.get("out")

    def stop(self) -> None:
        self._stopped.set()
        self._q.put((None, None, None))


@dataclass
class _LoopState:
    """Stan petli ramienia miedzy taktami (w watku albo krokami `Twin.step`)."""

    prev: float
    last_read: float = -1e9
    measured: dict[str, float] = field(default_factory=dict)
    measured_t: float = 0.0
    ticks: int = 0
    t_hz: float = 0.0
    #: Serwa przestaly odpowiadac - rozkazow nie wysylamy, bo czekalyby w buforze
    #: gniazda i przyszly do serw paczka, gdy lacze wroci.
    link_down: bool = False
    #: Cokolwiek juz ruszylo ramieniem od polaczenia (patrz `_tick`).
    moved: bool = False


class Twin:
    """Scena + ramie + kamery jednego stanowiska."""

    #: Przerwa miedzy taktami petli prawdziwego ramienia, po ktorej ruch jest
    #: przerywany [s]. Zmierzone: jeden niemy serwo przez socket:// blokowal
    #: odczyt na 1,75 s, a nastepny takt wysylal skok 28 st. na shoulder_pan.
    max_gap = 0.5

    def __init__(self, workspace: Workspace, loop_hz: float = 50.0):
        self.workspace = workspace
        self.lock = threading.RLock()
        self.loop_hz = loop_hz
        #: Dodatki sceny ponad stanowisko: karta w szczekach (`with_card`, `card_pose`),
        #: obiekty zadania (`objects`, `grasp_sensors`). Przezywaja przebudowe sceny.
        self.extras: dict = {}
        #: Zwiekszany przy kazdej przebudowie - widoki (panel) wiedza, ze trzeba odswiezyc siatki.
        self.version = 0
        self.renderer = RenderWorker()
        self.scene: sc.Scene = sc.build(workspace.scene_config())
        with self.lock:
            self.scene.set_joints(workspace.spec().home)
        #: (scena zrodlowa, kopia do renderu) - kopia ma wlasne MjData, render idzie poza blokada.
        self._view: tuple[sc.Scene, sc.Scene] | None = None
        self.cameras = CameraHub(workspace, sim_render=self.render)
        self.status = RobotStatus()

        self._backend: RobotBackend | None = None
        self._supervisor: SafetySupervisor | None = None
        #: Nadzor nie jest bezpieczny watkowo: krok w petli i `hold`/`begin_homing`
        #: z innych watkow ida pod ta blokada (inaczej krok nadpisywal swiezy `hold`).
        self._sup_lock = threading.RLock()
        self._target: dict[str, float] | None = None
        self._engaged = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._read_period = 0.04          # prawdziwe serwa czytamy ~25 Hz; kazdy odczyt to transakcja
        self._ls: _LoopState | None = None
        self._manual = False              # connect(threaded=False): takty z `step()`, bez watku
        self._alive = False
        self._clock = 0.0
        self._latest: dict[str, float] = {}

        # Wlasnosc ramienia (K3). `_gen` rosnie przy kazdym odebraniu ramienia
        # i kazdym (roz)laczeniu - ruch, ktory go zapamietal, wie, ze go przerwano.
        self._own_lock = threading.Lock()
        self._owner: str | None = None
        self._preempt_cb: Callable[[], None] | None = None
        self._gen = 0
        self._gen_reason = ""
        self._fault = ""
        self._note, self._note_t = "", 0.0
        self._base_limits: dict[str, tuple[float, float]] = {}

    # ------------------------------------------------------------ scena
    def rebuild(self) -> None:
        """Przebudowa sceny po zmianie stanowiska (kamery, stol) - stan ramienia i obiektow zostaje."""
        # Cala przebudowa w watku renderujacym: kontekst GL zamyka watek, ktory go
        # stworzyl. Blokada brana DOPIERO tam - czekanie na watek renderujacy
        # z blokada w reku zakleszczyloby sie z renderem, ktory czeka na blokade.
        cfg = self.workspace.scene_config(**self.extras)

        def job():
            new = sc.build(cfg)
            with self.lock:
                old = self.scene
                _copy_state(old, new)
                old.close()
                self.scene = new
                self._view = None
                self.version += 1
        self.renderer.call(job)

    def configure(self, **extras) -> None:
        """Nowe dodatki sceny (np. `with_card=True` albo kostka) i przebudowa."""
        self.extras = {k: v for k, v in extras.items() if v is not None}
        self.rebuild()

    def _render_view(self) -> sc.Scene:
        """Kopia stanu sceny do renderu: pod blokada tylko kopiowanie wektorow stanu.

        Wywolywane w watku renderujacym (tam tez jest przebudowa), wiec kopia nie
        zmieni sie pod reka. Model jest wspolny - render go tylko czyta.
        """
        with self.lock:
            src = self.scene
            if self._view is None or self._view[0] is not src:
                view = copy.copy(src)                     # wspolny model i renderery, wlasne MjData
                view.data = mujoco.MjData(src.model)
                self._view = (src, view)
            view = self._view[1]
            d, s = view.data, src.data
            d.qpos[:] = s.qpos
            d.qvel[:] = s.qvel
            d.mocap_pos[:] = s.mocap_pos
            d.mocap_quat[:] = s.mocap_quat
            d.time = s.time
        mujoco.mj_kinematics(view.model, d)
        mujoco.mj_camlight(view.model, d)
        return view

    def render(self, camera: str) -> np.ndarray:
        return self.renderer.call(lambda: self._render_view().render(camera))

    def render_with(self, fn):
        """Dowolny render na scenie (np. segmentacja) w watku renderujacym.

        `fn` dostaje KOPIE stanu sceny (wlasne MjData, wspolny model), bez blokady
        petli - moze zmieniac model (np. poze kamery), ale nie stan symulacji.
        """
        return self.renderer.call(lambda: fn(self._render_view()))

    def joints(self) -> dict[str, float]:
        with self.lock:
            return self.scene.joints()

    # ------------------------------------------------------ wlasnosc ramienia
    @property
    def owner(self) -> str | None:
        return self._owner

    @property
    def preempt_reason(self) -> str:
        """Dlaczego ostatnio odebrano ramie (do komunikatow przerwanych zadan)."""
        return self._gen_reason

    def claim(self, owner: str, preempt: Callable[[], None] | None = None) -> None:
        """Bierze ramie dla `owner`. Inny wlasciciel -> RuntimeError; ten sam - podmienia `preempt`.

        Jazda do domu w toku jest przerywana w miejscu: nowy wlasciciel zaczyna
        od zmierzonej pozy. Inaczej po koncu rampy nadzor przechodzil prosto do
        ACTIVE i doganial cel sterownika z pelna predkoscia.
        """
        with self._own_lock:
            if self._owner is not None and self._owner != owner:
                raise RuntimeError(f"ramie zajete: {self._owner}")
            self._owner, self._preempt_cb = owner, preempt
        sup = self._supervisor
        if sup is not None and not sup.is_homing_done:
            self.hold_measured()

    def release(self, owner: str) -> None:
        with self._own_lock:
            if self._owner == owner:
                self._owner, self._preempt_cb = None, None

    def preempt(self, reason: str) -> None:
        """Odbiera ramie wlascicielowi: ramie staje w ZMIERZONEJ pozie, sterownik dostaje `preempt`.

        Najpierw zatrzymanie (sprzeglo, poza zmierzona), dopiero potem wywolanie
        zwrotne - moze czekac na watek sterownika, a ramie ma stanac od razu.
        Nigdy pod `self.lock`. Z watku petli wywolanie idzie w osobnym watku:
        petla nie moze czekac 2 s na zatrzymanie polityki.
        """
        with self._own_lock:
            cb = self._preempt_cb
            self._owner, self._preempt_cb = None, None
            self._gen += 1
            self._gen_reason = reason
        with self._sup_lock:                               # razem - takt petli nie wejdzie miedzy nie
            self._engaged, self._target = False, None
            self.hold_measured()
        if cb is None:
            return
        if threading.current_thread() is self._thread:
            threading.Thread(target=self._call_preempt, args=(cb,), name="odebranie-ramienia", daemon=True).start()
        else:
            self._call_preempt(cb)

    @staticmethod
    def _call_preempt(cb: Callable[[], None]) -> None:
        try:
            cb()
        except Exception:                                  # pragma: no cover - sprzatanie sterownika nie moze wysypac STOP-u
            logger.exception("Blad przy przerywaniu wlasciciela ramienia")

    def hold_measured(self, measured: Mapping[str, float] | None = None) -> None:
        """Cel := zmierzona poza; rozkaz nadzoru i ograniczniki predkosci od nowa w niej - nic nie skacze."""
        sup = self._supervisor
        if sup is None:
            return
        if measured is None:
            b = self._backend
            measured = self.joints() if b is not None and b.info.simulated else dict(self._latest)
        measured = {k: float(v) for k, v in measured.items() if k in JOINT_NAMES}
        if not measured:
            return
        with self._sup_lock:
            sup.hold(measured)
            # Przy wylaczonym sprzegle cel musi zostac pusty: stary cel "zmierzony"
            # sprzed minut wrocilby po ponownym wlaczeniu sprzegla jako skok.
            self._target = dict(measured) if self._engaged else None

    # ----------------------------------------------------------- ramie
    def connect(self, backend: str | None = None, port: str | None = None, go_home: bool = False,
                *, threaded: bool = True) -> None:
        """Laczy z ramieniem. `sim` - blizniak sam jest ramieniem; inaczej prawdziwy sprzet.

        `threaded=False` (testy): bez watku petli - takty robi `step(dt)` w czasie
        symulowanym, wiec wynik nie zalezy od obciazenia procesora.
        """
        self.preempt("ponowne laczenie")
        self.disconnect()
        ws = self.workspace
        backend = (backend or ws.backend).lower()
        port = port if port is not None else ws.port
        cfg = load_config(overrides={"robot": {"backend": "sim" if backend == "sim" else backend,
                                               "port": port}})
        try:
            b: RobotBackend = SceneBackend(self) if backend == "sim" else create_backend(cfg)
            b.connect()
            measured = b.read_joints()
            _intersect_limits(cfg, b)
        except Exception as exc:
            self.status = RobotStatus(error=f"{type(exc).__name__}: {exc}")
            raise
        self._backend = b
        # Nadzor dostaje WLASNA kopie konfiguracji: limity backendu (`cfg`, ta sama,
        # ktora przycina `send_joints`) poszerzamy chwilowo dla stawu spoza zakresu
        # (`_sync_backend_limits`), a nadzor musi dalej znac prawdziwe limity.
        self._base_limits = {n: (cfg.joint(n).min, cfg.joint(n).max) for n in JOINT_NAMES}
        self._supervisor = SafetySupervisor(copy.deepcopy(cfg))
        # Staw poza limitami nie jest przycinany na starcie - patrz `SafetySupervisor.start`.
        self._supervisor.start(measured, go_home=go_home, keep_outside=True)
        self._target, self._engaged = None, False
        self._latest = dict(measured)
        self._fault, self._note = "", ""
        ws.backend, ws.port = backend, port
        self.status = RobotStatus(connected=True, backend=backend, simulated=b.info.simulated,
                                  state=self._supervisor.state.value, measured=measured,
                                  command=self._supervisor.command)
        self._stop.clear()
        self._manual = not threaded
        self._alive = True
        self._clock = 0.0
        now = self._clock if self._manual else time.monotonic()
        self._ls = _LoopState(prev=now, measured=dict(measured), measured_t=now, t_hz=now)
        if threaded:
            self._thread = threading.Thread(target=self._loop, name="blizniak-petla", daemon=True)
            self._thread.start()
        logger.info("Polaczono z ramieniem: %s", b.info.name)

    def disconnect(self) -> None:
        if self._backend is not None:
            self.preempt("rozlaczenie")
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._backend is not None:
            try:
                self._backend.disconnect()
            except Exception:                           # pragma: no cover - rozlaczenie nie moze wysypac UI
                logger.exception("Blad przy rozlaczaniu ramienia")
            self._backend = None
        with self._own_lock:
            self._gen += 1
        self._alive = False
        self._supervisor = None
        self.status = RobotStatus()

    @property
    def connected(self) -> bool:
        if self._backend is None:
            return False
        if self._manual:
            return self._alive
        return self._thread is not None and self._thread.is_alive()

    def joint_limits(self) -> dict[str, tuple[float, float]]:
        """Limity stawow, ktore naprawde obowiazuja [jednostki aplikacji]: konfiguracja
        przecieta z limitami backendu (EEPROM serw) przy polaczeniu."""
        cfg = self._supervisor.cfg if self._supervisor is not None else load_config()
        return {n: (cfg.joint(n).min, cfg.joint(n).max) for n in JOINT_NAMES}

    def set_engaged(self, engaged: bool) -> None:
        """Sprzeglo: bez niego cele z UI sa ignorowane, a ramie trzyma pozycje."""
        self._engaged = engaged
        if not engaged:
            self._target = None

    def set_target(self, joints: Mapping[str, float], owner: str | None = None) -> None:
        """Nowy cel. Z `owner` - tylko, jesli ten wlasciciel wciaz ma ramie (inaczej RuntimeError):
        watek sterownika, ktoremu odebrano ramie, nie moze po chwili nadpisac celu nowego."""
        cmd = self.status.command or self.joints()
        target = {**cmd, **{k: float(v) for k, v in joints.items()}}
        with self._own_lock:
            if owner is not None and self._owner != owner:
                raise RuntimeError(f"ramie odebrane ({self._gen_reason or 'inny wlasciciel'})")
            self._target = target

    def home(self) -> None:
        """Pozycja domowa: odbiera ramie wlascicielowi (polityka, fala, identyfikacja) i jedzie rampa."""
        if self._supervisor is None:
            return
        self.preempt("pozycja domowa")
        with self._sup_lock:
            self._supervisor.begin_homing()

    def estop(self, reason: str = "STOP awaryjny") -> None:
        sup = self._supervisor
        if sup is None:
            return
        # Najpierw sam STOP (natychmiast), potem odebranie ramienia i trzymanie
        # zmierzonej pozy - STOP trzymal ostatni rozkaz, czyli po kolizji docisk.
        with self._sup_lock:
            sup.trigger_estop()
        self.preempt(reason)

    def clear_estop(self) -> None:
        if self._supervisor is not None:
            with self._sup_lock:
                self._supervisor.clear_estop()
            self._fault, self._note = "", ""

    def move(self, joints: Mapping[str, float], duration: float, settle: float = 0.4,
             owner: str = "kalibracja") -> None:
        """Blokujacy przejazd rampa smoothstep - dla sesji kalibracji (`session.Robot`).

        Ruch nalezy do `owner`: gdy ramie jest wolne, bierze je (i zostawia - fala
        to wiele przejazdow, a konczy ja `home()`, ktory ramie odbiera); gdy ma
        je ten sam wlasciciel - jedzie; gdy inny - RuntimeError "ramie zajete".
        Bez tego fala i polityka pisaly cel na zmiane.

        RuntimeError, gdy w trakcie ramie odebrano (dom, STOP, inny wlasciciel),
        rozlaczono albo polaczono na nowo - wtedy fala nie moze liczyc na poze,
        ktorej ramie nie osiagnelo.
        """
        if not self.connected:
            raise RuntimeError("ramie nie jest polaczone")
        self._wait_for_homing()
        with self._own_lock:
            if self._owner is None:
                self._owner, self._preempt_cb = owner, None
            elif self._owner != owner:
                raise RuntimeError(f"ramie zajete: {self._owner}")
            gen = self._gen
            # Sprzeglo razem z wlasnoscia: odebranie ramienia (STOP) tuz po tym
            # bloku nie zostawi wlaczonego sprzegla bez wlasciciela.
            self._engaged = True
        start = dict(self.status.command or self.joints())
        goal = {k: float(joints.get(k, v)) for k, v in start.items()}
        t0 = time.monotonic()
        while True:
            self._check_move(gen)
            s = smoothstep((time.monotonic() - t0) / max(duration, 1e-3))
            target = {k: start[k] + s * (goal[k] - start[k]) for k in start}
            with self._own_lock:
                if self._gen != gen:
                    raise RuntimeError(f"ruch przerwany: {self._gen_reason}")
                self._target = target
            if s >= 1.0:
                break
            time.sleep(1.0 / self.loop_hz)
        t_end = time.monotonic() + settle
        while time.monotonic() < t_end:
            self._check_move(gen)
            time.sleep(min(0.05, settle))
        self._check_move(gen)

    def _check_move(self, gen: int) -> None:
        if self._supervisor is not None and self._supervisor.estopped:
            raise RuntimeError("stop awaryjny w trakcie ruchu")
        if self._gen != gen:
            raise RuntimeError(f"ruch przerwany: {self._gen_reason}")
        if not self.connected:
            raise RuntimeError("ramie rozlaczone w trakcie ruchu")

    def _wait_for_homing(self, timeout: float = 6.0) -> None:
        """Ruch zaczety w trakcie rampy do domu startowalby z pozy, ktora zaraz sie zmieni."""
        t_end = time.monotonic() + timeout
        while self._supervisor is not None and not self._supervisor.is_homing_done:
            if self._manual:
                raise RuntimeError("ramie jedzie do domu - poczekaj na koniec rampy")
            if time.monotonic() > t_end:
                raise RuntimeError("ramie wciaz jedzie do domu")
            time.sleep(0.02)

    # ---------------------------------------------------------- petla
    def step(self, dt: float) -> None:
        """Jeden takt petli w czasie symulowanym - tylko po `connect(threaded=False)`."""
        if not self._manual:
            raise RuntimeError("step() tylko po connect(..., threaded=False)")
        if not self.connected:
            raise RuntimeError("ramie nie jest polaczone")
        self._clock += dt
        if not self._tick(self._clock):
            self._alive = False

    def _loop(self) -> None:
        period = 1.0 / self.loop_hz
        while not self._stop.is_set():
            now = time.monotonic()
            if not self._tick(now):
                break
            time.sleep(max(0.0, period - (time.monotonic() - now)))

    def _tick(self, now: float) -> bool:
        """Jeden takt: odczyt, bledy serw, nadzor, rozkaz, scena. False = petla padla."""
        b, sup, ls = self._backend, self._supervisor, self._ls
        assert b is not None and sup is not None and ls is not None
        period = 1.0 / self.loop_hz
        gap = now - ls.prev
        # Krok nadzoru nie dluzszy niz 1,5 okresu: po przestoju (render, odczyt z
        # przerwanego lacza) ogranicznik predkosci puszczal max_vel * 0,2 s w jednym
        # rozkazie, a serwo jechalo tam z wlasna, pelna predkoscia.
        dt = min(max(gap, 0.0), 1.5 * period)
        ls.prev = now
        try:
            if b.info.simulated or now - ls.last_read >= self._read_period:
                ls.measured = b.read_joints()
                ls.last_read = ls.measured_t = now
                self._latest = dict(ls.measured)

            faults = _backend_faults(b)
            if faults:
                msg = "; ".join(faults)
                if not sup.estopped:
                    logger.error("Serwa zglaszaja blad - STOP: %s", msg)
                    self.estop(msg)
                self._fault = msg
            link_down = any("brak odpowiedzi" in f for f in faults)
            if ls.link_down and not link_down:
                self.hold_measured(ls.measured)            # lacze wrocilo: od swiezego odczytu
            ls.link_down = link_down

            if not b.info.simulated and gap > self.max_gap:
                self._note, self._note_t = f"przerwa w petli sterowania ({gap * 1000:.0f} ms) - ruch przerwany", now
                logger.warning(self._note)
                self.preempt(self._note)

            # Cel, sprzeglo i krok nadzoru razem pod blokada nadzoru: `preempt`/`hold_measured`
            # z innego watku miedzy odczytem celu a krokiem dawaly jeszcze jeden takt
            # w strone starego celu (po kolizji - dalej w przeszkode).
            with self._sup_lock:
                if not sup.is_homing_done:
                    self._target = None                    # cel sprzed rampy nie moze wrocic po jej koncu
                engaged = self._engaged
                desired = self._target if engaged else None
                # Obecnosc operatora w blizniaku to sprzeglo z UI, nie dlon w kadrze.
                command, report = sup.step(desired, dt, hand_present=True, engaged=engaged)
                outside = sup.outside
            if engaged or report.state in _MOVING:
                ls.moved = True
            sent_cmd = dict(command)
            # Prawdziwe ramie po "polacz bez ruchu" trzyma cel wpisany przy polaczeniu
            # (tam, gdzie stoi) - do pierwszego ruchu nic nie wysylamy: backend przycina
            # rozkaz do limitow, wiec staw spoza nich dostawal skok od razu po polaczeniu.
            if not link_down and (b.info.simulated or ls.moved):
                self._sync_backend_limits(b, outside)
                sent = b.send_joints(command) or {}
                sent_cmd.update({k: float(v) for k, v in sent.items() if k in sent_cmd})
                off = {k: v for k, v in sent_cmd.items() if abs(v - command[k]) > 0.5}
                if off:
                    # Serwo przycielo rozkaz (limit w EEPROM) - nadzor liczy dalej od
                    # tego, co naprawde poszlo, i tak samo nagrywa to identyfikacja.
                    with self._sup_lock:
                        sup.reseed(off)
            b.step(dt)
            if not b.info.simulated and self.lock.acquire(blocking=False):
                try:                                       # prawdziwe ramie: scena je tylko odzwierciedla
                    self.scene.set_joints(ls.measured)
                finally:
                    self.lock.release()
            ls.ticks += 1
            hz = self.status.loop_hz
            if now - ls.t_hz >= 1.0:
                hz = ls.ticks / (now - ls.t_hz)
                ls.ticks, ls.t_hz = 0, now
            if self._note and now - self._note_t > 10.0:
                self._note = ""
            self.status = RobotStatus(connected=True, backend=self.status.backend, simulated=b.info.simulated,
                                      state=report.state.value, engaged=self._engaged,
                                      measured=dict(ls.measured), command=sent_cmd,
                                      at_limit=list(report.at_limit), loop_hz=hz,
                                      error=self._fault or self._note, measured_t=ls.measured_t,
                                      owner=self._owner or "", faults=list(faults))
            return True
        except Exception as exc:                           # odczyt z portu padl, kabel wypadl...
            logger.exception("Petla ramienia przerwana")
            reason = f"{type(exc).__name__}: {exc}"
            self.status = RobotStatus(False, self.status.backend, error=reason)
            # Martwa petla nie wysyla juz nic - wlasciciel (identyfikacja, polityka,
            # fala) musi sie o tym dowiedziec, a nie pisac dalej cele w proznie.
            with self._own_lock:
                cb = self._preempt_cb
                self._owner, self._preempt_cb = None, None
                self._gen += 1
                self._gen_reason = f"petla ramienia padla: {reason}"
            self._engaged, self._target = False, None
            if cb is not None:
                if threading.current_thread() is self._thread:
                    threading.Thread(target=self._call_preempt, args=(cb,), daemon=True).start()
                else:
                    self._call_preempt(cb)
            return False

    def _sync_backend_limits(self, b: RobotBackend, outside: dict[str, tuple[float, float]]) -> None:
        """Limity, do ktorych backend przycina `send_joints` = limity nadzoru, poszerzone
        dla stawu, ktory stoi poza zakresem i dopiero do niego wraca.

        Nadzor prowadzi taki staw powoli od zmierzonej pozy, ale `FeetechArm.send_joints`
        przycina do limitow konfiguracji - bez poszerzenia pierwszy rozkaz i tak
        skakal do limitu (ramie nr 2: shoulder_lift -101 -> -95 st. z pelna predkoscia serwa).
        """
        bcfg = getattr(b, "cfg", None)
        if bcfg is None or not hasattr(bcfg, "joint") or not self._base_limits:
            return
        for name, (lo, hi) in self._base_limits.items():
            try:
                jc = bcfg.joint(name)
            except Exception:                              # backend z inna konfiguracja stawow
                continue
            s = outside.get(name)
            jc.min, jc.max = (min(lo, s[0]), max(hi, s[1])) if s else (lo, hi)

    @property
    def safety_state(self) -> SafetyState | None:
        return None if self._supervisor is None else self._supervisor.state

    def close(self) -> None:
        self.disconnect()
        self.cameras.close()

        def job():
            with self.lock:
                self.scene.close()
        try:
            self.renderer.call(job)
        finally:
            self.renderer.stop()


#: Stany nadzoru, w ktorych cos rusza ramieniem (STOP i bezczynnosc - nie).
_MOVING = (SafetyState.ACTIVE, SafetyState.STARTING, SafetyState.HOMING)


def _backend_faults(b: RobotBackend) -> list[str]:
    """`RobotBackend.faults()` - backend bez tej metody (albo z bledem w niej) nie ma bledow."""
    fn = getattr(b, "faults", None)
    if fn is None:
        return []
    try:
        return [str(f) for f in (fn() or [])]
    except Exception:                                      # pragma: no cover - diagnostyka nie moze zabic petli
        logger.exception("faults() backendu rzucilo wyjatek")
        return []


def _intersect_limits(cfg, b: RobotBackend) -> None:
    """Limity konfiguracji przeciete z twardymi limitami backendu (EEPROM serw).

    Serwo i tak przytnie rozkaz do swojego zakresu. Bez przeciecia nadzor,
    polityka i identyfikacja liczyly na cel, ktorego serwo nie przyjelo
    (ramie nr 2: wrist_flex do +88 st. w serwie, +95 w konfiguracji).
    """
    fn = getattr(b, "joint_limits", None)
    if fn is None:
        return
    try:
        limits = fn() or {}
    except Exception:                                      # pragma: no cover - brak limitow = brak przeciecia
        logger.exception("joint_limits() backendu rzucilo wyjatek")
        return
    for name, (lo, hi) in limits.items():
        if name not in cfg.joints:
            continue
        jc = cfg.joint(name)
        new_lo, new_hi = max(jc.min, float(lo)), min(jc.max, float(hi))
        if new_lo >= new_hi:
            logger.warning("%s: limity serwa %.1f..%.1f nie pokrywaja sie z konfiguracja %.1f..%.1f - zostaja "
                           "limity konfiguracji", name, lo, hi, jc.min, jc.max)
            continue
        if (new_lo, new_hi) != (jc.min, jc.max):
            logger.info("%s: limity %.1f..%.1f przyciete do limitow serwa -> %.1f..%.1f",
                        name, jc.min, jc.max, new_lo, new_hi)
        jc.min, jc.max = new_lo, new_hi


def _copy_state(old: sc.Scene, new: sc.Scene) -> None:
    """Stan symulacji ze starej sceny do nowej: kazdy staw i aktuator o tej samej nazwie.

    Wczesniej przechodzily tylko stawy ramienia - kostka w trakcie podnoszenia
    wracala po przebudowie (np. przelaczenie kamery) na miejsce startowe z `Box`,
    czasem wprost w szczeki.
    """
    om, od, nm, nd = old.model, old.data, new.model, new.data
    size = {int(mujoco.mjtJoint.mjJNT_FREE): (7, 6), int(mujoco.mjtJoint.mjJNT_BALL): (4, 3)}

    def key(m, j):
        # Wolne stawy obiektow (`add_freejoint`) nie maja nazwy - wtedy cialo i numer stawu w nim.
        name = m.joint(j).name
        if name:
            return name
        b = int(m.jnt_bodyid[j])
        return (m.body(b).name, j - int(m.body_jntadr[b])) if m.body(b).name else None

    old_ids = {k: o for o in range(om.njnt) if (k := key(om, o)) is not None}
    for j in range(nm.njnt):
        o = old_ids.get(key(nm, j))
        if o is None:
            continue
        if om.jnt_type[o] != nm.jnt_type[j]:
            continue
        nq, nv = size.get(int(nm.jnt_type[j]), (1, 1))
        qa, oqa = nm.jnt_qposadr[j], om.jnt_qposadr[o]
        va, ova = nm.jnt_dofadr[j], om.jnt_dofadr[o]
        nd.qpos[qa:qa + nq] = od.qpos[oqa:oqa + nq]
        nd.qvel[va:va + nv] = od.qvel[ova:ova + nv]
    for a in range(nm.nu):
        name = nm.actuator(a).name
        try:
            nd.ctrl[a] = od.ctrl[om.actuator(name).id]
        except KeyError:
            continue
    nd.time = od.time
    mujoco.mj_forward(nm, nd)
