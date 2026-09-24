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

MuJoCo nie jest bezpieczne watkowo, a scene czytaja: petla (fizyka), UI (poza
do rysowania) i kamery symulowane (render). Wszystko przez `self.lock`.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from ..config import load_config
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
        done, box = threading.Event(), {}
        self._q.put((fn, done, box))
        done.wait()
        if "err" in box:
            raise box["err"]
        return box.get("out")

    def stop(self) -> None:
        self._q.put((None, None, None))


class Twin:
    """Scena + ramie + kamery jednego stanowiska."""

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
        self.cameras = CameraHub(workspace, sim_render=self.render)
        self.status = RobotStatus()

        self._backend: RobotBackend | None = None
        self._supervisor: SafetySupervisor | None = None
        self._target: dict[str, float] | None = None
        self._engaged = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._read_period = 0.04          # prawdziwe serwa czytamy ~25 Hz; kazdy odczyt to transakcja

    # ------------------------------------------------------------ scena
    def rebuild(self) -> None:
        """Przebudowa sceny po zmianie stanowiska (kamery, stol) - poza ramienia zostaje."""
        # Cala przebudowa w watku renderujacym: kontekst GL zamyka watek, ktory go
        # stworzyl. Blokada brana DOPIERO tam - czekanie na watek renderujacy
        # z blokada w reku zakleszczyloby sie z renderem, ktory czeka na blokade.
        cfg = self.workspace.scene_config(**self.extras)

        def job():
            new = sc.build(cfg)
            with self.lock:
                joints = self.scene.joints()
                ctrl = self.scene.data.ctrl.copy()
                self.scene.close()
                self.scene = new
                self.scene.set_joints(joints)
                if len(ctrl) == len(self.scene.data.ctrl):
                    self.scene.data.ctrl[:] = ctrl
                self.version += 1
        self.renderer.call(job)

    def configure(self, **extras) -> None:
        """Nowe dodatki sceny (np. `with_card=True` albo kostka) i przebudowa."""
        self.extras = {k: v for k, v in extras.items() if v is not None}
        self.rebuild()

    def render(self, camera: str) -> np.ndarray:
        def job():
            with self.lock:
                return self.scene.render(camera)
        return self.renderer.call(job)

    def render_with(self, fn):
        """Dowolny render na scenie (np. segmentacja) w watku renderujacym, pod blokada."""
        def job():
            with self.lock:
                return fn(self.scene)
        return self.renderer.call(job)

    def joints(self) -> dict[str, float]:
        with self.lock:
            return self.scene.joints()

    # ----------------------------------------------------------- ramie
    def connect(self, backend: str | None = None, port: str | None = None, go_home: bool = False) -> None:
        """Laczy z ramieniem. `sim` - blizniak sam jest ramieniem; inaczej prawdziwy sprzet."""
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
        except Exception as exc:
            self.status = RobotStatus(error=f"{type(exc).__name__}: {exc}")
            raise
        self._backend = b
        self._supervisor = SafetySupervisor(cfg)
        self._supervisor.start(measured, go_home=go_home)
        self._target, self._engaged = None, False
        ws.backend, ws.port = backend, port
        self.status = RobotStatus(connected=True, backend=backend, simulated=b.info.simulated,
                                  state=self._supervisor.state.value, measured=measured,
                                  command=self._supervisor.command)
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="blizniak-petla", daemon=True)
        self._thread.start()
        logger.info("Polaczono z ramieniem: %s", b.info.name)

    def disconnect(self) -> None:
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
        self._supervisor = None
        self.status = RobotStatus()

    @property
    def connected(self) -> bool:
        return self._backend is not None and self._thread is not None and self._thread.is_alive()

    def set_engaged(self, engaged: bool) -> None:
        """Sprzeglo: bez niego cele z UI sa ignorowane, a ramie trzyma pozycje."""
        self._engaged = engaged
        if not engaged:
            self._target = None

    def set_target(self, joints: Mapping[str, float]) -> None:
        cmd = self.status.command or self.joints()
        self._target = {**cmd, **{k: float(v) for k, v in joints.items()}}

    def home(self) -> None:
        if self._supervisor is not None:
            self._target = None
            self._supervisor.begin_homing()

    def estop(self) -> None:
        if self._supervisor is not None:
            self._supervisor.trigger_estop()
            self._engaged, self._target = False, None

    def clear_estop(self) -> None:
        if self._supervisor is not None:
            self._supervisor.clear_estop()

    def move(self, joints: Mapping[str, float], duration: float, settle: float = 0.4) -> None:
        """Blokujacy przejazd rampa smoothstep - dla sesji kalibracji (`session.Robot`)."""
        if not self.connected:
            raise RuntimeError("ramie nie jest polaczone")
        start = dict(self.status.command or self.joints())
        goal = {k: float(joints.get(k, v)) for k, v in start.items()}
        self.set_engaged(True)
        t0 = time.monotonic()
        while True:
            if self._supervisor is not None and self._supervisor.estopped:
                raise RuntimeError("stop awaryjny w trakcie ruchu")
            s = smoothstep((time.monotonic() - t0) / max(duration, 1e-3))
            self._target = {k: start[k] + s * (goal[k] - start[k]) for k in start}
            if s >= 1.0:
                break
            time.sleep(1.0 / self.loop_hz)
        time.sleep(settle)

    # ---------------------------------------------------------- petla
    def _loop(self) -> None:
        period = 1.0 / self.loop_hz
        prev = time.monotonic()
        last_read, measured = 0.0, dict(self.status.measured)
        ticks, t_hz = 0, prev
        b, sup = self._backend, self._supervisor
        assert b is not None and sup is not None
        while not self._stop.is_set():
            now = time.monotonic()
            dt = min(now - prev, 0.2)
            prev = now
            try:
                if b.info.simulated or now - last_read >= self._read_period:
                    measured = b.read_joints()
                    last_read = now
                desired = self._target if self._engaged else None
                # Obecnosc operatora w blizniaku to sprzeglo z UI, nie dlon w kadrze.
                command, report = sup.step(desired, dt, hand_present=True, engaged=self._engaged)
                b.send_joints(command)
                b.step(dt)
                if not b.info.simulated:
                    with self.lock:                        # prawdziwe ramie: scena je tylko odzwierciedla
                        self.scene.set_joints(measured)
                ticks += 1
                if now - t_hz >= 1.0:
                    hz, ticks, t_hz = ticks / (now - t_hz), 0, now
                else:
                    hz = self.status.loop_hz
                self.status = RobotStatus(True, self.status.backend, b.info.simulated, report.state.value,
                                          self._engaged, dict(measured), dict(command),
                                          list(report.at_limit), hz)
            except Exception as exc:                       # odczyt z portu padl, kabel wypadl...
                logger.exception("Petla ramienia przerwana")
                self.status = RobotStatus(False, self.status.backend, error=f"{type(exc).__name__}: {exc}")
                break
            time.sleep(max(0.0, period - (time.monotonic() - now)))

    @property
    def safety_state(self) -> SafetyState | None:
        return None if self._supervisor is None else self._supervisor.state

    def close(self) -> None:
        self.disconnect()
        self.cameras.close()

        def job():
            with self.lock:
                self.scene.close()
        self.renderer.call(job)
        self.renderer.stop()
