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
(`preempt`) i zatrzymuja je tam, gdzie serwo sie zatrzyma - nie dalej niz ostatni
rozkaz, nie za zmierzona poza (`hold_measured`).

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
    #: Ostrzezenia bez STOP-u, trwale do czasu, az przestana obowiazywac (np. kalibracja
    #: chwytaka backendu inna niz blizniaka, chwytak odciazony po przeciazeniu).
    warnings: list[str] = field(default_factory=list)


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
    #: Lacze niepotwierdzone: odczyt bez zadnej odpowiedzi, backend wstrzymal wysylke albo
    #: petla stala dluzej niz `Twin.max_gap`. Nadzor stoi (krok 0 s), nic nie wysylamy, a dwa
    #: kolejne zgodne odczyty zaczynaja rozkaz od nowa od zmierzonej pozy
    #: (`Twin._confirm_after_silence`) - takze ponizej progu utraty lacza.
    silent: bool = False
    #: Rozkaz, ktory NAPRAWDE poszedl do serw ostatnim razem (to widzi `status.command`).
    last_sent: dict[str, float] = field(default_factory=dict)
    #: Poprzedni swiezy odczyt i jego chwila - predkosc stawow do trzymania pozy po STOP-ie.
    prev_measured: dict[str, float] = field(default_factory=dict)
    prev_measured_t: float = 0.0
    #: Od kiedy staw odjechal od rozkazu dalej niz `Twin.track_err_deg`.
    lag_since: dict[str, float] = field(default_factory=dict)
    #: Swieze odczyty z ostatniej ~sekundy (chwila, poza) - czy staw jedzie do rozkazu.
    hist: list[tuple[float, dict[str, float]]] = field(default_factory=list)
    #: Od kiedy serwo chwytaka zglasza blad (None = nie zglasza).
    grip_fault_since: float | None = None
    #: Od kiedy serwo chwytaka NIE zglasza bledu (None = zglasza albo nie bylo bledu).
    grip_ok_since: float | None = None
    #: Od kiedy rozkaz chwytaka jest ciasniejszy od pomiaru (None = nie jest) - `Twin._grip_squeezing`.
    grip_tight_since: float | None = None
    #: Po milczeniu: pierwszy swiezy odczyt czeka na potwierdzenie nastepnym (i od kiedy czeka).
    confirm: dict[str, float] | None = None
    confirm_since: float | None = None
    #: Lacze wrocilo po utracie (STOP): potwierdzona poza idzie przez `hold_measured`, nie `reseed`.
    recover_hold: bool = False
    #: Petla stoi (lacze niepotwierdzone) - `Twin.move` nie liczy wtedy czasu rampy.
    frozen: bool = False


class Twin:
    """Scena + ramie + kamery jednego stanowiska."""

    #: Przerwa miedzy taktami petli prawdziwego ramienia, po ktorej ruch jest
    #: przerywany [s]. Zmierzone: jeden niemy serwo przez socket:// blokowal
    #: odczyt na 1,75 s, a nastepny takt wysylal skok 28 st. na shoulder_pan.
    max_gap = 0.5
    #: Straznik rozjazdu dla KAZDEGO wlasciciela (nie tylko polityki): staw ramienia dalej
    #: od wyslanego rozkazu niz tyle stopni przez `track_err_s` sekund = STOP. Zmierzone
    #: w emulatorze: serwo, ktore stracilo moment bez bitu bledu, konczylo `Twin.move`
    #: "normalnie" 63 st. od celu. Rampy nadzoru (do 220 st./s) i odczyt co 40 ms daja
    #: rozjazd kilkunastu stopni przez ulamek sekundy - ponizej progu.
    track_err_deg = 25.0
    track_err_s = 0.5
    #: ...ale tylko, gdy staw przez ostatnie `track_err_s` NIE jechal do rozkazu co najmniej
    #: tyle st./s. Zmierzone (emulator STS3215 za mostem, 10 ms w jedna strone): suwak
    #: wrist_roll od konca do konca przy serwie 185 st./s (nadzor 220) albo shoulder_lift przy
    #: 70 st./s (nadzor 120) dawal rozjazd 40-50 st. - falszywy STOP, choc serwo jechalo pelna
    #: predkoscia. Staw zablokowany albo bez momentu stoi (albo odjezdza) - STOP jak dotad.
    track_min_speed = 15.0
    #: Przyspieszenie serwa do drogi hamowania przy trzymaniu pozy [st./s^2]. LeRobot zapisuje
    #: Acceleration=254 (~2230 st./s^2); nizsza wartosc = dluzsza droga = trzymanie blizej
    #: ostatniego rozkazu, czyli najwyzej tam, dokad serwo i tak jechalo.
    servo_accel = 2000.0
    #: Zapas na predkosc stawu z odczytow przy trzymaniu pozy (patrz `_measured_now`).
    vel_margin = 1.3
    #: Chwytak "sciska", gdy rozkaz jest ciasniejszy od pomiaru o wiecej niz tyle [0..100]...
    grip_squeeze_margin = 3.0
    #: ...i szczeka STOI: dwa kolejne swieze odczyty blizej niz tyle [0..100]. Zamykajaca sie
    #: szczeka tez odstaje od rozkazu (~20 jednostek przy 300/s) - STOP trzymal wtedy pelny
    #: rozkaz zamkniecia i szczeki zamykaly sie dalej o 15-27 jednostek (palec miedzy nimi).
    grip_still = 1.0
    #: ...i rozkaz jest ciasniejszy od pomiaru od co najmniej tylu sekund. Szczeka, ktorej
    #: rozkazano zamkniecie przed chwila, stoi jeszcze (opoznienie lacza, okres odczytu) -
    #: emulator: Dom 80 ms po rozkazie zamkniecia zostawial chwytak na 77,5 zamiast w domu.
    grip_block_s = 0.15
    #: Odciazenie chwytaka znika, gdy bledu nie ma od tylu sekund, a szczeka juz nie sciska.
    grip_clear_s = 1.0
    #: Po bledzie serwa chwytaka (przeciazenie przy mocnym chwycie) rozkaz szczek najwyzej
    #: tyle ciasniej niz pomiar [0..100, ~0,6 st. na jednostke] - docisk zostaje, prad spada.
    grip_ease = 8.0
    #: Blad chwytaka dluzszy niz tyle sekund (mimo odciazenia) = STOP.
    grip_fault_s = 2.0
    #: Roznica tikow chwytaka backend/blizniak, od ktorej panel ostrzega.
    grip_ticks_tol = 20.0
    #: Dwa kolejne odczyty przy polaczeniu (i po kazdym milczeniu lacza) musza sie zgadzac
    #: do tylu jednostek.
    start_read_tol = 2.0
    #: Po milczeniu lacza odczyty niezgodne dluzej niz tyle sekund = STOP (petla dalej stoi).
    confirm_timeout_s = 1.0
    #: `move` konczy sie dopiero, gdy stawy ramienia stoja (wolniej niz `move_still_speed`
    #: st./s) najwyzej `move_arrive_deg` od rozkazu; inaczej RuntimeError po `settle` +
    #: `move_arrive_s`. Zmierzone: przestoj lacza 1,1 s pod koniec przejazdu - `move` wracal
    #: "normalnie" z ramieniem 17 st. od celu, jadacym 130 st./s (fala brala wtedy kadry).
    move_arrive_deg = 10.0
    move_still_speed = 10.0
    move_arrive_s = 1.5

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
        #: Powod ostatniego odebrania dla `preempt_reason` - kasowany, gdy ktos wezmie ramie.
        self._preempt_note = ""
        #: Wlasciciel -> dlaczego odebrano mu ramie (do odmow jego spoznionych rozkazow).
        self._lost: dict[str, str] = {}
        self._fault = ""
        self._note, self._note_t = "", 0.0
        self._base_limits: dict[str, tuple[float, float]] = {}
        #: Trwale ostrzezenia dla panelu (`RobotStatus.warnings`), po kluczu.
        self._warnings: dict[str, str] = {}
        #: Najciasniejszy dozwolony rozkaz chwytaka po jego bledzie (None = bez ograniczenia).
        self._grip_floor: float | None = None
        #: STOP z powodu bledu chwytaka: szczeka trzyma zmierzone rozwarcie, bez docisku.
        self._grip_release = False

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
        """Dlaczego ostatnio odebrano ramie (do komunikatow przerwanych zadan).

        Pusty, gdy od tamtej pory ktos wzial ramie: odmowy podawaly powod sprzed
        minut ("ponowne laczenie"), choc ramie mial juz po prostu inny wlasciciel.
        """
        return self._preempt_note

    def _refusal(self, owner: str) -> str:
        """Dlaczego `owner` nie ma ramienia - aktualny powod, nie ostatni z historii."""
        if self._owner is not None:
            return f"ramie ma: {self._owner}"
        return self._lost.get(owner) or "nikt go nie ma"

    def claim(self, owner: str, preempt: Callable[[], None] | None = None) -> None:
        """Bierze ramie dla `owner`. Inny wlasciciel -> RuntimeError; ten sam - podmienia `preempt`.

        Jazda do domu w toku jest przerywana w miejscu: nowy wlasciciel zaczyna
        od zmierzonej pozy. Inaczej po koncu rampy nadzor przechodzil prosto do
        ACTIVE i doganial cel sterownika z pelna predkoscia.
        """
        with self._own_lock:
            if self._owner is not None and self._owner != owner:
                raise RuntimeError(f"ramie zajete: {self._owner}")
            if self._owner is None:
                self._new_owner()
            self._owner, self._preempt_cb = owner, preempt
            self._lost.pop(owner, None)
        sup = self._supervisor
        if sup is not None and not sup.is_homing_done:
            self.hold_measured()

    def release(self, owner: str) -> None:
        with self._own_lock:
            if self._owner == owner:
                self._owner, self._preempt_cb = None, None
                self._lost.pop(owner, None)

    def _new_owner(self) -> None:
        """Ramie bierze nowy wlasciciel (pod `_own_lock`): stare powody i odciazenie chwytaka nieaktualne."""
        self._preempt_note = ""
        self._grip_floor = None
        self._warnings.pop("grip_eased", None)

    def preempt(self, reason: str) -> None:
        """Odbiera ramie wlascicielowi: ramie staje w ZMIERZONEJ pozie, sterownik dostaje `preempt`.

        Najpierw zatrzymanie (sprzeglo, poza zmierzona), dopiero potem wywolanie
        zwrotne - moze czekac na watek sterownika, a ramie ma stanac od razu.
        Nigdy pod `self.lock`. Z watku petli wywolanie idzie w osobnym watku:
        petla nie moze czekac 2 s na zatrzymanie polityki.
        """
        with self._own_lock:
            cb = self._preempt_cb
            if self._owner is not None:
                self._lost[self._owner] = reason
            self._owner, self._preempt_cb = None, None
            self._gen += 1
            self._gen_reason = self._preempt_note = reason
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
        """Cel := poza, w ktorej ramie stanie; rozkaz nadzoru i ograniczniki predkosci od nowa w niej.

        Jedna regula dla STOP-u, Domu, odebrania ramienia i stopu na kolizji (bez
        `measured`): kazdy staw trzyma punkt, w ktorym serwo i tak sie zatrzyma - ostatni
        odczyt przesuniety o jego wiek i droge hamowania, najwyzej do ostatniego rozkazu,
        nigdy za zmierzona poze w strone przeciwna do ruchu (`_measured_now`). Staw w biegu
        nie zawraca wiec skokiem; staw zablokowany (kolizja, serwo bez momentu) ma predkosc
        zero i trzyma czysty pomiar - bez docisku do przeszkody.

        Chwytak, ktory sciska (rozkaz ciasniejszy od pomiaru, szczeka stoi), trzyma SWOJ
        rozkaz: rozkaz = zmierzony kat zablokowanej szczeki to zerowa sila, a kostka niesiona
        17-20 cm nad blatem wypadala po STOP-ie, Domu albo stopie polityki na rozjezdzie.
        Szczeka w trakcie zamykania NIE sciska - staje w zmierzonym miejscu (palec miedzy
        szczekami). Wyjatek: STOP od bledu chwytaka (`_grip_release`) - bez docisku.
        """
        sup = self._supervisor
        if sup is None:
            return
        if measured is None:
            measured = self._measured_now()
        measured = {k: float(v) for k, v in measured.items() if k in JOINT_NAMES}
        if not measured:
            return
        with self._sup_lock:
            pose = self._hold_pose(measured)
            sup.hold(pose)
            # Przy wylaczonym sprzegle cel musi zostac pusty: stary cel "zmierzony"
            # sprzed minut wrocilby po ponownym wlaczeniu sprzegla jako skok.
            self._target = dict(pose) if self._engaged else None

    def _hold_pose(self, measured: Mapping[str, float]) -> dict[str, float]:
        """Poza do trzymania: stawy ramienia jak podane, sciskajacy chwytak przy swoim rozkazie."""
        pose = dict(measured)
        sup = self._supervisor
        if GRIPPER in pose and sup is not None and not self._grip_release:
            if self._grip_squeezing(pose[GRIPPER]):
                pose[GRIPPER] = sup.command[GRIPPER]
            if self._grip_floor is not None:
                pose[GRIPPER] = max(pose[GRIPPER], self._grip_floor)
        return pose

    def _grip_squeezing(self, measured: float) -> bool:
        """Chwytak sciska: rozkaz ciasniejszy od pomiaru o wiecej niz margines I szczeka stoi.

        Bez warunku postoju zamykajaca sie szczeka (odstaje od rozkazu ~20 jednostek przy
        300/s) liczyla sie jako sciskajaca: STOP trzymal pelny rozkaz zamkniecia, a szczeki
        zamykaly sie dalej o 15-27 jednostek (emulator) - na palcu miedzy nimi. Dom zostawial
        chwytak w przypadkowym rozwarciu. Ustalony, nieruchomy chwyt zostaje jak dotad.
        """
        sup, ls = self._supervisor, self._ls
        cmd = None if sup is None else sup.command.get(GRIPPER)
        if cmd is None or measured - cmd <= self.grip_squeeze_margin:
            return False
        if ls is None:
            return True
        now = self._clock if self._manual else time.monotonic()
        if ls.grip_tight_since is None or now - ls.grip_tight_since < self.grip_block_s:
            return False
        a, b = ls.measured.get(GRIPPER), ls.prev_measured.get(GRIPPER)
        return a is None or b is None or abs(float(a) - float(b)) <= self.grip_still

    def _measured_now(self) -> dict[str, float]:
        """Gdzie kazdy staw sie zatrzyma, gdy teraz dostanie rozkaz "stoj" - poza do trzymania.

        Odczyt ma do 40-70 ms (+ takt do wyslania trzymania), a serwo w tym czasie jedzie
        dalej i potrzebuje jeszcze drogi hamowania v^2/2a (`servo_accel`). Zmierzone
        (emulator STS3215, ~2230 st./s^2): trzymanie punktu bez drogi hamowania cofalo
        wrist_roll o 6-14 st., Dom w biegu wysylal skok 7 st. wstecz. Wynik jest przyciety
        miedzy pomiar a ostatni wyslany rozkaz: dalej niz tam serwo i tak by nie pojechalo,
        a po kolizji predkosc jest zerowa i zostaje czysty pomiar (bez docisku do przeszkody).

        Chwytak: tylko w strone otwierania - zamykajaca sie szczeka trzyma pomiar, bo STOP
        nie moze domykac szczek (palec). Cofniecie o ulamek to lekkie otwarcie.

        Predkosc z dwoch odczytow co 40 ms stemplowanych poczatkiem taktu, a nie chwila
        probki - jitter lacza zanizal ja o ~25% i serwo i tak cofalo sie do 12,7 st.
        (emulator, 10 ms +-5 ms). Stad zapas `vel_margin`; nadmiar i tak przycina ostatni
        rozkaz. Granica to dalszy z (ostatni wyslany, biezacy rozkaz nadzoru): STOP z innego
        watku miedzy krokiem nadzoru a wysylka trzymal punkt krok za rozkazem, ktory zaraz
        poszedl (wrist_roll: 4,4 st. wstecz w jednym takcie).
        """
        ls, sup = self._ls, self._supervisor
        m = dict(self._latest)
        if ls is None or not ls.prev_measured:
            return m
        span = ls.measured_t - ls.prev_measured_t
        if span <= 1e-6:
            return m
        now = self._clock if self._manual else time.monotonic()
        horizon = min(max(now - ls.measured_t, 0.0) + 1.0 / self.loop_hz, 2 * self._read_period)
        cmd = sup.command if sup is not None else {}
        for k, v in m.items():
            if k not in ls.prev_measured or k not in ls.last_sent:
                continue
            vel = (v - ls.prev_measured[k]) / span * self.vel_margin
            if k == GRIPPER and vel < 0:
                continue
            est = v + vel * horizon + vel * abs(vel) / (2.0 * self.servo_accel)
            c = ls.last_sent[k]
            c2 = cmd.get(k, c)
            if (c2 - v) * (c - v) >= 0 and abs(c2 - v) > abs(c - v):
                c = c2
            m[k] = min(max(est, min(v, c)), max(v, c))
        return m

    def _squeeze_keep(self) -> dict[str, float]:
        """{chwytak: rozkaz}, gdy chwytak sciska - rampa do domu nie otwiera go po drodze."""
        sup = self._supervisor
        if sup is None or self._grip_release:
            return {}
        b = self._backend
        measured = self.joints() if b is not None and b.info.simulated else dict(self._latest)
        cmd = sup.command.get(GRIPPER)
        if cmd is None or GRIPPER not in measured:
            return {}
        if self._grip_squeezing(float(measured[GRIPPER])):
            return {GRIPPER: cmd}
        return {}

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
        # Bez `max_relative_target` (D3): LeRobot przycinal cel do +-12 st. od pozycji, wiec
        # rozjazd pomiar-rozkaz nie przekraczal ~12 st., straznik rozjazdu (25 st.) nie mogl
        # zadzialac, a serwo dociskalo do przeszkody bez konca (zmierzone na prawdziwym stosie
        # LeRobota nad emulatorem; bez limitu STOP po 0,78 s). Nadzor blizniaka i tak ogranicza
        # predkosc. Aplikacja dloni ma wlasna konfiguracje - bez zmian.
        cfg = load_config(overrides={"robot": {"backend": "sim" if backend == "sim" else backend,
                                               "port": port, "max_relative_target": None}})
        b: RobotBackend | None = None
        try:
            b = SceneBackend(self) if backend == "sim" else create_backend(cfg)
            b.connect()
            measured = _stable_read(b, self.start_read_tol)
            _intersect_limits(cfg, b)
        except Exception as exc:
            self.status = RobotStatus(error=f"{type(exc).__name__}: {exc}")
            if b is not None:
                try:                                       # port nie moze zostac otwarty po odmowie
                    b.disconnect()
                except Exception:                          # pragma: no cover - sprzatanie po bledzie
                    logger.exception("Blad przy rozlaczaniu po nieudanym polaczeniu")
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
        self._grip_floor, self._grip_release = None, False
        self._warnings = {}
        grip = _gripper_mismatch(b, self.grip_ticks_tol)
        if grip:
            logger.warning(grip)
            self._warnings["grip_ticks"] = grip
        # Backend `lerobot` liczy katy od srodka zakresu kalibracji, nie od tiku 2048 jak
        # blizniak: na typowej kalibracji -2 st. na shoulder_pan i -6,4 st. na elbow_flex,
        # przy jednostronnym zakresie (stary wrist_flex ramienia nr 1) ok. 35 st.
        try:
            calib = list(getattr(b, "calibration_warnings", lambda: [])() or [])
        except Exception:
            calib = []
        for i, w in enumerate(calib):
            logger.warning(w)
            self._warnings[f"calib_{i}"] = str(w)
        ws.backend, ws.port = backend, port
        self.status = RobotStatus(connected=True, backend=backend, simulated=b.info.simulated,
                                  state=self._supervisor.state.value, measured=measured,
                                  command=self._supervisor.command, warnings=list(self._warnings.values()))
        self._stop.clear()
        self._manual = not threaded
        self._alive = True
        self._clock = 0.0
        now = self._clock if self._manual else time.monotonic()
        self._ls = _LoopState(prev=now, measured=dict(measured), measured_t=now, t_hz=now,
                              last_sent=self._supervisor.command)
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

    def set_engaged(self, engaged: bool, owner: str | None = None) -> bool:
        """Sprzeglo: bez niego cele z UI sa ignorowane, a ramie trzyma pozycje.

        Z `owner` - tylko, jesli ten wlasciciel wciaz ma ramie (sprawdzenie i zmiana
        razem, pod blokada wlasnosci); zwraca, czy sprzeglo zmieniono. Bez tego
        runner odebrany STOP-em miedzy `claim` a sprzeglem wylaczal sprzeglo
        NOWEGO wlasciciela (panel), a jego pole wyboru dalej pokazywalo "wlaczone".
        """
        with self._own_lock:
            if owner is not None and self._owner != owner:
                return False
            self._engaged = engaged
            if not engaged:
                self._target = None
        return True

    def set_target(self, joints: Mapping[str, float], owner: str | None = None) -> None:
        """Nowy cel. Z `owner` - tylko, jesli ten wlasciciel wciaz ma ramie (inaczej RuntimeError):
        watek sterownika, ktoremu odebrano ramie, nie moze po chwili nadpisac celu nowego."""
        cmd = self.status.command or self.joints()
        target = {**cmd, **{k: float(v) for k, v in joints.items()}}
        with self._own_lock:
            if owner is not None and self._owner != owner:
                raise RuntimeError(f"ramie odebrane ({self._refusal(owner)})")
            self._target = target

    def home(self) -> None:
        """Pozycja domowa: odbiera ramie wlascicielowi (polityka, fala, identyfikacja) i jedzie rampa."""
        if self._supervisor is None:
            return
        self.preempt("pozycja domowa")
        with self._sup_lock:
            # Chwytak sciskajacy kostke jedzie do domu zamkniety: otwarcie w rampie
            # upuszczalo ja z wysokosci, na ktora ja podniesiono.
            self._supervisor.begin_homing(keep=self._squeeze_keep())

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
        """Kasuje STOP. Blad chwytaka, jesli trwa, liczy sie od nowa (znow `grip_fault_s` na odciazeniu).

        Bez restartu licznika STOP od bledu chwytaka wracal w nastepnym takcie: skasowanie,
        sprzeglo i suwak chwytaka na 100 byly nadpisane, zanim szczeki drgnely - kostki nie
        dalo sie wypuscic z panelu (emulator: 3 proby, 3 razy STOP po ~0 ms). Bit bledu,
        ktory serwo trzyma do ponownego wlaczenia momentu, kasuje dopiero "Polacz".
        """
        if self._supervisor is not None:
            with self._sup_lock:
                self._supervisor.clear_estop()
            self._fault, self._note = "", ""
            self._grip_floor = None                        # blad chwytaka, jesli trwa, wroci w nastepnym takcie
            self._grip_release = False
            self._warnings.pop("grip_eased", None)
            ls = self._ls
            if ls is not None:
                ls.grip_fault_since, ls.grip_ok_since = None, None

    def move(self, joints: Mapping[str, float], duration: float, settle: float = 0.4,
             owner: str = "kalibracja", take: bool = True) -> None:
        """Blokujacy przejazd rampa smoothstep - dla sesji kalibracji (`session.Robot`).

        Ruch nalezy do `owner`: gdy ramie jest wolne, bierze je (i zostawia - fala
        to wiele przejazdow, a konczy ja `home()`, ktory ramie odbiera); gdy ma
        je ten sam wlasciciel - jedzie; gdy inny - RuntimeError "ramie zajete".
        Bez tego fala i polityka pisaly cel na zmiane. `take=False`: wolnego ramienia
        nie bierze, tylko RuntimeError - dla fali, ktorej ramie odebrano miedzy
        przejazdami (inaczej jechalaby dalej na ramieniu polaczonym w tym czasie).

        RuntimeError, gdy w trakcie ramie odebrano (dom, STOP, inny wlasciciel),
        rozlaczono albo polaczono na nowo - wtedy fala nie moze liczyc na poze,
        ktorej ramie nie osiagnelo. Takze, gdy po `settle` (+ `move_arrive_s`) ramie nie
        stoi przy rozkazie (`_arrival`) - fala brala kadry i FK z ramienia w biegu.

        Zegar rampy stoi, gdy stoi petla (lacze niepotwierdzone): nadzor wtedy nie jedzie,
        a zegar liczony dalej konczyl rampe w czasie przestoju i po nim rozkaz gonil cel
        z pelna predkoscia.
        """
        if not self.connected:
            raise RuntimeError("ramie nie jest polaczone")
        self._wait_for_homing()
        with self._own_lock:
            if self._owner is None:
                if not take:
                    raise RuntimeError(f"ruch przerwany: ramie odebrane ({self._refusal(owner)})")
                self._new_owner()
                self._owner, self._preempt_cb = owner, None
                self._lost.pop(owner, None)
            elif self._owner != owner:
                raise RuntimeError(f"ramie zajete: {self._owner}")
            gen = self._gen
            # Sprzeglo razem z wlasnoscia: odebranie ramienia (STOP) tuz po tym
            # bloku nie zostawi wlaczonego sprzegla bez wlasciciela.
            self._engaged = True
        start = dict(self.status.command or self.joints())
        goal = {k: float(joints.get(k, v)) for k, v in start.items()}
        t0 = t_prev = time.monotonic()
        paused = 0.0
        while True:
            self._check_move(gen)
            now = time.monotonic()
            if self._frozen():
                paused += now - t_prev
            t_prev = now
            s = smoothstep((now - t0 - paused) / max(duration, 1e-3))
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
        deadline = t_end + self.move_arrive_s
        prev_cmd: dict[str, float] | None = None
        while True:
            self._check_move(gen)
            why, prev_cmd = self._arrival(prev_cmd)
            if not why:
                return
            if time.monotonic() > deadline:
                raise RuntimeError(f"ruch nie dojechal do celu w {settle + self.move_arrive_s:.1f} s: {why}")
            time.sleep(0.05)

    def _frozen(self) -> bool:
        """Petla stoi: lacze z serwami niepotwierdzone albo utracone."""
        ls = self._ls
        return ls is not None and ls.frozen

    def _arrival(self, prev_cmd: dict[str, float] | None) -> tuple[str, dict[str, float]]:
        """("", rozkaz), gdy stawy ramienia stoja przy rozkazie; inaczej (dlaczego nie, rozkaz).

        Rozkaz porownywany z poprzednim wywolaniem: nadzor moze jeszcze gonic cel
        (po przestoju, staw spoza limitow). Przy rozkazie = najwyzej `move_arrive_deg`
        (serwo pod ciezarem ma staly uchyb), stoi = wolniej niz `move_still_speed`.
        """
        st = self.status
        cmd, meas = dict(st.command), dict(st.measured)
        if self._frozen():
            return "lacze z serwami niepotwierdzone", cmd
        if prev_cmd is None:
            return "sprawdzanie", cmd
        arm = [k for k in cmd if k != GRIPPER]
        going = [k for k in arm if abs(cmd[k] - prev_cmd.get(k, cmd[k])) > 0.2]
        if going:
            return f"rozkaz jeszcze jedzie ({', '.join(going)})", cmd
        far = [(k, meas[k] - cmd[k]) for k in arm if k in meas and abs(meas[k] - cmd[k]) > self.move_arrive_deg]
        if far:
            return ", ".join(f"{k} {abs(e):.0f} st. od rozkazu" for k, e in far), cmd
        speed = self._joint_speed()
        fast = [(k, v) for k, v in speed.items() if k != GRIPPER and abs(v) > self.move_still_speed]
        if fast:
            return ", ".join(f"{k} jeszcze jedzie ({abs(v):.0f} st./s)" for k, v in fast), cmd
        return "", cmd

    def _joint_speed(self) -> dict[str, float]:
        """Predkosc stawow [jednostka/s] z dwoch ostatnich swiezych odczytow."""
        ls = self._ls
        if ls is None or not ls.prev_measured:
            return {}
        span = ls.measured_t - ls.prev_measured_t
        if span <= 1e-6:
            return {}
        return {k: (float(v) - float(ls.prev_measured[k])) / span
                for k, v in ls.measured.items() if k in ls.prev_measured}

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
            fresh = False
            if b.info.simulated or now - ls.last_read >= self._read_period:
                before, before_t = ls.measured, ls.measured_t
                ls.measured = b.read_joints()
                ls.last_read = now
                self._latest = dict(ls.measured)
                # Odczyt bez zadnej odpowiedzi (backend oddaje stare pozycje) nie jest
                # pomiarem: `measured_t` stoi, a identyfikacja nie nagrywa go jako nowego.
                fresh = not _read_silent(b)
                if fresh:
                    ls.prev_measured, ls.prev_measured_t = before, before_t
                    ls.measured_t = now
                    ls.hist.append((now, dict(ls.measured)))
                    if ls.hist[0][0] < now - 2.0 * self.track_err_s:
                        ls.hist = [h for h in ls.hist if h[0] >= now - 2.0 * self.track_err_s]
                    if ls.silent:
                        self._confirm_after_silence(sup, ls, now)
                else:
                    ls.silent, ls.confirm = True, None
            g, c = ls.measured.get(GRIPPER), sup.command.get(GRIPPER)
            if g is not None and c is not None and float(g) - c > self.grip_squeeze_margin:
                if ls.grip_tight_since is None:
                    ls.grip_tight_since = now
            else:
                ls.grip_tight_since = None

            faults = _backend_faults(b)
            grip_faults = [f for f in faults if _fault_joint(f) == GRIPPER]
            arm_faults = [f for f in faults if _fault_joint(f) != GRIPPER]
            if grip_faults:
                ls.grip_ok_since = None
                grip_faults_long = self._ease_gripper(sup, ls, now, grip_faults)
            else:
                ls.grip_fault_since, grip_faults_long = None, False
                self._end_easing(sup, ls, now)
            if grip_faults_long and not arm_faults and not self._grip_release:
                # STOP od bledu chwytaka: szczeka staje na zmierzonym rozwarciu. Docisk (nawet
                # odciazony) podtrzymywal przeciazenie, ktore do tego STOP-u doprowadzilo.
                self._grip_release = True
                g = ls.measured.get(GRIPPER)
                if sup.estopped and g is not None:     # STOP juz byl - tylko chwytak bez docisku
                    with self._sup_lock:
                        sup.reseed({GRIPPER: float(g)})
            if arm_faults or grip_faults_long:
                # Blad serwa ramienia (albo chwytaka, ktory mimo odciazenia trwa) = STOP.
                msg = "; ".join(faults)
                if not sup.estopped:
                    logger.error("Serwa zglaszaja blad - STOP: %s", msg)
                    self.estop(msg)
                self._fault = msg
            link_down = any("brak odpowiedzi" in f for f in arm_faults)
            if ls.link_down and not link_down:
                # Lacze wrocilo: trzymanie od POTWIERDZONEGO odczytu (dwa zgodne), nie od
                # pierwszego - ten bywa spozniona odpowiedzia sprzed utraty (`_confirm_after_silence`).
                ls.recover_hold = True
                if not ls.silent:                          # czesc serw milczala - ten odczyt to kandydat
                    ls.silent, ls.confirm, ls.confirm_since = True, dict(ls.measured), now
            ls.link_down = link_down

            if not b.info.simulated and gap > self.max_gap:
                # Przerwa w petli to tez klopot z laczem: odczyt, ktory trzymal petle 0,8 s, oddal
                # spozniona odpowiedz sprzed przestoju, a nastepny druga, identyczna (emulator) -
                # trzymanie z nich (predkosc zero = "zablokowany") cofalo serwo o 7 st. z 85 st./s.
                # Petla stoi do dwoch zgodnych odczytow, trzymanie idzie od nich.
                ls.silent, ls.confirm, ls.recover_hold = True, None, True
                self._note, self._note_t = f"przerwa w petli sterowania ({gap * 1000:.0f} ms) - ruch przerwany", now
                logger.warning(self._note)
                self.preempt(self._note)

            if fresh:
                self._check_tracking(b, sup, ls, now)

            # Lacze niepotwierdzone: nadzor stoi. Zmierzone przez most: przestoj 0,9 s
            # (ponizej progu utraty lacza) - nadzor liczyl dalej, nic nie szlo, a pierwszy
            # rozkaz po powrocie skakal 10,5 -> 27,2 st. i serwo jechalo tam z pelna predkoscia.
            # Stoi tez do potwierdzenia pozy dwoma zgodnymi odczytami.
            frozen = ls.silent or link_down
            ls.frozen = frozen
            # Cel, sprzeglo i krok nadzoru razem pod blokada nadzoru: `preempt`/`hold_measured`
            # z innego watku miedzy odczytem celu a krokiem dawaly jeszcze jeden takt
            # w strone starego celu (po kolizji - dalej w przeszkode).
            with self._sup_lock:
                if not sup.is_homing_done:
                    self._target = None                    # cel sprzed rampy nie moze wrocic po jej koncu
                engaged = self._engaged
                desired = self._target if engaged else None
                # Obecnosc operatora w blizniaku to sprzeglo z UI, nie dlon w kadrze.
                command, report = sup.step(desired, 0.0 if frozen else dt, hand_present=True, engaged=engaged)
                floor = self._grip_floor
                if floor is not None and command.get(GRIPPER, floor) < floor:
                    sup.reseed({GRIPPER: floor})           # chwytak odciazony: nie ciasniej niz prog
                    command = sup.command
                outside = sup.outside
            if engaged or report.state in _MOVING:
                ls.moved = True
            # Prawdziwe ramie po "polacz bez ruchu" trzyma cel wpisany przy polaczeniu
            # (tam, gdzie stoi) - do pierwszego ruchu nic nie wysylamy: backend przycina
            # rozkaz do limitow, wiec staw spoza nich dostawal skok od razu po polaczeniu.
            if not frozen and (b.info.simulated or ls.moved):
                self._sync_backend_limits(b, outside)
                sent = b.send_joints(command) or {}
                if command and not sent:
                    # Backend wstrzymal wysylke (lacze milczy) - od nastepnego taktu nadzor
                    # stoi, a pierwszy dobry odczyt zacznie rozkaz od zmierzonej pozy.
                    ls.silent = True
                else:
                    sent_cmd = dict(command)
                    sent_cmd.update({k: float(v) for k, v in sent.items() if k in sent_cmd})
                    off = {k: v for k, v in sent_cmd.items() if abs(v - command[k]) > 0.5}
                    if off:
                        # Serwo przycielo rozkaz (limit w EEPROM) - nadzor liczy dalej od
                        # tego, co naprawde poszlo, i tak samo nagrywa to identyfikacja.
                        with self._sup_lock:
                            sup.reseed(off)
                    ls.last_sent = sent_cmd
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
                                      measured=dict(ls.measured), command=dict(ls.last_sent),
                                      at_limit=list(report.at_limit), loop_hz=hz,
                                      error=self._fault or self._note, measured_t=ls.measured_t,
                                      owner=self._owner or "", faults=list(faults),
                                      warnings=list(self._warnings.values()))
            return True
        except Exception as exc:                           # odczyt z portu padl, kabel wypadl...
            logger.exception("Petla ramienia przerwana")
            reason = f"{type(exc).__name__}: {exc}"
            self.status = RobotStatus(False, self.status.backend, error=reason)
            # Martwa petla nie wysyla juz nic - wlasciciel (identyfikacja, polityka,
            # fala) musi sie o tym dowiedziec, a nie pisac dalej cele w proznie.
            with self._own_lock:
                cb = self._preempt_cb
                if self._owner is not None:
                    self._lost[self._owner] = f"petla ramienia padla: {reason}"
                self._owner, self._preempt_cb = None, None
                self._gen += 1
                self._gen_reason = self._preempt_note = f"petla ramienia padla: {reason}"
            self._engaged, self._target = False, None
            if cb is not None:
                if threading.current_thread() is self._thread:
                    threading.Thread(target=self._call_preempt, args=(cb,), daemon=True).start()
                else:
                    self._call_preempt(cb)
            return False

    def _confirm_after_silence(self, sup: SafetySupervisor, ls: _LoopState, now: float) -> None:
        """Swiezy odczyt po milczeniu lacza: petla rusza dopiero po DWOCH kolejnych zgodnych.

        Pierwsza "swieza" odpowiedz po przestoju bywa spozniona ramka sprzed niego: TCP
        oddaje ja po powrocie, a SYNC READ dopasowuje odpowiedzi tylko po ID serwa.
        Zmierzone (emulator za mostem, pan 140 st./s, przestoj 0,9 s): reseed do niej
        cofal rozkaz o 6-7 st. i serwo zawracalo z ~90 st./s - takze po STOP-ie
        wcisnietym w przestoju; jedna odpowiedz spozniona o 0,3 s cofala o 2-4 st.
        Petla stoi (nic nie wysyla), serwo dojezdza do ostatniego rozkazu i staje, a dwa
        odczyty zgodne do `start_read_tol` to juz poza, w ktorej ono naprawde jest.
        Niezgodne dluzej niz `confirm_timeout_s` - STOP z powodem (petla dalej stoi).
        """
        cand, ls.confirm = ls.confirm, dict(ls.measured)
        if cand is None:
            ls.confirm_since = now
            return
        common = cand.keys() & ls.measured.keys()
        worst = max((abs(float(ls.measured[k]) - float(cand[k])) for k in common), default=0.0)
        if worst <= self.start_read_tol:
            self._reseed_after_silence(sup, ls, now, cand)
            return
        since = ls.confirm_since if ls.confirm_since is not None else now
        if now - since >= self.confirm_timeout_s and not sup.estopped:
            msg = (f"odczyty serw po przerwie lacza sie nie zgadzaja (roznica {worst:.1f} przez "
                   f"{now - since:.1f} s) - ramie stoi; sprawdz lacze")
            logger.error("STOP: %s", msg)
            self._fault = msg
            self.estop(msg)

    def _reseed_after_silence(self, sup: SafetySupervisor, ls: _LoopState, now: float,
                              first: Mapping[str, float] | None = None) -> None:
        """Potwierdzona poza po milczeniu lacza: rozkaz od nowa od niej (`first` - pierwszy z dwoch odczytow).

        Dopoki lacze milczalo, nadzor stal, wiec rozkaz = ostatni wyslany. Serwo moglo
        jednak nie dojechac (przeszkoda) - start od pomiaru, jak po utracie lacza.
        Cel zostaje: ruch jedzie dalej od miejsca, gdzie ramie JEST, z limitem predkosci.
        Po utracie lacza (STOP) - `hold_measured` tej pozy.

        Ta sama regula co przy STOP-ie (`_measured_now`): staw, ktory miedzy dwoma odczytami
        jedzie do ostatniego wyslanego rozkazu albo stoi najwyzej krok nadzoru (max_vel x 1,5
        taktu) przed nim, trzyma ten rozkaz - serwo i tak tam dojedzie. Rozkaz wyslany tuz przed
        przestojem potrafi dojsc do serwa PO nim (TCP); reseed do pomiaru cofal wtedy serwo
        (emulator: +4,2 st. do spoznionego rozkazu, zaraz potem -5,5 st. z powrotem, do
        ~90 st./s). Staw stojacy dalej (przeszkoda) - pomiar, bez docisku.
        """
        ls.silent, ls.confirm, ls.confirm_since = False, None, None
        ls.lag_since.clear()
        ls.hist = [(now, dict(ls.measured))]
        pose = {k: float(v) for k, v in ls.measured.items()}
        for k, v in pose.items():
            c = ls.last_sent.get(k)
            if k == GRIPPER or c is None or k not in sup.cfg.joints:
                continue
            step = sup.cfg.joint(k).max_vel * sup.cfg.safety.velocity_scale * 1.5 / self.loop_hz
            toward = 0.0 if first is None or k not in first else (v - float(first[k])) * (1.0 if c >= v else -1.0)
            if abs(c - v) <= step + 0.5 or toward > 0.3:
                pose[k] = c
        if ls.recover_hold:
            ls.recover_hold = False
            self.hold_measured(pose)
        else:
            with self._sup_lock:
                sup.reseed(self._hold_pose(pose))
        logger.info("Lacze z serwami wrocilo - rozkaz od potwierdzonej, zmierzonej pozy.")

    def _end_easing(self, sup: SafetySupervisor, ls: _LoopState, now: float) -> None:
        """Odciazenie chwytaka znika, gdy bledu nie ma od `grip_clear_s`, a szczeka juz nie sciska.

        Zostawalo na cala sesje panelu: po chwilowym przeciazeniu pusty chwytak zamykal sie
        tylko do starego progu (22 zamiast 0) i ostrzezenie wisialo. Dopoki szczeka trzyma
        (rozkaz = prog, ciasniej niz pomiar) - zostaje, bo powrot pelnego docisku znow by
        przeciazyl serwo. Otwarcie (rozkaz nie ciasniej niz pomiar) = obiekt puszczony.
        """
        if self._grip_floor is None:
            return
        if ls.grip_ok_since is None:
            ls.grip_ok_since = now
        g, cmd = ls.measured.get(GRIPPER), sup.command.get(GRIPPER)
        if g is None or cmd is None or now - ls.grip_ok_since < self.grip_clear_s:
            return
        if float(g) - cmd <= self.grip_squeeze_margin:
            self._grip_floor = None
            self._warnings.pop("grip_eased", None)
            logger.info("Chwytak puscil - odciazenie po bledzie zdjete.")

    def _ease_gripper(self, sup: SafetySupervisor, ls: _LoopState, now: float, faults: list[str]) -> bool:
        """Blad TYLKO serwa chwytaka: szczeki luzniej zamiast STOP-u. True = trwa za dlugo (STOP).

        Mocny chwyt kostki wyzwala przeciazenie serwa chwytaka (LeRobot zapisuje mu
        niski prog w EEPROM). STOP na kazdy blad trzymal zmierzony kat szczek, czyli
        zwalnial chwyt - w emulatorze kostka niesiona w powietrzu wypadala. Tu rozkaz
        chwytaka idzie najwyzej `grip_ease` ciasniej niz pomiar (chwyt zostaje, prad
        spada), panel dostaje ostrzezenie, a STOP dopiero, gdy blad trwa `grip_fault_s`.
        """
        if ls.grip_fault_since is None:
            ls.grip_fault_since = now
            logger.warning("Blad serwa chwytaka - docisk zmniejszony: %s", "; ".join(faults))
        measured = ls.measured.get(GRIPPER)
        if measured is not None:
            floor = float(measured) - self.grip_ease
            if self._grip_floor is None or floor > self._grip_floor:
                self._grip_floor = floor
            with self._sup_lock:
                if sup.command.get(GRIPPER, floor) < self._grip_floor:
                    sup.reseed({GRIPPER: self._grip_floor})
            self._warnings["grip_eased"] = (f"{'; '.join(faults)} - docisk chwytaka zmniejszony (rozkaz nie ciasniej "
                                            f"niz {self._grip_floor:.0f}); STOP, jesli blad potrwa "
                                            f"{self.grip_fault_s:.0f} s")
        return now - ls.grip_fault_since >= self.grip_fault_s

    def _check_tracking(self, b: RobotBackend, sup: SafetySupervisor, ls: _LoopState, now: float) -> None:
        """Straznik rozjazdu pomiar-rozkaz dla kazdego wlasciciela (patrz `track_err_deg`).

        STOP, gdy staw jest dalej od rozkazu niz `track_err_deg` od `track_err_s` I przez
        ostatnie `track_err_s` nie jechal do rozkazu co najmniej `track_min_speed`. Samo
        "daleko od rozkazu" to takze serwo wolniejsze od nadzoru i odczyt sprzed 40-70 ms
        (most) - pelne przejazdy suwakiem dawaly falszywy STOP. Staw zablokowany albo bez
        momentu stoi albo odjezdza, wiec STOP przychodzi jak dotad (~0,5-1 s).

        Chwytak pominiety - szczeka zablokowana na kostce odstaje od rozkazu z definicji.
        """
        if not ls.moved or sup.estopped or not (self._engaged or sup.state in _MOVING):
            ls.lag_since.clear()
            return
        stale = _stale_joints(b)
        # Odczyt sprzed ~track_err_s (najstarszy, jesli historia krotsza) - postep stawu.
        past = ls.hist[0] if ls.hist else None
        for t_h, pose in ls.hist:
            if t_h > now - self.track_err_s:
                break
            past = (t_h, pose)
        late = []
        for name in JOINT_NAMES:
            if name == GRIPPER or name in stale or name not in ls.measured or name not in ls.last_sent:
                continue
            err = ls.last_sent[name] - ls.measured[name]
            if abs(err) <= self.track_err_deg:
                ls.lag_since.pop(name, None)
                continue
            t0 = ls.lag_since.setdefault(name, now)
            if now - t0 < self.track_err_s:
                continue
            if past is not None and name in past[1] and now - past[0] > 1e-6:
                toward = (ls.measured[name] - past[1][name]) * (1.0 if err > 0 else -1.0)
                if toward >= self.track_min_speed * (now - past[0]):
                    continue                               # jedzie do rozkazu - wolne serwo, nie kolizja
            late.append((name, now - t0, err))
        if not late:
            return
        name, held, err = late[0]
        msg = (f"{name}: {abs(err):.0f} st. od rozkazu przez {held:.1f} s i nie jedzie do niego - serwo "
               f"stracilo moment albo kolizja")
        logger.error("Rozjazd pomiar-rozkaz - STOP: %s", msg)
        ls.lag_since.clear()
        self._fault = msg
        self.estop(msg)

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

#: Staw chwytaka w jednostkach aplikacji (0 = zamkniety, 100 = otwarty).
GRIPPER = "gripper"


def _fault_joint(fault: str) -> str | None:
    """Staw, ktorego dotyczy zdanie z `faults()` ("gripper (serwo 6): ..." albo "gripper: ...")."""
    head = fault.split(":", 1)[0].split(" (", 1)[0].strip()
    return head if head in JOINT_NAMES else None


def _read_silent(b: RobotBackend) -> bool:
    """Czy ostatni odczyt backendu nie dostal ZADNEJ odpowiedzi (wartosci sa stare).

    `FeetechArm` zwraca wtedy poprzednie pozycje i wstrzymuje wysylke. Backend bez
    tej wiedzy (sim, lerobot) - zawsze swiezy.
    """
    flag = getattr(b, "link_silent", None)
    if flag is None:
        flag = getattr(b, "_link_silent", False)
    try:
        return bool(flag() if callable(flag) else flag)
    except Exception:                                      # pragma: no cover - diagnostyka nie moze zabic petli
        return False


def _stale_joints(b: RobotBackend) -> set[str]:
    """Stawy bez swiezej odpowiedzi w ostatnim odczycie (`FeetechArm._stale`)."""
    try:
        return set(getattr(b, "_stale", None) or ())
    except Exception:                                      # pragma: no cover
        return set()


def _stable_read(b: RobotBackend, tol: float, tries: int = 6) -> dict[str, float]:
    """Poza startowa z dwoch KOLEJNYCH zgodnych odczytow.

    Z niej startuje nadzor, a staw spoza limitow nie jest przycinany (`keep_outside`).
    Jeden odczyt z przeklamana ramka o poprawnej sumie kontrolnej (0 tikow na
    serwie 1) dawal rozkaz -180 st. przy ramieniu stojacym na 0 i skok z pelna
    predkoscia po wlaczeniu sprzegla. Niezgodne odczyty - odmowa polaczenia.
    """
    prev = b.read_joints()
    for _ in range(tries):
        cur = b.read_joints()
        common = prev.keys() & cur.keys()
        worst = max((abs(float(cur[k]) - float(prev[k])) for k in common), default=0.0)
        if worst <= tol and not _read_silent(b):
            return cur
        prev = cur
    raise RuntimeError(f"odczyty stawow przy polaczeniu sie nie zgadzaja (roznica {worst:.1f}) - "
                       "sprawdz lacze i czy ramie stoi")


def _gripper_mismatch(b: RobotBackend, tol: float) -> str:
    """Ostrzezenie, gdy backend mapuje chwytak 0..100 na inne tiki niz blizniak (kontrakt C1)."""
    fn = getattr(b, "gripper_ticks", None)
    if fn is None:
        return ""
    try:
        got = fn()
    except Exception:                                      # pragma: no cover - diagnostyka nie moze psuc polaczenia
        logger.exception("gripper_ticks() backendu rzucilo wyjatek")
        return ""
    if not got:
        return ""
    from .kinematics import backend_gripper_ticks

    want = backend_gripper_ticks()
    if all(abs(float(a) - float(w)) <= tol for a, w in zip(got, want)):
        return ""
    fmt = "zamkniety {:.0f}, otwarty {:.0f}, zero {:.0f}"
    return (f"chwytak: backend ma tiki ({fmt.format(*got)}), a blizniak i polityki ({fmt.format(*want)}) - "
            "kat szczek w blizniaku nie zgadza sie z serwem; popraw gripper_*_ticks w konfiguracji")


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
