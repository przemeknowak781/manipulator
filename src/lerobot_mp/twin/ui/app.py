"""Panel cyfrowego blizniaka w przegladarce (viser): ramie, kamery, kalibracja, mapa, trening, polityki.

    lerobot-twin ui                      # http://localhost:8080 (tez zdalnie: --host 0.0.0.0)

Jedna petla odswieza widok (poza sceny ~30 Hz, kadry ~5 Hz, mapa ~3 Hz),
a wszystko, co trwa dluzej - fala kalibracyjna, intrynsyki, identyfikacja,
trening, ewaluacja - idzie w tle (`jobs`), wiec zaden przycisk nie blokuje
interfejsu ani petli sterowania ramieniem.

Zasada bezpieczenstwa, ta sama co w reszcie aplikacji: kazdy ruch ramienia
- z suwakow, z uchwytu w 3D, z polityki, z fali - idzie przez
`runtime.Twin` i jego `SafetySupervisor`. Przycisk STOP jest nad zakladkami,
zawsze widoczny.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from functools import wraps
from pathlib import Path
from typing import Any

import cv2
import mujoco
import numpy as np
import viser
from viser import uplot

from .. import scene as sc
from ..calib.card import Card
from ..calib.intrinsics import Board
from ..kinematics import inverse, pose
from ..perception import CubeDetector, CubeTracker, TableMapper
from ..runtime import Twin
from ..workspace import CameraRecord, Workspace, nominal_K
from . import jobs
from .bridge import SceneMirror, mat_to_wxyz
from .watch import CameraWatch, arm_mask, distort_mask

logger = logging.getLogger(__name__)

#: Wlasciciel ramienia (`Twin.claim`), gdy steruje panel: suwaki i uchwyt TCP ze sprzeglem.
PANEL_OWNER = "panel"
POLICY_OWNER = "polityka"
#: Najwiekszy skok stawu [st.] miedzy kolejnymi rozwiazaniami IK uchwytu TCP. Zmierzone:
#: przeciagniecie uchwytu o 1-2 cm z shoulder_pan ~108 st. dawalo "poprawne" IK na
#: drugiej galezi - 178-180 st. zmiany, cale ramie przez stol w ~1,3 s.
GIZMO_MAX_STEP_DEG = 15.0
#: Po takiej przerwie w przeciaganiu uchwyt TCP wraca na koncowke (Dom, polityka,
#: suwaki, przebudowa - uchwyt nie moze zostac w starym miejscu jako "cel").
GIZMO_IDLE_S = 0.5
#: Kadr starszy niz tyle [s] nie jest kadrem "na zywo" (zamrozony strumien).
FRAME_MAX_AGE = 1.0
#: Detekcja kostki starsza niz tyle [s] nie trafia do polityki jako nowa.
CUBE_MAX_AGE = 0.5
#: Niepewnosc chwili kadru [s] wzgledem jego znacznika czasu (opoznienie kamery i USB).
MASK_WINDOW = (-0.15, 0.25)
#: Najdalsze przesuniecie [px] sylwetki ramienia w masce - dalej to raczej zla historia katow.
MASK_MAX_SHIFT_PX = 60.0
#: Co tyle [px] kopia maski na drodze ramienia - ogniwo ma w kadrze kilkadziesiat px, bez przerw.
MASK_STEP_PX = 6.0


def _skip_unchanged_markdown() -> None:
    """Viser wysyla tresc markdownu przy kazdym przypisaniu, takze tej samej.

    Panel odswieza kilkanascie opisow 5 razy na sekunde - bez tego lecialo to
    przez websocket stale, nawet gdy nic sie nie zmienialo.
    """
    from viser._gui_handles import GuiMarkdownHandle

    prop = GuiMarkdownHandle.content
    if getattr(prop.fset, "_dedup", False):
        return

    def fset(self, content: str) -> None:
        if getattr(self, "_content", None) != content:
            prop.fset(self, content)
    fset._dedup = True
    GuiMarkdownHandle.content = property(prop.fget, fset)


_skip_unchanged_markdown()

GREEN, ORANGE, GREY, BLUE, RED = (40, 190, 90), (240, 160, 30), (150, 150, 150), (60, 140, 255), (230, 50, 50)
CUBE_COLORS = {"czerwona": ((0, 120, 70), (12, 255, 255), (170, 120, 70), (180, 255, 255)),
               "zielona": ((40, 80, 60), (85, 255, 255), (40, 80, 60), (85, 255, 255)),
               "niebieska": ((95, 100, 60), (130, 255, 255), (95, 100, 60), (130, 255, 255)),
               "zolta": ((20, 100, 90), (35, 255, 255), (20, 100, 90), (35, 255, 255))}


def wxyz_to_mat(q) -> np.ndarray:
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, float))
    return R.reshape(3, 3)


def fov_of(K: np.ndarray, height: int) -> float:
    return float(2 * np.arctan(height / 2 / K[1, 1]))


def thumb(img: np.ndarray, width: int) -> np.ndarray:
    h, w = img.shape[:2]
    return cv2.resize(img, (width, max(1, int(h * width / w))), interpolation=cv2.INTER_AREA)


def _px(shift: float) -> str:
    """Przesuniecie kadru z `CameraWatch` do tabeli; nieskonczone = kadr nie pasuje do odniesienia."""
    return f"{shift:.1f} px" if np.isfinite(shift) else "kadr nie pasuje do odniesienia"


#: Parametry dynamiki w panelu: (pole `Dynamics`, etykieta).
DYN_PARAMS = (("kp", "kp"), ("damping", "tlumienie"), ("armature", "armatura"), ("frictionloss", "tarcie"),
              ("delay", "opoznienie"))


def fitted_names(dyn, sysid_default: bool = False) -> tuple[str, ...]:
    """Parametry, ktore identyfikacja naprawde dopasowala (`Dynamics.fitted`).

    Starsza `Dynamics` bez tego pola: dla swiezego wyniku identyfikacji - lista,
    z ktora ja liczono (`sysid.IDENTIFIED`); dla zapisanej dynamiki - nic (nie wiadomo).
    """
    names = getattr(dyn, "fitted", None)
    if names is None and sysid_default:
        from ..rl import sysid
        names = getattr(sysid, "IDENTIFIED", ())
    return tuple(names or ())


def dynamics_parts(dyn, fitted) -> tuple[list[str], list[str]]:
    """(dopasowane, z modelu) jako teksty "tlumienie x0.95", "opoznienie 25 ms".

    Panel pisal "Dopasowano: kp x1.00 ... tarcie x1.00" takze dla parametrow, ktorych
    identyfikacja NIE ruszala (kp i tarcie zostaja z modelu) - wygladalo to jak
    pomiar "kp idealnie jak w modelu", a to tylko wartosc startowa.
    """
    fitted = set(fitted)
    a, b = [], []
    for name, label in DYN_PARAMS:
        v = float(getattr(dyn, name))
        txt = f"{label} {v / 20 * 1000:.0f} ms" if name == "delay" else f"{label} x{v:.2f}"
        (a if name in fitted else b).append(txt)
    return a, b


def smear_mask(mask: np.ndarray, shifts: list[tuple[float, float]]) -> np.ndarray:
    """Suma maski i jej kopii przesunietych o `shifts` [px] - sylwetka na drodze ramienia."""
    out = mask.copy()
    if not shifts or not mask.any():
        return out
    h, w = mask.shape
    x, y, bw, bh = cv2.boundingRect(mask.astype(np.uint8))
    crop = mask[y:y + bh, x:x + bw]                         # tylko prostokat ramienia - kilka razy szybciej
    for dx, dy in {(int(round(dx)), int(round(dy))) for dx, dy in shifts}:
        x0, y0 = x + dx, y + dy
        sx, sy = max(0, -x0), max(0, -y0)
        ex, ey = min(bw, w - x0), min(bh, h - y0)
        if (dx or dy) and sx < ex and sy < ey:
            out[y0 + sy:y0 + ey, x0 + sx:x0 + ex] |= crop[sy:ey, sx:ex]
    return out


def no_frame_image(text: str = "brak kadru", size: tuple[int, int] = (320, 240)) -> np.ndarray:
    img = np.zeros((size[1], size[0], 3), np.uint8)
    cv2.putText(img, text, (20, size[1] // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (230, 60, 60), 2, cv2.LINE_AA)
    return img


class TwinApp:
    def __init__(self, workspace: str | Path | None = None, host: str = "127.0.0.1", port: int = 8080):
        self.host = host
        self.ws = Workspace.load(workspace)
        self.ws_path = Path(self.ws.path).resolve() if self.ws.path else None
        self.twin = Twin(self.ws)
        self.watch = CameraWatch()
        self.frames: dict[str, np.ndarray] = {}
        #: Chwila wykonania kazdego kadru z `frames` (time.monotonic) - wiek detekcji kostki.
        self.frame_times: dict[str, float] = {}
        self._t_grab = 0.0
        self.frame_lock = threading.Lock()
        self.calib_job, self.intr_job = jobs.Job("kalibracja"), jobs.Job("intrynsyki")
        self.sysid_job, self.eval_job = jobs.Job("identyfikacja"), jobs.Job("ewaluacja")
        from ..rl.policy import DEFAULT_DIR
        self.policies_dir = DEFAULT_DIR.resolve()
        self.train = jobs.TrainingJob(self.policies_dir, self.ws_path)
        self.runner = None
        self.policy = None
        self.cube_det = CubeDetector()
        self.cube_tracker = CubeTracker()
        self.last_cube = None
        #: Chwila kadrow ostatniej detekcji podanej polityce - ta sama detekcja nie idzie drugi raz.
        self._cube_used_t = 0.0
        from ..kinematics import RobotKinematics
        self._kin_vision = RobotKinematics(self.ws.spec())
        # Punkty ramienia do przesuniecia maski w kadrze - wlasna kinematyka (maski licza
        # petla panelu i przyciski visera, a `_kin_vision` jest w watku polityki).
        self._kin_mask = RobotKinematics(self.ws.spec())
        self._mask_lock = threading.Lock()
        # Uchwyt TCP: wlasna kinematyka (callbacki visera ida z wielu watkow naraz,
        # a `ik` pisze do swojego MjData) i ostatnie przyjete rozwiazanie jako start IK.
        self._kin_gizmo = RobotKinematics(self.ws.spec())
        self._gizmo_lock = threading.Lock()
        self._gizmo_seed: dict[str, float] | None = None
        self._gizmo_t = 0.0
        self._engage_t = 0.0
        #: (t, katy) z ostatnich ~3 s - maska ramienia w pozie z chwili kadru, nie renderu.
        self._joint_hist: deque[tuple[float, dict[str, float]]] = deque(maxlen=120)
        self._goal_base: np.ndarray | None = None
        self._reach_task = None                         # obszar celow reach bez uruchomionej polityki
        self._map_pose_key = None
        self._intr_solve = threading.Event()
        self._dirty_rebuild = 0.0                       # kamera spoza modelu przeciagana - przebudowa po chwili
        self.mapper: TableMapper | None = None
        self._mapper_key = None
        self._stop = threading.Event()
        self.frustums: dict[str, Any] = {}
        self._frustum_key: dict[str, Any] = {}
        self._img_sent: dict[str, int] = {}
        #: Piramidy kamer tworza i usuwaja callbacki (inne watki), a obrazy wstawia petla -
        #: bez blokady petla trafiala w usuniety uchwyt i caly panel padal.
        self.scene_lock = threading.RLock()
        self._tick_errors: set[str] = set()
        self.estimates: dict[str, Any] = {}
        self._dirty_save = 0.0                          # przeciaganie kamery: zapis po chwili spokoju
        t_start = time.monotonic()

        # Serwer powstaje DOPIERO teraz, gdy cale ciezkie przygotowanie (scena, torch,
        # stanowisko) jest za nami: viser slucha od chwili utworzenia, a przegladarka
        # polaczona w polowie budowy panelu potrafila zostac z pustym widokiem.
        # Potem od razu panel (lekki), a na koncu siatki ramienia (~350 tys. trojkatow).
        self.server = viser.ViserServer(host=host, port=port, label="Blizniak SO-101", verbose=False)
        s = self.server
        s.gui.configure_theme(control_width="large", dark_mode=True, show_logo=False,
                              show_share_button=False, brand_color=(255, 196, 0))
        if hasattr(s.gui, "main_panel"):
            s.gui.main_panel.dock_right()
        s.scene.set_up_direction("+z")
        T = self.twin.scene.T_base2world
        s.initial_camera.position = tuple(T[:3, 3] + np.array([0.75, -0.65, 0.55]))
        s.initial_camera.look_at = tuple(T[:3, 3] + np.array([0.18, 0.0, 0.05]))

        self._build_header()
        tabs = s.gui.add_tab_group()
        with tabs.add_tab("Ramie", icon=viser.Icon.ROBOT):
            self._build_arm()
        with tabs.add_tab("Kamery", icon=viser.Icon.CAMERA):
            self._build_cameras()
        with tabs.add_tab("Kalibracja", icon=viser.Icon.TARGET):
            self._build_calibration()
        with tabs.add_tab("Mapa", icon=viser.Icon.MAP):
            self._build_map()
        with tabs.add_tab("Trening", icon=viser.Icon.BRAIN):
            self._build_training()
        with tabs.add_tab("Polityki", icon=viser.Icon.PLAYER_PLAY):
            self._build_policies()
        with tabs.add_tab("Sim-Real", icon=viser.Icon.ARROWS_DIFF):
            self._build_simreal()

        # Oswietlenie widoku (nie symulacji): rozproszone z nieba (+z) i jedno kierunkowe z gory.
        # three.js bierze kierunek swiatla polkuli z POLOZENIA swiatla (obrot nic nie zmienia):
        # polozenie (0, 0, 1) = niebo w +z. Obrot przy polozeniu (0, 0, 0) dawal zerowy
        # kierunek i plaskie pol na pol nieba i ziemi na kazdej powierzchni.
        s.scene.add_light_ambient("/swiatlo/otoczenie", intensity=0.5)
        s.scene.add_light_hemisphere("/swiatlo/niebo", sky_color=(235, 240, 255), ground_color=(70, 65, 60),
                                     intensity=1.6, position=(0.0, 0.0, 1.0))
        s.scene.add_light_directional("/swiatlo/gora", intensity=1.4, position=(0.6, -0.8, 2.5), cast_shadow=True)
        self.base_frame = s.scene.add_frame("/podstawa", axes_length=0.08, axes_radius=0.003,
                                            position=T[:3, 3], wxyz=mat_to_wxyz(T[:3, :3]))
        self.mirror = None
        self._mirror_version = -1
        self._sync_mirror()
        self.twin.cameras.sync()
        self._refresh_cameras()
        logger.info("Panel gotowy w %.1f s", time.monotonic() - t_start)

    # ================================================================ narzedzia
    def _safe(self, fn):
        """Wyjatek w przycisku ma trafic do uzytkownika, a nie zginac w watku visera."""
        @wraps(fn)
        def wrapper(event, *a, **kw):
            try:
                return fn(event, *a, **kw)
            except Exception as exc:
                logger.exception("Blad w panelu")
                self._notify(event, "Blad", f"{type(exc).__name__}: {exc}", error=True)
        return wrapper

    def _notify(self, event, title: str, body: str, error: bool = False) -> None:
        client = getattr(event, "client", None)
        targets = [client] if client is not None else list(self.server.get_clients().values())
        for c in targets:
            c.add_notification(title, body, auto_close_seconds=6.0 if not error else 12.0,
                               color="red" if error else "yellow")

    def _img(self, handle, key: str, img: np.ndarray) -> None:
        """Obraz do panelu/sceny tylko, gdy sie zmienil (stojaca scena = ten sam kadr)."""
        digest = hash(img.tobytes()[::97])
        if self._img_sent.get(key) != digest:
            self._img_sent[key] = digest
            handle.image = img

    def _save(self) -> None:
        self.ws.save(self.ws_path)
        self.ws_path = Path(self.ws.path).resolve()
        self.train.workspace_path = self.ws_path

    @property
    def T_b2w(self) -> np.ndarray:
        return self.twin.scene.T_base2world

    # ================================================================ naglowek
    def _build_header(self) -> None:
        g = self.server.gui
        self.status_md = g.add_markdown("**Blizniak** - uruchamianie...")
        stop = g.add_button("STOP", color="red", icon=viser.Icon.HAND_STOP,
                            hint="Zatrzymanie awaryjne: ramie staje, polityka i fala sie koncza")

        @stop.on_click
        async def _(event):
            # Asynchronicznie, w petli zdarzen visera, a sama robota od razu w nowym watku:
            # zwykly callback czekal w tej samej puli 32 watkow co reszta panelu - za
            # przebudowami sceny z przeciagania kamery STOP potrafil ruszyc po kilkudziesieciu s.
            threading.Thread(target=self._emergency_stop, args=(event,), name="STOP", daemon=True).start()

    def _emergency_stop(self, event=None) -> None:
        """STOP: najpierw nadzor (ramie staje w zmierzonej pozie), dopiero potem sprzatanie watkow."""
        try:
            self.twin.estop("STOP z panelu")
        finally:
            if self.runner is not None:
                self.runner.stop("STOP z panelu")
            self.calib_job.stop()
            self.sysid_job.stop()                       # tylko nagranie - dopasowanie jest chronione
            self.arm_engage.value = False
            self._gizmo_seed = None
        self._notify(event, "STOP", "Ramie zatrzymane. Skasuj STOP w zakladce Ramie, zeby ruszyc dalej.", error=True)

    # ------------------------------------------------ kto steruje ramieniem
    def _busy(self, me: str) -> str:
        """Kto INNY teraz rusza ramieniem ("" = nikt). Panel (sprzeglo) oddaje ramie sam - nie blokuje.

        Zadania patrzymy obok `Twin.owner`: fala bierze ramie dopiero przy pierwszym
        przejezdzie, a do tego czasu (przebudowa sceny, sprawdzacz kolizji) jest juz w toku.
        """
        if me != jobs.CALIB_OWNER and self.calib_job.running:
            return "fala kalibracyjna"
        if me != jobs.SYSID_OWNER and self.sysid_job.running and "recording" not in self.sysid_job.data:
            return "identyfikacja dynamiki"
        if me != POLICY_OWNER and self.runner is not None and self.runner.status.running:
            return "polityka"
        owner = self.twin.owner
        if owner is not None and owner not in (PANEL_OWNER, me):
            return owner
        return ""

    def _release_panel(self) -> None:
        """Panel puszcza ramie (sprzeglo off) - tylko jesli wciaz je ma."""
        self.arm_engage.value = False
        self._gizmo_seed = None
        if self.twin.owner == PANEL_OWNER:
            self.twin.set_engaged(False)
            self.twin.release(PANEL_OWNER)

    def _take_arm(self, owner: str, preempt=None) -> None:
        """Ramie dla zadania `owner` - albo RuntimeError z tym, kto je ma. Panel oddaje je bez pytania."""
        busy = self._busy(owner)
        if busy:
            raise RuntimeError(f"ramie zajete: {busy} - najpierw je zatrzymaj (albo STOP)")
        self._release_panel()
        self.twin.claim(owner, preempt=preempt)

    def _panel_preempted(self) -> None:
        """Twin odebral ramie panelowi (Dom, STOP, polaczenie, petla padla) - sprzeglo w panelu off."""
        self.arm_engage.value = False
        self._gizmo_seed = None

    def _stop_motion(self, reason: str, wait: float = 3.0) -> None:
        """Konczy wszystko, co rusza ramieniem, i czeka na zadania - przed (roz)laczeniem.

        Bez tego fala albo identyfikacja z sim jechala dalej na swiezo polaczonym
        prawdziwym ramieniu (odtworzone: shoulder_pan 14,8 -> 40 -> -39 -> 33 st.
        bez zadnej akcji operatora), z pominieciem potwierdzen z ich startu.
        """
        if self.runner is not None:
            self.runner.stop(reason)
        self.calib_job.stop()
        # Identyfikacja: przerywa tylko nagranie. Po nim zadanie jest chronione
        # (`Job.protect`) - dopasowanie ramienia nie rusza i konczy sie z wynikiem.
        self.sysid_job.stop()
        self.twin.preempt(reason)
        for job in (self.calib_job, self.sysid_job):
            if job.running and "recording" not in job.data and not job.wait(wait):
                logger.warning("Zadanie %s nie skonczylo sie w %.0f s (%s)", job.name, wait, reason)
        self._release_panel()

    def _reset_confirmations(self) -> None:
        """Kazdy przejazd prawdziwego ramienia wymaga swiezego potwierdzenia (karta, wolne miejsce)."""
        for box in (self.calib_confirm, self.dyn_confirm, self.pol_confirm):
            box.value = False

    def _status_text(self) -> str:
        st = self.twin.status
        arm = (f"{st.backend} - {st.state} - {st.loop_hz:.0f} Hz" + (f", steruje: **{self.twin.owner}**"
                                                                      if self.twin.owner else "")
               + (f" - **{st.error}**" if st.error else "") if st.connected
               else ("rozlaczone" + (f" ({st.error})" if st.error else "")))
        cams = [c for c in self.ws.cameras if c.enabled]
        trusted = sum(1 for c in cams if c.trusted)
        moved = [c.name for c in cams if self.watch.moved(c.name)]
        cam_txt = f"{len(cams)} ({trusted} skalibr.)" + (f" - **przestawiona: {', '.join(moved)}**" if moved else "")
        pol = "-"
        if self.runner is not None:
            rs = self.runner.status
            pol = (f"jedzie ({rs.step} krokow, {rs.hz:.0f} Hz)" if rs.running
                   else f"stoi ({rs.stopped_because})")
        tr = "-"
        if self.train.running:
            p = self.train.progress() or {}
            tr = f"{p.get('iteration', 0)}/{p.get('iterations', '?')} it., sukces {p.get('success', 0):.0%}"
        # Trwale ostrzezenia ramienia (kalibracja chwytaka niezgodna z blizniakiem, chwytak
        # poluzowany po przeciazeniu) - `RobotStatus.warnings`; starszy status ich nie ma.
        warns = [str(w) for w in (getattr(st, "warnings", None) or [])]
        warn_txt = f" | **Uwaga:** {'; '.join(warns)}" if warns else ""
        return f"**Ramie:** {arm} | **Kamery:** {cam_txt} | **Polityka:** {pol} | **Trening:** {tr}{warn_txt}"

    # ================================================================== ramie
    def _build_arm(self) -> None:
        g = self.server.gui
        with g.add_folder("Polaczenie"):
            self.arm_backend = g.add_dropdown("Ramie", ("sim", "feetech", "lerobot"), initial_value=self.ws.backend
                                              if self.ws.backend in ("sim", "feetech", "lerobot") else "sim",
                                              hint="sim - blizniak jest ramieniem; feetech - serwa wprost przez port")
            self.arm_port = g.add_text("Port", self.ws.port or "",
                                       hint="COM12, /dev/ttyACM0 albo socket://adres:5555 (most lerobot-mp-bridge)")
            self.ports_md = g.add_markdown("")
            scan = g.add_button("Wykryj porty", icon=viser.Icon.SEARCH)
            self.arm_home_on_connect = g.add_checkbox("Po polaczeniu jedz do pozycji domowej", False)
            connect = g.add_button("Polacz", icon=viser.Icon.PLUG_CONNECTED, color="green")
            disconnect = g.add_button("Rozlacz", icon=viser.Icon.PLUG_CONNECTED_X)

        @scan.on_click
        @self._safe
        def _(event):
            import serial.tools.list_ports as lp
            lines = []
            for p in lp.comports():
                if p.vid is None:
                    continue
                tag = " **(CH343 - SO-101)**" if p.vid == 0x1A86 else ""
                lines.append(f"- `{p.device}` {p.description}{tag}")
                if p.vid == 0x1A86 and not self.arm_port.value:
                    self.arm_port.value = p.device
            self.ports_md.content = "\n".join(lines) or (
                "Brak przejsciowek USB-serial. Na Shadow: przepusc urzadzenie USB w kliencie albo uzyj mostu "
                "`lerobot-mp-bridge` i portu `socket://adres:5555`.")

        @connect.on_click
        @self._safe
        def _(event):
            self._connect(self.arm_backend.value, self.arm_port.value.strip() or None, self.arm_home_on_connect.value)
            self._notify(event, "Polaczono", f"{self.twin.status.backend}: {self.twin.status.state}")

        @disconnect.on_click
        @self._safe
        def _(event):
            self._stop_motion("rozlaczenie")
            self._reset_confirmations()
            self.twin.disconnect()

        with g.add_folder("Sterowanie"):
            self.arm_engage = g.add_checkbox("Sprzeglo: panel steruje ramieniem", False,
                                             hint="Bez sprzegla ramie trzyma pozycje, a suwaki tylko pokazuja katy")
            home = g.add_button("Pozycja domowa", icon=viser.Icon.HOME)
            clear = g.add_button("Skasuj STOP", icon=viser.Icon.RESTORE)
            self.tcp_gizmo_on = g.add_checkbox("Uchwyt koncowki w 3D", False,
                                               hint="Przeciagnij koncowke; katy liczy odwrotna kinematyka")
            self.tcp_md = g.add_markdown("")
            spec = self.ws.spec()
            kin = self.twin.scene.kin
            self.sliders = {}
            self.slider_range = {}
            for k, name in enumerate(spec.joints):
                if name == spec.gripper:
                    lo, hi = 0.0, 100.0
                else:
                    # Zaokraglone do 0,5 st. do srodka zakresu MJCF - ladne liczby na suwaku, zadnej poza zakresem.
                    lo = float(np.ceil(np.degrees(kin.lo[k]) * 2) / 2)
                    hi = float(np.floor(np.degrees(kin.hi[k]) * 2) / 2)
                s = g.add_slider(name, lo, hi, 0.5, float(np.clip(spec.home.get(name, 0.0), lo, hi)))
                self.sliders[name] = s
                self.slider_range[name] = (lo, hi)

                def on_slide(event, name=name):
                    if event.client is None or not self.arm_engage.value:
                        return                          # zmiana z petli (sprzezenie zwrotne) albo bez sprzegla
                    self._gizmo_seed = None             # cel zmienil sie obok uchwytu TCP
                    try:
                        self.twin.set_target({name: float(self.sliders[name].value)}, owner=PANEL_OWNER)
                    except RuntimeError:                # ramie odebrane panelowi w miedzyczasie
                        self.arm_engage.value = False
                s.on_update(on_slide)

        @self.arm_engage.on_update
        def _(event):
            self._on_engage(event)

        @home.on_click
        @self._safe
        def _(event):
            self.twin.home()                            # odbiera ramie kazdemu (panel, polityka, fala)
            self._gizmo_seed = None

        @clear.on_click
        def _(event):
            self.twin.clear_estop()

        self._build_table()

    def _connect(self, backend: str, port: str | None, go_home: bool = False, **kw) -> None:
        """Polacz: najpierw koniec wszystkiego, co jezdzi, i nowe potwierdzenia (moze to byc inne ramie)."""
        self._stop_motion("ponowne laczenie")
        self._reset_confirmations()
        self.twin.connect(backend, port, go_home=go_home, **kw)
        self._save()

    def _on_engage(self, event) -> None:
        """Sprzeglo z panelu (tylko zmiany od uzytkownika - te z petli maja `client` None)."""
        if event.client is None:
            return                                      # zmiana z petli / z kodu panelu
        self._engage_t = time.monotonic()
        if not self.arm_engage.value:
            self._release_panel()
            return
        if not self.twin.connected:
            self.arm_engage.value = False
            self._notify(event, "Brak ramienia", "Najpierw polacz ramie (np. sim).", error=True)
            return
        # Sprzeglo = panel bierze ramie na wlasnosc. Gdy jedzie fala, identyfikacja albo
        # polityka - odmowa z powodem; wczesniej suwaki pisaly cel na zmiane z nimi.
        try:
            busy = self._busy(PANEL_OWNER)
            if busy:
                raise RuntimeError(f"ramie zajete: {busy}")
            self.twin.claim(PANEL_OWNER, preempt=self._panel_preempted)
        except RuntimeError as exc:
            self.arm_engage.value = False
            self._notify(event, "Sprzeglo", f"{exc} - zatrzymaj je najpierw.", error=True)
            return
        self._gizmo_seed = None
        self._gizmo_t = 0.0                         # uchwyt TCP od razu na koncowce
        self.twin.set_engaged(True)
        if self.twin.owner != PANEL_OWNER:          # odebrane (STOP, Dom) miedzy claim a sprzeglem
            if self.twin.owner is None:
                self.twin.set_engaged(False)
            self.arm_engage.value = False

    def _build_table(self) -> None:
        """Stol i uchwyt TCP w 3D."""
        g = self.server.gui
        with g.add_folder("Stanowisko (stol)", expand_by_default=False):
            t = self.ws.table_obj()
            g.add_markdown("Blat i to, gdzie na nim stoi podstawa - zmierz na biurku. Kamery sa "
                           "skalibrowane wzgledem PODSTAWY, wiec zmiana stolu ich nie rusza.")
            self.tab_size = g.add_vector2("Blat: szerokosc, glebokosc [m]", tuple(t.size), min=(0.2, 0.2),
                                          max=(3.0, 3.0), step=0.01)
            self.tab_base = g.add_vector2("Podstawa od srodka blatu [m]", tuple(t.base_xy), min=(-1.5, -1.5),
                                          max=(1.5, 1.5), step=0.005)
            self.tab_yaw = g.add_number("Obrot podstawy [st.]", float(np.degrees(t.base_yaw)), min=-180.0,
                                        max=180.0, step=1.0)
            self.tab_h = g.add_number("Wysokosc blatu nad podloga [m]", float(t.height), min=0.3, max=1.5,
                                      step=0.01)
            apply_table = g.add_button("Zastosuj", icon=viser.Icon.CHECK)

        @apply_table.on_click
        @self._safe
        def _(event):
            self.ws.table.update(size=list(self.tab_size.value), base_xy=list(self.tab_base.value),
                                 base_yaw=float(np.radians(self.tab_yaw.value)), height=float(self.tab_h.value))
            self._save()
            self.twin.rebuild()
            T_new = self.twin.scene.T_base2world
            self.base_frame.position, self.base_frame.wxyz = T_new[:3, 3], mat_to_wxyz(T_new[:3, :3])
            self._refresh_cameras()
            if self._goal_base is not None:           # cel reach jest w ukladzie podstawy - kula za nia
                self._set_goal(self._goal_base)
            self._notify(event, "Stanowisko", "Stol zapisany, scena przebudowana.")

        self.tcp_gizmo = self.server.scene.add_transform_controls("/uchwyt_tcp", scale=0.12, disable_rotations=True,
                                                                  visible=False)

        @self.tcp_gizmo_on.on_update
        def _(event):
            self.tcp_gizmo.visible = self.tcp_gizmo_on.value
            self._gizmo_seed, self._gizmo_t = None, 0.0

        @self.tcp_gizmo.on_update
        def _(event):
            self._on_tcp_gizmo()

    def _on_tcp_gizmo(self) -> None:
        """Przeciagniety uchwyt TCP -> cel stawow, tylko na tej samej galezi IK co ramie teraz.

        IK startuje WYLACZNIE z ostatniego przyjetego rozwiazania (albo z rozkazu, gdy
        przeciaganie sie zaczyna) i bez losowych startow; rozwiazanie dalej niz
        `GIZMO_MAX_STEP_DEG` od startu jest odrzucane - to druga galaz, nie ruch o 1 cm.
        """
        if not (self.arm_engage.value and self.tcp_gizmo_on.value and self.twin.connected
                and self.twin.owner == PANEL_OWNER):
            return
        p_base = (inverse(self.T_b2w) @ np.r_[self.tcp_gizmo.position, 1.0])[:3]
        gripper = self.ws.spec().gripper
        with self._gizmo_lock:
            self._gizmo_t = time.monotonic()
            cmd = {k: float(v) for k, v in (self.twin.status.command or self.twin.joints()).items()}
            seed = dict(self._gizmo_seed) if self._gizmo_seed is not None else cmd
            # restarts=0: `ik` probuje wtedy start z `seed` i z pozycji domowej - ta druga
            # wygrywa tylko, gdy start z `seed` nie trafil, i wtedy odrzuca ja limit skoku.
            sol = self._kin_gizmo.ik(p_base, seed=seed, restarts=0)
            if not sol.ok:
                self.tcp_md.content = "uchwyt poza zasiegiem ramienia - cel bez zmian"
                return
            arm = {k: float(v) for k, v in sol.joints.items() if k != gripper}
            jump = max((abs(v - float(seed.get(k, v))) for k, v in arm.items()), default=0.0)
            if jump > GIZMO_MAX_STEP_DEG:
                self.tcp_md.content = (f"odrzucone: IK chcialo skoku {jump:.0f} st. (inna galaz) - "
                                       f"przeciagaj mniejszymi krokami")
                return
            self._gizmo_seed = {**seed, **arm}
        self.tcp_md.content = ""
        try:
            self.twin.set_target(arm, owner=PANEL_OWNER)
        except RuntimeError:                            # ramie odebrane panelowi w miedzyczasie
            self.arm_engage.value = False

    def _tick_arm(self) -> None:
        st = self.twin.status
        joints = st.measured if st.connected and st.measured else self.twin.joints()
        owner = self.twin.owner
        if self.arm_engage.value and owner != PANEL_OWNER and time.monotonic() - self._engage_t > 0.5:
            # Ramie odebrane panelowi (STOP, Dom, polaczenie, petla padla) albo wziete przez zadanie.
            self.arm_engage.value = False
            self._gizmo_seed = None
        other = owner is not None and owner != PANEL_OWNER
        if self.arm_engage.disabled != other:
            self.arm_engage.disabled = other
        if not self.arm_engage.value:
            for name, s in self.sliders.items():
                if name in joints:
                    s.value = round(float(np.clip(joints[name], *self.slider_range[name])), 1)
        driving = (self.arm_engage.value and owner == PANEL_OWNER
                   and time.monotonic() - self._gizmo_t < GIZMO_IDLE_S)
        if self.tcp_gizmo_on.value and not driving:
            # Uchwyt wraca na koncowke, gdy nikt go nie ciagnie: po Domu, przebudowie, polityce
            # czy suwakach nastepne male przesuniecie celowalo w STARE miejsce koncowki.
            with self.twin.lock:
                d, site = self.twin.scene.data, self.twin.scene.kin.site_id
                p = d.site_xpos[site].copy()
            if np.abs(np.asarray(self.tcp_gizmo.position) - p).max() > 1e-4:
                self.tcp_gizmo.position = p
            self._gizmo_seed = None

    # ================================================================= kamery
    def _build_cameras(self) -> None:
        g = self.server.gui
        self.cams_md = g.add_markdown("")
        self.cam_pick = g.add_dropdown("Kamera", ("-",), initial_value="-")
        self.cam_preview = g.add_image(np.zeros((240, 320, 3), np.uint8), label="Podglad", format="jpeg",
                                       jpeg_quality=70)
        self.cam_info = g.add_markdown("")
        with g.add_folder("Wybrana kamera"):
            self.cam_enabled = g.add_checkbox("Wlaczona", True)
            self.cam_move = g.add_checkbox("Przesuwaj w 3D (kamera symulowana)", False)
            here = g.add_button("Ustaw w miejscu widoku 3D", icon=viser.Icon.CAMERA_SELFIE,
                                hint="Kamera symulowana staje tam, skad teraz patrzysz na scene")
            remember = g.add_button("Zapamietaj kadr odniesienia", icon=viser.Icon.PHOTO_CHECK,
                                    hint="Od tego kadru liczone jest wykrywanie przestawienia kamery")
            truth = g.add_button("Symulowana: uznaj prawdziwa poze za kalibracje", icon=viser.Icon.CHECK)
            remove = g.add_button("Usun kamere", icon=viser.Icon.TRASH, color="red")
        with g.add_folder("Dodaj kamere", expand_by_default=False):
            probe = g.add_button("Szukaj kamer USB", icon=viser.Icon.SEARCH)
            self.usb_pick = g.add_dropdown("Znalezione", ("-",), initial_value="-")
            add_usb = g.add_button("Dodaj kamere USB", icon=viser.Icon.PLUS)
            self.sim_fov = g.add_slider("Pole widzenia kamery sym. [st.]", 40.0, 90.0, 1.0, 62.0)
            add_view = g.add_button("Dodaj symulowana w miejscu widoku 3D", icon=viser.Icon.CAMERA_PLUS)
            add_front = g.add_button("Dodaj symulowana przed ramieniem", icon=viser.Icon.CAMERA_PLUS)
        self.cam_gizmo = self.server.scene.add_transform_controls("/uchwyt_kamery", scale=0.1, visible=False)
        self._found_usb: list[dict] = []

        @self.cam_pick.on_update
        def _(event):
            self._select_camera()

        @self.cam_enabled.on_update
        @self._safe
        def _(event):
            if event.client is None:
                return
            rec = self._cam()
            if rec is not None:
                rec.enabled = self.cam_enabled.value
                self._save()
                self.twin.rebuild()
                self.twin.cameras.sync()
                self._refresh_cameras()

        @self.cam_move.on_update
        def _(event):
            self._select_camera()

        @self.cam_gizmo.on_update
        def _(event):
            rec = self._cam()
            if rec is None or not rec.simulated or not rec.enabled or not self.cam_move.value:
                return
            T_w = pose(wxyz_to_mat(self.cam_gizmo.wxyz), np.asarray(self.cam_gizmo.position))
            self._place_sim_camera(rec, T_w, rebuild=False)

        @here.on_click
        @self._safe
        def _(event):
            rec = self._cam()
            if rec is None or not rec.simulated:
                raise RuntimeError("to dziala dla kamery symulowanej - prawdziwa stoi tam, gdzie ja postawiles")
            self._place_sim_camera(rec, self._viewer_pose(event), rebuild=True)

        @remember.on_click
        @self._safe
        def _(event):
            rec = self._cam()
            if rec is None:
                return
            self._remember_reference(rec.name)
            self._notify(event, "Zapamietano", f"{rec.name}: kadr odniesienia do wykrywania przestawienia.")

        @truth.on_click
        @self._safe
        def _(event):
            rec = self._cam()
            if rec is None or not rec.simulated or rec.sim_pose is None:
                raise RuntimeError("tylko kamera symulowana z ustawiona poza")
            rec.T_cam2base = [list(r) for r in rec.sim_pose]
            rec.calibration = {"trusted": True, "reason": "", "rms_px": 0.0, "source": "prawda symulacji",
                               "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
            self._save()
            self._refresh_cameras()
            self._remember_reference(rec.name)

        @remove.on_click
        @self._safe
        def _(event):
            rec = self._cam()
            if rec is None:
                return
            self.ws.remove_camera(rec.name)
            self.watch.forget(rec.name)
            self._save()
            self.twin.rebuild()
            self.twin.cameras.sync()
            self._refresh_cameras()

        @probe.on_click
        @self._safe
        def _(event):
            from ..cameras import probe_devices
            used = {c.source for c in self.ws.cameras}
            self._found_usb = [d for d in probe_devices() if str(d["index"]) not in used]
            opts = tuple(f"{d['index']}: {d['width']}x{d['height']}" for d in self._found_usb) or ("-",)
            self.usb_pick.options = opts
            self.usb_pick.value = opts[0]
            if not self._found_usb:
                self._notify(event, "Brak nowych kamer", "Na Shadow kamere trzeba przepuscic w kliencie (USB).")

        @add_usb.on_click
        @self._safe
        def _(event):
            if not self._found_usb or self.usb_pick.value == "-":
                raise RuntimeError("najpierw wyszukaj kamery")
            d = self._found_usb[[f"{x['index']}: {x['width']}x{x['height']}" for x in self._found_usb]
                                .index(self.usb_pick.value)]
            rec = self.ws.add_camera(CameraRecord(self.ws.free_name("kamera"), str(d["index"]), d["width"],
                                                  d["height"]))
            self._save()
            self.twin.cameras.sync()
            self._refresh_cameras(select=rec.name)

        @add_view.on_click
        @self._safe
        def _(event):
            self._add_sim_camera(self._viewer_pose(event))

        @add_front.on_click
        @self._safe
        def _(event):
            n = sum(1 for c in self.ws.cameras if c.simulated)
            az = [-0.8, 0.8, -1.5, 1.5, 0.0][n % 5]
            centre = np.array([0.2, 0.0, 0.05])
            eye = centre + np.array([0.55 * np.cos(az), 0.55 * np.sin(az), 0.35])
            z = centre - eye
            z /= np.linalg.norm(z)
            x = np.cross(z, [0, 0, 1.0])
            x /= np.linalg.norm(x)
            self._add_sim_camera(self.T_b2w @ pose(np.column_stack([x, np.cross(z, x), z]), eye))

    def _viewer_pose(self, event) -> np.ndarray:
        cam = event.client.camera
        return pose(wxyz_to_mat(cam.wxyz), np.asarray(cam.position, float))

    def _add_sim_camera(self, T_world: np.ndarray) -> None:
        W, H = 640, 480
        f = (H / 2) / np.tan(np.radians(self.sim_fov.value) / 2)
        # Kamera symulowana ma "prawdziwe" K lekko nieidealne, jak kazda kamera na biurku.
        K = np.array([[f, 0, (W - 1) / 2 + 3.0], [0, f * 0.995, (H - 1) / 2 - 2.0], [0, 0, 1]])
        rec = CameraRecord(self.ws.free_name("sym"), "sim", W, H, K=K.tolist(), intrinsics_from="symulacja")
        rec.sim_pose = (inverse(self.T_b2w) @ T_world).tolist()
        self.ws.add_camera(rec)
        self._save()
        self.twin.rebuild()
        self._refresh_cameras(select=rec.name)

    def _place_sim_camera(self, rec: CameraRecord, T_world: np.ndarray, rebuild: bool) -> None:
        rec.sim_pose = (inverse(self.T_b2w) @ T_world).tolist()
        if rebuild:
            self._save()
            self.twin.rebuild()
        else:
            # Przeciaganie: tylko poza kamery w skompilowanym modelu - przebudowa co ruch myszy bylaby za ciezka.
            def move(scene):
                cid = scene.model.camera(rec.name).id
                scene.model.cam_pos[cid] = T_world[:3, 3]
                scene.model.cam_quat[cid] = mat_to_wxyz(T_world[:3, :3] @ sc.CV_TO_MJ)
                mujoco.mj_kinematics(scene.model, scene.data)
                mujoco.mj_camlight(scene.model, scene.data)
            try:
                self.twin.render_with(move)
            except KeyError:
                # Kamery nie ma w skompilowanym modelu: przebudowa RAZ, gdy przeciaganie ucichnie
                # (petla panelu). Przebudowa w kazdym zdarzeniu myszy (~250 ms, 60 zdarzen/s)
                # zapychala pule watkow visera - STOP czekal za nimi dziesiatki sekund.
                self._dirty_rebuild = time.monotonic()
            self._dirty_save = time.monotonic()
        self._update_frustum(rec)

    def _cam(self) -> CameraRecord | None:
        try:
            return self.ws.camera(self.cam_pick.value)
        except KeyError:
            return None

    def _remember_reference(self, name: str) -> None:
        img = self.twin.cameras.frame(name)
        if img is None:
            raise RuntimeError(f"{name}: brak kadru")
        mask = (self._arm_masks([name]) or {}).get(name)
        self.watch.remember(name, img, mask)

    # ------------------------------------------------------ maska ramienia
    def _joints_at(self, t: float | None) -> dict[str, float] | None:
        """Katy ramienia z chwili `t` (historia z petli panelu); None = brak historii / teraz."""
        if t is None or not self._joint_hist:
            return None
        hist = list(self._joint_hist)
        if t - hist[-1][0] > 0.3:
            return None                                     # historia urwana (rozlaczone) - poza z teraz
        before = [j for (ti, j) in hist if ti <= t]
        return before[-1] if before else hist[0][1]

    def _arm_points(self, joints: dict[str, float]) -> np.ndarray:
        """Punkty ramienia (srodki bryl, poczatki ogniw, TCP) w ukladzie podstawy, (N, 3)."""
        kin = self._kin_mask
        with self._mask_lock:
            kin._apply(kin.to_q(joints))
            d, m = kin.data, kin.model
            body = np.asarray(m.geom_bodyid) > 0                # bez bryl swiata (podloga)
            P = np.vstack([d.geom_xpos[body], d.xpos[1:], d.site_xpos])
            Ti = kin._base_inv()
        return P @ Ti[:3, :3].T + Ti[:3, 3]

    def _arm_shifts(self, rec: CameraRecord, t: float, ref: dict[str, float] | None) -> list[tuple[float, float]]:
        """Przesuniecia [px] sylwetki ramienia w kadrze `rec` w niepewnosci chwili kadru `t`.

        Z historii katow w `MASK_WINDOW` wokol `t`: dla kazdej pozy najdalej przesuniety
        punkt ramienia (rzut przez K i poze kamery) wzgledem pozy renderu `ref`. Maska
        rozciaga sie wiec TYLKO wzdluz drogi ramienia i tylko o tyle, ile ono naprawde
        przejechalo w kadrze. Wczesniej: kwadratowe poszerzenie z najwiekszego ruchu
        STAWU (lacznie z wrist_roll, ktory prawie nie rusza sylwetki) - przy ~16 st./s
        juz na limicie 40 px, maska rosla o ~22 px na kazda strone i chowala kostke
        przy szczekach we wszystkich kamerach naraz, dokladnie w chwili chwytu.
        """
        near = [(ti, j) for (ti, j) in list(self._joint_hist) if t + MASK_WINDOW[0] <= ti <= t + MASK_WINDOW[1]]
        T = rec.true_pose()
        if len(near) < 2 or T is None:
            return []
        K, _ = rec.intrinsics()
        R, p = T[:3, :3], T[:3, 3]

        def uv(P):
            c = (P - p) @ R                                     # uklad kamery (OpenCV, z do przodu)
            z = c[:, 2]
            ok = z > 0.05
            zs = np.where(ok, z, 1.0)
            return np.stack([K[0, 0] * c[:, 0] / zs + K[0, 2], K[1, 1] * c[:, 1] / zs + K[1, 2]], 1), ok

        uv0, ok0 = uv(self._arm_points(ref if ref is not None else near[-1][1]))
        path: list[np.ndarray] = []
        placed = False
        for ti, j in near:
            if not placed and ti > t:
                path.append(np.zeros(2))                        # poza renderu - na swoim miejscu w czasie
                placed = True
            uvj, okj = uv(self._arm_points(j))
            ok = ok0 & okj
            if not ok.any():
                continue
            dv = uvj[ok] - uv0[ok]
            v = dv[int(np.argmax(np.linalg.norm(dv, axis=1)))]
            n = float(np.linalg.norm(v))
            path.append(v * (MASK_MAX_SHIFT_PX / n) if n > MASK_MAX_SHIFT_PX else v)
        if not placed:
            path.append(np.zeros(2))
        shifts = [tuple(path[0])]
        for a, b in zip(path, path[1:]):
            steps = max(1, int(np.ceil(np.linalg.norm(b - a) / MASK_STEP_PX)))
            shifts += [tuple(a + (b - a) * k / steps) for k in range(1, steps + 1)]
        return shifts

    def _arm_masks(self, names: list[str], t: float | dict[str, float] | None = None,
                   dilate: int = 9) -> dict[str, np.ndarray] | None:
        """Maski ramienia (piksele zasloniete) w geometrii SUROWYCH kadrow kamer `names`.

        Render blizniaka to kamera otworkowa w pozie z chwili renderu, a kadr jest
        z dystorsja i sprzed 0-200 ms (plus opoznienie kamery). Dlatego: poza ramienia
        z chwili kadru (historia katow) - KAZDEJ kamery z chwili jej wlasnego kadru
        (`t` jako {kamera: chwila}; jedna chwila najstarszego kadru renderowala maske
        swiezszej kamery w pozie sprzed ~150 ms), maska rozciagnieta wzdluz drogi
        ramienia w niepewnosci chwili kadru (`_arm_shifts`) i przepuszczona przez
        dystorsje kamery.
        """
        names = [n for n in names if n in {c.name for c in self.ws.cameras}]
        if not names:
            return None
        times = t if isinstance(t, dict) else {n: t for n in names}
        poses = {n: self._joints_at(times.get(n)) for n in names}
        shifts = {}
        for n in names:
            tn = times.get(n)
            if tn is not None:
                shifts[n] = self._arm_shifts(self.ws.camera(n), tn, poses[n])

        def render(s):
            q0 = s.data.qpos.copy()
            out = {}
            for n in names:
                if poses[n]:
                    s.set_joints(poses[n])                  # kopia stanu do renderu - scena bez zmian
                elif not np.array_equal(s.data.qpos, q0):
                    s.data.qpos[:] = q0                     # bez historii: poza z teraz, nie poprzedniej kamery
                    mujoco.mj_kinematics(s.model, s.data)
                out[n] = arm_mask(s, n, dilate=dilate)
            return out
        try:
            masks = self.twin.render_with(render)
        except KeyError:                                    # kamery nie ma w modelu (wylaczona, bez pozy)
            return None
        out = {}
        for n, m in masks.items():
            m = smear_mask(m, shifts.get(n, []))
            rec = self.ws.camera(n)
            if not rec.simulated:
                K, dist = rec.intrinsics()
                m = distort_mask(m, K, dist)
            out[n] = m
        return out

    def _refresh_cameras(self, select: str | None = None) -> None:
        names = tuple(c.name for c in self.ws.cameras) or ("-",)
        self.cam_pick.options = names
        if select is not None:
            self.cam_pick.value = select
        elif self.cam_pick.value not in names:
            self.cam_pick.value = names[0]
        rows = ["| kamera | zrodlo | K | poza | przestawiona |", "|---|---|---|---|---|"]
        for c in self.ws.cameras:
            if not c.calibrated:
                cal = "nieskalibrowana"
            elif c.trusted:
                cal = f"zaufana ({c.calibration.get('rms_px', 0):.2f} px)"
            else:
                # Zapisana zaufana, ale K kamery inne niz to, z ktorym ja liczono (`CameraRecord.trusted`).
                why = (c.calibration.get("reason", "") if not c.calibration.get("trusted")
                       else "intrynsyki zmienione od kalibracji polozenia - skalibruj ponownie")
                cal = f"niezaufana: {why}"
            sh = self.watch.shift.get(c.name)
            moved = "-" if sh is None else (f"**TAK {_px(sh)}**" if self.watch.moved(c.name) else f"nie ({_px(sh)})")
            rows.append(f"| {c.name}{'' if c.enabled else ' (wyl.)'} | {c.source} | {c.intrinsics_from} | {cal} | {moved} |")
        self.cams_md.content = "\n".join(rows) if self.ws.cameras else \
            "Brak kamer. Dodaj kamere USB albo symulowana (ponizej)."
        with self.scene_lock:
            for n in list(self.frustums):
                if n not in {c.name for c in self.ws.cameras}:
                    self.frustums.pop(n).remove()
                    self._frustum_key.pop(n, None)
                    if n in self.estimates:
                        self.estimates.pop(n).remove()
        for c in self.ws.cameras:
            self._update_frustum(c)
        self._select_camera()
        self._mapper_key = None

    def _update_frustum(self, c: CameraRecord) -> None:
        with self.scene_lock:
            self._update_frustum_locked(c)

    def _update_frustum_locked(self, c: CameraRecord) -> None:
        T = c.true_pose()
        if T is None or not c.enabled:
            if c.name in self.frustums:
                self.frustums[c.name].visible = False
            return
        Tw = self.T_b2w @ T
        K, _ = c.intrinsics()
        color = GREEN if c.trusted else (ORANGE if c.calibrated else (BLUE if c.simulated else GREY))
        if self.watch.moved(c.name):
            color = RED
        h = self.frustums.get(c.name)
        key = (color, round(fov_of(K, c.height), 4))
        if h is not None and self._frustum_key.get(c.name) != key:
            h.remove()                                    # kolor i pole widzenia ustawia sie przy tworzeniu
            h = None
        if h is None:
            h = self.server.scene.add_camera_frustum(f"/kamery/{c.name}", fov=fov_of(K, c.height),
                                                     aspect=c.width / c.height, scale=0.06, color=color,
                                                     thickness=0.002, wxyz=mat_to_wxyz(Tw[:3, :3]),
                                                     position=Tw[:3, 3])
            self.frustums[c.name] = h
            self._frustum_key[c.name] = key
        else:
            h.position, h.wxyz, h.visible = Tw[:3, 3], mat_to_wxyz(Tw[:3, :3]), True
        # Symulowana: gdzie ja widzi kalibracja (szara piramida) obok prawdy.
        if c.simulated and c.calibrated:
            Te = self.T_b2w @ np.asarray(c.T_cam2base, float)
            e = self.estimates.get(c.name)
            if e is None:
                self.estimates[c.name] = self.server.scene.add_camera_frustum(
                    f"/kalibracja/{c.name}", fov=fov_of(K, c.height), aspect=c.width / c.height, scale=0.06,
                    color=(200, 200, 200), thickness=0.001, wxyz=mat_to_wxyz(Te[:3, :3]), position=Te[:3, 3])
            else:
                e.position, e.wxyz = Te[:3, 3], mat_to_wxyz(Te[:3, :3])
        elif c.name in self.estimates:
            self.estimates.pop(c.name).remove()

    def _select_camera(self) -> None:
        rec = self._cam()
        if rec is None:
            self.cam_info.content = ""
            self.cam_gizmo.visible = False
            return
        self.cam_enabled.value = rec.enabled
        K, dist = rec.intrinsics()
        lines = [f"**{rec.name}** - zrodlo `{rec.source}`, {rec.width}x{rec.height}, "
                 f"fx {K[0, 0]:.0f} fy {K[1, 1]:.0f} cx {K[0, 2]:.0f} cy {K[1, 2]:.0f} ({rec.intrinsics_from})"]
        if rec.intrinsics_info:
            ii = rec.intrinsics_info
            lines.append(f"intrynsyki: residuum {ii.get('rms_px', 0):.2f} px, {ii.get('n_views', 0)} kadrow, "
                         f"pokrycie {ii.get('coverage', 0):.0%}{'' if ii.get('trusted') else ' - ' + ii.get('reason', '')}")
        if rec.simulated and rec.calibrated and rec.sim_pose is not None:
            from ..calib.handeye import pose_error
            dt, dr = pose_error(np.asarray(rec.sim_pose), np.asarray(rec.T_cam2base))
            lines.append(f"kalibracja wzgledem prawdy: **{dt * 1000:.2f} mm, {np.degrees(dr):.3f} st.**")
        err = self.twin.cameras.error(rec.name)
        if err:
            lines.append(f"blad kamery: {err}")
        self.cam_info.content = "  \n".join(lines)
        T = rec.true_pose()
        # Wylaczonej kamery nie ma w scenie - uchwyt przy niej przebudowywal scene co ruch myszy.
        show = rec.simulated and rec.enabled and self.cam_move.value and T is not None
        if show:
            Tw = self.T_b2w @ T
            self.cam_gizmo.position, self.cam_gizmo.wxyz = Tw[:3, 3], mat_to_wxyz(Tw[:3, :3])
        self.cam_gizmo.visible = show

    # ============================================================= kalibracja
    def _build_calibration(self) -> None:
        g = self.server.gui
        g.add_markdown("Kolejnosc dla nowej kamery: **1** intrynsyki (tablica w reku), **2** polozenie "
                       "(karta w chwytaku, ramie macha). Kamera symulowana ma znane K - wystarczy krok 2.")
        with g.add_folder("1. Intrynsyki - tablica ChArUco"):
            self.board_mm = g.add_number("Zmierzony bok kwadratu [mm]", 28.0, min=5.0, max=100.0, step=0.1)
            sheet = g.add_button("Pobierz arkusz tablicy (A4, PNG)", icon=viser.Icon.DOWNLOAD)
            self.intr_cam = g.add_dropdown("Kamera", ("-",), initial_value="-")
            start = g.add_button("Zbieraj kadry", icon=viser.Icon.PLAYER_RECORD)
            solve = g.add_button("Oblicz i zapisz K", icon=viser.Icon.CALCULATOR, color="green")
            self.intr_md = g.add_markdown("")
            self.intr_img = g.add_image(np.zeros((240, 320, 3), np.uint8), format="jpeg", jpeg_quality=70,
                                        visible=False)
        with g.add_folder("2. Polozenie kamer - karta w chwytaku"):
            self.tag_mm = g.add_number("Zmierzony bok taga [mm]", float(self.ws.card_obj().tag_size * 1000),
                                       min=10.0, max=120.0, step=0.1)
            card = g.add_button("Pobierz arkusz karty (A4, PNG)", icon=viser.Icon.DOWNLOAD)
            self.calib_confirm = g.add_checkbox("Karta w szczekach, przestrzen nad stolem wolna", False,
                                                hint="Prawdziwe ramie: pierwszy ruch zamyka chwytak na karcie")
            wave = g.add_button("Start fali (wszystkie wlaczone kamery)", icon=viser.Icon.WAVE_SINE, color="green")
            self.reloc_cam = g.add_dropdown("Szybka relokalizacja kamery", ("-",), initial_value="-")
            reloc = g.add_button("Relokalizuj wybrana (krotka fala)", icon=viser.Icon.CURRENT_LOCATION)
            cancel = g.add_button("Przerwij", icon=viser.Icon.PLAYER_STOP)
            self.calib_bar = g.add_progress_bar(0.0, animated=True, visible=False)
            self.calib_md = g.add_markdown("")
            apply = g.add_button("Zapisz wynik kalibracji", icon=viser.Icon.DEVICE_FLOPPY, color="green",
                                 visible=False)
            self.calib_apply = apply

        @sheet.on_click
        @self._safe
        def _(event):
            img = Board(square=self.board_mm.value / 1000, marker=0.75 * self.board_mm.value / 1000).image()
            ok, buf = cv2.imencode(".png", img)
            event.client.send_file_download("tablica_charuco.png", buf.tobytes())

        @card.on_click
        @self._safe
        def _(event):
            sheet_img = Card(tag_size=self.tag_mm.value / 1000).sheet(dpi=300)
            ok, buf = cv2.imencode(".png", sheet_img[..., ::-1])
            event.client.send_file_download("karta_kalibracyjna.png", buf.tobytes())

        @start.on_click
        @self._safe
        def _(event):
            name = self.intr_cam.value
            rec = self.ws.camera(name)
            board = Board(square=self.board_mm.value / 1000, marker=0.75 * self.board_mm.value / 1000)
            solve_evt = threading.Event()
            self.intr_job.start(lambda job: jobs.run_intrinsics(job, self.twin.cameras, name, board,
                                                                 (rec.width, rec.height), solve=solve_evt))
            self._intr_solve = solve_evt
            self.intr_img.visible = True

        @solve.on_click
        @self._safe
        def _(event):
            if not self.intr_job.running:
                raise RuntimeError("najpierw zbieraj kadry")
            self._intr_solve.set()                      # "Przerwij" to osobny sygnal - bez liczenia K

        @wave.on_click
        @self._safe
        def _(event):
            self._start_card_calibration(event, [c.name for c in self.ws.cameras
                                                 if c.enabled and self.twin.cameras.frame(c.name) is not None],
                                         quick=False)

        @reloc.on_click
        @self._safe
        def _(event):
            self._start_card_calibration(event, [self.reloc_cam.value], quick=True)

        @cancel.on_click
        def _(event):
            self.calib_job.stop()
            if self.intr_job.running:
                self.intr_job.stop()                    # przerwanie: K zostaje, jakie bylo

        @apply.on_click
        @self._safe
        def _(event):
            names = self._apply_calibration()
            bad = [f"{n}: {self.ws.camera(n).calibration.get('reason', '')}" for n in names
                   if not self.ws.camera(n).trusted]
            self._notify(event, "Zapisano", f"Kalibracja kamer: {', '.join(names)}"
                         + (f". NIEZAUFANE: {'; '.join(bad)}" if bad else ""), error=bool(bad))

    def _apply_calibration(self) -> list[str]:
        """Zapis wyniku fali - z bokiem taga i K, z ktorymi ja liczono."""
        fit = self.calib_job.result
        if fit is None:
            raise RuntimeError("brak wyniku kalibracji")
        used = self.calib_job.data.get("tag_size")
        if used is None:
            raise RuntimeError("wynik bez boku taga - uruchom fale ponownie")
        if abs(self.tag_mm.value / 1000 - used) > 1e-6:
            # Poza policzona z innym bokiem taga jest przeskalowana (50,0 -> 49,2 mm: 1,6%
            # translacji) - zapis z nowym bokiem ukrylby te niezgodnosc na zawsze.
            raise RuntimeError(f"bok taga zmieniony od fali ({used * 1000:.1f} -> {self.tag_mm.value:.1f} mm) "
                               f"- uruchom fale ponownie z nowym bokiem")
        self.ws.card["tag_size"] = used
        names = self.ws.apply_fit(fit, tag_size=used, intrinsics=self.calib_job.data.get("intrinsics"))
        self._save()
        self.twin.rebuild()
        for n in names:
            try:
                self._remember_reference(n)
            except RuntimeError:
                pass
        self._refresh_cameras()
        self.calib_apply.visible = False
        return names

    def _start_card_calibration(self, event, cameras: list[str], quick: bool) -> None:
        if not self.twin.connected:
            raise RuntimeError("polacz ramie (sim albo prawdziwe) w zakladce Ramie")
        if not cameras or cameras == ["-"]:
            raise RuntimeError("zadna wlaczona kamera nie daje kadru")
        if not self.twin.status.simulated and not self.calib_confirm.value:
            raise RuntimeError("potwierdz, ze karta jest w szczekach, a przestrzen nad stolem wolna")
        if self.calib_job.running:
            raise RuntimeError("fala juz trwa")
        # Fala bierze ramie na wlasnosc PRZED startem: jedzie polityka albo identyfikacja -
        # odmowa z powodem (wczesniej trzy watki pisaly cel na zmiane i ramie skakalo).
        # Odebranie ramienia (Dom, STOP, polaczenie) przerywa fale przez `calib_job.stop`.
        self._take_arm(jobs.CALIB_OWNER, preempt=self.calib_job.stop)
        self.ws.card["tag_size"] = self.tag_mm.value / 1000
        self.calib_apply.visible = False
        try:
            self.calib_job.start(lambda job: jobs.run_card_calibration(job, self.twin, cameras, quick))
        except Exception:
            self.twin.release(jobs.CALIB_OWNER)
            raise
        self.calib_confirm.value = False                # nastepna fala (np. relokalizacja) - nowe potwierdzenie

    def _tick_calibration(self) -> None:
        j = self.calib_job
        self.calib_bar.visible = j.running
        if j.running:
            self.calib_bar.value = 100 * j.progress
            prog = j.data.get("progress", {})
            rows = [f"- {c}: {o} obs., {p} poz, rozrzut {s:.0f} st." for c, (o, p, s) in prog.items()]
            self.calib_md.content = f"{j.message}\n\n" + "\n".join(rows)
        elif j.state == jobs.DONE and j.result is not None and not self.calib_apply.visible and \
                j.data.get("shown") is not j.result:
            fit = j.result
            rows = ["| kamera | residuum | obs. | rozrzut | werdykt |", "|---|---|---|---|---|"]
            for n, c in fit.cameras.items():
                extra = ""
                rec = next((r for r in self.ws.cameras if r.name == n), None)
                if rec is not None and rec.simulated and rec.sim_pose is not None:
                    from ..calib.handeye import pose_error
                    dt, dr = pose_error(np.asarray(rec.sim_pose), c.T_cam2base)
                    extra = f" (wzgl. prawdy {dt * 1000:.2f} mm, {np.degrees(dr):.3f} st.)"
                # Ten sam werdykt, ktory zapisze `Workspace.apply_fit`: K, z ktorym LICZONO poze
                # (z fali), musi byc zaufane i wciaz takie samo jak w kamerze.
                k_problem = rec.fit_problem((j.data.get("intrinsics") or {}).get(n)) if rec is not None else ""
                verdict = f"NIE: {c.reason}" if not c.trusted else (f"NIE: {k_problem}" if k_problem else "zaufana")
                rows.append(f"| {n} | {c.rms_px:.2f} px | {c.n_obs} | {c.spread_deg:.0f} st. | {verdict}{extra} |")
            self.calib_md.content = "\n".join(rows)
            self.calib_apply.visible = True
            j.data["shown"] = fit
        elif j.state in (jobs.FAILED, jobs.CANCELLED) and j.data.get("shown") != j.state:
            self.calib_md.content = f"**{j.state}**: {j.error or j.message}"
            j.data["shown"] = j.state

        ij = self.intr_job
        if ij.running:
            col = ij.data.get("collector")
            img = ij.data.get("last")
            if img is not None:
                vis = img.copy()
                det = ij.data.get("last_det")
                if det is not None:
                    for u, v in det[1]:
                        cv2.circle(vis, (int(u), int(v)), 3, (0, 255, 0), -1)
                self.intr_img.image = thumb(vis, 480)
            if col is not None:
                self.intr_md.content = (f"kadry: **{len(col.views)}** (zalecane 12+), pokrycie kadru "
                                        f"**{col.coverage:.0%}** - {ij.message}. Przesuwaj, przybliz i pochylaj "
                                        f"tablice; kadr zapisuje sie sam, gdy wnosi nowe ujecie.")
        elif ij.state == jobs.DONE and ij.result is not None and ij.data.get("saved") is not ij.result:
            res = ij.result
            ij.data["saved"] = res
            # Kamera, z ktorej ZBIERANO kadry - lista w panelu mogla sie w tym czasie zmienic
            # (K kamery1 ladowalo w kamerze2 i odbieralo jej zaufanie do pozy).
            name = ij.data.get("camera")
            try:
                rec = self.ws.camera(name)
            except KeyError:
                self.intr_md.content = f"**Kamery {name} juz nie ma** - K nie zapisane"
                return
            rec.K, rec.dist = res.K.tolist(), res.dist.tolist()
            rec.intrinsics_from = "szachownica"
            rec.intrinsics_info = {"rms_px": res.rms_px, "n_views": res.n_views, "coverage": res.coverage,
                                   "trusted": res.trusted, "reason": res.reason,
                                   "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
            if rec.calibrated:
                rec.calibration["reason"] = "intrynsyki zmienione po kalibracji polozenia - skalibruj ponownie"
                rec.calibration["trusted"] = False
            self._save()
            self.twin.rebuild()
            self._refresh_cameras()
            self.intr_md.content = (f"Zapisano K ({name}): fx {res.K[0, 0]:.1f}, fy {res.K[1, 1]:.1f}, "
                                    f"cx {res.K[0, 2]:.1f}, cy {res.K[1, 2]:.1f}; residuum **{res.rms_px:.3f} px**, "
                                    f"{res.n_views} kadrow" + ("" if res.trusted else f" - uwaga: {res.reason}"))
        elif ij.state == jobs.FAILED and ij.data.get("saved") != "err":
            self.intr_md.content = f"**Blad**: {ij.error}"
            ij.data["saved"] = "err"
        elif ij.state == jobs.CANCELLED and ij.data.get("saved") != "cancel":
            self.intr_md.content = "Przerwano - intrynsyki kamery bez zmian."
            ij.data["saved"] = "cancel"

    # ================================================================== mapa
    def _build_map(self) -> None:
        g = self.server.gui
        g.add_markdown("Kadry wszystkich skalibrowanych kamer przerysowane na blat i zszyte - przedmiot na stole "
                       "trafia w swoje prawdziwe (x, y), skadkolwiek patrzy kamera.")
        self.map_on = g.add_checkbox("Mapa na zywo", True)
        self.map_3d = g.add_checkbox("Pokaz mape na blacie w 3D", True)
        self.map_img = g.add_image(np.zeros((224, 224, 3), np.uint8), label="Mapa blatu", format="jpeg",
                                   jpeg_quality=80)
        self.cube_on = g.add_checkbox("Szukaj kostki", True)
        self.cube_color = g.add_dropdown("Kolor kostki", tuple(CUBE_COLORS), initial_value="czerwona")
        self.map_md = g.add_markdown("")
        self.map_node = None
        self.cube_node = self.server.scene.add_box("/percepcja/kostka", color=(255, 80, 200), dimensions=(0.03,) * 3,
                                                   opacity=0.5, wireframe=True, visible=False)

        @self.cube_color.on_update
        def _(event):
            lo, hi, lo2, hi2 = CUBE_COLORS[self.cube_color.value]
            self.cube_det.hsv_lo, self.cube_det.hsv_hi, self.cube_det.hsv_lo2, self.cube_det.hsv_hi2 = lo, hi, lo2, hi2

        def vision_off(event):
            # Bez mapy nie ma detekcji kostki - polityka z kostka "z kamer" jechalaby do
            # ostatniego polozenia. Zatrzymujemy ja jawnie, zamiast czekac na `hold_s`.
            if event.client is None or (self.map_on.value and self.cube_on.value):
                return
            if self._vision_policy_running():
                self.runner.stop("mapa albo szukanie kostki wylaczone - kostka z kamer niedostepna")
                self._notify(event, "Polityka zatrzymana", "Bez mapy i szukania kostki polityka nie wie, gdzie ona jest.")
        self.map_on.on_update(vision_off)
        self.cube_on.on_update(vision_off)

    def _vision_policy_running(self) -> bool:
        return (self.runner is not None and self.runner.status.running
                and self.runner.cube_provider == self._vision_cube)

    def _mapper_now(self) -> TableMapper | None:
        key = tuple((c.name, c.enabled, c.trusted, str(c.T_cam2base), str(c.K)) for c in self.ws.cameras)
        if key != self._mapper_key:
            self._mapper_key = key
            m = TableMapper.from_workspace(self.ws, only_trusted=True)
            self.mapper = m if m.cameras else None
        return self.mapper

    def _tick_map(self, frames: dict[str, np.ndarray]) -> None:
        """Mapa i detekcja kostki. `last_cube` dostaje NOWA detekcje albo None - nigdy stara.

        Wczesniej `last_cube` zostawal z ostatniej detekcji, gdy mapa byla wylaczona,
        brakowalo kamer albo cos rzucilo - polityka dostawala ja co takt jako swieza,
        `hold_s` trackera nie mijal i zgubiona kostka nigdy nie zatrzymywala ramienia.
        Przypisanie na koncu (zamiast None na poczatku) - watek polityki nie widzi
        "nie ma kostki" przez caly czas dopasowania.
        """
        det = None
        try:
            det = self._map_and_detect(frames)
        finally:
            self.last_cube = det

    def _map_and_detect(self, frames: dict[str, np.ndarray]):
        mapper = self._mapper_now() if self.map_on.value else None
        if mapper is None:
            self.map_md.content = "Brak zaufanej, skalibrowanej kamery - najpierw kalibracja." if self.map_on.value \
                else ""
            if self.map_node is not None:
                self.map_node.visible = False
            self.cube_node.visible = False
            return None
        table, wsum = mapper.fuse(frames)
        self._img(self.map_img, "mapa", table)
        cover = float((wsum > 1e-6).mean())
        txt = f"pokrycie blatu: **{cover:.0%}** z {len(mapper.cameras)} kamer"
        # Wiersz 0 mapy to brzeg +y podstawy, a viser kladzie wiersz 0 obrazu po stronie -y
        # wezla: bez odwrocenia wierszy mapa na blacie byla lustrem w y (kostka z +0,10 m
        # rysowala sie na -0,10 m, obok poprawnej ramki kostki).
        table3d = np.ascontiguousarray(table[::-1])
        Tm = self.T_b2w @ pose(np.eye(3), np.array([mapper.centre[0], mapper.centre[1], 0.0015]))
        if self.map_3d.value:
            if self.map_node is None:
                self.map_node = self.server.scene.add_image("/mapa", table3d, mapper.side, mapper.side,
                                                            format="jpeg", position=Tm[:3, 3],
                                                            wxyz=mat_to_wxyz(Tm[:3, :3]))
                self._map_pose_key = Tm.round(6).tobytes()
            else:
                self._img(self.map_node, "mapa3d", table3d)
                key = Tm.round(6).tobytes()
                if key != self._map_pose_key:          # stol przestawiony ("Zastosuj") - mapa za podstawa
                    self.map_node.position, self.map_node.wxyz = Tm[:3, 3], mat_to_wxyz(Tm[:3, :3])
                    self._map_pose_key = key
                self.map_node.visible = True
        elif self.map_node is not None:
            self.map_node.visible = False
        det = None
        if self.cube_on.value:
            names = [n for n in mapper.cameras if n in frames]
            now = time.monotonic()
            with self.frame_lock:
                times = dict(self.frame_times)
            # Chwila detekcji = chwila NAJSTARSZEGO uzytego kadru (konsument ocenia jej wiek).
            t_cams = {n: times.get(n, now) for n in names}
            t_frames = min(t_cams.values(), default=now)
            # Piksele zasloniete ramieniem (maska z blizniaka, w pozie z chwili kadru KAZDEJ kamery) nie glosuja.
            occ = self._arm_masks(names, t_cams, dilate=5) if names else None
            if names:
                det = self.cube_det.detect_frames({n: frames[n] for n in names}, mapper, occ, t=t_frames)
            if det is not None:
                Tw = self.T_b2w @ pose(det.rot, det.pos)
                self.cube_node.position, self.cube_node.wxyz = Tw[:3, 3], mat_to_wxyz(Tw[:3, :3])
                self.cube_node.visible = True
                yaw = np.degrees(np.arctan2(det.rot[1, 0], det.rot[0, 0]))
                # n_cameras -1 = detekcja z samej mapy (nie wiadomo, ile kamer ja widzialo)
                cams = f"{det.n_cameras} kam." if det.n_cameras >= 0 else "z mapy"
                txt += (f"  \nkostka: x {det.pos[0] * 100:.1f} cm, y {det.pos[1] * 100:.1f} cm, obrot {yaw:.0f} st., "
                        f"pewnosc {det.confidence:.2f}, {cams}, "
                        f"kadr sprzed {1000 * (time.monotonic() - det.t):.0f} ms")
            else:
                self.cube_node.visible = False
                txt += "  \nkostka: nie widac"
        else:
            self.cube_node.visible = False
        self.map_md.content = txt
        return det

    # =============================================================== trening
    def _build_training(self) -> None:
        g = self.server.gui
        with g.add_folder("Nowy trening (PPO na GPU, MuJoCo Warp)"):
            self.tr_task = g.add_dropdown("Zadanie", ("reach", "lift"), initial_value="reach")
            self.tr_envs = g.add_dropdown("Swiatow naraz", ("1024", "2048", "4096", "8192"), initial_value="4096")
            self.tr_iters = g.add_number("Iteracje", 200, min=10, max=5000, step=10,
                                         hint="reach: ~150 wystarcza (2 min); lift: 400-600 (15-25 min)")
            self.tr_rand = g.add_checkbox("Randomizacja dziedziny", True)
            self.tr_spread = g.add_slider("Szerokosc randomizacji", 0.0, 2.0, 0.1, 1.0,
                                          hint="Wokol dynamiki zmierzonej na ramieniu (ponizej)")
            self.tr_name = g.add_text("Nazwa (puste = zadanie + data)", "")
            self.tr_init = g.add_dropdown("Start z polityki", ("(od zera)",), initial_value="(od zera)",
                                          hint="Douczanie - np. po identyfikacji dynamiki na ramieniu. "
                                               "Zadanie bierze sie wtedy z polityki.")
            start = g.add_button("Ucz", icon=viser.Icon.PLAYER_PLAY, color="green")
            stop = g.add_button("Zatrzymaj (zapisze polityke)", icon=viser.Icon.PLAYER_STOP)
            self.tr_bar = g.add_progress_bar(0.0, animated=True, visible=False)
            self.tr_md = g.add_markdown("")
            self.tr_plot = g.add_uplot(
                (np.array([0.0]), np.array([0.0])),
                (uplot.Series(label="mln krokow"), uplot.Series(label="sukces", stroke="#ffc400", width=2)),
                title="Sukces w trakcie treningu", aspect=1.6, visible=False,
                scales={"y": uplot.Scale(range=(0, 1))})
        with g.add_folder("Dynamika serw (sim-to-real)"):
            self.dyn_md = g.add_markdown("")
            ident = g.add_button("Identyfikuj na polaczonym ramieniu (~20 s ruchu)", icon=viser.Icon.ACTIVITY)
            self.dyn_confirm = g.add_checkbox("Prawdziwe ramie: przestrzen wokol wolna", False)
            self.dyn_bar = g.add_progress_bar(0.0, visible=False)
            self.dyn_res = g.add_markdown("")
            keep = g.add_button("Zapisz jako dynamike stanowiska", icon=viser.Icon.DEVICE_FLOPPY, color="green",
                                visible=False)
            reset = g.add_button("Wroc do modelu Menagerie", icon=viser.Icon.RESTORE)
            self.dyn_keep = keep
        self._show_dynamics()

        @start.on_click
        @self._safe
        def _(event):
            name = self.tr_name.value.strip() or f"{self.tr_task.value}-{time.strftime('%Y%m%d-%H%M%S')}"
            init = None
            if self.tr_init.value != "(od zera)":
                from ..rl.policy import list_policies
                init = next(p["path"] for p in list_policies(self.policies_dir) if p["name"] == self.tr_init.value)
            self._save()
            self.train.start(self.tr_task.value, int(self.tr_envs.value), int(self.tr_iters.value),
                             float(self.tr_spread.value), self.tr_rand.value, name, init=init)
            self._notify(event, "Trening ruszyl", f"{name}: pierwsza iteracja po kompilacji kerneli (~20 s)")

        @stop.on_click
        def _(event):
            self.train.stop()

        @ident.on_click
        @self._safe
        def _(event):
            if not self.twin.connected:
                raise RuntimeError("polacz ramie w zakladce Ramie")
            if not self.twin.status.simulated and not self.dyn_confirm.value:
                raise RuntimeError("potwierdz, ze wokol ramienia jest wolne miejsce")
            if self.sysid_job.running:
                if "recording" in self.sysid_job.data:
                    raise RuntimeError("trwa dopasowanie poprzedniego nagrania - poczekaj na wynik")
                raise RuntimeError("identyfikacja juz trwa")
            # Ramie dla identyfikacji albo odmowa z powodem - fala czy polityka w toku mieszaly
            # swoje cele z pobudzeniem, a nagranie z obu szlo do dopasowania dynamiki.
            self._take_arm(jobs.SYSID_OWNER, preempt=self.sysid_job.stop)
            self.dyn_keep.visible = False
            try:
                self.sysid_job.start(lambda job: jobs.run_sysid(job, self.twin))
            except Exception:
                self.twin.release(jobs.SYSID_OWNER)
                raise
            self.dyn_confirm.value = False

        @keep.on_click
        @self._safe
        def _(event):
            dyn = self.sysid_job.result
            self.ws.dynamics = dyn.to_dict()
            self._save()
            self.dyn_keep.visible = False
            self._show_dynamics()

        @reset.on_click
        @self._safe
        def _(event):
            self.ws.dynamics = {}
            self._save()
            self._show_dynamics()

    def _show_dynamics(self) -> None:
        from ..rl.randomize import Dynamics
        d = Dynamics.from_dict(self.ws.dynamics)
        names = fitted_names(d)
        if names:
            fitted, fixed = dynamics_parts(d, names)
            vals = f"dopasowane: {', '.join(fitted)}" + (f"; z modelu: {', '.join(fixed)}" if fixed else "")
        else:                                           # model albo zapis bez listy dopasowanych
            vals = ", ".join(sum(dynamics_parts(d, ()), []))
        self.dyn_md.content = (f"Srodek randomizacji: **{d.source}** - {vals}"
                               + (f", blad dopasowania {d.fit_deg:.2f} st." if d.fit_deg else ""))

    def _tick_training(self) -> None:
        p = self.train.progress()
        running = self.train.running
        self.tr_bar.visible = running
        if p:
            self.tr_bar.value = 100 * p["iteration"] / max(1, p["iterations"])
            hist = p.get("history", [])
            if hist:
                self.tr_plot.visible = True
                self.tr_plot.data = (np.array([h["steps"] / 1e6 for h in hist]), np.array([h["success"] for h in hist]))
            state = "trwa" if running else ("koniec" if self.train.exit_code == 0 else
                                            f"proces zakonczony kodem {self.train.exit_code}")
            self.tr_md.content = (f"**{self.train.run_dir.name}** ({state}): iteracja {p['iteration']}/{p['iterations']}, "
                                  f"{p['steps'] / 1e6:.1f} mln krokow, {p['fps'] / 1e3:.0f} tys./s, "
                                  f"sukces **{p['success']:.0%}**, {p['elapsed']:.0f} s")
            if not running and self.train.exit_code is not None:
                tail = self.train.tail(4)
                if "Ewaluacja" in tail or "Zapisano" in tail:
                    self.tr_md.content += f"\n\n```\n{tail}\n```"
        elif running:
            self.tr_md.content = "kompilacja kerneli MuJoCo Warp i pierwsza iteracja..."
        j = self.sysid_job
        self.dyn_bar.visible = j.running
        if j.running:
            self.dyn_bar.value = 100 * j.progress
            self.dyn_res.content = j.message
        elif j.state == jobs.DONE and j.result is not None and j.data.get("shown") is not j.result:
            d = j.result
            fitted, fixed = dynamics_parts(d, fitted_names(d, sysid_default=True))
            head = (f"Dopasowano: {', '.join(fitted)}" if fitted else
                    "Wynik (nie wiadomo, ktore parametry dopasowano)") + \
                   (f"; z modelu (nie dopasowane): {', '.join(fixed)}" if fixed else "")
            self.dyn_res.content = (f"{head}. Blad symulacji wzgledem nagrania: "
                                    f"**{j.data.get('base', float('nan')):.2f} st. -> {d.fit_deg:.2f} st.**")
            self.dyn_keep.visible = True
            j.data["shown"] = d
        elif j.state == jobs.FAILED and j.data.get("shown") != "err":
            self.dyn_res.content = f"**Blad**: {j.error}"
            self.dyn_keep.visible = False               # "Zapisz" zapisywalby wynik, ktorego nie ma
            j.data["shown"] = "err"
        elif j.state == jobs.CANCELLED and j.data.get("shown") != "cancel":
            # Wczesniej przerwane zadanie znikalo bez slowa - pasek gasl, a wyniku nie bylo.
            self.dyn_res.content = f"**Przerwane**: {j.message or 'identyfikacja przerwana'}"
            self.dyn_keep.visible = False
            j.data["shown"] = "cancel"

    # =============================================================== polityki
    def _build_policies(self) -> None:
        g = self.server.gui
        self.pol_pick = g.add_dropdown("Polityka", ("-",), initial_value="-")
        refresh = g.add_button("Odswiez liste", icon=viser.Icon.REFRESH)
        self.pol_md = g.add_markdown("")
        ev = g.add_button("Ewaluuj na CPU (zwykle MuJoCo, 50 epizodow)", icon=viser.Icon.CHART_BAR)
        self.pol_eval_md = g.add_markdown("")
        with g.add_folder("Uruchom na blizniaku"):
            self.pol_cube_src = g.add_dropdown("lift: skad polozenie kostki", ("symulacja", "kamery"),
                                               initial_value="symulacja",
                                               hint="kamery = ta sama percepcja co na biurku (zakladka Mapa)")
            self.pol_confirm = g.add_checkbox("Prawdziwe ramie: rozumiem, ze sie ruszy", False)
            run = g.add_button("Uruchom", icon=viser.Icon.PLAYER_PLAY, color="green")
            halt = g.add_button("Zatrzymaj", icon=viser.Icon.PLAYER_STOP)
            new_goal = g.add_button("reach: losowy cel", icon=viser.Icon.TARGET)
            new_cube = g.add_button("lift (sim): poloz kostke losowo", icon=viser.Icon.CUBE)
            self.pol_run_md = g.add_markdown("")
        self.goal_node = self.server.scene.add_icosphere("/cel", radius=0.012, color=(255, 196, 0), visible=False)
        self.goal_gizmo = self.server.scene.add_transform_controls("/cel_uchwyt", scale=0.08, disable_rotations=True,
                                                                   visible=False)
        self._refresh_policies()

        @refresh.on_click
        def _(event):
            self._refresh_policies()

        @self.pol_pick.on_update
        def _(event):
            self._show_policy()

        @ev.on_click
        @self._safe
        def _(event):
            path = self._policy_path()
            self.eval_job.start(lambda job: jobs.run_eval(job, path, self.ws))

        @self.goal_gizmo.on_update
        def _(event):
            self._set_goal((inverse(self.T_b2w) @ np.r_[self.goal_gizmo.position, 1.0])[:3])

        @new_goal.on_click
        @self._safe
        def _(event):
            from ..rl import task as tk
            g_ = tk.sample_goals(self.twin.scene.kin, tk.make_task("reach"), np.random.default_rng(), 1)[0]
            self._set_goal(g_)

        @new_cube.on_click
        @self._safe
        def _(event):
            self._place_cube_randomly()

        @run.on_click
        @self._safe
        def _(event):
            self._start_policy(event)

        @halt.on_click
        def _(event):
            if self.runner is not None:
                self.runner.stop("zatrzymana z panelu")

    def _policy_path(self) -> str:
        from ..rl.policy import list_policies
        for p in list_policies(self.policies_dir):
            if p["name"] == self.pol_pick.value:
                return p["path"]
        raise RuntimeError("wybierz polityke")

    def _refresh_policies(self) -> None:
        from ..rl.policy import list_policies
        items = list_policies(self.policies_dir)
        opts = tuple(p["name"] for p in items) or ("-",)
        cur = self.pol_pick.value
        self.pol_pick.options = opts
        self.pol_pick.value = cur if cur in opts else opts[0]
        if hasattr(self, "tr_init"):
            init_opts = ("(od zera)",) + tuple(p["name"] for p in items)
            if self.tr_init.options != init_opts:
                cur = self.tr_init.value
                self.tr_init.options = init_opts
                self.tr_init.value = cur if cur in init_opts else "(od zera)"
        self._show_policy()

    def _show_policy(self) -> None:
        from ..rl.policy import list_policies
        p = next((x for x in list_policies(self.policies_dir) if x["name"] == self.pol_pick.value), None)
        if p is None:
            self.pol_md.content = "Brak polityk - naucz pierwsza w zakladce Trening."
            return
        ev = p["evals"]
        lines = [f"**{p['name']}** - zadanie `{p['task']}`, {p['steps'] / 1e6:.1f} mln krokow, {p['created']}",
                 f"- GPU (trening, z eksploracja): {p['success'] or 0:.0%}"]
        for k, label in (("cpu_nominal", "CPU bez randomizacji"), ("cpu_rand", "CPU z randomizacja")):
            if k in ev:
                e = ev[k]
                extra = ", ".join(f"{kk} {vv:.1f}" for kk, vv in e.items() if kk.endswith("_mm"))
                lines.append(f"- {label}: **{e['success']:.0%}** ({extra})")
        self.pol_md.content = "\n".join(lines)

    def _set_goal(self, p_base: np.ndarray) -> None:
        """Cel reach: rzutowany na obszar, z ktorego losowano cele w treningu, i tam pokazany.

        Uchwyt celu mozna bylo zaciagnac pod blat - polityka bez limitu epizodu wciskala
        wtedy szczeki w stol, az serwo zablokowalo sie na 25 st. rozjazdu. Uchwyt
        wraca na rzutowany punkt, zeby bylo widac, dokad ramie naprawde jedzie.
        """
        from ..rl import task as tk
        from ..rl.runner import project_goal
        runner = self.runner if self.runner is not None and self.runner.task.name == "reach" else None
        if runner is not None:
            runner.goal = np.asarray(p_base, float)       # setter rzutuje na obszar zadania tej polityki
            g = np.asarray(runner.goal, float)
        else:
            if self._reach_task is None:
                self._reach_task = tk.make_task("reach")
            g = project_goal(self._reach_task, np.asarray(p_base, float))
        self._goal_base = g
        pw = (self.T_b2w @ np.r_[g, 1.0])[:3]
        self.goal_node.position = pw
        if np.abs(np.asarray(self.goal_gizmo.position) - pw).max() > 1e-4:
            self.goal_gizmo.position = pw

    def _cube_in_scene(self) -> bool:
        with self.twin.lock:
            return "cube" in [self.twin.scene.model.body(b).name for b in range(self.twin.scene.model.nbody)]

    def _place_cube_randomly(self) -> None:
        from ..rl import task as tk
        task = tk.make_task("lift")
        if not self._cube_in_scene():
            h = task.cube_half
            self.twin.configure(objects=[sc.Box("cube", (h, h, h), (0.2, 0.0), rgba=(0.85, 0.25, 0.2, 1.0),
                                                mass=task.cube_mass)], grasp_sensors=["cube"])
        pos, quat = tk.sample_cubes(task, np.random.default_rng(), 1)
        with self.twin.lock:
            s = self.twin.scene
            m, d = s.model, s.data
            b = m.body("cube").id
            a, dv = m.jnt_qposadr[m.body_jntadr[b]], m.jnt_dofadr[m.body_jntadr[b]]
            T = s.T_base2world
            qb, qw = np.zeros(4), np.zeros(4)
            mujoco.mju_mat2Quat(qb, T[:3, :3].ravel())
            mujoco.mju_mulQuat(qw, qb, quat[0])
            d.qpos[a:a + 3] = T[:3, :3] @ pos[0] + T[:3, 3]
            d.qpos[a + 3:a + 7] = qw
            d.qvel[dv:dv + 6] = 0.0
            mujoco.mj_forward(m, d)

    def _sim_cube(self):
        with self.twin.lock:
            s = self.twin.scene
            b = s.model.body("cube").id
            Ti = inverse(s.T_base2world)
            return (Ti[:3, :3] @ s.data.xpos[b] + Ti[:3, 3]).copy(), (Ti[:3, :3] @ s.data.xmat[b].reshape(3, 3)).copy()

    def _vision_cube(self):
        """Kostka z kamer dla polityki - z dlonia, gdy szczeki ja zaslaniaja albo niosa.

        Do trackera idzie tylko detekcja NOWA (z kadrow pozniejszych niz poprzednio
        podana) i mlodsza niz `CUBE_MAX_AGE`. Stara detekcja podawana co takt jako
        swieza odnawiala `hold_s` w nieskonczonosc - zamrozona kamera albo wylaczona
        mapa i polityka jechala po kostke, ktorej juz tam nie bylo.
        """
        kin = self._kin_vision
        joints = dict(self.twin.status.measured) or self.twin.joints()
        q = kin.to_q(joints)
        cmd = self.runner.q_cmd[5] if self.runner is not None else q[5]
        now = time.monotonic()
        det = self.last_cube
        if det is not None and (det.confidence <= 0.4 or not det.t or det.t <= self._cube_used_t
                                or now - det.t > CUBE_MAX_AGE):
            det = None
        if det is not None:
            self._cube_used_t = det.t
        return self.cube_tracker.update(det, kin.tcp(joints), q[5], cmd, self._grip_closed_q(), now)

    def _grip_closed_q(self) -> float:
        """Kat szczeki [rad] pustego, zamknietego chwytaka (chwytak 0) - `grip_closed` trackera.

        Ten sam, co w polityce (`Limits.lo` jej biegacza), a nie dolny kraniec MJCF:
        po przeliczeniu chwytaka przez tiki serwa zamknieta szczeka stoi na -5,4 st.,
        a MJCF ma -10 st. - warunek "szczeka stoi szerzej niz zamknieta" byl wtedy
        zawsze prawdziwy i pusta dlon mogla wyjsc na "w dloni".
        """
        runner = self.runner
        if runner is not None and getattr(runner, "limits", None) is not None:
            return float(runner.limits.lo[5])
        kin = self._kin_vision
        return float(kin.to_q({kin.spec.gripper: 0.0})[kin.spec.joints.index(kin.spec.gripper)])

    def _start_policy(self, event) -> None:
        from ..rl.policy import Policy
        from ..rl.runner import PolicyRunner
        if not self.twin.connected:
            raise RuntimeError("polacz ramie w zakladce Ramie (sim, zeby bezpiecznie sprawdzic)")
        if not self.twin.status.simulated and not self.pol_confirm.value:
            raise RuntimeError("prawdziwe ramie: zaznacz potwierdzenie, ze sie ruszy")
        if self.twin.safety_state is not None and self.twin.safety_state.value == "ESTOP":
            raise RuntimeError("aktywny STOP - skasuj go w zakladce Ramie")
        busy = self._busy(POLICY_OWNER)
        if busy:
            raise RuntimeError(f"ramie zajete: {busy} - najpierw je zatrzymaj")
        if self.runner is not None:
            self.runner.stop("nowa polityka")
        pol = Policy.load(self._policy_path())
        provider = None
        if pol.task.name == "lift":
            if self.pol_cube_src.value == "symulacja":
                if not self.twin.status.simulated:
                    raise RuntimeError("na prawdziwym ramieniu kostka musi pochodzic z kamer")
                if not self._cube_in_scene():
                    self._place_cube_randomly()
                provider = self._sim_cube
            else:
                if self.mapper is None:
                    raise RuntimeError("kostka z kamer wymaga co najmniej jednej skalibrowanej kamery (zakladka Mapa)")
                if not (self.map_on.value and self.cube_on.value):
                    raise RuntimeError("kostka z kamer wymaga wlaczonej mapy i szukania kostki (zakladka Mapa)")
                self.cube_tracker = CubeTracker()
                self._cube_used_t = 0.0
                provider = self._vision_cube
        self._release_panel()
        self.policy = pol
        self.runner = PolicyRunner(self.twin, pol, cube_provider=provider)
        if pol.task.name == "reach":
            p = (inverse(self.T_b2w) @ np.r_[self.goal_gizmo.position, 1.0])[:3]
            if not self.goal_node.visible:
                from ..rl import task as tk
                p = tk.sample_goals(self.twin.scene.kin, pol.task, np.random.default_rng(), 1)[0]
            self._set_goal(p)
            self.runner.episode_limit = False                # cel przeciagany na zywo - bez limitu epizodu
        self.goal_node.visible = self.goal_gizmo.visible = pol.task.name == "reach"
        self.runner.start()                                  # bierze ramie ("polityka") albo RuntimeError
        self.pol_confirm.value = False                       # nastepne uruchomienie - nowe potwierdzenie

    def _tick_policies(self) -> None:
        if self.runner is not None:
            rs = self.runner.status
            txt = (f"{'**jedzie**' if rs.running else 'stoi'} - krok {rs.step}, {rs.hz:.0f} Hz")
            if self.runner.task.name == "reach" and np.isfinite(rs.distance):
                txt += f", do celu **{rs.distance * 1000:.0f} mm**" + (" (w celu)" if rs.success else "")
                if rs.goal_clamped:
                    txt += " - cel przyciety do obszaru z treningu"
            if self.runner.cube_provider == self._vision_cube:
                txt += f", kostka: **{self.cube_tracker.source}**"
            if self.runner.task.name == "lift" and self.twin.status.simulated and self._cube_in_scene():
                pos, _ = self._sim_cube()
                txt += f"  \nprawda symulacji: kostka **{100 * (pos[2] - self.runner.task.cube_half):.1f} cm** nad blatem"
            if not rs.running and rs.stopped_because:
                txt += f"  \nzatrzymana: {rs.stopped_because}"
            self.pol_run_md.content = txt
        j = self.eval_job
        if j.running:
            self.pol_eval_md.content = f"ewaluacja... {j.message}"
        elif j.state == jobs.DONE and j.data.get("shown") is not j.result:
            self._show_policy()
            lines = []
            for k, label in (("cpu_nominal", "bez randomizacji"), ("cpu_rand", "z randomizacja")):
                e = (j.result or {}).get(k)
                if e:
                    lines.append(f"- CPU {label}: **{e['success']:.0%}** z {e['episodes']} epizodow")
            self.pol_eval_md.content = "\n".join(lines)
            j.data["shown"] = j.result
        elif j.state == jobs.FAILED and j.data.get("shown") != "err":
            self.pol_eval_md.content = f"**Blad**: {j.error}"
            j.data["shown"] = "err"

    # =============================================================== sim-real
    def _build_simreal(self) -> None:
        g = self.server.gui
        g.add_markdown("Prawdziwy kadr i render blizniaka z TEJ SAMEJ kamery (skalibrowana poza, K, katy "
                       "zmierzone na serwach). Rozjazd krawedzi to blad kalibracji albo modelu - widac go od razu.")
        self.sr_cam = g.add_dropdown("Kamera", ("-",), initial_value="-")
        self.sr_alpha = g.add_slider("Przezroczystosc renderu", 0.0, 1.0, 0.05, 0.5)
        self.sr_edges = g.add_checkbox("Krawedzie symulacji zamiast renderu", True)
        self.sr_img = g.add_image(np.zeros((240, 320, 3), np.uint8), format="jpeg", jpeg_quality=75)
        self.sr_md = g.add_markdown("")

    def _tick_simreal(self, frames: dict[str, np.ndarray]) -> None:
        name = self.sr_cam.value
        try:
            rec = self.ws.camera(name)
        except KeyError:
            self.sr_md.content = "Wybierz kamere."
            return
        if rec.simulated:
            self.sr_md.content = ("Kamera symulowana - jej porownanie z prawda (mm, st.) jest w zakladce Kamery.")
            return
        real = frames.get(name)
        if real is None or not rec.calibrated:
            self.sr_md.content = "Potrzebny kadr i skalibrowana poza kamery."
            return
        K, dist = rec.intrinsics()
        if dist is not None and np.any(dist):
            # Render blizniaka to kamera otworkowa (samo K) - kadr musi byc bez dystorsji,
            # inaczej przy k1 = -0,25 krawedzie przy brzegu rozjezdzaly sie o dziesiatki px
            # przy dobrej kalibracji, a wynik ponizej kazal ja powtarzac.
            real = cv2.undistort(real, K, dist)
        sim = self.twin.render(name)
        if sim.shape != real.shape:
            sim = cv2.resize(sim, (real.shape[1], real.shape[0]))
        a = float(self.sr_alpha.value)
        if self.sr_edges.value:
            e_sim = cv2.Canny(cv2.cvtColor(sim, cv2.COLOR_RGB2GRAY), 60, 160) > 0
            out = real.copy()
            out[e_sim] = (0.3 * out[e_sim] + 0.7 * np.array([255, 196, 0])).astype(np.uint8)
            e_real = cv2.Canny(cv2.cvtColor(real, cv2.COLOR_RGB2GRAY), 60, 160)
            dt = cv2.distanceTransform(255 - e_real, cv2.DIST_L2, 3)
            score = float(np.median(dt[e_sim])) if e_sim.any() else float("nan")
            self.sr_md.content = f"mediana odleglosci krawedzi symulacji od krawedzi kadru: **{score:.1f} px**"
        else:
            out = cv2.addWeighted(real, 1 - a, sim, a, 0)
            self.sr_md.content = ""
        self.sr_img.image = thumb(out, 640)

    # ================================================================== petla
    def _grab(self) -> dict[str, np.ndarray]:
        """Kadry wlaczonych kamer z chwilami ich wykonania (`frame_times`).

        Kadr starszy niz `FRAME_MAX_AGE` jest pomijany: strumien, ktory stanal
        (przekazanie USB na Shadow), oddawal te sama klatke w nieskonczonosc,
        a mapa co takt "widziala" w niej kostke od nowa.
        """
        frames, times = {}, {}
        now = time.monotonic()
        for c in self.ws.cameras:
            if not c.enabled:
                continue
            try:
                img, t = self.twin.cameras.frame_t(c.name)
            except Exception:
                img, t = None, 0.0
            if img is None or (t and now - t > FRAME_MAX_AGE):
                continue
            frames[c.name] = img
            times[c.name] = t or now
        with self.frame_lock:
            self.frames, self.frame_times = frames, times
            self._t_grab = now
        return frames

    def _fresh_frames(self, max_age: float) -> dict[str, np.ndarray]:
        """Kadry z ostatniego `_grab`, jesli sa mlodsze niz `max_age` [s], inaczej nowe."""
        with self.frame_lock:
            if time.monotonic() - self._t_grab < max_age:
                return dict(self.frames)
        return self._grab()

    def _tick_slow(self, frames: dict[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
        self.status_md.content = self._status_text()
        opts = tuple(c.name for c in self.ws.cameras) or ("-",)
        for dd in (self.intr_cam, self.reloc_cam, self.sr_cam):
            if dd.options != opts:
                cur = dd.value
                dd.options = opts
                dd.value = cur if cur in opts else opts[0]
        if frames is None:
            frames = self._grab()
        rec = self._cam()
        if rec is not None:
            # Wybrana kamera bez kadru (nie otworzyla sie, wylaczona, strumien stanal) - napis,
            # a nie ostatni kadr POPRZEDNIEJ kamery pod nazwa nowej.
            img = frames.get(rec.name)
            self._img(self.cam_preview, "podglad", thumb(img, 480) if img is not None else no_frame_image())
        else:
            # Usunieta ostatnia kamera: podglad zostawal z jej ostatnim kadrem, jakby wciaz byla.
            self._img(self.cam_preview, "podglad", no_frame_image("brak kamer"))
        with self.scene_lock:
            for name, img in frames.items():
                h = self.frustums.get(name)
                if h is not None and h.visible:
                    self._img(h, f"piramida/{name}/{id(h)}", thumb(img, 160))
        self._tick_calibration()
        self._tick_training()
        self._tick_policies()
        return frames

    def _tick_watch(self, frames: dict[str, np.ndarray]) -> None:
        changed = False
        for c in self.ws.cameras:
            if c.name not in frames or c.name not in self.watch.refs:
                continue
            was = self.watch.moved(c.name)
            with self.frame_lock:
                t = self.frame_times.get(c.name)
            mask = (self._arm_masks([c.name], t) or {}).get(c.name)
            self.watch.check(c.name, frames[c.name], mask)
            if self.watch.moved(c.name) != was:
                changed = True
                if not was:
                    self._notify(None, "Kamera przestawiona",
                                 f"{c.name}: kadr przesunal sie o {_px(self.watch.shift[c.name])} od kalibracji. "
                                 f"Zrob szybka relokalizacje (zakladka Kalibracja).", error=True)
        if changed:
            self._refresh_cameras()

    def _guard(self, what: str, fn, *args):
        """Blad jednego odswiezenia nie moze zabic petli panelu - pod nim jezdzi ramie."""
        try:
            return fn(*args)
        except Exception:
            if what not in self._tick_errors:                # jeden wpis na rodzaj, nie co takt
                self._tick_errors.add(what)
                logger.exception("Blad w odswiezaniu panelu (%s) - petla dziala dalej", what)
            return None

    def _sync_mirror(self) -> None:
        """Widok 3D za scena: przebudowa po `twin.rebuild`, potem tylko pozy cial.

        Wersja i model czytane RAZEM pod blokada: przebudowa w trakcie budowy widoku
        (~0,6 s) zapisywala nowa wersje przy siatkach ze starego modelu - widok zostawal
        na starym modelu na zawsze, a `update` czytal ciala nowego modelu po starych numerach.
        """
        with self.twin.lock:
            v, model = self.twin.version, self.twin.scene.model
        if v != self._mirror_version:
            if self.mirror is not None:
                self.mirror.remove()
            self.mirror = SceneMirror(self.server, model)
            self._mirror_version = v
        with self.twin.lock:
            if self.twin.version == self._mirror_version:
                self.mirror.update(self.twin.scene.data)

    def _record_joints(self, now: float) -> None:
        """Historia katow do masek ramienia w chwili kadru (`_arm_masks`)."""
        st = self.twin.status
        if st.connected and st.measured:
            self._joint_hist.append((now, dict(st.measured)))

    def run(self) -> None:
        port = self.server.get_port()
        remote = "  (zdalnie: http://<adres-maszyny>:{port})" if self.host not in ("127.0.0.1", "localhost") else ""
        print(f"Panel blizniaka: http://localhost:{port}{remote.format(port=port)}", flush=True)
        t_slow = t_map = t_watch = t_list = 0.0
        frames: dict[str, np.ndarray] = {}
        try:
            while not self._stop.is_set():
                t0 = time.monotonic()
                self._guard("scena", self._sync_mirror)
                self._record_joints(t0)
                self._guard("ramie", self._tick_arm)
                vision_policy = self._vision_policy_running()
                if t0 - t_slow > 0.2:
                    t_slow = t0
                    # Kadry osobno od reszty odswiezania: blad w zakladce Trening nie moze
                    # zostawic mapy z kadrami sprzed minuty (wczesniej `... or frames`).
                    frames = self._guard("kadry", self._fresh_frames, 0.1) or {}
                    self._guard("odswiezanie", self._tick_slow, frames)
                if t0 - t_map > (0.12 if vision_policy else 0.33):
                    t_map = t0
                    if vision_policy:                          # polityka z kamer: kadry swieze na te mape
                        frames = self._guard("kadry", self._fresh_frames, 0.05) or {}
                    self._guard("mapa", self._tick_map, frames)
                    self._guard("sim-real", self._tick_simreal, frames)
                if t0 - t_watch > 2.0:
                    t_watch = t0
                    self._guard("straznik kamer", self._tick_watch, frames)
                if t0 - t_list > 5.0:
                    t_list = t0
                    if self.train.exit_code is not None and self.train.run_dir is not None:
                        self._guard("lista polityk", self._refresh_policies)
                if self._dirty_save and t0 - self._dirty_save > 1.0:
                    self._dirty_save = 0.0
                    self._guard("zapis", self._save)
                if self._dirty_rebuild and t0 - self._dirty_rebuild > 1.0:
                    self._dirty_rebuild = 0.0
                    self._guard("przebudowa", self.twin.rebuild)
                time.sleep(max(0.0, 1 / 30 - (time.monotonic() - t0)))
        except KeyboardInterrupt:
            pass
        finally:
            self.close()

    def close(self) -> None:
        """Najpierw ramie (polityka, zadania, rozlaczenie), dopiero potem trening i serwer.

        Wczesniej `train.shutdown` (do 8 s na zapis polityki + 5 s na zabicie) szlo
        PRZED rozlaczeniem ramienia, a drugie Ctrl+C w tym czasie pomijalo
        `twin.close()` - proces konczyl sie bez czystego rozlaczenia serw.
        """
        self._stop.set()
        try:
            if self.runner is not None:
                self.runner.stop("zamkniecie panelu")
            for job in (self.calib_job, self.sysid_job, self.intr_job, self.eval_job):
                job.stop()
        finally:
            try:
                self.twin.close()
            finally:
                try:
                    # Trening w osobnym procesie bez konsoli - bez tego zostawal po zamknieciu panelu.
                    self._guard("trening", self.train.shutdown)
                finally:
                    self.server.stop()


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="lerobot-twin ui", description="Panel cyfrowego blizniaka w przegladarce.")
    # Panel steruje ramieniem i nie ma hasla: z sieci tylko na zyczenie (--host 0.0.0.0).
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--workspace", default=None)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # Sondy HTTP (HEAD, niedokonczone polaczenia) nie sa bledami panelu - websockets
    # drukuje po kilkadziesiat linii stosu na kazda, zasypujac to, co wazne.
    logging.getLogger("websockets.server").setLevel(logging.CRITICAL)
    TwinApp(a.workspace, a.host, a.port).run()
    return 0
