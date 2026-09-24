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
from .watch import CameraWatch, arm_mask

logger = logging.getLogger(__name__)


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


class TwinApp:
    def __init__(self, workspace: str | Path | None = None, host: str = "0.0.0.0", port: int = 8080):
        self.ws = Workspace.load(workspace)
        self.ws_path = Path(self.ws.path).resolve() if self.ws.path else None
        self.twin = Twin(self.ws)
        self.watch = CameraWatch()
        self.frames: dict[str, np.ndarray] = {}
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
        from ..kinematics import RobotKinematics
        self._kin_vision = RobotKinematics(self.ws.spec())
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
        s.scene.add_light_ambient("/swiatlo/otoczenie", intensity=0.5)
        s.scene.add_light_hemisphere("/swiatlo/niebo", sky_color=(235, 240, 255), ground_color=(70, 65, 60),
                                     intensity=1.6, wxyz=(np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0))
        s.scene.add_light_directional("/swiatlo/gora", intensity=1.4, position=(0.6, -0.8, 2.5), cast_shadow=True)
        self.base_frame = s.scene.add_frame("/podstawa", axes_length=0.08, axes_radius=0.003,
                                            position=T[:3, 3], wxyz=mat_to_wxyz(T[:3, :3]))
        self.mirror = SceneMirror(s, self.twin.scene.model)
        self._mirror_version = self.twin.version
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
        def _(event):
            if self.runner is not None:
                self.runner.stop("STOP z panelu")
            self.calib_job.stop()
            self.sysid_job.stop()
            self.twin.estop()
            self.arm_engage.value = False
            self._notify(event, "STOP", "Ramie zatrzymane. Skasuj STOP w zakladce Ramie, zeby ruszyc dalej.",
                         error=True)

    def _status_text(self) -> str:
        st = self.twin.status
        arm = (f"{st.backend} - {st.state} - {st.loop_hz:.0f} Hz" if st.connected
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
        return f"**Ramie:** {arm} | **Kamery:** {cam_txt} | **Polityka:** {pol} | **Trening:** {tr}"

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
            if self.runner is not None:
                self.runner.stop("ponowne laczenie")
            port = self.arm_port.value.strip() or None
            self.twin.connect(self.arm_backend.value, port, go_home=self.arm_home_on_connect.value)
            self._save()
            self._notify(event, "Polaczono", f"{self.twin.status.backend}: {self.twin.status.state}")

        @disconnect.on_click
        @self._safe
        def _(event):
            if self.runner is not None:
                self.runner.stop("rozlaczenie")
            self.twin.disconnect()

        with g.add_folder("Sterowanie"):
            self.arm_engage = g.add_checkbox("Sprzeglo: panel steruje ramieniem", False,
                                             hint="Bez sprzegla ramie trzyma pozycje, a suwaki tylko pokazuja katy")
            home = g.add_button("Pozycja domowa", icon=viser.Icon.HOME)
            clear = g.add_button("Skasuj STOP", icon=viser.Icon.RESTORE)
            self.tcp_gizmo_on = g.add_checkbox("Uchwyt koncowki w 3D", False,
                                               hint="Przeciagnij koncowke; katy liczy odwrotna kinematyka")
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
                    self.twin.set_target({name: float(self.sliders[name].value)})
                s.on_update(on_slide)

        @self.arm_engage.on_update
        def _(event):
            if event.client is not None:
                if self.arm_engage.value and not self.twin.connected:
                    self.arm_engage.value = False
                    self._notify(event, "Brak ramienia", "Najpierw polacz ramie (np. sim).", error=True)
                    return
                self.twin.set_engaged(self.arm_engage.value)

        @home.on_click
        def _(event):
            self.twin.home()

        @clear.on_click
        def _(event):
            self.twin.clear_estop()

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
            self._notify(event, "Stanowisko", "Stol zapisany, scena przebudowana.")

        self.tcp_gizmo = self.server.scene.add_transform_controls("/uchwyt_tcp", scale=0.12, disable_rotations=True,
                                                                  visible=False)

        @self.tcp_gizmo_on.on_update
        def _(event):
            self.tcp_gizmo.visible = self.tcp_gizmo_on.value

        @self.tcp_gizmo.on_update
        def _(event):
            if not (self.arm_engage.value and self.tcp_gizmo_on.value and self.twin.connected):
                return
            p_base = (inverse(self.T_b2w) @ np.r_[self.tcp_gizmo.position, 1.0])[:3]
            with self.twin.lock:
                seed = self.twin.scene.joints()
            sol = self.twin.scene.kin.ik(p_base, seed=seed, restarts=2)
            if sol.ok:
                self.twin.set_target({k: v for k, v in sol.joints.items() if k != self.ws.spec().gripper})

    def _tick_arm(self) -> None:
        st = self.twin.status
        joints = st.measured if st.connected and st.measured else self.twin.joints()
        if not self.arm_engage.value:
            for name, s in self.sliders.items():
                if name in joints:
                    s.value = round(float(np.clip(joints[name], *self.slider_range[name])), 1)
        if self.tcp_gizmo_on.value and not self.arm_engage.value:
            with self.twin.lock:
                d, site = self.twin.scene.data, self.twin.scene.kin.site_id
                p = d.site_xpos[site].copy()
            if np.abs(np.asarray(self.tcp_gizmo.position) - p).max() > 1e-4:
                self.tcp_gizmo.position = p
        if self.arm_engage.value and not st.engaged and st.connected and st.state == "ESTOP":
            self.arm_engage.value = False

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
            if rec is None or not rec.simulated or not self.cam_move.value:
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
                self.twin.rebuild()
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
        mask = None
        try:
            mask = self.twin.render_with(lambda s: arm_mask(s, name))
        except KeyError:
            pass
        self.watch.remember(name, img, mask)

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
                cal = f"niezaufana: {c.calibration.get('reason', '')}"
            sh = self.watch.shift.get(c.name)
            moved = "-" if sh is None else (f"**TAK {sh:.1f} px**" if self.watch.moved(c.name) else f"nie ({sh:.1f} px)")
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
        show = rec.simulated and self.cam_move.value and T is not None
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
            self.intr_job.start(lambda job: jobs.run_intrinsics(job, self.twin.cameras, name, board,
                                                                 (rec.width, rec.height)))
            self.intr_img.visible = True

        @solve.on_click
        @self._safe
        def _(event):
            if not self.intr_job.running:
                raise RuntimeError("najpierw zbieraj kadry")
            self.intr_job.stop()

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
                self.intr_job.cancel.set()

        @apply.on_click
        @self._safe
        def _(event):
            fit = self.calib_job.result
            if fit is None:
                raise RuntimeError("brak wyniku kalibracji")
            self.ws.card["tag_size"] = self.tag_mm.value / 1000
            names = self.ws.apply_fit(fit)
            self._save()
            self.twin.rebuild()
            for n in names:
                try:
                    self._remember_reference(n)
                except RuntimeError:
                    pass
            self._refresh_cameras()
            self.calib_apply.visible = False
            self._notify(event, "Zapisano", f"Kalibracja kamer: {', '.join(names)}")

    def _start_card_calibration(self, event, cameras: list[str], quick: bool) -> None:
        if not self.twin.connected:
            raise RuntimeError("polacz ramie (sim albo prawdziwe) w zakladce Ramie")
        if not cameras or cameras == ["-"]:
            raise RuntimeError("zadna wlaczona kamera nie daje kadru")
        if not self.twin.status.simulated and not self.calib_confirm.value:
            raise RuntimeError("potwierdz, ze karta jest w szczekach, a przestrzen nad stolem wolna")
        self.ws.card["tag_size"] = self.tag_mm.value / 1000
        if self.runner is not None:
            self.runner.stop("kalibracja")
        self.arm_engage.value = False
        self.calib_apply.visible = False
        self.calib_job.start(lambda job: jobs.run_card_calibration(job, self.twin, cameras, quick))

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
                verdict = "zaufana" if c.trusted else f"NIE: {c.reason}"
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
            rec = self.ws.camera(self.intr_cam.value)
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
            self.intr_md.content = (f"Zapisano K: fx {res.K[0, 0]:.1f}, fy {res.K[1, 1]:.1f}, cx {res.K[0, 2]:.1f}, "
                                    f"cy {res.K[1, 2]:.1f}; residuum **{res.rms_px:.3f} px**, {res.n_views} kadrow"
                                    + ("" if res.trusted else f" - uwaga: {res.reason}"))
            ij.data["saved"] = res
        elif ij.state == jobs.FAILED and ij.data.get("saved") != "err":
            self.intr_md.content = f"**Blad**: {ij.error}"
            ij.data["saved"] = "err"

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

    def _mapper_now(self) -> TableMapper | None:
        key = tuple((c.name, c.enabled, c.trusted, str(c.T_cam2base), str(c.K)) for c in self.ws.cameras)
        if key != self._mapper_key:
            self._mapper_key = key
            m = TableMapper.from_workspace(self.ws, only_trusted=True)
            self.mapper = m if m.cameras else None
        return self.mapper

    def _tick_map(self, frames: dict[str, np.ndarray]) -> None:
        mapper = self._mapper_now() if self.map_on.value else None
        if mapper is None:
            self.map_md.content = "Brak zaufanej, skalibrowanej kamery - najpierw kalibracja." if self.map_on.value \
                else ""
            if self.map_node is not None:
                self.map_node.visible = False
            self.cube_node.visible = False
            return
        table, wsum = mapper.fuse(frames)
        self._img(self.map_img, "mapa", table)
        cover = float((wsum > 1e-6).mean())
        txt = f"pokrycie blatu: **{cover:.0%}** z {len(mapper.cameras)} kamer"
        Tm = self.T_b2w @ pose(np.eye(3), np.array([mapper.centre[0], mapper.centre[1], 0.0015]))
        if self.map_3d.value:
            if self.map_node is None:
                self.map_node = self.server.scene.add_image("/mapa", table, mapper.side, mapper.side,
                                                            format="jpeg", position=Tm[:3, 3],
                                                            wxyz=mat_to_wxyz(Tm[:3, :3]))
            else:
                self._img(self.map_node, "mapa3d", table)
                self.map_node.visible = True
        elif self.map_node is not None:
            self.map_node.visible = False
        self.last_cube = None
        if self.cube_on.value:
            # Piksele zasloniete ramieniem (maska z blizniaka, w pozie z serw) nie glosuja.
            names = [n for n in mapper.cameras if n in frames]
            try:
                occ = self.twin.render_with(lambda s: {n: arm_mask(s, n, dilate=5) for n in names})
            except KeyError:
                occ = None
            det = self.cube_det.detect_frames(frames, mapper, occ)
            if det is not None:
                self.last_cube = det
                Tw = self.T_b2w @ pose(det.rot, det.pos)
                self.cube_node.position, self.cube_node.wxyz = Tw[:3, 3], mat_to_wxyz(Tw[:3, :3])
                self.cube_node.visible = True
                yaw = np.degrees(np.arctan2(det.rot[1, 0], det.rot[0, 0]))
                txt += (f"  \nkostka: x {det.pos[0] * 100:.1f} cm, y {det.pos[1] * 100:.1f} cm, obrot {yaw:.0f} st., "
                        f"pewnosc {det.confidence:.2f}")
            else:
                self.cube_node.visible = False
                txt += "  \nkostka: nie widac"
        self.map_md.content = txt

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
            if self.runner is not None:
                self.runner.stop("identyfikacja")
            self.arm_engage.value = False
            self.dyn_keep.visible = False
            self.sysid_job.start(lambda job: jobs.run_sysid(job, self.twin))

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
        self.dyn_md.content = (f"Srodek randomizacji: **{d.source}** - kp x{d.kp:.2f}, tlumienie x{d.damping:.2f}, "
                               f"armatura x{d.armature:.2f}, tarcie x{d.frictionloss:.2f}, opoznienie "
                               f"{d.delay:.2f} taktu" + (f", blad dopasowania {d.fit_deg:.2f} st." if d.fit_deg else ""))

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
            self.dyn_res.content = (f"Dopasowano: kp x{d.kp:.2f}, tlumienie x{d.damping:.2f}, armatura "
                                    f"x{d.armature:.2f}, tarcie x{d.frictionloss:.2f}, opoznienie {d.delay / 20 * 1000:.0f} ms. "
                                    f"Blad symulacji wzgledem nagrania: **{j.data.get('base', float('nan')):.2f} st. -> "
                                    f"{d.fit_deg:.2f} st.**")
            self.dyn_keep.visible = True
            j.data["shown"] = d
        elif j.state == jobs.FAILED and j.data.get("shown") != "err":
            self.dyn_res.content = f"**Blad**: {j.error}"
            j.data["shown"] = "err"

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
            p = (inverse(self.T_b2w) @ np.r_[self.goal_gizmo.position, 1.0])[:3]
            if self.runner is not None:
                self.runner.goal = p
            self.goal_node.position = self.goal_gizmo.position

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
        pw = (self.T_b2w @ np.r_[p_base, 1.0])[:3]
        self.goal_node.position = pw
        self.goal_gizmo.position = pw
        if self.runner is not None:
            self.runner.goal = np.asarray(p_base, float)

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
        """Kostka z kamer dla polityki - z dlonia, gdy szczeki ja zaslaniaja albo niosa."""
        kin = self._kin_vision
        joints = dict(self.twin.status.measured) or self.twin.joints()
        q = kin.to_q(joints)
        cmd = self.runner.q_cmd[5] if self.runner is not None else q[5]
        det = self.last_cube if self.last_cube is not None and self.last_cube.confidence > 0.4 else None
        return self.cube_tracker.update(det, kin.tcp(joints), q[5], cmd, kin.lo[5], time.monotonic())

    def _start_policy(self, event) -> None:
        from ..rl.policy import Policy
        from ..rl.runner import PolicyRunner
        if not self.twin.connected:
            raise RuntimeError("polacz ramie w zakladce Ramie (sim, zeby bezpiecznie sprawdzic)")
        if not self.twin.status.simulated and not self.pol_confirm.value:
            raise RuntimeError("prawdziwe ramie: zaznacz potwierdzenie, ze sie ruszy")
        if self.twin.safety_state is not None and self.twin.safety_state.value == "ESTOP":
            raise RuntimeError("aktywny STOP - skasuj go w zakladce Ramie")
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
                self.cube_tracker = CubeTracker()
                provider = self._vision_cube
        self.arm_engage.value = False
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
        self.runner.start()

    def _tick_policies(self) -> None:
        if self.runner is not None:
            rs = self.runner.status
            txt = (f"{'**jedzie**' if rs.running else 'stoi'} - krok {rs.step}, {rs.hz:.0f} Hz")
            if self.runner.task.name == "reach" and np.isfinite(rs.distance):
                txt += f", do celu **{rs.distance * 1000:.0f} mm**" + (" (w celu)" if rs.success else "")
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
        frames = {}
        for c in self.ws.cameras:
            if not c.enabled:
                continue
            try:
                img = self.twin.cameras.frame(c.name)
            except Exception:
                img = None
            if img is not None:
                frames[c.name] = img
        with self.frame_lock:
            self.frames = frames
        return frames

    def _tick_slow(self) -> None:
        self.status_md.content = self._status_text()
        opts = tuple(c.name for c in self.ws.cameras) or ("-",)
        for dd in (self.intr_cam, self.reloc_cam, self.sr_cam):
            if dd.options != opts:
                cur = dd.value
                dd.options = opts
                dd.value = cur if cur in opts else opts[0]
        frames = self._grab()
        rec = self._cam()
        if rec is not None and rec.name in frames:
            self._img(self.cam_preview, "podglad", thumb(frames[rec.name], 480))
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
            mask = None
            try:
                mask = self.twin.render_with(lambda s, n=c.name: arm_mask(s, n))
            except KeyError:
                pass
            self.watch.check(c.name, frames[c.name], mask)
            if self.watch.moved(c.name) != was:
                changed = True
                if not was:
                    self._notify(None, "Kamera przestawiona",
                                 f"{c.name}: kadr przesunal sie o {self.watch.shift[c.name]:.1f} px od kalibracji. "
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

    def run(self) -> None:
        port = self.server.get_port()
        print(f"Panel blizniaka: http://localhost:{port}  (zdalnie: http://<adres-maszyny>:{port})", flush=True)
        t_slow = t_map = t_watch = t_list = 0.0
        frames: dict[str, np.ndarray] = {}
        try:
            while not self._stop.is_set():
                t0 = time.monotonic()
                if self.twin.version != self._mirror_version:
                    self.mirror.remove()
                    self.mirror = SceneMirror(self.server, self.twin.scene.model)
                    self._mirror_version = self.twin.version
                with self.twin.lock:
                    data = self.twin.scene.data
                    self._guard("scena", self.mirror.update, data)
                self._guard("ramie", self._tick_arm)
                if t0 - t_slow > 0.2:
                    t_slow = t0
                    frames = self._guard("odswiezanie", self._tick_slow) or frames
                vision_policy = (self.runner is not None and self.runner.status.running
                                 and self.runner.cube_provider == self._vision_cube)
                if t0 - t_map > (0.12 if vision_policy else 0.33):
                    t_map = t0
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
                time.sleep(max(0.0, 1 / 30 - (time.monotonic() - t0)))
        except KeyboardInterrupt:
            pass
        finally:
            self.close()

    def close(self) -> None:
        self._stop.set()
        if self.runner is not None:
            self.runner.stop("zamkniecie panelu")
        self.calib_job.stop()
        self.twin.close()
        self.server.stop()


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="lerobot-twin ui", description="Panel cyfrowego blizniaka w przegladarce.")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--workspace", default=None)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # Sondy HTTP (HEAD, niedokonczone polaczenia) nie sa bledami panelu - websockets
    # drukuje po kilkadziesiat linii stosu na kazda, zasypujac to, co wazne.
    logging.getLogger("websockets.server").setLevel(logging.CRITICAL)
    TwinApp(a.workspace, a.host, a.port).run()
    return 0
