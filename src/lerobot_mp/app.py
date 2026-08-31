"""Petla glowna: kamera -> MediaPipe -> mapowanie -> nadzor -> robot.

Wazny szczegol architektury: *wizja i sterowanie chodza w roznym tempie*.
Petla sterowania tyka ze stalą czestotliwoscia (`loop_hz`) niezaleznie od tego,
czy kamera zdazyla dostarczyc nowa klatke. Dzieki temu ograniczenia predkosci,
plynne dojscia i watchdog dzialaja poprawnie takze wtedy, gdy detekcja
chwilowo zwolni - robot nie szarpie, tylko dojezdza tam, gdzie mial dojechac.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import cv2
import numpy as np

from .config import AppConfig
from .control.mapping import ControlOutput, HandToJointMapper
from .control.safety import SafetyState, SafetySupervisor
from .robot import create_backend
from .robot.base import RobotBackend
from .ui.arm_view import compose, draw_arm_view
from .ui.hud import HudData, draw_hand_skeleton, draw_hud
from .utils.rate import FpsMeter, LoopRate
from .vision.camera import CameraStream
from .vision.features import HandFeatures, extract_features
from .vision.tracker import HandTracker

logger = logging.getLogger(__name__)

ESC = 27


@dataclass
class _Preview:
    """Podglad ramienia - model 3D albo rysunek schematyczny."""

    renderer: object | None = None
    model: object | None = None
    image: np.ndarray | None = None
    last_render: float = 0.0
    enabled: bool = True

    @property
    def is_3d(self) -> bool:
        return self.renderer is not None


class TeleopApp:
    """Sterowanie ramieniem SO-101 gestami dloni."""

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.mapper = HandToJointMapper(cfg)
        self.supervisor = SafetySupervisor(cfg)
        self.fps = FpsMeter()

        self._features = HandFeatures.absent()
        self._features_time = 0.0
        self._last_frame_index = -1
        self._last_hands: list = []
        self._measured: dict[str, float] = {}
        self._last_read = 0.0
        self._message = ""
        self._message_until = 0.0
        self._latency_ms = 0.0
        self._running = True
        self._preview = _Preview(enabled=cfg.ui.draw_arm_view)
        self._writer: cv2.VideoWriter | None = None

    # ------------------------------------------------------------------ setup
    def _setup_preview(self) -> None:
        """Przygotowuje podglad 3D, jesli model jest dostepny."""
        if not self.cfg.ui.draw_arm_view or not self.cfg.ui.preview_3d:
            return
        from .preview.model import load_model
        from .preview.render import Renderer3D

        model = load_model(self.cfg.ui.preview_asset)
        if model is None:
            return
        height = int(self.cfg.camera.height)
        self._preview.model = model
        self._preview.renderer = Renderer3D(
            model, size=(self.cfg.ui.preview_width, max(height, 240))
        )

    # ------------------------------------------------------------------- run
    def run(self) -> int:
        camera = CameraStream(self.cfg.camera)
        backend = create_backend(self.cfg)
        tracker: HandTracker | None = None

        try:
            camera.open()
            first = camera.wait_for_frame(timeout=8.0)
            logger.info(
                "Kamera %s: %dx%d", self.cfg.camera.source, first.image.shape[1], first.image.shape[0]
            )

            tracker = HandTracker(self.cfg.tracker)
            self._setup_preview()

            backend.connect()
            self.supervisor.start(backend.read_joints())
            self._measured = backend.read_joints()

            if self.cfg.ui.show:
                cv2.namedWindow(self.cfg.ui.window_name, cv2.WINDOW_AUTOSIZE)
            else:
                logger.info("Tryb bez podgladu - klawiszologia niedostepna.")

            self._loop(camera, tracker, backend)
        except KeyboardInterrupt:
            logger.info("Przerwano z klawiatury.")
        finally:
            self._shutdown(camera, tracker, backend)
        return 0

    def _loop(self, camera: CameraStream, tracker: HandTracker, backend: RobotBackend) -> None:
        rate = LoopRate(self.cfg.loop_hz)
        previous = time.monotonic()
        deadline = (
            previous + self.cfg.max_runtime_s if self.cfg.max_runtime_s else float("inf")
        )

        while self._running:
            now = time.monotonic()
            if now >= deadline:
                logger.info("Uplynal zadany czas dzialania - koncze.")
                break
            dt = min(now - previous, 0.25)  # zabezpieczenie po zawieszeniu okna
            previous = now
            self.fps.tick(now)

            frame = camera.read()
            if frame is None:
                if not camera.is_running:
                    logger.warning("Strumien obrazu sie skonczyl - koncze.")
                    break
                rate.sleep()
                continue

            if frame.index != self._last_frame_index:
                self._last_frame_index = frame.index
                self._process_frame(frame, tracker)

            features = self._current_features(now)
            output = self.mapper.update(features, self.supervisor.command, dt)

            command, report = self.supervisor.step(
                output.targets,
                dt,
                hand_present=features.present,
                engaged=output.engaged,
            )
            backend.send_joints(command)
            backend.step(dt)
            self._refresh_measured(backend, now)

            if self.cfg.ui.show or self.cfg.ui.record_path:
                canvas = self._render(frame.image, output, command, report, now)
                self._record(canvas)
                if self.cfg.ui.show:
                    cv2.imshow(self.cfg.ui.window_name, canvas)
                    if not self._handle_keys(cv2.waitKey(1) & 0xFF, backend):
                        break

            rate.sleep()

    # --------------------------------------------------------------- wizja
    def _process_frame(self, frame, tracker: HandTracker) -> None:
        started = time.perf_counter()
        result = tracker.process(frame.image, frame.timestamp)
        self._latency_ms = (time.perf_counter() - started) * 1000.0

        self._last_hands = result.hands
        hand = result.pick(self.cfg.tracker.preferred_hand)
        if hand is None:
            self._features = HandFeatures.absent()
            return

        height, width = frame.image.shape[:2]
        self._features = extract_features(
            hand,
            (width, height),
            previous=self._features if self._features.present else None,
            curl_threshold=self.cfg.clutch.curl_threshold,
        )
        self._features_time = time.monotonic()

    def _current_features(self, now: float) -> HandFeatures:
        """Cechy dloni z kontrola swiezosci - stara detekcja to brak dloni."""
        if not self._features.present:
            return self._features
        if now - self._features_time > self.cfg.safety.hold_timeout_s:
            return HandFeatures.absent()
        return self._features

    def _refresh_measured(self, backend: RobotBackend, now: float) -> None:
        """Odczyt faktycznej pozycji stawow - rzadziej niz petla sterowania."""
        period = 1.0 / max(self.cfg.robot.read_state_hz, 0.1)
        if now - self._last_read < period:
            return
        self._last_read = now
        try:
            self._measured = backend.read_joints()
        except Exception:
            logger.exception("Nie udalo sie odczytac pozycji stawow")

    # ------------------------------------------------------------------ widok
    def _render(
        self,
        frame: np.ndarray,
        output: ControlOutput,
        command: dict[str, float],
        report,
        now: float,
    ) -> np.ndarray:
        """Sklada klatke podgladu: obraz z kamery + HUD + panel ramienia."""
        canvas = frame.copy()

        if self.cfg.ui.draw_skeleton:
            for hand in self._last_hands:
                draw_hand_skeleton(canvas, hand, output.engaged)

        if self.cfg.ui.draw_hud:
            draw_hud(canvas, self._hud_data(output, command, report, now), self.cfg)

        panel = self._preview_panel(command, output, now)
        if panel is not None:
            canvas = compose(canvas, panel)

        if abs(self.cfg.ui.display_scale - 1.0) > 1e-3:
            canvas = cv2.resize(
                canvas,
                None,
                fx=self.cfg.ui.display_scale,
                fy=self.cfg.ui.display_scale,
                interpolation=cv2.INTER_AREA,
            )
        return canvas

    def _record(self, canvas: np.ndarray) -> None:
        """Dopisuje klatke do pliku wideo (otwiera go przy pierwszym uzyciu)."""
        path = self.cfg.ui.record_path
        if not path:
            return
        if self._writer is None:
            height, width = canvas.shape[:2]
            self._writer = cv2.VideoWriter(
                path, cv2.VideoWriter_fourcc(*"mp4v"), self.cfg.loop_hz, (width, height)
            )
            if not self._writer.isOpened():
                logger.error("Nie udalo sie otworzyc pliku nagrania: %s", path)
                self.cfg.ui.record_path = None
                self._writer = None
                return
            logger.info("Nagrywam podglad do %s", path)
        self._writer.write(canvas)

    def _hud_data(self, output: ControlOutput, command: dict, report, now: float) -> HudData:
        message = self._message if now < self._message_until else ""
        return HudData(
            state=self.supervisor.state,
            engaged=output.engaged,
            hand_present=output.hand_present,
            reason=output.reason,
            features=output.features,
            command=command,
            measured=self._measured,
            report=report,
            fps=self.fps.fps,
            latency_ms=self._latency_ms,
            backend=self._backend_label(),
            tracker=self._tracker_label(),
            mode=self.cfg.mapping.mode,
            message=message,
            ik_clamped=bool(output.ik and output.ik.clamped),
        )

    def _preview_panel(
        self, command: dict[str, float], output: ControlOutput, now: float
    ) -> np.ndarray | None:
        """Panel podgladu; model 3D odswiezamy rzadziej niz petle sterowania."""
        if not self._preview.enabled:
            return None

        if not self._preview.is_3d:
            return draw_arm_view(
                self.cfg,
                self.mapper.kinematics,
                command,
                size=(self.cfg.ui.preview_width, max(self.cfg.camera.height, 240)),
                ee_target=output.ee_target,
            )

        period = 1.0 / max(self.cfg.ui.preview_hz, 1.0)
        if self._preview.image is None or now - self._preview.last_render >= period:
            renderer = self._preview.renderer
            model = self._preview.model
            self._preview.image = renderer.render(  # type: ignore[union-attr]
                model.from_lerobot(command),  # type: ignore[union-attr]
                ee_target=output.ee_target,
            )
            self._preview.last_render = now
        return self._preview.image

    def _backend_label(self) -> str:
        return "symulator" if self.cfg.robot.backend == "sim" else self.cfg.robot.backend

    def _tracker_label(self) -> str:
        return self.cfg.tracker.backend

    def _notify(self, text: str, seconds: float = 2.0) -> None:
        self._message = text
        self._message_until = time.monotonic() + seconds
        logger.info(text)

    # ------------------------------------------------------------- klawisze
    def _handle_keys(self, key: int, backend: RobotBackend) -> bool:
        """Obsluga klawiatury. Zwraca False, gdy aplikacja ma sie zakonczyc."""
        if key in (255, -1):
            return True

        char = chr(key).lower() if 32 <= key < 127 else ""

        if key == ESC or char == "q":
            return False
        if key == 32:  # spacja
            engaged = self.mapper.toggle_key_clutch()
            self._notify("Sterowanie WLACZONE" if engaged else "Sterowanie WYLACZONE")
        elif char == "h":
            self.supervisor.begin_homing()
            self.mapper.release_anchor()
            self._notify("Powrot do pozycji domowej")
        elif char == "x":
            if self.supervisor.estopped:
                self.supervisor.clear_estop()
                self._notify("Stop awaryjny skasowany")
            else:
                self.supervisor.trigger_estop()
                self.mapper.set_key_clutch(False)
                self._notify("STOP AWARYJNY", 4.0)
        elif char == "c":
            self.mapper.release_anchor()
            self._notify("Nowe zaczepienie dloni")
        elif char in ("o", "p"):
            which = "open" if char == "o" else "closed"
            features = self.mapper.last_features
            if features and self.mapper.calibrate_pinch(features, which):
                self._notify(
                    f"Chwytak: zapisano {'otwarcie' if which == 'open' else 'zamkniecie'} "
                    f"({features.pinch:.2f})"
                )
            else:
                self._notify("Pokaz dlon, zanim skalibrujesz chwytak")
        elif char == "m":
            self.cfg.mapping.mode = "ik" if self.cfg.mapping.mode == "direct" else "direct"
            self.mapper.release_anchor()
            self._notify(f"Tryb mapowania: {self.cfg.mapping.mode}")
        elif char == "v":
            self._preview.enabled = not self._preview.enabled
            self._notify(f"Podglad ramienia: {'wl.' if self._preview.enabled else 'wyl.'}")
        elif char in ("j", "l", "i", "k", ",", "."):
            self._move_camera(char)
        elif char in ("-", "_"):
            self.cfg.safety.velocity_scale = max(0.1, self.cfg.safety.velocity_scale - 0.1)
            self._notify(f"Limit predkosci: {self.cfg.safety.velocity_scale:.1f}x")
        elif char in ("=", "+"):
            self.cfg.safety.velocity_scale = min(2.0, self.cfg.safety.velocity_scale + 0.1)
            self._notify(f"Limit predkosci: {self.cfg.safety.velocity_scale:.1f}x")
        return True

    def _move_camera(self, char: str) -> None:
        renderer = self._preview.renderer
        if renderer is None:
            return
        camera = renderer.camera  # type: ignore[attr-defined]
        if char == "j":
            camera.orbit(-8.0, 0.0)
        elif char == "l":
            camera.orbit(8.0, 0.0)
        elif char == "i":
            camera.orbit(0.0, 5.0)
        elif char == "k":
            camera.orbit(0.0, -5.0)
        elif char == ",":
            camera.zoom(1.1)
        elif char == ".":
            camera.zoom(1 / 1.1)
        self._preview.image = None  # wymus przerysowanie

    # ------------------------------------------------------------ zamkniecie
    def _shutdown(
        self, camera: CameraStream, tracker: HandTracker | None, backend: RobotBackend
    ) -> None:
        try:
            if backend.is_connected and self.cfg.safety.home_on_exit:
                self._go_home(backend)
        except Exception:
            logger.exception("Blad przy powrocie do pozycji domowej")

        if self._writer is not None:
            self._writer.release()
            self._writer = None
            logger.info("Nagranie zapisane: %s", self.cfg.ui.record_path)

        for name, close in (
            ("robot", backend.disconnect),
            ("kamera", camera.close),
            ("tracker", tracker.close if tracker else lambda: None),
        ):
            try:
                close()
            except Exception:
                logger.exception("Blad przy zamykaniu: %s", name)

        if self.cfg.ui.show:
            cv2.destroyAllWindows()
        logger.info("Zakonczono.")

    def _go_home(self, backend: RobotBackend, timeout: float = 6.0) -> None:
        """Plynny powrot do pozycji domowej przed rozlaczeniem."""
        logger.info("Wracam do pozycji domowej ...")
        self.supervisor.clear_estop()
        self.supervisor.begin_homing()

        rate = LoopRate(self.cfg.loop_hz)
        deadline = time.monotonic() + timeout
        previous = time.monotonic()
        while time.monotonic() < deadline:
            now = time.monotonic()
            dt = min(now - previous, 0.25)
            previous = now
            command, _ = self.supervisor.step(None, dt, hand_present=False, engaged=False)
            backend.send_joints(command)
            backend.step(dt)
            if self.supervisor.state is not SafetyState.HOMING:
                break
            rate.sleep()


def run_app(cfg: AppConfig) -> int:
    return TeleopApp(cfg).run()
