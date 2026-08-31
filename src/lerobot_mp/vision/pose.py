"""Sledzenie sylwetki operatora - bark, lokiec i nadgarstek.

Sledzenie samej dloni mowi, GDZIE jest reka, ale nie mowi, JAK jest ulozone
ramie. Do sterowania antropomorficznego - moj lokiec zgina sie tak, jak lokiec
robota - potrzebny jest caly lancuch bark -> lokiec -> nadgarstek. Daje go
MediaPipe Pose.

Jak w module `tracker`, obslugujemy oba API MediaPipe i sprowadzamy je do
jednego formatu, wiec reszta aplikacji nie wie, ktora wersja jest zainstalowana.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

import numpy as np

from ..config import ArmTrackingConfig
from .tracker import download_model

logger = logging.getLogger(__name__)

# Indeksy punktow sylwetki wg MediaPipe Pose (33 punkty).
# Uwaga: "LEFT"/"RIGHT" sa z perspektywy OPERATORA, nie kamery.
NOSE = 0
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_HIP, RIGHT_HIP = 23, 24

NUM_POSE_LANDMARKS = 33

#: Punkty ramienia, ktore musza byc widoczne, zeby uznac je za sledzone.
ARM_POINTS = {
    "Left": (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST),
    "Right": (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST),
}

#: Polaczenia do rysowania ramienia na podgladzie.
ARM_CONNECTIONS = ((LEFT_SHOULDER, LEFT_ELBOW), (LEFT_ELBOW, LEFT_WRIST))


@dataclass
class PoseSample:
    """Sylwetka operatora w jednej klatce.

    Attributes:
        landmarks: (33, 3) wspolrzedne znormalizowane do kadru.
        world_landmarks: (33, 3) wspolrzedne metryczne [m] w ukladzie sylwetki
            (poczatek miedzy biodrami). To z nich licza sie katy - wspolrzedne
            obrazu splaszczylyby ruch "do kamery".
        visibility: (33,) wiarygodnosc kazdego punktu, 0..1.
        timestamp: czas KLATKI, z ktorej sylwetka powstala [s, monotoniczny].
            W trybie `live_stream` jest starszy niz chwila odbioru wyniku.
    """

    landmarks: np.ndarray
    world_landmarks: np.ndarray | None = None
    visibility: np.ndarray = field(default_factory=lambda: np.zeros(NUM_POSE_LANDMARKS))
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        self.landmarks = np.asarray(self.landmarks, dtype=np.float32).reshape(-1, 3)
        if self.world_landmarks is not None:
            self.world_landmarks = np.asarray(self.world_landmarks, dtype=np.float32).reshape(-1, 3)
        self.visibility = np.asarray(self.visibility, dtype=np.float32).reshape(-1)

    @property
    def has_world(self) -> bool:
        return self.world_landmarks is not None

    def arm_visibility(self, side: str) -> float:
        """Najslabiej widoczny punkt danego ramienia - o nim decyduje prog."""
        points = ARM_POINTS.get(side)
        if points is None:
            return 0.0
        return float(min(self.visibility[i] for i in points))


class _PoseTasksBackend:
    """Nowe API: `mediapipe.tasks.python.vision.PoseLandmarker`."""

    name = "tasks"

    def __init__(self, cfg: ArmTrackingConfig, live: bool):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        self._live = live
        self._lock = threading.Lock()
        self._async_result: PoseSample | None = None
        self._last_ms = -1

        mode = vision.RunningMode.LIVE_STREAM if live else vision.RunningMode.VIDEO
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=str(download_model(cfg.model_path, cfg.model_url, "Pose"))
            ),
            running_mode=mode,
            num_poses=1,
            min_pose_detection_confidence=cfg.min_detection_confidence,
            min_pose_presence_confidence=cfg.min_presence_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence,
            result_callback=self._on_result if live else None,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)

    @staticmethod
    def _convert(result: object) -> PoseSample | None:
        poses = getattr(result, "pose_landmarks", []) or []
        if not poses:
            return None
        points = np.array([[p.x, p.y, p.z] for p in poses[0]], dtype=np.float32)
        visibility = np.array([getattr(p, "visibility", 1.0) for p in poses[0]], dtype=np.float32)

        world = getattr(result, "pose_world_landmarks", []) or []
        world_points = None
        if world and world[0]:
            world_points = np.array([[p.x, p.y, p.z] for p in world[0]], dtype=np.float32)
        return PoseSample(points, world_points, visibility)

    def _on_result(self, result: object, _image: object, _timestamp_ms: int) -> None:
        converted = self._convert(result)
        # Tak samo jak przy dloni: liczy sie czas klatki, a nie czas odbioru.
        if converted is not None:
            converted.timestamp = _timestamp_ms / 1000.0
        with self._lock:
            self._async_result = converted

    def process(self, image_rgb: np.ndarray, timestamp_ms: int) -> PoseSample | None:
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=image_rgb)
        timestamp_ms = max(timestamp_ms, self._last_ms + 1)
        self._last_ms = timestamp_ms

        if self._live:
            self._landmarker.detect_async(mp_image, timestamp_ms)
            with self._lock:
                return self._async_result
        sample = self._convert(self._landmarker.detect_for_video(mp_image, timestamp_ms))
        if sample is not None:
            sample.timestamp = timestamp_ms / 1000.0
        return sample

    def close(self) -> None:
        try:
            self._landmarker.close()
        except Exception:  # pragma: no cover - sprzatanie MediaPipe bywa halasliwe
            logger.debug("Blad przy zamykaniu PoseLandmarker", exc_info=True)


class _PoseLegacyBackend:
    """Stare API: `mediapipe.solutions.pose.Pose` (MediaPipe <= 0.10.x)."""

    name = "legacy"

    def __init__(self, cfg: ArmTrackingConfig, live: bool):  # noqa: ARG002
        import mediapipe as mp

        self._pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=cfg.min_detection_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence,
        )

    def process(self, image_rgb: np.ndarray, timestamp_ms: int) -> PoseSample | None:
        result = self._pose.process(image_rgb)
        if not result.pose_landmarks:
            return None
        points = np.array(
            [[p.x, p.y, p.z] for p in result.pose_landmarks.landmark], dtype=np.float32
        )
        visibility = np.array(
            [p.visibility for p in result.pose_landmarks.landmark], dtype=np.float32
        )
        world = None
        if getattr(result, "pose_world_landmarks", None):
            world = np.array(
                [[p.x, p.y, p.z] for p in result.pose_world_landmarks.landmark], dtype=np.float32
            )
        return PoseSample(points, world, visibility, timestamp_ms / 1000.0)

    def close(self) -> None:
        self._pose.close()


class PoseTracker:
    """Detektor sylwetki niezalezny od wersji MediaPipe."""

    def __init__(self, cfg: ArmTrackingConfig, running_mode: str = "video"):
        self.cfg = cfg
        live = running_mode.lower() == "live_stream"
        try:
            self._backend: _PoseTasksBackend | _PoseLegacyBackend = _PoseTasksBackend(cfg, live)
        except Exception as tasks_exc:
            logger.warning("Pose przez Tasks API niedostepne (%s) - probuje starego API.", tasks_exc)
            try:
                self._backend = _PoseLegacyBackend(cfg, live)
            except Exception as legacy_exc:
                raise RuntimeError(
                    "Nie udalo sie uruchomic sledzenia sylwetki (MediaPipe Pose).\n"
                    f"  tasks : {tasks_exc}\n"
                    f"  legacy: {legacy_exc}"
                ) from legacy_exc
        logger.info("Backend MediaPipe Pose: %s", self._backend.name)

    @property
    def backend_name(self) -> str:
        return self._backend.name

    def process(self, image_bgr: np.ndarray, timestamp: float) -> PoseSample | None:
        import cv2

        rgb = np.ascontiguousarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        return self.process_rgb(rgb, timestamp)

    def process_rgb(self, image_rgb: np.ndarray, timestamp: float) -> PoseSample | None:
        """To samo, ale na gotowej klatce RGB - patrz `HandTracker.process_rgb`."""
        return self._backend.process(image_rgb, int(timestamp * 1000.0))

    def close(self) -> None:
        self._backend.close()

    def __enter__(self) -> "PoseTracker":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
