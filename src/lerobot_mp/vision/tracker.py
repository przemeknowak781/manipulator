"""Detekcja dloni w MediaPipe - z obsluga obu API biblioteki.

MediaPipe 1.x usunelo stare `mediapipe.solutions.hands`, a MediaPipe 0.10.x
jest wciaz bardzo popularne. Ta klasa wybiera dostepny backend automatycznie
i w obu przypadkach zwraca ten sam `TrackResult`.
"""

from __future__ import annotations

import logging
import threading
import urllib.request
from pathlib import Path

import numpy as np

from ..config import TrackerConfig
from .landmarks import HandSample, TrackResult

logger = logging.getLogger(__name__)


def ensure_model(cfg: TrackerConfig) -> Path:
    """Zwraca sciezke do modelu `.task`, pobierajac go przy pierwszym uzyciu."""
    path = Path(cfg.model_path).expanduser()
    if path.is_file() and path.stat().st_size > 0:
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Pobieram model MediaPipe do %s ...", path)
    tmp = path.with_suffix(path.suffix + ".part")
    try:
        with urllib.request.urlopen(cfg.model_url, timeout=60) as response:  # noqa: S310
            tmp.write_bytes(response.read())
        tmp.replace(path)
    except Exception as exc:  # pragma: no cover - zalezne od sieci
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"Nie udalo sie pobrac modelu MediaPipe ({exc}).\n"
            f"Pobierz go recznie i zapisz jako {path}:\n"
            f"  curl -L -o {path} {cfg.model_url}"
        ) from exc
    logger.info("Model pobrany (%.1f MB).", path.stat().st_size / 1e6)
    return path


class _TasksBackend:
    """Nowe API: `mediapipe.tasks.python.vision.HandLandmarker`."""

    name = "tasks"

    def __init__(self, cfg: TrackerConfig):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        self._vision = vision
        self.cfg = cfg
        self._live = cfg.running_mode.lower() == "live_stream"
        self._lock = threading.Lock()
        self._async_result: TrackResult | None = None
        self._last_ms = -1

        mode = vision.RunningMode.LIVE_STREAM if self._live else vision.RunningMode.VIDEO
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(ensure_model(cfg))),
            running_mode=mode,
            num_hands=cfg.num_hands,
            min_hand_detection_confidence=cfg.min_detection_confidence,
            min_hand_presence_confidence=cfg.min_presence_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence,
            result_callback=self._on_result if self._live else None,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)

    # --------------------------------------------------------------- helpers
    @staticmethod
    def _convert(result: object) -> TrackResult:
        hands: list[HandSample] = []
        hand_landmarks = getattr(result, "hand_landmarks", []) or []
        world = getattr(result, "hand_world_landmarks", []) or []
        handedness = getattr(result, "handedness", []) or []

        for i, lms in enumerate(hand_landmarks):
            pts = np.array([[p.x, p.y, p.z] for p in lms], dtype=np.float32)
            wpts = None
            if i < len(world) and world[i]:
                wpts = np.array([[p.x, p.y, p.z] for p in world[i]], dtype=np.float32)
            label, score = "Unknown", 0.0
            if i < len(handedness) and handedness[i]:
                label = handedness[i][0].category_name or "Unknown"
                score = float(handedness[i][0].score)
            hands.append(HandSample(pts, wpts, label, score))
        return TrackResult(hands=hands)

    def _on_result(self, result: object, _image: object, _timestamp_ms: int) -> None:
        converted = self._convert(result)
        with self._lock:
            self._async_result = converted

    # ------------------------------------------------------------------ api
    def process(self, image_rgb: np.ndarray, timestamp_ms: int) -> TrackResult:
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=image_rgb)
        # MediaPipe wymaga scisle rosnacych znacznikow czasu.
        timestamp_ms = max(timestamp_ms, self._last_ms + 1)
        self._last_ms = timestamp_ms

        if self._live:
            self._landmarker.detect_async(mp_image, timestamp_ms)
            with self._lock:
                return self._async_result or TrackResult()
        return self._convert(self._landmarker.detect_for_video(mp_image, timestamp_ms))

    def close(self) -> None:
        try:
            self._landmarker.close()
        except Exception:  # pragma: no cover - sprzatanie MediaPipe bywa halasliwe
            logger.debug("Blad przy zamykaniu HandLandmarker", exc_info=True)


class _LegacyBackend:
    """Stare API: `mediapipe.solutions.hands.Hands` (MediaPipe <= 0.10.x)."""

    name = "legacy"

    def __init__(self, cfg: TrackerConfig):
        import mediapipe as mp

        self.cfg = cfg
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=cfg.num_hands,
            model_complexity=1,
            min_detection_confidence=cfg.min_detection_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence,
        )

    def process(self, image_rgb: np.ndarray, timestamp_ms: int) -> TrackResult:  # noqa: ARG002
        result = self._hands.process(image_rgb)
        hands: list[HandSample] = []
        multi = result.multi_hand_landmarks or []
        world = result.multi_hand_world_landmarks or []
        handed = result.multi_handedness or []

        for i, lms in enumerate(multi):
            pts = np.array([[p.x, p.y, p.z] for p in lms.landmark], dtype=np.float32)
            wpts = None
            if i < len(world):
                wpts = np.array([[p.x, p.y, p.z] for p in world[i].landmark], dtype=np.float32)
            label, score = "Unknown", 0.0
            if i < len(handed):
                cls = handed[i].classification[0]
                label = cls.label or "Unknown"
                score = float(cls.score)
            hands.append(HandSample(pts, wpts, label, score))
        return TrackResult(hands=hands)

    def close(self) -> None:
        self._hands.close()


class HandTracker:
    """Wysokopoziomowy detektor dloni niezalezny od wersji MediaPipe."""

    def __init__(self, cfg: TrackerConfig):
        self.cfg = cfg
        self._backend = self._make_backend(cfg)
        logger.info("Backend MediaPipe: %s", self._backend.name)

    @staticmethod
    def _make_backend(cfg: TrackerConfig) -> _TasksBackend | _LegacyBackend:
        wanted = cfg.backend.lower()
        if wanted == "tasks":
            return _TasksBackend(cfg)
        if wanted == "legacy":
            return _LegacyBackend(cfg)
        if wanted != "auto":
            raise ValueError(f"Nieznany backend trackera: {cfg.backend!r} (auto|tasks|legacy)")

        try:
            return _TasksBackend(cfg)
        except Exception as tasks_exc:
            logger.warning("Tasks API niedostepne (%s) - probuje starego API.", tasks_exc)
            try:
                return _LegacyBackend(cfg)
            except Exception as legacy_exc:
                raise RuntimeError(
                    "Nie udalo sie uruchomic zadnego backendu MediaPipe.\n"
                    f"  tasks : {tasks_exc}\n"
                    f"  legacy: {legacy_exc}\n"
                    "Zainstaluj mediapipe: pip install 'mediapipe>=0.10.9'"
                ) from legacy_exc

    @property
    def backend_name(self) -> str:
        return self._backend.name

    def process(self, image_bgr: np.ndarray, timestamp: float) -> TrackResult:
        """Wykrywa dlonie na klatce BGR (jak z OpenCV)."""
        import cv2

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        result = self._backend.process(rgb, int(timestamp * 1000.0))
        result.timestamp = timestamp
        return result

    def close(self) -> None:
        self._backend.close()

    def __enter__(self) -> "HandTracker":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
