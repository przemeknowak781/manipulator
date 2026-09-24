"""Straznik przestawionej kamery, maska ramienia w geometrii surowego kadru i wiek kadrow z kamer.

Bez sprzetu i bez przegladarki: syntetyczne kadry z tekstura (jak blat z przedmiotami)
i atrapa strumienia kamery.
"""

from __future__ import annotations

import time

import cv2
import numpy as np
import pytest

from lerobot_mp.twin.ui.watch import CameraWatch, distort_mask


def textured(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    g = cv2.GaussianBlur((rng.random((480, 640)) * 255).astype(np.uint8), (0, 0), 3)
    g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)
    for _ in range(30):
        cv2.circle(g, (int(rng.integers(0, 640)), int(rng.integers(0, 480))), int(rng.integers(5, 40)),
                   int(rng.integers(0, 255)), -1)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)


def moved(img: np.ndarray, roll_deg: float = 0.0, zoom: float = 1.0, tx: float = 0.0) -> np.ndarray:
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), roll_deg, zoom)
    M[0, 2] += tx
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)


@pytest.mark.parametrize("roll, zoom", [(2.0, 1.0), (3.0, 1.0), (0.0, 1.04)])
def test_roll_and_motion_along_the_optical_axis_are_caught(roll, zoom):
    """Korelacja fazowa widziala tu 0,9 / 1,6 / 0,55 px (prog 3 px), a narozniki jechaly o 14 / 21 / 16 px."""
    img = textured()
    w = CameraWatch()
    w.remember("k", img)
    assert w.check("k", moved(img, roll, zoom)) > 8.0
    assert w.moved("k")


def test_translation_is_measured_and_a_still_camera_stays_quiet_after_many_checks():
    img = textured(1)
    w = CameraWatch()
    w.remember("k", img)
    # Wiele sprawdzen tym samym odniesieniem: `phaseCorrelate` mnozyl odniesienie przez okno
    # Hanninga W MIEJSCU, a po kilku sprawdzeniach zostawal z niego sam srodek kadru.
    for _ in range(8):
        assert w.check("k", img) < 0.5
    assert not w.moved("k")
    assert w.check("k", moved(img, tx=5.0)) == pytest.approx(5.0, abs=0.5)
    assert w.check("k", moved(img, roll_deg=2.0)) > 8.0


def test_a_different_picture_is_not_reported_as_a_still_camera():
    w = CameraWatch()
    w.remember("k", textured(2))
    assert w.check("k", textured(3)) > 100.0
    assert w.moved("k")


def test_masked_arm_moving_does_not_count_as_camera_motion():
    img = textured(4)
    mask = np.zeros(img.shape[:2], bool)
    mask[150:350, 250:420] = True
    cur = img.copy()
    cur[mask] = 255 - cur[mask]                         # "ramie" w innym miejscu tego samego obszaru
    w = CameraWatch()
    w.remember("k", img, mask)
    assert w.check("k", cur, mask) < 1.0


def test_render_mask_is_warped_into_the_distorted_raw_frame():
    """Punkt sceny: w renderze (bez dystorsji) i w surowym kadrze (z dystorsja) - maska ma byc tam, gdzie kadr."""
    K = np.array([[600.0, 0, 319.5], [0, 600.0, 239.5], [0, 0, 1]])
    dist = np.array([-0.3, 0.1, 0.0, 0.0, 0.0])
    p = np.array([[[0.35, 0.28, 1.0]]])                   # daleko od srodka - tu dystorsja jest duza
    pin, _ = cv2.projectPoints(p, np.zeros(3), np.zeros(3), K, None)
    raw, _ = cv2.projectPoints(p, np.zeros(3), np.zeros(3), K, dist)
    (u0, v0), (u1, v1) = pin.ravel(), raw.ravel()
    assert np.hypot(u1 - u0, v1 - v0) > 10                # przypadek, w ktorym to ma znaczenie
    mask = np.zeros((480, 640), bool)
    cv2.circle(mask.view(np.uint8), (int(round(u0)), int(round(v0))), 4, 1, -1)
    out = distort_mask(mask, K, dist)
    ys, xs = np.nonzero(out)
    assert np.hypot(xs.mean() - u1, ys.mean() - v1) < 1.5
    assert distort_mask(mask, K, None) is mask            # bez dystorsji - bez zmian


class _FakeStream:
    def __init__(self):
        from lerobot_mp.vision.camera import Frame
        self.Frame = Frame
        self.frame = None
        self.error = None
        self.is_running = True

    def read(self):
        return self.frame


def _hub_with_fake_camera():
    from lerobot_mp.twin import cameras as cams
    from lerobot_mp.twin.workspace import CameraRecord, Workspace

    ws = Workspace()
    ws.add_camera(CameraRecord("usb", "0"))
    hub = cams.CameraHub(ws, stale_after=0.5)
    live = cams._Live.__new__(cams._Live)
    live.stream = _FakeStream()
    hub._live["usb"] = live
    return hub, live.stream


def test_camera_hub_reports_capture_time_and_drops_a_frozen_stream():
    """Zamrozony strumien (USB przez Shadow) oddawal ostatnia klatke w nieskonczonosc jako nowa."""
    hub, stream = _hub_with_fake_camera()
    img = np.zeros((4, 4, 3), np.uint8)
    t = time.monotonic()
    stream.frame = stream.Frame(image=img, timestamp=t, index=1)
    got, t_got = hub.frame_t("usb")
    assert got is not None and t_got == pytest.approx(t)
    assert hub.frame("usb") is not None and hub.error("usb") is None
    stream.frame = stream.Frame(image=img, timestamp=t - 2.0, index=1)
    assert hub.frame("usb") is None
    assert "brak nowych klatek" in hub.error("usb")
    got, t_got = hub.frame_t("usb")                       # frame_t oddaje - z wiekiem, konsument decyduje
    assert got is not None and t_got == pytest.approx(t - 2.0)


def test_camera_hub_gives_no_frame_from_a_dead_stream():
    hub, stream = _hub_with_fake_camera()
    stream.frame = stream.Frame(image=np.zeros((4, 4, 3), np.uint8), timestamp=time.monotonic(), index=1)
    stream.error, stream.is_running = "Koniec strumienia obrazu.", False
    assert hub.frame_t("usb") == (None, 0.0)
    assert "przerwany" in hub.error("usb")
