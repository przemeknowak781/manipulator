"""Ekstrakcja cech sterujacych z punktow dloni."""

from __future__ import annotations

import math

import numpy as np
import pytest

from lerobot_mp.vision.features import extract_features
from lerobot_mp.vision.landmarks import HandSample, NUM_LANDMARKS, TrackResult

from conftest import make_hand


def test_palm_position_follows_hand(frame_size):
    left = extract_features(make_hand(center=(0.3, 0.4)), frame_size)
    right = extract_features(make_hand(center=(0.7, 0.4)), frame_size)
    assert left.present and right.present
    assert right.x > left.x
    assert right.y == pytest.approx(left.y, abs=1e-6)


def test_scale_grows_when_hand_is_closer(frame_size):
    near = extract_features(make_hand(scale=0.18), frame_size)
    far = extract_features(make_hand(scale=0.09), frame_size)
    assert near.scale > far.scale * 1.5


def test_roll_follows_hand_rotation(frame_size):
    """Obrot dloni ma sie przekladac na `roll` mniej wiecej 1:1."""
    base = extract_features(make_hand(roll_deg=0.0), frame_size)
    turned = extract_features(make_hand(roll_deg=30.0), frame_size)
    delta = math.degrees(turned.roll - base.roll)
    assert delta == pytest.approx(30.0, abs=3.0)


def test_roll_is_consistent_between_hands(frame_size):
    """Lewa dlon jest odbiciem prawej - ten sam gest ma dac ten sam obrot."""
    for angle in (-30.0, 0.0, 25.0):
        right = extract_features(make_hand(roll_deg=angle, handedness="Right"), frame_size)
        left = extract_features(make_hand(roll_deg=angle, handedness="Left"), frame_size)
        assert math.degrees(abs(right.roll - left.roll)) == pytest.approx(0.0, abs=0.5)


def test_roll_direction_is_the_same_for_both_hands(frame_size):
    """Obrot nadgarstka w te sama strone ma dawac ten sam znak zmiany."""
    for handedness in ("Right", "Left"):
        base = extract_features(make_hand(roll_deg=0.0, handedness=handedness), frame_size)
        turned = extract_features(make_hand(roll_deg=20.0, handedness=handedness), frame_size)
        assert math.degrees(turned.roll - base.roll) == pytest.approx(20.0, abs=3.0)


def test_pinch_reflects_finger_distance(frame_size):
    wide = extract_features(make_hand(pinch=0.9), frame_size)
    tight = extract_features(make_hand(pinch=0.1), frame_size)
    assert wide.pinch > tight.pinch * 3


def test_pinch_is_independent_of_distance_from_camera(frame_size):
    """Chwytak nie moze reagowac na zblizanie dloni do kamery."""
    near = extract_features(make_hand(scale=0.20, pinch=0.5), frame_size)
    far = extract_features(make_hand(scale=0.08, pinch=0.5), frame_size)
    assert near.pinch == pytest.approx(far.pinch, rel=0.05)


def test_curled_fingers_are_detected(frame_size):
    assert extract_features(make_hand(curl=0.9), frame_size).curled is False
    assert extract_features(make_hand(curl=0.25), frame_size).curled is True


def test_pinching_does_not_look_like_a_fist(frame_size):
    """Kluczowe rozroznienie: szczypanie steruje chwytakiem, piesc pauzuje."""
    pinching = extract_features(make_hand(curl=0.9, pinch=0.05), frame_size)
    assert pinching.curled is False


def test_curl_detection_has_hysteresis(frame_size):
    """Na granicy progu stan nie moze migotac miedzy klatkami."""
    open_hand = extract_features(make_hand(curl=0.9), frame_size)
    threshold = 0.58
    borderline = make_hand(curl=0.5)
    from_open = extract_features(borderline, frame_size, previous=open_hand, curl_threshold=threshold)
    from_curled = extract_features(
        borderline,
        frame_size,
        previous=extract_features(make_hand(curl=0.2), frame_size),
        curl_threshold=threshold,
    )
    # Ta sama klatka moze dac inny wynik zaleznie od stanu poprzedniego.
    assert from_open.extension == pytest.approx(from_curled.extension)


def test_aspect_ratio_is_corrected(frame_size):
    """Kadr 16:9 sciska os X - bez korekty `pinch` zalezalby od obrotu dloni."""
    wide = extract_features(make_hand(roll_deg=0.0), (1920, 1080))
    tall = extract_features(make_hand(roll_deg=90.0), (1920, 1080))
    assert wide.scale == pytest.approx(tall.scale, rel=0.02)


def test_degenerate_hand_is_reported_absent(frame_size):
    flat = HandSample(np.zeros((NUM_LANDMARKS, 3), dtype=np.float32))
    assert extract_features(flat, frame_size).present is False


def test_absent_features_have_safe_defaults():
    from lerobot_mp.vision.features import HandFeatures

    absent = HandFeatures.absent()
    assert absent.present is False
    assert absent.curled is False


def test_track_result_picks_requested_hand():
    left = HandSample(np.zeros((NUM_LANDMARKS, 3)), handedness="Left", score=0.6)
    right = HandSample(np.zeros((NUM_LANDMARKS, 3)), handedness="Right", score=0.9)
    result = TrackResult([left, right])
    assert result.pick("Left") is left
    assert result.pick("any") is right          # najpewniejsza
    assert result.pick() is right
    assert TrackResult([left]).pick("Right") is None
    assert TrackResult().pick() is None
