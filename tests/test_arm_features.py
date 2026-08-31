"""Katy ramienia operatora liczone w ukladzie tulowia."""

from __future__ import annotations

import numpy as np
import pytest

from lerobot_mp.vision.arm_features import (
    ArmFeatures,
    extract_arm_features,
    pick_arm,
    torso_frame,
)
from lerobot_mp.vision.pose import NUM_POSE_LANDMARKS, PoseSample

from conftest import make_pose

FRAME = (1280, 720)


def features(**kwargs) -> ArmFeatures:
    return extract_arm_features(make_pose(**kwargs), kwargs.get("side", "Right"), FRAME)


def test_torso_frame_faces_the_camera():
    """Os "do przodu" ma wskazywac tam, gdzie patrzy operator - w kamere.

    W ukladzie MediaPipe blizej kamery znaczy MNIEJSZE Z, wiec skladowa Z
    tej osi musi byc ujemna. To pinuje konwencje, na ktorej stoi caly azymut.
    """
    up, right, forward = torso_frame(make_pose().world_landmarks)
    assert up[1] < 0          # w gore = mniejsze Y
    assert right[0] < 0       # prawa strona operatora jest po lewej obrazu
    assert forward[2] < 0     # do przodu = w strone kamery
    for a, b in ((up, right), (right, forward), (forward, up)):
        assert abs(float(np.dot(a, b))) < 1e-6
        assert float(np.linalg.norm(a)) == pytest.approx(1.0, abs=1e-6)


def test_hanging_arm_is_ninety_degrees_down():
    assert features(elevation_deg=-90.0).elevation == pytest.approx(-90.0, abs=0.5)


def test_elevation_matches_the_input():
    for angle in (-90.0, -45.0, 0.0, 45.0, 80.0):
        assert features(elevation_deg=angle).elevation == pytest.approx(angle, abs=0.5)


def test_azimuth_zero_points_away_from_the_body():
    assert features(elevation_deg=0.0, azimuth_deg=0.0).azimuth == pytest.approx(0.0, abs=0.5)


def test_azimuth_ninety_points_at_the_camera():
    assert features(elevation_deg=0.0, azimuth_deg=90.0).azimuth == pytest.approx(90.0, abs=0.5)


def test_azimuth_matches_the_input():
    for angle in (-80.0, -30.0, 0.0, 30.0, 80.0):
        assert features(elevation_deg=0.0, azimuth_deg=angle).azimuth == pytest.approx(angle, abs=0.5)


def test_elbow_bend_matches_the_input():
    for angle in (0.0, 30.0, 90.0, 140.0):
        assert features(elevation_deg=0.0, elbow_deg=angle).elbow == pytest.approx(angle, abs=0.5)


def test_both_arms_give_the_same_angles_for_the_same_gesture():
    """Lewa i prawa reka sa lustrzane - ten sam gest ma dac te same liczby."""
    for angle in (-30.0, 0.0, 45.0):
        right = extract_arm_features(
            make_pose(elevation_deg=angle, azimuth_deg=25.0, side="Right"), "Right", FRAME
        )
        left = extract_arm_features(
            make_pose(elevation_deg=angle, azimuth_deg=25.0, side="Left"), "Left", FRAME
        )
        assert left.elevation == pytest.approx(right.elevation, abs=0.5)
        assert left.azimuth == pytest.approx(right.azimuth, abs=0.5)


def test_low_visibility_is_rejected():
    """Zaslonieta reka ma dac "brak", a nie zgadniete katy."""
    pose = make_pose(visibility=0.3)
    assert extract_arm_features(pose, "Right", FRAME, min_visibility=0.6).present is False


def test_missing_world_landmarks_are_rejected():
    pose = PoseSample(
        np.zeros((NUM_POSE_LANDMARKS, 3)), None, np.ones(NUM_POSE_LANDMARKS)
    )
    assert extract_arm_features(pose, "Right", FRAME).present is False


def test_degenerate_skeleton_is_rejected():
    pose = PoseSample(
        np.zeros((NUM_POSE_LANDMARKS, 3)),
        np.zeros((NUM_POSE_LANDMARKS, 3)),
        np.ones(NUM_POSE_LANDMARKS),
    )
    assert extract_arm_features(pose, "Right", FRAME).present is False


def test_unknown_side_is_an_error():
    with pytest.raises(ValueError):
        extract_arm_features(make_pose(), "middle", FRAME)


def test_pixel_points_are_inside_the_frame():
    points = features().points_px
    assert len(points) == 3
    for x, y in points:
        assert 0 <= x <= FRAME[0] and 0 <= y <= FRAME[1]


def test_pick_arm_honours_the_request():
    pose = make_pose()
    assert pick_arm(pose, "Left") == "Left"
    assert pick_arm(pose, "Right") == "Right"


def test_pick_arm_auto_needs_visibility():
    assert pick_arm(make_pose(visibility=1.0), "auto") in ("Left", "Right")
    assert pick_arm(make_pose(visibility=0.2), "auto") is None
    assert pick_arm(make_pose(visibility=0.2), "Right") is None


def test_angles_ignore_where_the_operator_stands():
    """Katy sa liczone w ukladzie tulowia, wiec obrot calej sylwetki ich nie zmienia.

    To wlasnie po to uklad tulowia istnieje: przechylenie sie na krzesle albo
    obrot bokiem do kamery nie ma zmieniac zadanej pozy robota. Obracamy tu
    caly szkielet - reka i tulow razem - i sprawdzamy, ze liczby stoja.
    """
    import math

    base = make_pose(elevation_deg=-30.0, azimuth_deg=25.0, elbow_deg=60.0)
    reference = extract_arm_features(base, "Right", FRAME)

    for angle_deg in (-40.0, -15.0, 15.0, 40.0):
        angle = math.radians(angle_deg)
        # Obrot calej sylwetki wokol osi patrzenia (Z).
        rotation = np.array(
            [
                [math.cos(angle), -math.sin(angle), 0.0],
                [math.sin(angle), math.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        turned = PoseSample(
            base.landmarks,
            base.world_landmarks @ rotation.T,
            base.visibility,
        )
        moved = extract_arm_features(turned, "Right", FRAME)
        assert moved.elevation == pytest.approx(reference.elevation, abs=0.5)
        assert moved.azimuth == pytest.approx(reference.azimuth, abs=0.5)
        assert moved.elbow == pytest.approx(reference.elbow, abs=0.5)
