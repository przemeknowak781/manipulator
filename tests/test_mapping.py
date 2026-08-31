"""Mapowanie ruchu dloni na zadane pozycje stawow."""

from __future__ import annotations

import math

import pytest

from lerobot_mp.config import JOINT_NAMES, load_config
from lerobot_mp.control.mapping import HandToJointMapper
from lerobot_mp.vision.features import HandFeatures, extract_features

from conftest import make_hand

DT = 1.0 / 30.0
FRAME = (1280, 720)


def features(**kwargs) -> HandFeatures:
    return extract_features(make_hand(**kwargs), FRAME)


def engaged_mapper(cfg, **overrides) -> HandToJointMapper:
    cfg.clutch.mode = "always"
    for key, value in overrides.items():
        setattr(cfg.mapping, key, value)
    return HandToJointMapper(cfg)


def drive(mapper, feature: HandFeatures, joints: dict, steps: int = 40):
    """Podaje te sama klatke wielokrotnie, zeby filtry doszly do wartosci ustalonej."""
    output = None
    for _ in range(steps):
        output = mapper.update(feature, joints, DT)
    return output


def zeros() -> dict[str, float]:
    return {name: 0.0 for name in JOINT_NAMES}


def test_no_hand_gives_no_target(cfg):
    mapper = HandToJointMapper(cfg)
    output = mapper.update(HandFeatures.absent(), zeros(), DT)
    assert output.targets is None
    assert output.engaged is False
    assert "brak dloni" in output.reason


def test_clutch_key_mode_requires_space(cfg):
    cfg.clutch.mode = "key"
    mapper = HandToJointMapper(cfg)
    assert drive(mapper, features(), zeros()).targets is None
    mapper.toggle_key_clutch()
    assert drive(mapper, features(), zeros()).targets is not None


def test_clutch_gesture_pauses_on_curled_fingers(cfg):
    cfg.clutch.mode = "gesture"
    mapper = HandToJointMapper(cfg)
    mapper.toggle_key_clutch()
    assert drive(mapper, features(curl=0.9), zeros()).engaged is True

    paused = drive(mapper, features(curl=0.2), zeros())
    assert paused.engaged is False
    assert "pauza" in paused.reason


def test_clutch_debounce_needs_time(cfg):
    cfg.clutch.mode = "always"
    mapper = HandToJointMapper(cfg)
    # Jedna klatka nie wystarczy - stan musi sie utrzymac przez `debounce_s`.
    assert mapper.update(features(), zeros(), DT).engaged is False
    assert drive(mapper, features(), zeros()).engaged is True


def test_relative_mode_starts_from_current_pose(cfg):
    """Po zalaczeniu robot nie moze przeskoczyc - cel startuje z jego pozycji."""
    mapper = engaged_mapper(cfg)
    start = dict(zeros(), shoulder_pan=33.0)
    output = drive(mapper, features(center=(0.5, 0.5)), start)
    assert output.targets["shoulder_pan"] == pytest.approx(33.0, abs=1.0)


def test_hand_moving_right_turns_the_base(cfg):
    mapper = engaged_mapper(cfg)
    drive(mapper, features(center=(0.5, 0.5)), zeros())
    moved = drive(mapper, features(center=(0.75, 0.5)), zeros())
    assert moved.targets["shoulder_pan"] > 5.0


def test_hand_moving_up_lifts_the_arm(cfg):
    mapper = engaged_mapper(cfg)
    drive(mapper, features(center=(0.5, 0.5)), zeros())
    moved = drive(mapper, features(center=(0.5, 0.25)), zeros())
    assert moved.targets["shoulder_lift"] > 5.0


def test_hand_closer_to_camera_extends_the_elbow(cfg):
    mapper = engaged_mapper(cfg)
    drive(mapper, features(scale=0.10), zeros())
    closer = drive(mapper, features(scale=0.18), zeros())
    assert closer.targets["elbow_flex"] > 5.0


def test_wrist_rotation_is_geared_one_to_one(cfg):
    """Domyslne przelozenie nadgarstka to 1:1 - obrot dloni o 30 stopni."""
    mapper = engaged_mapper(cfg)
    drive(mapper, features(roll_deg=0.0), zeros())
    turned = drive(mapper, features(roll_deg=30.0), zeros(), steps=90)
    assert turned.targets["wrist_roll"] == pytest.approx(30.0, abs=6.0)


def test_deadzone_absorbs_small_jitter(cfg):
    mapper = engaged_mapper(cfg, deadzone=0.05)
    drive(mapper, features(center=(0.5, 0.5)), zeros())
    jitter = drive(mapper, features(center=(0.51, 0.5)), zeros())
    assert jitter.targets["shoulder_pan"] == pytest.approx(0.0, abs=1.0)


def test_invert_flips_direction(cfg):
    mapper = engaged_mapper(cfg)
    drive(mapper, features(center=(0.5, 0.5)), zeros())
    normal = drive(mapper, features(center=(0.75, 0.5)), zeros()).targets["shoulder_pan"]

    cfg2 = load_config(overrides={"joints": {"shoulder_pan": {"invert": True}}})
    mapper2 = engaged_mapper(cfg2)
    drive(mapper2, features(center=(0.5, 0.5)), zeros())
    inverted = drive(mapper2, features(center=(0.75, 0.5)), zeros()).targets["shoulder_pan"]
    assert normal == pytest.approx(-inverted, abs=1e-6)


def test_gripper_follows_pinch(cfg):
    mapper = engaged_mapper(cfg)
    wide = drive(mapper, features(pinch=1.0), zeros()).targets["gripper"]
    tight = drive(mapper, features(pinch=0.05), zeros()).targets["gripper"]
    assert wide > 80.0
    assert tight < 20.0


def test_gripper_can_be_inverted(cfg):
    mapper = engaged_mapper(cfg, gripper_invert=True)
    wide = drive(mapper, features(pinch=1.0), zeros()).targets["gripper"]
    assert wide < 20.0


def test_gripper_calibration_uses_your_own_hand(cfg):
    mapper = engaged_mapper(cfg)
    output = drive(mapper, features(pinch=0.4), zeros())
    assert mapper.calibrate_pinch(output.features, "open")
    assert mapper.pinch_open == pytest.approx(output.features.pinch)
    # Po kalibracji ta sama dlon oznacza chwytak w pelni otwarty.
    assert drive(mapper, features(pinch=0.4), zeros()).targets["gripper"] > 95.0


def test_gripper_calibration_keeps_thresholds_ordered(cfg):
    mapper = engaged_mapper(cfg)
    output = drive(mapper, features(pinch=0.4), zeros())
    mapper.calibrate_pinch(output.features, "closed")
    mapper.calibrate_pinch(output.features, "open")
    assert mapper.pinch_open > mapper.pinch_closed


def test_gripper_calibration_needs_a_hand(cfg):
    mapper = HandToJointMapper(cfg)
    assert mapper.calibrate_pinch(HandFeatures.absent(), "open") is False
    with pytest.raises(ValueError):
        mapper.calibrate_pinch(features(), "sideways")


def test_releasing_the_anchor_rebases_the_motion(cfg):
    """Po `C` ta sama pozycja dloni ma oznaczac "zostan tam, gdzie jestes"."""
    mapper = engaged_mapper(cfg)
    drive(mapper, features(center=(0.5, 0.5)), zeros())
    moved = drive(mapper, features(center=(0.75, 0.5)), zeros())
    pose = dict(zeros(), shoulder_pan=moved.targets["shoulder_pan"])

    mapper.release_anchor()
    rebased = drive(mapper, features(center=(0.75, 0.5)), pose)
    assert rebased.targets["shoulder_pan"] == pytest.approx(pose["shoulder_pan"], abs=1.0)


def test_losing_the_hand_releases_the_anchor(cfg):
    mapper = engaged_mapper(cfg)
    drive(mapper, features(), zeros())
    assert mapper.anchored
    mapper.update(HandFeatures.absent(), zeros(), DT)
    assert not mapper.anchored


def test_absolute_mode_ignores_current_pose(cfg):
    """W trybie bezwzglednym srodek kadru zawsze oznacza poze domowa."""
    mapper = engaged_mapper(cfg, relative=False)
    output = drive(mapper, features(center=(0.5, 0.5)), dict(zeros(), shoulder_pan=50.0))
    assert output.targets["shoulder_pan"] == pytest.approx(cfg.safety.home["shoulder_pan"], abs=1.0)


def test_ik_mode_produces_reachable_targets(cfg):
    mapper = engaged_mapper(cfg, mode="ik")
    output = drive(mapper, features(center=(0.5, 0.5)), dict(cfg.safety.home))
    assert output.ik is not None
    assert output.ee_target is not None
    assert set(output.targets) == set(JOINT_NAMES)
    assert not output.ik.clamped


def test_ik_and_direct_agree_on_direction(cfg):
    """Ten sam ruch dloni ma obracac podstawe w te sama strone w obu trybach."""
    directions = {}
    for mode in ("direct", "ik"):
        cfg_mode = load_config(overrides={"mapping": {"mode": mode}, "clutch": {"mode": "always"}})
        mapper = HandToJointMapper(cfg_mode)
        home = dict(cfg_mode.safety.home)
        base = drive(mapper, features(center=(0.5, 0.5)), home)
        moved = drive(mapper, features(center=(0.78, 0.5)), home)
        directions[mode] = moved.targets["shoulder_pan"] - base.targets["shoulder_pan"]

    assert directions["direct"] * directions["ik"] > 0, directions
    assert abs(directions["direct"]) > 1.0 and abs(directions["ik"]) > 1.0


def test_ik_clamps_targets_outside_the_workspace(cfg):
    mapper = engaged_mapper(cfg, mode="ik", depth_gain=40.0)
    drive(mapper, features(scale=0.10), dict(cfg.safety.home))
    far = drive(mapper, features(scale=0.30), dict(cfg.safety.home))
    assert far.ik.clamped
    assert all(math.isfinite(v) for v in far.targets.values())


def test_unknown_clutch_mode_is_an_error(cfg):
    cfg.clutch.mode = "telepathy"
    with pytest.raises(ValueError):
        HandToJointMapper(cfg).update(features(), zeros(), DT)
