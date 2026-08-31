"""Kinematyka prosta i odwrotna SO-101."""

from __future__ import annotations

import dataclasses
import itertools
import math

import pytest

from lerobot_mp.config import ArmGeometryConfig
from lerobot_mp.control.kinematics import ArmKinematics

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex")


@pytest.fixture
def kin(cfg):
    return ArmKinematics(cfg.geometry)


def test_rejects_degenerate_geometry():
    with pytest.raises(ValueError):
        ArmKinematics(dataclasses.replace(ArmGeometryConfig(), upper_arm=0.0))


def test_forward_at_zero_matches_real_arm(kin):
    """Poza zerowa: ramie wyciagniete, koncowka ~40 cm przed podstawa.

    Liczby pochodza z prawdziwego zlozenia SO-101 (patrz scripts/derive_geometry.py).
    """
    x, y, z = kin.forward(0.0, 0.0, 0.0, 0.0)
    assert x == pytest.approx(0.3975, abs=2e-3)
    assert y == pytest.approx(0.0, abs=2e-3)
    assert z == pytest.approx(0.2241, abs=2e-3)


def test_pan_rotates_about_its_own_axis(kin, cfg):
    """Os obrotu podstawy jest wysunieta do przodu - promien liczy sie od niej."""
    axis_x = cfg.geometry.pan_axis_x
    reference = kin.forward(0.0, 0.0, 0.0, 0.0)
    radius = math.hypot(reference[0] - axis_x, reference[1])
    for pan in (-90.0, -30.0, 30.0, 90.0):
        x, y, _ = kin.forward(pan, 0.0, 0.0, 0.0)
        assert math.hypot(x - axis_x, y) == pytest.approx(radius, abs=1e-6)


def test_positive_pan_turns_to_negative_y(kin):
    """Zwrot zgodny z kalibracja SO-101 - sprawdzony wobec modelu 3D."""
    assert kin.forward(30.0, 0.0, 0.0, 0.0)[1] < kin.forward(0.0, 0.0, 0.0, 0.0)[1]


def test_inverse_round_trip_is_exact(kin):
    """IK ma dokladnie odwracac FK w calym uzywanym zakresie."""
    worst = 0.0
    for pan, lift, elbow, wrist in itertools.product(
        (-60.0, 0.0, 60.0), (-60.0, -20.0, 40.0), (-60.0, 0.0, 60.0), (-40.0, 0.0, 40.0)
    ):
        target = kin.forward(pan, lift, elbow, wrist)
        pitch = kin.tool_pitch(lift, elbow, wrist)
        result = kin.inverse(*target, pitch)
        if result.clamped:
            continue
        worst = max(worst, max(abs(a - b) for a, b in zip(result.reached, target)))
    assert worst < 1e-9


def test_inverse_reports_and_clamps_unreachable_targets(kin):
    result = kin.inverse(2.0, 0.0, 0.5, 0.0)
    assert result.clamped
    reached = kin.forward(**result.as_dict())
    assert math.dist(reached, (2.0, 0.0, 0.5)) > 1.0  # daleko, ale bez wyjatku
    assert all(math.isfinite(v) for v in result.as_dict().values())


def test_inverse_handles_target_on_the_rotation_axis(kin, cfg):
    result = kin.inverse(cfg.geometry.pan_axis_x, 0.0, 0.2, 0.0)
    assert result.clamped
    assert all(math.isfinite(v) for v in result.as_dict().values())


def test_workspace_is_mostly_reachable_within_joint_limits(cfg):
    """Domyslna przestrzen robocza ma sie miescic w zakresach stawow.

    Nie wymagamy 100% - naroza szescianu przy skrajnym kacie narzedzia
    wychodza poza zakres i sa przycinane przez nadzor. Wymagamy, zeby
    zdecydowana wiekszosc byla osiagalna, bo inaczej sterowanie w trybie `ik`
    non stop uderzaloby w limity.
    """
    kin = ArmKinematics(cfg.geometry)
    limits = {name: (cfg.joint(name).min, cfg.joint(name).max) for name in JOINTS}
    ws = cfg.workspace
    cx, cy, cz = ws.center

    reachable = total = 0
    for dx, dy, dz, pitch in itertools.product(
        (-0.06, 0.0, 0.06),
        (-0.09, 0.0, 0.09),
        (-0.07, 0.0, 0.07),
        (ws.pitch_min_deg, -30.0, ws.pitch_max_deg),
    ):
        result = kin.inverse(cx + dx, cy + dy, cz + dz, pitch)
        total += 1
        values = result.as_dict()
        if not result.clamped and all(
            limits[name][0] <= values[name] <= limits[name][1] for name in JOINTS
        ):
            reachable += 1
    assert reachable / total >= 0.85


def test_chain_points_are_connected_and_match_forward(kin, cfg):
    points = kin.chain_points(-20.0, 30.0, 10.0)
    assert len(points) == 5
    assert points[0] == (cfg.geometry.pan_axis_x, 0.0)

    lengths = [math.dist(a, b) for a, b in zip(points[1:], points[2:])]
    assert lengths[0] == pytest.approx(cfg.geometry.upper_arm, abs=1e-9)
    assert lengths[1] == pytest.approx(cfg.geometry.forearm, abs=1e-9)

    # Ostatni punkt lancucha to ta sama koncowka, ktora zwraca FK przy pan=0.
    x, _, z = kin.forward(0.0, -20.0, 30.0, 10.0)
    assert points[-1][0] == pytest.approx(x, abs=1e-9)
    assert points[-1][1] == pytest.approx(z, abs=1e-9)


def test_tool_pitch_follows_wrist(kin):
    base = kin.tool_pitch(0.0, 0.0, 0.0)
    assert kin.tool_pitch(0.0, 0.0, 10.0) == pytest.approx(base - 10.0, abs=1e-9)


def test_elbow_branch_changes_solution(cfg):
    up = ArmKinematics(dataclasses.replace(cfg.geometry, elbow_up=True))
    down = ArmKinematics(dataclasses.replace(cfg.geometry, elbow_up=False))
    target = (0.30, 0.0, 0.15)
    assert up.inverse(*target, -30.0).elbow_flex != down.inverse(*target, -30.0).elbow_flex
    # Obie galezie musza trafiac w ten sam punkt.
    for solver in (up, down):
        result = solver.inverse(*target, -30.0)
        assert math.dist(solver.forward(**result.as_dict()), target) < 1e-9
