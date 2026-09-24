"""Opis ramienia i kinematyka blizniaka.

Najwazniejszy test jest pierwszy: MJCF z MuJoCo Menagerie ma sie zgadzac
z niezaleznym modelem Articulusa co do polozenia kazdego czlonu w dowolnej
pozie. Articulus jest tym, na czym stoi podglad 3D aplikacji i przeliczenia
backendu `feetech`, wiec zgodnosc znaczy, ze katy z prawdziwego ramienia
wpisane w symulacje daja w niej te sama poze. Bez tego blizniak klamie.
"""

from __future__ import annotations

import numpy as np
import pytest

# Przed importem blizniaka: kinematics importuje mujoco, a bez niego (instalacja
# bez [twin]) caly plik ma sie pominac, a nie przerwac zbieranie testow.
mujoco = pytest.importorskip("mujoco")

from lerobot_mp.twin.kinematics import RobotKinematics, inverse  # noqa: E402
from lerobot_mp.twin.robots import SO101, get_spec  # noqa: E402


@pytest.fixture(scope="module")
def kin() -> RobotKinematics:
    return RobotKinematics(SO101)


def random_joints(kin: RobotKinematics, rng: np.random.Generator) -> dict[str, float]:
    joints = {
        name: float(rng.uniform(np.degrees(lo), np.degrees(hi)))
        for name, lo, hi in zip(SO101.joints, kin.lo, kin.hi)
    }
    joints["gripper"] = float(rng.uniform(0.0, 100.0))
    return joints


def test_mjcf_matches_the_articulus_reference_in_every_pose(kin):
    preview = pytest.importorskip("lerobot_mp.preview.model")
    art = preview.load_model()
    if art is None:
        pytest.skip("brak modelu podgladu 3D w repozytorium")

    base = art.link_names.index("base")
    links = ["shoulder", "upper_arm", "lower_arm", "wrist", "gripper"]
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(100):
        joints = random_joints(kin, rng)
        Ta = art.link_transforms(art.from_lerobot(joints))
        for name in links:
            ref = (inverse(Ta[base]) @ Ta[art.link_names.index(name)])[:3, 3]
            worst = max(worst, float(np.linalg.norm(ref - kin.body(name, joints)[:3, 3])))
    assert worst < 1e-5, f"czlony rozjezdzaja sie o {worst * 1000:.3f} mm"


def test_units_round_trip(kin):
    rng = np.random.default_rng(1)
    for _ in range(20):
        joints = random_joints(kin, rng)
        back = kin.from_q(kin.to_q(joints))
        for name in SO101.joints:
            assert back[name] == pytest.approx(joints[name], abs=1e-9)


def test_gripper_angle_is_the_servo_angle_of_the_feetech_backend(kin):
    """0..100 chwytaka ma byc w blizniaku TYM SAMYM katem szczeki, co na serwie.

    Wczesniej 0..100 szlo liniowo na caly zakres MJCF (-10..100 st.), a backend
    `feetech` rozciaga je na tiki 1986..2670 (60 st. skoku serwa) - jednostka
    chwytaka byla w symulacji 1,8 raza wieksza niz na ramieniu.
    """
    from lerobot_mp.config import load_config
    from lerobot_mp.robot.feetech import FeetechArm

    cfg = load_config(overrides={"robot": {"backend": "feetech", "port": "COM_TEST"}})
    arm = FeetechArm(cfg, bus=object())                   # tylko przeliczenia, bez portu
    rc = cfg.robot
    k = SO101.joints.index("gripper")
    for ticks in (rc.gripper_closed_ticks, 2100, 2300, rc.gripper_open_ticks):
        servo_deg = (ticks - rc.center_ticks) * 360.0 / 4096
        units = arm._to_units("gripper", ticks)
        assert np.degrees(kin.to_q({"gripper": units})[k]) == pytest.approx(servo_deg, abs=1e-6)
        assert kin.from_q(kin.to_q({"gripper": units}))["gripper"] == pytest.approx(units, abs=1e-6)
    # Poza zakresem stawu MJCF kat jest przyciety, a nie zawiniety.
    assert kin.to_q({"gripper": 1000.0})[k] == pytest.approx(kin.hi[k])
    assert kin.to_q({"gripper": -1000.0})[k] == pytest.approx(kin.lo[k])


def test_approach_axis_points_out_of_the_jaws(kin):
    """Krok wzdluz podejscia ma oddalac TCP od nadgarstka - inaczej karta
    kalibracyjna sterczalaby w strone ramienia."""
    joints = dict(SO101.home)
    tcp = kin.tcp(joints)
    wrist = kin.body("wrist", joints)[:3, 3]
    approach, _ = kin.tool_axes(joints)
    before = np.linalg.norm(tcp[:3, 3] - wrist)
    after = np.linalg.norm(tcp[:3, 3] + 0.05 * approach - wrist)
    assert after > before + 0.045


def test_closing_axis_separates_the_jaw_tips(kin):
    """Os zamykania to kierunek, w ktorym rozchodza sie czubki szczek."""
    joints = dict(SO101.home, gripper=50.0)
    _, closing = kin.tool_axes(joints)
    d = kin.data
    fixed = d.geom_xpos[kin.model.geom("fixed_jaw_sph_tip1").id]
    moving = d.geom_xpos[kin.model.geom("moving_jaw_sph_tip1").id]
    gap = (moving - fixed) / np.linalg.norm(moving - fixed)
    assert float(gap @ closing) > 0.9


def test_ik_reaches_poses_the_arm_can_take(kin):
    """Cel wziety z kinematyki prostej jest osiagalny z definicji - IK ma w niego trafic."""
    rng = np.random.default_rng(2)
    misses = []
    for _ in range(25):
        joints = random_joints(kin, rng)
        target = kin.tcp(joints)[:3, 3]
        sol = kin.ik(target, seed=SO101.home, rng=np.random.default_rng(3))
        reached = kin.tcp(sol.joints)[:3, 3]
        if not sol.ok:
            misses.append(sol.pos_err)
        else:
            assert np.linalg.norm(reached - target) < 1.5e-3
    assert len(misses) <= 2, f"IK nie trafil w {len(misses)}/25 osiagalnych celow: {misses}"


def test_ik_keeps_the_gripper_as_it_was(kin):
    target = kin.tcp(SO101.home)[:3, 3] + np.array([0.0, 0.03, 0.02])
    sol = kin.ik(target, seed=dict(SO101.home, gripper=80.0))
    assert sol.joints["gripper"] == pytest.approx(80.0, abs=1e-6)


def test_ik_treats_orientation_as_soft_for_a_five_joint_arm(kin):
    """Pelnej orientacji SO-101 nie osiagnie - pozycja ma byc trafiona mimo to."""
    home_tcp = kin.tcp(SO101.home)
    target = home_tcp[:3, 3] + np.array([0.02, 0.05, 0.0])
    # Obrot o 90 stopni wokol pionu: tego piec stawow w tej pozycji nie zrobi.
    turn = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]) @ home_tcp[:3, :3]
    sol = kin.ik(target, turn, seed=SO101.home)
    assert sol.ok
    assert np.linalg.norm(kin.tcp(sol.joints)[:3, 3] - target) < 1.5e-3


def test_ik_matches_orientation_when_the_arm_can_take_it(kin):
    """Pierwszenstwo pozycji nie moze znaczyc ignorowania obrotu: poza wzieta
    z kinematyki prostej jest osiagalna w calosci i ma byc trafiona w calosci."""
    rng = np.random.default_rng(4)
    rot_errors = []
    for _ in range(15):
        joints = random_joints(kin, rng)
        T = kin.tcp(joints)
        sol = kin.ik(T[:3, 3], T[:3, :3], seed=SO101.home, rng=np.random.default_rng(5))
        if sol.ok:
            rot_errors.append(sol.rot_err)
    assert len(rot_errors) >= 13
    assert float(np.median(rot_errors)) < 0.02


def test_unknown_robot_is_reported_with_the_known_ones():
    with pytest.raises(KeyError, match="so101"):
        get_spec("ur5")
