"""Kalibracja kamer z karty w chwytaku: detekcja, solver, karta, pelna sesja.

Solver sprawdzamy na obserwacjach syntetycznych (szybko, bez renderu), a cala
sesje - ramie, fala, render, detekcja, dopasowanie, bramki - w symulacji,
wzgledem prawdziwych poz kamer i karty, ktorych sesja nie widzi.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mujoco")

from lerobot_mp.twin.calib.card import Card, perturb, pinch_point  # noqa: E402
from lerobot_mp.twin.calib.handeye import (  # noqa: E402
    Observation,
    judge,
    pose_error,
    rotation_spread_R,
    solve,
)
from lerobot_mp.twin.calib.simulate import run, sample_cameras  # noqa: E402
from lerobot_mp.twin.calib.tags import corners_in_tag, detect, project, render  # noqa: E402
from lerobot_mp.twin.kinematics import RobotKinematics, inverse  # noqa: E402
from lerobot_mp.twin.robots import SO101  # noqa: E402


@pytest.fixture(scope="module")
def kin() -> RobotKinematics:
    return RobotKinematics(SO101)


def synthetic(kin, cameras, truth, card, rng, n_poses=48, noise=0.2, only_roll=False):
    """Obserwacje takie, jakie dalaby prawdziwa fala: rogi taga rzutowane przez K z szumem."""
    obs = []
    X = np.c_[corners_in_tag(card.tag_size), np.ones(4)]
    for _ in range(n_poses):
        joints = dict(SO101.home, gripper=0.0)
        if only_roll:
            joints["wrist_roll"] = float(rng.uniform(-150, 150))
        else:
            for name, (lo, hi) in SO101.wave_ranges.items():
                joints[name] = float(rng.uniform(lo, hi))
            joints["wrist_roll"] = float(rng.uniform(-150, 150))
        T_tcp = kin.tcp(joints)
        for view in cameras:
            for tid, T_tag in card.tag_poses().items():
                T = inverse(view.T_cam2base) @ T_tcp @ truth @ T_tag
                if T[2, 3] <= 0.05 or T[:3, 2] @ -T[:3, 3] <= 0.0:     # za kamera albo tylem
                    continue
                px = project((T @ X.T).T[:, :3], view.K)
                if not ((px >= 0) & (px < [view.width, view.height])).all():
                    continue
                obs.append(Observation(T_tcp, T_tag, px + rng.normal(0, noise, px.shape), tid,
                                       "card", view.name))
    return obs


# ----------------------------------------------------------------- tagi
def test_detected_corners_follow_the_pinhole_pixel_convention():
    """Krawedzie czarnego kwadratu (piksele 140..459) leza na 139,5 i 459,5.

    Doszlifowanie APRILTAG oddaje rogi o pol piksela dalej; bez poprawki solver
    kompensowal to obrotem kamery - i blad obrotu w symulacji byl dwa razy wiekszy.
    """
    img = np.full((600, 600), 255, np.uint8)
    img[100:500, 100:500] = render(0, 400)
    corners = detect(img)[0]
    assert np.abs(corners - [[139.5, 139.5], [459.5, 139.5], [459.5, 459.5], [139.5, 459.5]]).max() < 0.05


# ----------------------------------------------------------------- karta
def test_card_faces_are_back_to_back_with_opposite_normals():
    card = Card()
    poses = card.tag_poses()
    front, back = poses[card.ids[0]], poses[card.ids[1]]
    assert np.allclose(front[:2, 3], back[:2, 3])                      # ten sam srodek w plaszczyznie
    assert front[2, 3] - back[2, 3] == pytest.approx(card.thickness)
    assert front[:3, 2] @ back[:3, 2] == pytest.approx(-1.0)


def test_printable_sheet_carries_both_tags():
    card = Card()
    assert sorted(detect(card.sheet(dpi=150))) == sorted(card.ids)


def test_nominal_card_sticks_out_past_the_jaws(kin):
    """Srodek taga ma lezec przed czubkami szczek, na linii ich uchwytu."""
    pinch = pinch_point(kin)
    nominal = Card().nominal(pinch)
    assert nominal[0, 3] > pinch[0] + 0.02
    assert nominal[2, 3] == pytest.approx(pinch[2])


def test_misplacement_stays_within_its_limits():
    rng = np.random.default_rng(0)
    for _ in range(50):
        d = perturb(np.eye(4), rng)
        dt, dr = pose_error(np.eye(4), d)
        assert dt <= np.sqrt(3) * 0.012 + 1e-9
        assert dr <= np.sqrt(3) * np.radians(10.0) + 1e-9


# ----------------------------------------------------------------- solver
@pytest.mark.parametrize("n_cameras", [1, 2])
def test_solver_recovers_cameras_and_card_from_noisy_corners(kin, n_cameras):
    rng = np.random.default_rng(10 + n_cameras)
    card = Card()
    nominal = card.nominal(pinch_point(kin))
    truth = perturb(nominal, rng)
    cameras = sample_cameras(rng, n_cameras)
    obs = synthetic(kin, cameras, truth, card, rng)
    fit = judge(solve(obs, {v.name: (v.K, None) for v in cameras}, card.tag_size, {"card": nominal}))

    for view in cameras:
        dt, dr = pose_error(view.T_cam2base, fit.cameras[view.name].T_cam2base)
        assert dt < 1.0e-3, f"{view.name}: {dt * 1000:.2f} mm"
        assert np.degrees(dr) < 0.15, f"{view.name}: {np.degrees(dr):.3f} st"
        assert fit.cameras[view.name].trusted, fit.cameras[view.name].reason
    ct, cr = pose_error(truth, fit.mounts["card"])
    assert ct < 1.0e-3 and np.degrees(cr) < 0.2


def test_a_wave_that_only_rolls_the_wrist_is_refused(kin):
    """Obrot wokol jednej osi zostawia kamere wolna wzdluz tej osi - dopasowanie
    wychodzi pewne siebie i bledne. Bramka ma to odrzucic z podaniem powodu."""
    rng = np.random.default_rng(3)
    card = Card()
    nominal = card.nominal(pinch_point(kin))
    cameras = sample_cameras(rng, 1)
    obs = synthetic(kin, cameras, perturb(nominal, rng), card, rng, only_roll=True)
    assert rotation_spread_R([o.T_frame2base[:3, :3] for o in obs]) < 1.0
    fit = judge(solve(obs, {cameras[0].name: (cameras[0].K, None)}, card.tag_size, {"card": nominal}))
    cam = fit.cameras[cameras[0].name]
    assert not cam.trusted
    assert "rozrzut" in cam.reason


# ----------------------------------------------------------------- sesja
def test_full_session_in_simulation_calibrates_the_camera():
    """Cala sesja: fala, render, detekcja, dopasowanie i werdykt - wzgledem prawdy."""
    res = run(seed=0, n_cameras=1)
    cam = next(iter(res.trusted))
    assert res.trusted[cam], res.reasons[cam]
    assert res.err_mm[cam] < 1.0
    assert res.err_deg[cam] < 0.15
    assert res.poses < 40, "fala ma sie zmiescic w kilkudziesieciu pozach, nie w maksimum"
