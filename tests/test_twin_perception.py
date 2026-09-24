"""Percepcja z kalibrowanych kamer: kostka na mapie stolu, sledzenie w dloni, przestawiona kamera."""

from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
cv2 = pytest.importorskip("cv2")

from lerobot_mp.twin import scene as sc  # noqa: E402
from lerobot_mp.twin.kinematics import pose  # noqa: E402
from lerobot_mp.twin.perception import CubeDetection, CubeDetector, CubeTracker, TableMapper  # noqa: E402
from lerobot_mp.twin.robots import SO101  # noqa: E402

K = np.array([[560.0, 0, 322.0], [0, 560.0, 236.0], [0, 0, 1]])
OUT_OF_VIEW = {"shoulder_pan": 110.0, "shoulder_lift": -95.0, "elbow_flex": 90.0}


def look_at(eye, target):
    eye = np.asarray(eye, float)
    z = np.asarray(target, float) - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    return pose(np.column_stack([x, np.cross(z, x), z]), eye)


VIEWS = [sc.CameraView("a", K, 640, 480, look_at([0.55, -0.45, 0.45], [0.2, 0, 0])),
         sc.CameraView("b", K, 640, 480, look_at([0.5, 0.5, 0.5], [0.2, 0, 0]))]


def render_cube(xy, yaw):
    cfg = sc.SceneConfig(SO101, cameras=VIEWS, objects=[sc.Box("cube", (0.015,) * 3, xy, rgba=(0.85, 0.25, 0.2, 1))])
    with sc.build(cfg) as s:
        s.set_joints(OUT_OF_VIEW)
        b = s.model.body("cube").id
        a = s.model.jnt_qposadr[s.model.body_jntadr[b]]
        s.data.qpos[a + 3:a + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        mujoco.mj_forward(s.model, s.data)
        return {v.name: s.render(v.name) for v in VIEWS}


def test_cube_is_found_from_two_cameras_to_a_few_millimetres():
    """Czesc wspolna masek z kamer: sama gorna sciana, bez rozmazanych bokow."""
    rng = np.random.default_rng(0)
    mapper = TableMapper({v.name: (v.K, None, v.T_cam2base) for v in VIEWS}, n=280)
    errs, yaws = [], []
    for _ in range(5):
        xy, yaw = (rng.uniform(0.14, 0.28), rng.uniform(-0.15, 0.15)), rng.uniform(-0.7, 0.7)
        det = CubeDetector().detect_frames(render_cube(xy, yaw), mapper)
        assert det is not None
        errs.append(np.linalg.norm(det.pos[:2] - xy))
        d = np.arctan2(det.rot[1, 0], det.rot[0, 0]) - yaw
        yaws.append(abs((d + np.pi / 4) % (np.pi / 2) - np.pi / 4))
    assert np.median(errs) < 0.003 and max(errs) < 0.006
    assert np.degrees(np.median(yaws)) < 3.0


def _det(p):
    return CubeDetection(np.asarray(p, float), np.eye(3), 100.0, 1.0)


def test_tracker_carries_the_cube_with_the_hand_when_cameras_lose_it():
    """Kamery widza kostke tylko z daleka - przy chwycie szczeki zaslaniaja gorna sciane."""
    tr = CubeTracker()
    closed = -0.17
    cube = np.array([0.205, 0.0, 0.015])
    # TCP jeszcze 5 cm nad kostka, szczeki otwarte: widac ja
    T_above = pose(np.eye(3), np.array([0.2, 0.0, 0.065]))
    got = tr.update(_det(cube), T_above, grip_q=0.8, grip_cmd=0.8, grip_closed=closed, now=0.0)
    assert np.allclose(got[0], cube)
    # TCP zjechal, szczeki zasloniely kostke i ZAMYKAJA SIE (jeszcze jada) - ostatnie widziane
    T_grasp = pose(np.eye(3), np.array([0.2, 0.0, 0.02]))
    for t, q in ((0.3, 0.5), (0.4, 0.2)):
        got = tr.update(None, T_grasp, grip_q=q, grip_cmd=closed, grip_closed=closed, now=t)
        assert tr.source == "ostatnie widziane" and np.allclose(got[0], cube)
    # szczeka stanela na kostce, choc rozkaz zamyka dalej - od teraz kostka jedzie z dlonia
    tr.update(None, T_grasp, grip_q=0.19, grip_cmd=closed, grip_closed=closed, now=0.5)
    assert tr.source == "w dloni"
    T_up = pose(np.eye(3), np.array([0.2, 0.0, 0.12]))
    got = tr.update(None, T_up, grip_q=0.19, grip_cmd=closed, grip_closed=closed, now=0.9)
    assert tr.source == "w dloni"
    assert got[0] == pytest.approx(cube + [0.0, 0.0, 0.10], abs=1e-9)


def test_tracker_forgets_a_cube_nobody_holds():
    tr = CubeTracker(hold_s=1.0)
    T_far = pose(np.eye(3), np.array([0.1, 0.2, 0.2]))
    tr.update(_det([0.25, 0.0, 0.015]), T_far, 1.0, 1.0, -0.17, now=0.0)
    assert tr.update(None, T_far, 1.0, 1.0, -0.17, now=0.5) is not None      # chwilowo zaslonieta
    assert tr.update(None, T_far, 1.0, 1.0, -0.17, now=1.5) is None         # nie ma jej juz za dlugo


def test_moved_camera_is_detected_and_moving_arm_is_not():
    from lerobot_mp.twin.ui.watch import CameraWatch, arm_mask

    T = look_at([0.6, -0.45, 0.4], [0.15, 0, 0.05])
    cfg = sc.SceneConfig(SO101, cameras=[sc.CameraView("c", K, 640, 480, T)],
                         objects=[sc.Box("k", (0.02,) * 3, (0.25, 0.1))])
    with sc.build(cfg) as s:
        s.set_joints(SO101.home)
        w = CameraWatch()
        w.remember("c", s.render("c"), arm_mask(s, "c"))
        s.set_joints(dict(SO101.home, shoulder_pan=40.0, elbow_flex=10.0))
        assert w.check("c", s.render("c"), arm_mask(s, "c")) < 1.5
        assert not w.moved("c")
        cam = s.model.camera("c").id
        a = np.radians(1.0)
        R = T[:3, :3] @ np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
        Tw = s.T_base2world @ pose(R, T[:3, 3])
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, (Tw[:3, :3] @ sc.CV_TO_MJ).ravel())
        s.model.cam_quat[cam] = q
        mujoco.mj_forward(s.model, s.data)
        shift = w.check("c", s.render("c"), arm_mask(s, "c"))
    assert shift == pytest.approx(560 * a, rel=0.1)
    assert w.moved("c")
