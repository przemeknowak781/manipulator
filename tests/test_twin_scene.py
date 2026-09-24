"""Scena blizniaka: kamery w scenie maja widziec dokladnie to, co przewiduje K.

To jest fundament sim-2-real w tym projekcie: kamera skalibrowana na prawdziwym
stanowisku trafia do symulacji ze swoja poza i macierza K, a polityka widzi
w symulacji ten sam kadr, ktory zobaczy na biurku. Jesli render rozjezdza sie
z rzutem przez K, to caly most klamie po cichu - dlatego mierzymy go na kulkach
w znanych punktach, dla roznych K, co do ulamka piksela.
"""

from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
cv2 = pytest.importorskip("cv2")

from lerobot_mp.twin import scene as sc  # noqa: E402
from lerobot_mp.twin.kinematics import inverse, pose  # noqa: E402
from lerobot_mp.twin.robots import SO101  # noqa: E402

W, H = 640, 480
POINTS = [(0.15, 0.05), (0.25, -0.05), (0.08, 0.14), (0.30, 0.10), (0.20, -0.12)]
OUT_OF_VIEW = {"shoulder_pan": 100.0, "shoulder_lift": -95.0, "elbow_flex": 90.0}


def look_at(eye, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Poza kamery OpenCV (z do przodu, y w dol) patrzacej z `eye` na `target`."""
    z = np.asarray(target, float) - np.asarray(eye, float)
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    return pose(np.column_stack([x, np.cross(z, x), z]), np.asarray(eye, float))


#: Identyfikatory kulek w renderze segmentacji - poza zakresem geomow sceny.
PROBE_ID = 1000


def sphere_centres(scene: sc.Scene, camera: str, points) -> list[np.ndarray]:
    """Renderuje male kulki w punktach stolu i zwraca srodki ich MASEK segmentacji.

    Maska, a nie plama koloru: dolna polowa kulki jest w cieniu i odpada na
    progu "czerwony", co podnosilo srodek plamy o ~0,4 px - i przykrylo
    polpikselowy blad punktu glownego w pionie. Maska to czysta geometria.
    """
    view = scene.camera(camera)
    r = mujoco.Renderer(scene.model, height=view.height, width=view.width)
    try:
        r.enable_segmentation_rendering()
        r.update_scene(scene.data, camera=camera)
        for k, (x, y) in enumerate(points):
            p = scene.T_base2world @ np.array([x, y, 0.004, 1.0])
            g = r.scene.geoms[r.scene.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.004, 0.0, 0.0]), p[:3],
                                np.eye(3).ravel(), np.array([1.0, 0.0, 0.0, 1.0], np.float32))
            # `mjv_initGeom` zostawia segid 0 - kulka dostalaby kolor pierwszego geomu sceny.
            g.segid = r.scene.ngeom
            g.objid = PROBE_ID + k
            r.scene.ngeom += 1
        seg = r.render()[..., 0]
    finally:
        r.close()
    centres = []
    for k in range(len(points)):
        v, u = np.nonzero(seg == PROBE_ID + k)
        if len(u) > 3:
            centres.append(np.array([u.mean(), v.mean()]))
    return centres


@pytest.mark.parametrize(
    "K",
    [
        np.array([[600.0, 0, 319.5], [0, 600.0, 239.5], [0, 0, 1]]),     # punkt glowny w srodku
        np.array([[610.0, 0, 350.0], [0, 575.0, 215.0], [0, 0, 1]]),     # fx != fy, przesuniety
        np.array([[520.0, 0, 280.0], [0, 540.0, 270.0], [0, 0, 1]]),     # przesuniety w druga strone
    ],
    ids=["srodek", "przesuniety", "odwrotnie"],
)
def test_sim_camera_renders_where_its_intrinsics_project(K):
    T = look_at([0.55, -0.40, 0.30], [0.18, 0.02, 0.02])
    with sc.build(sc.SceneConfig(SO101, cameras=[sc.CameraView("probe", K, W, H, T)])) as scene:
        scene.set_joints(OUT_OF_VIEW)
        blobs = sphere_centres(scene, "probe", POINTS)
    assert len(blobs) == len(POINTS)
    errs = []
    for x, y in POINTS:
        pc = inverse(T) @ np.array([x, y, 0.004, 1.0])
        uv = (K @ pc[:3])[:2] / pc[2]
        errs.append(min(blobs, key=lambda b: np.linalg.norm(b - uv)) - uv)
    errs = np.array(errs)
    # Srodek maski kuli to rzut jej srodka tylko w przyblizeniu; systematyczne
    # przesuniecie (srednia) ma byc zerowe, rozrzut - pojedyncze dziesiate piksela.
    assert np.abs(errs.mean(axis=0)).max() < 0.15, f"systematyczne przesuniecie {errs.mean(axis=0)} px"
    assert np.linalg.norm(errs, axis=1).max() < 0.4


def test_camera_lands_in_the_scene_where_the_config_put_it():
    T = look_at([0.5, 0.3, 0.4], [0.15, 0.0, 0.0])
    K = np.array([[600.0, 0, 319.5], [0, 600.0, 239.5], [0, 0, 1]])
    table = sc.Table(base_xy=(0.1, -0.2), base_yaw=0.6)                   # ramie nie w poczatku swiata
    with sc.build(sc.SceneConfig(SO101, table=table, cameras=[sc.CameraView("c", K, W, H, T)])) as scene:
        got = scene.camera_pose("c")
    assert np.linalg.norm(got[:3, 3] - T[:3, 3]) < 1e-9
    assert np.abs(got[:3, :3] - T[:3, :3]).max() < 1e-9


def test_scene_takes_the_physics_options_the_robot_was_tuned_with():
    """Menagerie stroi SO-101 pod manipulacje - scena nie moze tego zgubic."""
    robot = mujoco.MjSpec.from_file(str(SO101.mjcf_path))
    with sc.build(sc.SceneConfig(SO101)) as scene:
        opt = scene.model.opt
        assert opt.cone == robot.option.cone
        assert opt.impratio == pytest.approx(robot.option.impratio)
        assert opt.timestep == pytest.approx(robot.option.timestep)


def test_servos_hold_and_reach_commanded_joints():
    with sc.build(sc.SceneConfig(SO101)) as scene:
        scene.set_joints(SO101.home)
        target = dict(SO101.home, shoulder_pan=25.0, elbow_flex=30.0)
        scene.command(target)
        scene.step(2.0)
        got = scene.joints()
    for name in ("shoulder_pan", "elbow_flex"):
        assert got[name] == pytest.approx(target[name], abs=2.0)
