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


def render_cube(xy, yaw, lift=0.0):
    cfg = sc.SceneConfig(SO101, cameras=VIEWS, objects=[sc.Box("cube", (0.015,) * 3, xy, rgba=(0.85, 0.25, 0.2, 1))])
    with sc.build(cfg) as s:
        s.set_joints(OUT_OF_VIEW)
        b = s.model.body("cube").id
        a = s.model.jnt_qposadr[s.model.body_jntadr[b]]
        s.data.qpos[a + 2] += lift                   # kostka "w szczekach" nad blatem
        s.data.qpos[a + 3:a + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        mujoco.mj_forward(s.model, s.data)
        return {v.name: s.render(v.name) for v in VIEWS}


@pytest.mark.render
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


def _det(p, n_cameras=2):
    return CubeDetection(np.asarray(p, float), np.eye(3), 100.0, 1.0, n_cameras)


@pytest.mark.render
def test_detection_says_how_many_cameras_saw_the_cube():
    """Podniesiona kostka jest "lezaca dalej" dla jednej kamery - bramka IoU tego nie widzi."""
    xy = (0.22, 0.03)
    both = TableMapper({v.name: (v.K, None, v.T_cam2base) for v in VIEWS}, n=280)
    det = CubeDetector().detect_frames(render_cube(xy, 0.3), both, t=12.5)
    assert det.n_cameras == 2 and det.t == 12.5
    # Dwie kamery: kostka 2 cm nad blatem sie nie zgadza - odrzucona.
    lifted = render_cube(xy, 0.3, lift=0.02)
    assert CubeDetector().detect_frames(lifted, both) is None
    # Jedna kamera: ta sama kostka przechodzi bramke kilka cm od prawdy - ale wie, ze jest z jednej.
    only_a = TableMapper({v.name: (v.K, None, v.T_cam2base) for v in VIEWS if v.name == "a"}, n=280)
    det = CubeDetector().detect_frames({"a": lifted["a"]}, only_a)
    assert det is not None and det.n_cameras == 1
    assert np.linalg.norm(det.pos[:2] - xy) > 0.015
    # Druga kamera jest, ale ramie zaslania jej kostke: tez jeden swiadek.
    m = CubeDetector().mask(lifted["b"]) > 0
    ys, xs = np.nonzero(m)
    occ = np.zeros(m.shape, bool)
    occ[max(0, ys.min() - 30):ys.max() + 30, max(0, xs.min() - 30):xs.max() + 30] = True
    det = CubeDetector().detect_frames(lifted, both, {"b": occ})
    assert det is not None and det.n_cameras == 1


def test_tracker_ignores_a_single_camera_cube_at_the_hand():
    """Kostka w szczekach, jedna kamera "widzi" ja na blacie 5 cm dalej - polityka dostaje dlon."""
    tr = CubeTracker()
    closed = -0.17
    cube = np.array([0.22, 0.03, 0.015])
    T_far = pose(np.eye(3), np.array([0.1, -0.15, 0.2]))
    # Z daleka jedna kamera wystarcza: nic poza dlonia nie podnosi kostki.
    got = tr.update(_det(cube, 1), T_far, grip_q=0.8, grip_cmd=0.8, grip_closed=closed, now=0.0)
    assert tr.source == "kamery" and np.allclose(got[0], cube)
    T_grasp = pose(np.eye(3), cube + [0.0, 0.0, 0.005])
    ghost = cube + np.array([-0.034, 0.048, 0.0])            # zmierzone: podniesiona o 4 cm, kamera a
    # Szczeka zamyka sie na kostce (jeszcze jedzie), potem stoi na niej.
    for t, q in ((0.3, 0.5), (0.4, 0.19), (0.5, 0.19)):
        got = tr.update(_det(ghost, 1), T_grasp, grip_q=q, grip_cmd=closed, grip_closed=closed, now=t)
        assert tr.source != "kamery" and np.allclose(got[0], cube)
    assert tr.source == "w dloni"
    T_up = pose(np.eye(3), cube + [0.0, 0.0, 0.045])
    got = tr.update(_det(ghost, 1), T_up, grip_q=0.19, grip_cmd=closed, grip_closed=closed, now=0.6)
    assert tr.source == "w dloni" and got[0] == pytest.approx(cube + [0.0, 0.0, 0.04], abs=1e-9)
    # Dwie kamery przy dloni dalej sa prawda (bramka IoU odrzuca podniesiona kostke sama).
    tr2 = CubeTracker()
    tr2.update(_det(cube, 1), T_far, 0.8, 0.8, closed, now=0.0)
    moved = cube + [0.01, 0.0, 0.0]
    got = tr2.update(_det(moved, 2), T_grasp, 0.8, 0.8, closed, now=0.1)
    assert tr2.source == "kamery" and np.allclose(got[0], moved)


def test_single_camera_confirmation_keeps_a_resting_cube_alive():
    """Dlon dlugo krazy nad kostka widziana jedna kamera - kostka nie znika po `hold_s`."""
    tr = CubeTracker(hold_s=1.0)
    cube = np.array([0.22, 0.03, 0.015])
    tr.update(_det(cube, 1), pose(np.eye(3), np.array([0.1, -0.15, 0.2])), 0.8, 0.8, -0.17, now=0.0)
    T_above = pose(np.eye(3), cube + [0.0, 0.0, 0.05])
    for t in np.arange(0.5, 3.0, 0.5):
        got = tr.update(_det(cube + [0.003, 0.0, 0.0], 1), T_above, 0.8, 0.8, -0.17, now=float(t))
        # Szczeka otwarta: po `settle_n` zgodnych detekcjach kostka moze tez przejsc na polozenie
        # z kamery (3 mm dalej) - byle nie znikala.
        assert got is not None and np.linalg.norm(got[0] - cube) < 0.004
    # Szczeka przymknieta (moglaby trzymac kostke): tylko potwierdzenie starego polozenia.
    tr = CubeTracker(hold_s=1.0)
    tr.update(_det(cube, 1), pose(np.eye(3), np.array([0.1, -0.15, 0.2])), 0.3, 0.3, -0.17, now=0.0)
    for t in np.arange(0.5, 3.0, 0.5):
        got = tr.update(_det(cube + [0.003, 0.0, 0.0], 1), T_above, 0.3, 0.3, -0.17, now=float(t))
        assert got is not None and np.allclose(got[0], cube)


CLOSED = -0.17
T_FAR = pose(np.eye(3), np.array([0.1, -0.15, 0.2]))


def _holding(tr, cube):
    """Tracker z kostka w dloni: widziana z daleka, potem szczeka zablokowana na niej."""
    tr.update(_det(cube, 2), T_FAR, 0.8, 0.8, CLOSED, now=0.0)
    T_grasp = pose(np.eye(3), cube + [0.0, 0.0, 0.005])
    for t, q in ((0.3, 0.5), (0.4, 0.19), (0.5, 0.19)):
        tr.update(None, T_grasp, grip_q=q, grip_cmd=CLOSED, grip_closed=CLOSED, now=t)
    assert tr.source == "w dloni"


@pytest.mark.parametrize("n_cameras", [0, -1])
def test_detection_without_two_witnesses_never_replaces_the_cube_in_hand(n_cameras):
    """Dopasowanie sylwetki daje n_cameras=0, gdy ramie zaslania kostke w szczekach KAZDEJ kamerze.

    Wczesniej 0 znaczylo "nie wiadomo" i przechodzilo jak dwie kamery: duch kostki
    7 cm obok i 6 cm nizej zostawal "kamery", a kostka wypadala z dloni trackera.
    """
    tr = CubeTracker()
    cube = np.array([0.22, 0.03, 0.015])
    _holding(tr, cube)
    ghost = cube + [-0.05, 0.07, 0.0]
    # Szczeka drgnela (nie stoi), ale dalej sciska - duch nie moze wejsc w `last`
    # ani zdjac kostki z dloni.
    T = pose(np.eye(3), cube + [0.0, 0.0, 0.005])
    got = tr.update(_det(ghost, n_cameras), T, grip_q=0.25, grip_cmd=CLOSED, grip_closed=CLOSED, now=0.6)
    assert tr.source != "kamery" and np.allclose(got[0], cube)
    for k, t in enumerate((0.7, 0.8, 0.9)):
        T = pose(np.eye(3), cube + [0.0, 0.0, 0.005 + 0.02 * k])
        got = tr.update(_det(ghost, n_cameras), T, grip_q=0.25, grip_cmd=CLOSED, grip_closed=CLOSED, now=t)
        assert tr.source == "w dloni" and got[0] == pytest.approx(cube + [0.0, 0.0, 0.02 * k], abs=1e-9)


@pytest.mark.render
def test_map_detection_counts_as_one_witness():
    """`detect` z mapy nie wie, ile kamer widzialo kostke - tracker ma to traktowac jak jedna."""
    xy = (0.22, 0.03)
    mapper = TableMapper({v.name: (v.K, None, v.T_cam2base) for v in VIEWS}, n=280).at_height(0.03)
    det = CubeDetector().detect(mapper.fuse(render_cube(xy, 0.3))[0], mapper)
    assert det is not None and det.n_cameras == -1
    tr = CubeTracker()
    tr.update(_det([0.22, 0.03, 0.015], 2), T_FAR, 0.8, 0.8, CLOSED, now=0.0)
    T_near = pose(np.eye(3), np.array([0.22, 0.03, 0.06]))
    det.pos = np.array([0.25, 0.03, 0.015])
    tr.update(det, T_near, 0.8, 0.8, CLOSED, now=0.1)
    assert tr.source.startswith("ostatnie widziane")


def test_pushed_cube_is_reacquired_when_it_stays_put_while_the_hand_moves():
    """Kostka potracona 3 cm, jedna kamera, dlon przy niej: wczesniej stala w starym miejscu 6 s,
    potem None."""
    tr = CubeTracker()
    cube = np.array([0.22, 0.03, 0.015])
    tr.update(_det(cube, 2), T_FAR, 0.8, 0.8, CLOSED, now=0.0)
    pushed = cube + [0.03, 0.0, 0.0]
    # Dlon stoi 5 cm nad starym miejscem, szczeka przymknieta (0,3 rad, szczelina 38 mm - w takiej
    # moglaby byc kostka): nic nie dowodzi, ze to nie duch - zostaje stare polozenie, ale zrodlo
    # mowi wprost, ze kamera widzi kostke gdzie indziej.
    T_hover = pose(np.eye(3), cube + [0.0, 0.0, 0.05])
    for t in (0.1, 0.2, 0.3, 0.4):
        jitter = [0.002 * (-1) ** int(10 * t), 0.0, 0.0]
        got = tr.update(_det(pushed + jitter, 1), T_hover, 0.3, 0.3, CLOSED, now=t)
        assert np.allclose(got[0], cube)
    assert "gdzie indziej" in tr.source
    # Dlon odjezdza w poziomie, kostka stoi (szum 2-4 mm) - przyjeta z kamer.
    for k, t in enumerate((0.5, 0.6, 0.7)):
        T = pose(np.eye(3), cube + [-0.008 * k, -0.006 * k, 0.05])
        got = tr.update(_det(pushed + [0.0, 0.003 * (-1) ** k, 0.0], 1), T, 0.3, 0.3, CLOSED, now=t)
    assert tr.source == "kamery" and np.linalg.norm(got[0] - pushed) < 0.004


OPEN = 0.8        # szczeka otwarta (szczelina 77 mm); kostka 30 mm trzymana to 0,19-0,20 rad


def test_pushed_cube_is_reacquired_under_a_still_hand_with_an_open_jaw():
    """Kostka potracona, dlon zawisla nad nia z otwarta szczeka (panel, jedna kamera): 1 z 8 prob
    wisiala 6,9 s na "ostatnie widziane (1 kamera widzi ja gdzie indziej)", potem "kamery jej nie widza".
    Duch podniesionej kostki potrzebuje kostki w szczekach - otwarta szeroko jej nie trzyma."""
    tr = CubeTracker()
    cube = np.array([0.22, 0.03, 0.015])
    tr.update(_det(cube, 2), T_FAR, OPEN, OPEN, CLOSED, now=0.0)
    pushed = cube + [0.03, 0.0, 0.0]
    T_hover = pose(np.eye(3), cube + [0.0, 0.0, 0.05])
    for t in (0.1, 0.2, 0.3):
        got = tr.update(_det(pushed + [0.002 * (-1) ** int(10 * t), 0.0, 0.0], 1), T_hover, OPEN, OPEN,
                        CLOSED, now=t)
    assert tr.source == "kamery" and np.linalg.norm(got[0] - pushed) < 0.003


def test_restart_next_to_the_cube_with_one_camera_takes_the_cube():
    """STOP 6,6 cm od kostki, ponowne Uruchom: swiezy tracker, runner nie rusza ramieniem, dopoki
    nie ma kostki. Wczesniej pomijal detekcje z pewnoscia 0,99 i po 3 s "kamery jej nie widza"."""
    tr = CubeTracker()
    cube = np.array([0.20, 0.03, 0.015])
    T_still = pose(np.eye(3), cube + [0.03, -0.02, 0.055])
    got = None
    for k, t in enumerate((0.1, 0.25, 0.4)):
        got = tr.update(_det(cube + [0.0, 0.002 * (-1) ** k, 0.0], 1), T_still, OPEN, OPEN, CLOSED, now=t)
    assert tr.source == "kamery" and np.linalg.norm(got[0] - cube) < 0.003


def test_a_still_hand_with_a_closed_jaw_keeps_the_ghost_out_and_says_why():
    """Szczeka na wysokosci kostki (albo zacisnieta), dlon stoi: jedna kamera przy dloni dalej nie jest
    przyjmowana - to moze byc duch podniesionej kostki. Tak jest przy restarcie z kostka w szczekach:
    rozkaz chwytaka = zmierzony kat, wiec tracker nie widzi "sciskania". Zamiast "brak" (runner:
    "kamery jej nie widza") zrodlo mowi, co widzi i co zrobic."""
    held = 0.19                                         # zmierzony kat szczeki na kostce 30 mm
    cube = np.array([0.20, 0.03, 0.015])
    T_up = pose(np.eye(3), cube + [0.0, 0.0, 0.045])
    ghost = cube + [-0.034, 0.048, 0.0]               # zmierzone: podniesiona o 4 cm, kamera a
    tr = CubeTracker()
    for k in range(12):
        got = tr.update(_det(ghost + [0.001 * (-1) ** k, 0.0, 0.0], 1), T_up, held, held, CLOSED,
                        now=0.1 * (k + 1))
        assert got is None and tr.source.startswith("brak (")
    assert "Dom" in tr.source and "druga kamere" in tr.source
    # Kilka sekund bez zadnej detekcji: powod wygasa - teraz naprawde nikt jej nie widzi.
    assert tr.update(None, T_up, held, held, CLOSED, now=5.0) is None and tr.source == "brak"
    # Szczeka zacisnieta na kostce (rozkaz ciasniej) i dlon stoi: duch nigdy nie wchodzi.
    tr = CubeTracker()
    _holding(tr, cube)
    T_up = pose(np.eye(3), cube + [0.0, 0.0, 0.085])
    for k in range(12):
        got = tr.update(_det(ghost, 1), T_up, held, CLOSED, CLOSED, now=0.6 + 0.1 * k)
        assert tr.source == "w dloni" and got[0][2] == pytest.approx(0.095)


def test_ghost_moving_with_the_hand_is_never_accepted():
    """Duch kostki w szczekach jedzie z dlonia - nawet gdy tracker nie wie, ze szczeka cos trzyma."""
    tr = CubeTracker()
    cube = np.array([0.22, 0.03, 0.015])
    tr.update(_det(cube, 2), T_FAR, 0.8, 0.8, CLOSED, now=0.0)
    for k in range(10):
        move = np.array([0.004 * k, -0.003 * k, 0.0])
        tcp = cube + [0.0, 0.0, 0.04] + move
        # Rzut kostki z wysokosci dloni wzdluz promienia kamery: w poziomie jedzie SZYBCIEJ niz dlon.
        ghost = cube + [-0.04, 0.05, 0.0] + 1.2 * move
        got = tr.update(_det(ghost, 1), pose(np.eye(3), tcp), 0.8, 0.8, CLOSED, now=0.1 * (k + 1))
        assert tr.source != "kamery" and np.allclose(got[0], cube)


def test_cube_that_falls_from_the_hand_does_not_hang_in_the_air():
    """Szczeka zamknela sie na pustym 8 cm nad blatem: wczesniej "ostatnie widziane" w powietrzu,
    a runner liczyl kostke za podniesiona i konczyl "zadanie wykonane" z kostka na blacie."""
    tr = CubeTracker()
    cube = np.array([0.22, 0.03, 0.015])
    _holding(tr, cube)
    T_up = pose(np.eye(3), cube + [0.0, 0.0, 0.085])
    got = tr.update(None, T_up, grip_q=0.19, grip_cmd=CLOSED, grip_closed=CLOSED, now=0.6)
    assert tr.source == "w dloni" and got[0][2] == pytest.approx(0.095)
    # Drgniecie przy sciskaniu (jeszcze trzyma) - bez upadku, kostka dalej w dloni.
    got = tr.update(None, T_up, grip_q=0.13, grip_cmd=CLOSED, grip_closed=CLOSED, now=0.7)
    assert got[0][2] == pytest.approx(0.095)
    # ...ale w nastepnym takcie szczeka jest juz zamknieta na pustym: kostka lezy pod dlonia.
    got = tr.update(None, T_up, grip_q=-0.15, grip_cmd=CLOSED, grip_closed=CLOSED, now=0.8)
    assert tr.source == "ostatnie widziane (upuszczona)"
    assert got[0] == pytest.approx([0.22, 0.03, 0.015], abs=1e-9)
    # Drgniecie, po ktorym szczeka znow stoi na kostce - dalej w dloni, nie upadek.
    tr2 = CubeTracker()
    _holding(tr2, cube)
    tr2.update(None, T_up, 0.19, CLOSED, CLOSED, now=0.6)
    tr2.update(None, T_up, 0.13, CLOSED, CLOSED, now=0.7)
    got = tr2.update(None, T_up, 0.13, CLOSED, CLOSED, now=0.8)
    assert tr2.source == "w dloni" and got[0][2] == pytest.approx(0.095)


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
    tr.update(_det([0.25, 0.0, 0.015], 1), T_far, 1.0, 1.0, -0.17, now=0.0)
    assert tr.update(None, T_far, 1.0, 1.0, -0.17, now=0.5) is not None      # chwilowo zaslonieta
    assert tr.update(None, T_far, 1.0, 1.0, -0.17, now=1.5) is None         # nie ma jej juz za dlugo


@pytest.mark.render
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
        assert not w.moved("c")                 # "przestawiona" dopiero po drugim zgodnym sprawdzeniu
        w.check("c", s.render("c"), arm_mask(s, "c"))
    # Miara to najwieksze przesuniecie naroznika kadru: przy pochyleniu naroznik idzie troche
    # dalej niz srodek (10,7 px przy 560*a = 9,8 px), stad 15%, a nie 10%.
    assert shift == pytest.approx(560 * a, rel=0.15)
    assert w.moved("c")


@pytest.mark.render
def test_lift_from_one_camera_never_feeds_a_lifted_ghost_and_recovers_a_dropped_cube():
    """Caly lancuch lift-v3 z TYLKO kamera `a` (jak `test_twin_lift_from_cameras`, ale jeden swiadek).

    Seed 5: pierwszy chwyt wypuszcza kostke 3-5 cm dalej. Przed poprawka: duch z
    n_cameras=0 szedl do polityki jako "kamery" przy kostce 1,6 cm nad blatem, a potem
    lezaca obok kostka (widziana dobrze, ale przez jedna kamere przy dloni) byla
    pomijana - 130 taktow chwytania pustego miejsca i stop "kamery jej nie widza".
    """
    pytest.importorskip("torch")
    from lerobot_mp.twin.kinematics import RobotKinematics, inverse
    from lerobot_mp.twin.rl import task as tk
    from lerobot_mp.twin.rl.policy import Policy, bundled_dir
    from lerobot_mp.twin.rl.runner import PolicyRunner
    from lerobot_mp.twin.runtime import Twin
    from lerobot_mp.twin.ui.watch import arm_mask
    from lerobot_mp.twin.workspace import CameraRecord, Workspace

    path = bundled_dir() / "lift-v3" / "policy.pt"
    if not path.is_file():
        pytest.skip("brak bazowej polityki lift-v3 w assets/policies")
    ws = Workspace()
    T = look_at([0.55, -0.45, 0.45], [0.2, 0, 0]).tolist()
    ws.add_camera(CameraRecord("a", "sim", 640, 480, K=K.tolist(), sim_pose=T, T_cam2base=T,
                               calibration={"trusted": True}))
    pol = Policy.load(path)
    h = pol.task.cube_half
    twin = Twin(ws)
    try:
        twin.configure(objects=[sc.Box("cube", (h, h, h), (0.2, 0.0), rgba=(0.85, 0.25, 0.2, 1.0),
                                       mass=pol.task.cube_mass)], grasp_sensors=["cube"])
        twin.connect("sim", threaded=False)
        mapper = TableMapper.from_workspace(ws)
        kin = RobotKinematics(ws.spec())
        pos, quat = tk.sample_cubes(pol.task, np.random.default_rng(5), 1)
        with twin.lock:
            s = twin.scene
            b = s.model.body("cube").id
            a = s.model.jnt_qposadr[s.model.body_jntadr[b]]
            qb, qw = np.zeros(4), np.zeros(4)
            mujoco.mju_mat2Quat(qb, s.T_base2world[:3, :3].ravel())
            mujoco.mju_mulQuat(qw, qb, quat[0])
            s.data.qpos[a:a + 3] = s.T_base2world[:3, :3] @ pos[0] + s.T_base2world[:3, 3]
            s.data.qpos[a + 3:a + 7] = qw
            mujoco.mj_forward(s.model, s.data)

        def truth():
            with twin.lock:
                s = twin.scene
                Ti = inverse(s.T_base2world)
                return Ti[:3, :3] @ s.data.xpos[s.model.body("cube").id] + Ti[:3, 3]

        tracker, det = CubeTracker(), CubeDetector()
        last, clock, log = {"det": None}, {"t": 0.0}, []

        def provider():
            joints = dict(twin.status.measured) or twin.joints()
            q = kin.to_q(joints)
            d, last["det"] = last["det"], None                # tylko swieze detekcje, jak w panelu
            out = tracker.update(d, kin.tcp(joints), q[5], runner.q_cmd[5], runner.limits.lo[5], clock["t"])
            log.append((truth()[2] - h, tracker.source))
            return out

        runner = PolicyRunner(twin, pol, cube_provider=provider)
        runner.start(threaded=False)
        for k in range(1, int(20.0 / 0.01) + 1):
            clock["t"] = k * 0.01
            if k % 2 == 0:
                twin.step(0.02)
            if k % 10 == 0:
                occ = twin.render_with(lambda sc_: {"a": arm_mask(sc_, "a", dilate=5)})
                last["det"] = det.detect_frames({"a": twin.render("a")}, mapper, occ, t=clock["t"])
            if k % 5 == 0 and not runner.step_once(clock["t"]):
                break
        lifted_from_cameras = [lift for lift, src in log if lift > 0.01 and src == "kamery"]
        assert not lifted_from_cameras, f"podniesiona kostka jako 'kamery' {len(lifted_from_cameras)} razy"
        assert runner.status.stopped_because == "zadanie wykonane", runner.status.stopped_because
        assert truth()[2] - h > pol.task.lift_height
    finally:
        twin.close()


# Kamery jak w panelu (`TwinApp._add_sim_camera`: pole widzenia 62 st., K lekko poza srodkiem)
# i w e2e_panel - dwie kamery z przodu po obu stronach ramienia.
_F = 240.0 / np.tan(np.radians(31.0))
K_PANEL = np.array([[_F, 0, 319.5 + 3.0], [0, _F * 0.995, 239.5 - 2.0], [0, 0, 1]])
PANEL_VIEWS = [sc.CameraView("a", K_PANEL, 640, 480, look_at([0.55, -0.45, 0.45], [0.2, 0, 0])),
               sc.CameraView("b", K_PANEL, 640, 480, look_at([0.5, 0.5, 0.5], [0.2, 0, 0]))]


def _sweep_at_rest(spots, lift=0.0):
    """Kostka w `spots` [(x, y, obrot)], ramie w pozie domowej (jak po polaczeniu blizniaka)."""
    from lerobot_mp.twin.ui.watch import arm_mask

    cfg = sc.SceneConfig(SO101, cameras=PANEL_VIEWS,
                         objects=[sc.Box("cube", (0.015,) * 3, (0.2, 0.0), rgba=(0.85, 0.25, 0.2, 1))])
    mapper = TableMapper({v.name: (v.K, None, v.T_cam2base) for v in PANEL_VIEWS})
    out = []
    with sc.build(cfg) as s:
        s.set_joints(SO101.home)
        b = s.model.body("cube").id
        a = s.model.jnt_qposadr[s.model.body_jntadr[b]]
        occ = {v.name: arm_mask(s, v.name, dilate=5) for v in PANEL_VIEWS}      # jak w panelu
        T = s.T_base2world
        for x, y, yaw in spots:
            s.data.qpos[a:a + 3] = T[:3, :3] @ np.array([x, y, 0.015 + lift]) + T[:3, 3]
            qz, qw = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]), np.zeros(4)
            mujoco.mju_mat2Quat(qw, T[:3, :3].ravel())
            mujoco.mju_mulQuat(qw, qw.copy(), qz)
            s.data.qpos[a + 3:a + 7] = qw
            mujoco.mj_forward(s.model, s.data)
            frames = {v.name: s.render(v.name) for v in PANEL_VIEWS}
            out.append(CubeDetector().detect_frames(frames, mapper, occ))
    return out


def _edges_of_lift_region():
    """Brzegi obszaru treningu `lift` (promien 0,15-0,25 m, kierunek +-0,8 rad) i srodek promienia."""
    from lerobot_mp.twin.rl import task as tk

    task = tk.make_task("lift")
    spots = []
    for r in (task.cube_radius[0], 0.19, task.cube_radius[1]):
        for bearing in (task.cube_bearing[0], np.radians(-40), np.radians(40), task.cube_bearing[1]):
            spots += [(r * np.cos(bearing), r * np.sin(bearing), np.radians(yaw)) for yaw in range(0, 90, 10)]
    # Dwa polozenia z przegladu (e2e_lift_post, proby 8 i 14 - "kamery jej nie widza").
    spots += [(0.136, -0.134, np.radians(yaw)) for yaw in range(0, 90, 10)]
    spots += [(0.148, -0.123, np.radians(yaw)) for yaw in range(0, 90, 10)]
    return spots


@pytest.mark.render
def test_resting_cube_is_found_at_every_yaw_on_the_edges_of_the_lift_region():
    """Kostka lezy nieruchomo, ramie w spoczynku zaslania ja jednej z dwoch kamer prawie cala.

    Przed poprawka 59 z 360 polozen/obrotow (siatka co 5 st.) bez detekcji, z tego 14 z 18
    obrotow na r = 0,19 m, -45 st.: skrawek kostki widziany przez zasloniona kamere wchodzil
    do sredniej IoU z pelna waga. Dopasowanie stawalo 12 mm obok (IoU 0,70), a przy dobrym
    polozeniu szum krawedzi skrawka ciagnal srednia pod prog. Runner lift z kamer stawal
    wtedy w 2 z 20 prob panelu na "kamery jej nie widza".
    """
    spots = _edges_of_lift_region()
    dets = _sweep_at_rest(spots)
    missed = [(round(x, 3), round(y, 3), round(float(np.degrees(yaw))))
              for (x, y, yaw), d in zip(spots, dets)
              if d is None or np.hypot(d.pos[0] - x, d.pos[1] - y) > 0.006]
    assert not missed, f"{len(missed)} z {len(spots)} bez detekcji albo > 6 mm od prawdy: {missed}"


@pytest.mark.render
def test_lifted_cube_on_the_edges_is_never_a_two_camera_detection():
    """Ta sama siatka, kostka 2 i 4 cm nad blatem: bramka dwoch swiadkow dalej odrzuca ducha.

    Poprawka waz glos kamery tym, ile kostki widzi - dwie kamery widzace wiekszosc sylwetki
    glosuja jak wczesniej, wiec podniesiona kostka nie moze przejsc jako `n_cameras >= 2`.
    """
    spots = _edges_of_lift_region()[::3]
    for lift in (0.02, 0.04):
        dets = _sweep_at_rest(spots, lift=lift)
        passed = [(round(x, 3), round(y, 3), round(d.confidence, 2))
                  for (x, y, _), d in zip(spots, dets) if d is not None and d.n_cameras >= 2]
        assert not passed, f"podniesiona o {lift * 100:.0f} cm przeszla jako dwie kamery: {passed}"
