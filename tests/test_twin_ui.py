"""Panel blizniaka: startuje, odswieza sie i obsluguje kamery symulowane bez przegladarki.

Dalej - to, czego panel pilnuje przed prawdziwym ramieniem: jeden wlasciciel ramienia,
uchwyt TCP na tej samej galezi IK, swiezosc kadrow i detekcji kostki, zadania w tle.
"""

from __future__ import annotations

import socket
import subprocess
import sys
import time

import numpy as np
import pytest

pytest.importorskip("viser")
pytest.importorskip("mujoco")

from lerobot_mp.twin.ui import jobs  # noqa: E402


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _Client:
    def __init__(self):
        self.notes: list[tuple[str, str]] = []

    def add_notification(self, title, body, **kw):
        self.notes.append((title, body))


class _Event:
    """Zdarzenie od uzytkownika (z `client`), jak z przegladarki."""

    def __init__(self):
        self.client = _Client()


def _look_at_pose(eye, target):
    from lerobot_mp.twin.kinematics import pose
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    z = (target - eye) / np.linalg.norm(target - eye)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    return pose(np.column_stack([x, np.cross(z, x), z]), eye)


def test_panel_starts_adds_a_simulated_camera_and_maps_the_table(tmp_path):
    from lerobot_mp.twin.ui.app import TwinApp

    app = TwinApp(tmp_path / "twin.json", host="127.0.0.1", port=free_port())
    try:
        app._tick_slow()
        app._add_sim_camera(app.T_b2w @ _look_at_pose([0.6, -0.4, 0.45], [0.2, 0.0, 0.0]))
        rec = app.ws.cameras[0]
        assert rec.simulated and rec.sim_pose is not None
        frames = app._tick_slow()
        assert rec.name in frames and frames[rec.name].shape == (480, 640, 3)
        assert app.frame_times[rec.name] > 0
        # "uznaj prawde za kalibracje" -> kamera zaufana -> mapa stolu z niej
        rec.T_cam2base = [list(r) for r in rec.sim_pose]
        rec.calibration = {"trusted": True, "reason": "", "rms_px": 0.0}
        app._refresh_cameras()
        app._tick_map(frames)
        assert app.mapper is not None and rec.name in app.mapper.cameras
        assert (tmp_path / "twin.json").is_file()
        app.twin.connect("sim")
        app._tick_arm()
        assert app.twin.connected
    finally:
        app.close()


# ------------------------------------------------------------------ jeden panel na reszte testow
@pytest.fixture(scope="module")
def shared_app(tmp_path_factory):
    from lerobot_mp.twin.ui.app import TwinApp

    app = TwinApp(tmp_path_factory.mktemp("panel") / "twin.json", host="127.0.0.1", port=free_port())
    yield app
    app.close()


@pytest.fixture
def app(shared_app):
    a = shared_app
    yield a
    if a.runner is not None:
        a.runner.stop("koniec testu")
    for job in (a.calib_job, a.sysid_job, a.intr_job):
        job.stop()
        job.wait(10.0)
        job.state, job.result, job.data = jobs.IDLE, None, {}
    a.twin.disconnect()
    a.arm_engage.value = False
    a.tcp_gizmo_on.value = False
    for owner in (a.twin.owner,):
        if owner is not None:
            a.twin.release(owner)


def test_tcp_gizmo_rejects_a_jump_to_the_other_ik_branch(app):
    """Z shoulder_pan 109,5 st. przeciagniecie o 1 cm dawalo "poprawne" IK 180 st. dalej."""
    spec = app.ws.spec()
    start = dict(spec.home, shoulder_pan=109.5)
    kin = app._kin_gizmo
    p = kin.tcp(start)[:3, 3]
    tang = np.array([-p[1], p[0], 0.0]) / np.hypot(p[0], p[1])
    old = kin.ik(p - 0.01 * tang, seed=start, restarts=2)      # tak liczyl uchwyt przed poprawka
    assert old.ok and max(abs(old.joints[k] - start[k]) for k in start if k != spec.gripper) > 90

    with app.twin.lock:
        app.twin.scene.set_joints(start)
    app.twin.connect("sim", threaded=False)
    app.tcp_gizmo_on.value = True
    app.arm_engage.value = True
    app._on_engage(_Event())
    assert app.twin.owner == "panel"

    def drag(p_base):
        app.tcp_gizmo.position = (app.T_b2w @ np.r_[p_base, 1.0])[:3]
        app._on_tcp_gizmo()
        return app.twin._target

    assert drag(p - 0.01 * tang) is None                        # odrzucone, cel bez zmian
    app._gizmo_seed = None
    target = drag(p + 0.01 * tang)                              # w druga strone - ta sama galaz
    assert target is not None
    assert max(abs(target[k] - start[k]) for k in start if k != spec.gripper) < 5.0


def test_clutch_and_jobs_refuse_the_arm_while_someone_else_drives_it(app):
    app.twin.connect("sim", threaded=False)
    app.twin.claim(jobs.CALIB_OWNER)                            # jedzie fala
    ev = _Event()
    app.arm_engage.value = True
    app._on_engage(ev)
    assert not app.arm_engage.value and app.twin.owner == jobs.CALIB_OWNER
    assert "kalibracja" in ev.client.notes[-1][1]
    with pytest.raises(RuntimeError, match="zajete"):
        app._start_policy(_Event())
    with pytest.raises(RuntimeError, match="zajete"):
        app._take_arm(jobs.SYSID_OWNER)
    app.twin.release(jobs.CALIB_OWNER)
    # Panel (sprzeglo) oddaje ramie zadaniu bez pytania - to operator je uruchomil.
    app.arm_engage.value = True
    app._on_engage(_Event())
    assert app.twin.owner == "panel"
    app._take_arm(jobs.SYSID_OWNER)
    assert app.twin.owner == jobs.SYSID_OWNER and not app.arm_engage.value


def test_reconnect_ends_a_running_wave_and_it_never_drives_the_new_arm(app):
    """Fala w sim + "Polacz" = reszta fali na prawdziwym ramieniu, bez potwierdzenia karty."""
    home = dict(app.ws.spec().home)
    app.twin.connect("sim")
    app._take_arm(jobs.CALIB_OWNER, preempt=app.calib_job.stop)
    moves = []

    def wave(job):
        arm = jobs._OwnedArm(app.twin, job)
        for k in range(1000):
            arm.move(dict(home, shoulder_pan=20.0 if k % 2 else -20.0), 0.3)
            moves.append(k)

    app.calib_job.start(wave)
    time.sleep(0.8)
    app.calib_confirm.value = True
    app._connect("sim", None)
    assert app.calib_job.wait(5.0)
    assert app.calib_job.state == jobs.FAILED
    assert not app.calib_confirm.value                          # nowe ramie - nowe potwierdzenie
    n = len(moves)
    time.sleep(0.5)
    assert len(moves) == n and app.twin.owner is None and not app.twin.status.engaged


def test_card_wave_goes_home_only_while_it_still_owns_the_arm(app, monkeypatch):
    import lerobot_mp.twin.calib.session as session_mod

    home = dict(app.ws.spec().home)
    app.twin.connect("sim", threaded=False)

    class InterruptedSession:
        def __init__(self, robot, *a, **kw):
            self.robot, self.done = robot, False

        def step(self):
            app.twin.preempt("STOP w trakcie fali")             # miedzy przejazdami
            self.robot.move(home, 0.2)

    homes = []
    monkeypatch.setattr(session_mod, "Session", InterruptedSession)
    monkeypatch.setattr(app.twin, "home", lambda: homes.append(1))
    app.twin.claim(jobs.CALIB_OWNER, preempt=app.calib_job.stop)
    app.calib_job.start(lambda job: jobs.run_card_calibration(job, app.twin, [], quick=True))
    assert app.calib_job.wait(60.0)
    assert app.calib_job.state == jobs.FAILED and "odebrane" in app.calib_job.error
    assert homes == [] and app.twin.owner is None               # ramie nie nasze - bez jazdy do domu
    monkeypatch.undo()

    class DoneSession:
        def __init__(self, robot, *a, **kw):
            self.done = True

        def solve(self):
            return "wynik"

    monkeypatch.setattr(session_mod, "Session", DoneSession)
    app.twin.claim(jobs.CALIB_OWNER, preempt=app.calib_job.stop)
    app.calib_job.start(lambda job: jobs.run_card_calibration(job, app.twin, [], quick=True))
    assert app.calib_job.wait(60.0)
    # Dom po udanej fali odbiera ramie - ale nie moze oznaczyc fali jako przerwanej.
    assert app.calib_job.state == jobs.DONE and app.calib_job.result == "wynik"
    assert app.twin.owner is None and app.calib_job.data["tag_size"] == pytest.approx(app.ws.card_obj().tag_size)


def test_intrinsics_abort_keeps_K_and_the_result_goes_to_the_measured_camera(app, monkeypatch):
    import threading

    from lerobot_mp.twin.calib import intrinsics
    from lerobot_mp.twin.workspace import CameraRecord

    class Hub:
        def frame(self, name):
            return None

    board = intrinsics.Board(square=0.028, marker=0.021)
    solved = []
    monkeypatch.setattr(intrinsics.Collector, "solve", lambda self: solved.append(1) or "K")
    job = jobs.Job("intrynsyki")
    job.start(lambda j: jobs.run_intrinsics(j, Hub(), "kam1", board, (640, 480), period=0.01))
    time.sleep(0.05)
    job.stop()                                                  # "Przerwij"
    assert job.wait(2.0) and job.state == jobs.CANCELLED and job.result is None and solved == []
    solve = threading.Event()
    job.start(lambda j: jobs.run_intrinsics(j, Hub(), "kam1", board, (640, 480), period=0.01, solve=solve))
    solve.set()                                                 # "Oblicz i zapisz K"
    assert job.wait(2.0) and job.state == jobs.DONE and job.result == "K" and job.data["camera"] == "kam1"

    class Res:
        K = np.array([[610.0, 0, 320], [0, 611.0, 240], [0, 0, 1]])
        dist = np.zeros(5)
        rms_px, n_views, coverage, trusted, reason = 0.3, 14, 0.7, True, ""

    for n in ("kam1", "kam2"):
        app.ws.add_camera(CameraRecord(n, "sim"))
    try:
        app._tick_slow()
        app.intr_cam.value = "kam2"                             # lista w panelu zmieniona w trakcie
        ij = app.intr_job
        ij.state, ij.result, ij.data = jobs.DONE, Res(), {"camera": "kam1"}
        app._tick_calibration()
        assert app.ws.camera("kam1").K[0][0] == pytest.approx(610.0)
        assert app.ws.camera("kam2").K is None
    finally:
        for n in ("kam1", "kam2"):
            app.ws.remove_camera(n)
        app.intr_job.state = jobs.IDLE


def test_stale_or_repeated_cube_detection_does_not_keep_the_policy_going(app):
    from lerobot_mp.twin.perception import CubeDetection, CubeTracker

    def det(age: float) -> CubeDetection:
        return CubeDetection(np.array([0.25, 0.0, 0.015]), np.eye(3), 100.0, 0.9, n_cameras=2,
                             t=time.monotonic() - age)

    app.cube_tracker, app._cube_used_t = CubeTracker(hold_s=0.2), 0.0
    app.last_cube = det(0.0)
    assert app._vision_cube() is not None
    time.sleep(0.3)
    # Ta sama detekcja podawana co takt (mapa wylaczona, kamera zamrozona) odnawiala hold_s bez konca.
    assert app._vision_cube() is None and app.cube_tracker.source == "brak"
    app.last_cube = det(2.0)                                    # detekcja z kadrow sprzed 2 s
    assert app._vision_cube() is None
    # Mapa wylaczona: zadnej starej detekcji w `last_cube`.
    app.last_cube = det(0.0)
    app.map_on.value = False
    try:
        app._tick_map({})
        assert app.last_cube is None
    finally:
        app.map_on.value = True


def test_frozen_camera_frames_are_not_used(app, monkeypatch):
    from lerobot_mp.twin.workspace import CameraRecord

    img = np.zeros((48, 64, 3), np.uint8)
    app.ws.add_camera(CameraRecord("usb", "0"))
    try:
        monkeypatch.setattr(app.twin.cameras, "frame_t", lambda name: (img, time.monotonic() - 5.0))
        assert app._grab() == {}
        t = time.monotonic()
        monkeypatch.setattr(app.twin.cameras, "frame_t", lambda name: (img, t))
        assert "usb" in app._grab() and app.frame_times["usb"] == pytest.approx(t)
    finally:
        app.ws.remove_camera("usb")


def test_reach_goal_from_the_gizmo_is_kept_in_the_trained_region(app):
    from lerobot_mp.twin.rl import task as tk

    task = tk.make_task("reach")
    app._set_goal(np.array([0.25, 0.0, -0.05]))                 # pod blatem
    g = app._goal_base
    assert task.goal_height[0] - 1e-9 <= g[2] <= task.goal_height[1] + 1e-9
    assert np.allclose(app.goal_gizmo.position, (app.T_b2w @ np.r_[g, 1.0])[:3], atol=1e-6)


def test_saving_a_card_fit_refuses_a_tag_size_changed_since_the_wave(app):
    from lerobot_mp.twin.calib.handeye import CameraFit, Fit
    from lerobot_mp.twin.workspace import CameraRecord

    app.ws.add_camera(CameraRecord("front", "sim"))
    try:
        T = np.eye(4)
        T[:3, 3] = [0.5, 0.0, 0.3]
        app.calib_job.result = Fit({"front": CameraFit(T, 0.2, 0.5, 30, 20.0, trusted=True)})
        app.calib_job.data = {"tag_size": 0.050}
        app.tag_mm.value = 49.2
        with pytest.raises(RuntimeError, match="bok taga"):
            app._apply_calibration()
        app.tag_mm.value = 50.0
        assert app._apply_calibration() == ["front"]
        assert app.ws.camera("front").calibration["tag_size"] == pytest.approx(0.050)
    finally:
        app.ws.remove_camera("front")
        app.calib_job.result, app.calib_job.data = None, {}


def test_stop_takes_the_arm_from_the_panel_and_holds(app):
    app.twin.connect("sim", threaded=False)
    app.arm_engage.value = True
    app._on_engage(_Event())
    assert app.twin.owner == "panel"
    app._emergency_stop(None)
    assert app.twin.owner is None and app.twin.safety_state.value == "ESTOP" and not app.arm_engage.value


def test_disabled_sim_camera_drag_does_not_rebuild_and_preview_says_no_frame(app):
    from lerobot_mp.twin.ui.app import no_frame_image

    app._add_sim_camera(app.T_b2w @ _look_at_pose([0.6, 0.4, 0.45], [0.2, 0.0, 0.0]))
    rec = app.ws.cameras[-1]
    try:
        rec.enabled = False
        app.twin.rebuild()
        app.cam_move.value = True
        app._refresh_cameras(select=rec.name)
        assert not app.cam_gizmo.visible                         # wylaczona - nie ma czego przesuwac
        v = app.twin.version
        for dx in (0.0, 0.01, 0.02):
            T = app.T_b2w @ _look_at_pose([0.6 + dx, 0.4, 0.45], [0.2, 0.0, 0.0])
            app._place_sim_camera(rec, T, rebuild=False)
        assert app.twin.version == v and app._dirty_rebuild > 0
        app._tick_slow()
        assert app._img_sent["podglad"] == hash(no_frame_image().tobytes()[::97])
        app._sync_mirror()
        assert app._mirror_version == app.twin.version
    finally:
        app.cam_move.value = False
        app.ws.remove_camera(rec.name)
        app._dirty_rebuild = 0.0
        app.twin.rebuild()
        app._refresh_cameras()


def test_training_process_is_ended_with_the_panel(tmp_path):
    tj = jobs.TrainingJob(tmp_path, None)
    tj.run_dir = tmp_path
    tj.proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        tj.shutdown(timeout=0.5)
        assert tj.proc.poll() is not None
        assert (tmp_path / "STOP").is_file()                    # najpierw prosba o zapis polityki
    finally:
        if tj.proc.poll() is None:
            tj.proc.kill()
