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


# ------------------------------------------------------------------ poprawki przed prawdziwym ramieniem
def _static_mask(app, name, joints, dilate=5):
    from lerobot_mp.twin.ui.watch import arm_mask

    def render(s):
        s.set_joints(joints)
        return arm_mask(s, name, dilate=dilate)
    return app.twin.render_with(render)


def test_arm_mask_grows_along_the_arm_path_and_each_camera_at_its_own_frame_time(app):
    """Kwadratowe poszerzenie z predkosci stawu (tez wrist_roll) chowalo kostke przy szczekach.

    Przed poprawka: wrist_roll 50 st. w oknie - maska 2,5 raza wieksza (5 -> 45 px jadra),
    a maski wszystkich kamer w pozie z chwili NAJSTARSZEGO kadru.
    """
    home = dict(app.ws.spec().home)
    app._add_sim_camera(app.T_b2w @ _look_at_pose([0.55, -0.35, 0.40], [0.2, 0.0, 0.05]))
    app._add_sim_camera(app.T_b2w @ _look_at_pose([0.55, 0.35, 0.40], [0.2, 0.0, 0.05]))
    a, b = app.ws.cameras[-2].name, app.ws.cameras[-1].name

    def swing(joint, span):
        t0 = time.monotonic()
        app._joint_hist.clear()
        for k in range(-12, 13):                                 # 30 Hz, +-0,4 s wokol kadru
            app._joint_hist.append((t0 + k / 30, dict(home, **{joint: home[joint] + span * k / 12})))
        return t0

    try:
        still = _static_mask(app, a, home)
        t0 = swing("wrist_roll", 50.0)
        roll = app._arm_masks([a], t0, dilate=5)[a]
        assert roll.sum() < 1.6 * still.sum()                    # przed poprawka ~2,5x
        t0 = swing("shoulder_pan", 20.0)
        pan = app._arm_masks([a], t0, dilate=5)[a]
        # Ramie na koncu okna niepewnosci (kadr 0,23 s pozniej) wciaz w masce.
        far = _static_mask(app, a, dict(home, shoulder_pan=home["shoulder_pan"] + 20.0 * 7 / 12))
        assert (far & ~pan).sum() < 0.03 * far.sum()

        # Dwie kamery, kadry z roznych chwil: kazda maska w pozie ze swojej chwili.
        t0 = time.monotonic()
        pose_a, pose_b = dict(home, shoulder_pan=home["shoulder_pan"] - 25.0), dict(home)
        app._joint_hist.clear()
        for k in range(30):
            app._joint_hist.append((t0 - 1.0 + k / 30, pose_a if k < 15 else pose_b))
        masks = app._arm_masks([a, b], {a: t0 - 0.9, b: t0 - 0.05}, dilate=5)
        assert np.array_equal(masks[a], _static_mask(app, a, pose_a))
        assert np.array_equal(masks[b], _static_mask(app, b, pose_b))
    finally:
        app._joint_hist.clear()
        for n in (a, b):
            app.ws.remove_camera(n)
        app.twin.rebuild()
        app._refresh_cameras()


def test_preview_says_no_cameras_after_the_last_one_is_removed(app):
    from lerobot_mp.twin.ui.app import no_frame_image

    app._add_sim_camera(app.T_b2w @ _look_at_pose([0.6, 0.4, 0.45], [0.2, 0.0, 0.0]))
    name = app.ws.cameras[-1].name
    app._tick_slow()
    assert app._img_sent["podglad"] != hash(no_frame_image("brak kamer").tobytes()[::97])
    for n in [c.name for c in app.ws.cameras]:
        app.ws.remove_camera(n)
    app.twin.rebuild()
    app._refresh_cameras()
    app._tick_slow()
    assert app.cam_pick.value == "-" and name not in app.frustums
    assert app._img_sent["podglad"] == hash(no_frame_image("brak kamer").tobytes()[::97])


def _click(app, label: str) -> _Event:
    """Przycisk panelu bez przegladarki - te same callbacki, co klikniecie."""
    hs = [h for h in app.server.gui._gui_input_handle_from_uuid.values() if getattr(h, "label", None) == label]
    assert len(hs) == 1, label
    ev = _Event()
    for cb in hs[0]._impl.update_cb:
        cb(ev)
    return ev


def test_sysid_fit_survives_stop_and_reconnect_and_fixed_params_are_not_called_fitted(app, monkeypatch):
    """STOP / "Polacz" w trakcie minutowego dopasowania (ramie juz wolne) gubily wynik bez slowa."""
    import threading

    import lerobot_mp.twin.rl.sysid as sysid
    from lerobot_mp.twin.rl.randomize import Dynamics

    go = threading.Event()

    def slow_fit(rec, on_progress=None):
        go.wait(10.0)
        dyn = Dynamics(damping=0.9, armature=1.2, delay=0.5, source="test", fit_deg=0.4)
        if hasattr(dyn, "fitted"):
            dyn.fitted = ("damping", "armature", "delay")
        return dyn, 1.5

    monkeypatch.setattr(sysid, "record", lambda twin, plan, **kw: "nagranie")
    monkeypatch.setattr(sysid, "fit", slow_fit)
    app.twin.connect("sim", threaded=False)
    app._take_arm(jobs.SYSID_OWNER, preempt=app.sysid_job.stop)
    app.sysid_job.start(lambda job: jobs.run_sysid(job, app.twin))
    t_end = time.monotonic() + 5.0
    while "recording" not in app.sysid_job.data and time.monotonic() < t_end:
        time.sleep(0.01)
    assert app._busy("polityka") == ""                           # ramie wolne w trakcie dopasowania
    app._emergency_stop(None)
    app.twin.clear_estop()
    app._connect("sim", None)
    ev = _click(app, "Identyfikuj na polaczonym ramieniu (~20 s ruchu)")
    assert ev.client.notes and "dopasowanie poprzedniego" in ev.client.notes[-1][1]
    go.set()
    assert app.sysid_job.wait(10.0) and app.sysid_job.state == jobs.DONE
    app._tick_training()
    txt = app.dyn_res.content
    assert app.dyn_keep.visible and "Dopasowano: " in txt
    fitted, fixed = txt.split("z modelu")
    assert "tlumienie x0.90" in fitted and "armatura x1.20" in fitted and "kp" not in fitted
    assert "kp x1.00" in fixed and "tarcie x1.00" in fixed


def test_sysid_stopped_during_recording_ends_as_cancelled_and_says_so(app, monkeypatch):
    import lerobot_mp.twin.rl.sysid as sysid

    fits = []

    def record(twin, plan, should_stop=None, **kw):
        t_end = time.monotonic() + 5.0
        while not should_stop() and time.monotonic() < t_end:
            time.sleep(0.01)
        return "nagranie"                                        # nagranie skonczone w chwili STOP

    monkeypatch.setattr(sysid, "record", record)
    monkeypatch.setattr(sysid, "fit", lambda rec, on_progress=None: fits.append(1))
    app.twin.connect("sim", threaded=False)
    app.sysid_job.start(lambda job: jobs.run_sysid(job, app.twin))
    time.sleep(0.05)
    app._stop_motion("ponowne laczenie")
    assert app.sysid_job.wait(5.0) and app.sysid_job.state == jobs.CANCELLED and fits == []
    app._tick_training()
    assert "Przerwane" in app.dyn_res.content and not app.dyn_keep.visible


def test_close_disconnects_the_arm_before_waiting_for_the_training():
    """Drugie Ctrl+C w czasie do 13 s czekania na trening pomijalo `twin.close()`."""
    import threading
    from types import SimpleNamespace

    from lerobot_mp.twin.ui.app import TwinApp

    calls = []

    def shutdown():
        calls.append("trening")
        raise KeyboardInterrupt                                  # drugie Ctrl+C w trakcie czekania

    st = SimpleNamespace(_stop=threading.Event(), runner=None, _tick_errors=set(),
                         calib_job=jobs.Job("a"), sysid_job=jobs.Job("b"), intr_job=jobs.Job("c"),
                         eval_job=jobs.Job("d"), twin=SimpleNamespace(close=lambda: calls.append("ramie")),
                         train=SimpleNamespace(shutdown=shutdown),
                         server=SimpleNamespace(stop=lambda: calls.append("serwer")))
    st._guard = lambda what, fn, *a: TwinApp._guard(st, what, fn, *a)
    with pytest.raises(KeyboardInterrupt):
        TwinApp.close(st)
    assert calls == ["ramie", "trening", "serwer"]


def test_cube_tracker_gets_the_effective_closed_jaw_angle(app, monkeypatch):
    """Nie dolny kraniec MJCF (-10 st.), a kat szczeki przy chwytaku 0 (-5,4 st.), jak w polityce."""
    from types import SimpleNamespace

    from lerobot_mp.twin.rl import task as tk

    seen = []
    monkeypatch.setattr(app.cube_tracker, "update", lambda det, T, q, cmd, closed, now: seen.append(closed))
    kin = app._kin_vision
    old = app.runner
    try:
        app.runner = None
        app._vision_cube()
        assert seen[-1] == pytest.approx(kin.to_q({"gripper": 0.0})[5])
        assert seen[-1] > kin.lo[5] + np.radians(3.0)
        lim = tk.Limits.of(kin)
        app.runner = SimpleNamespace(limits=lim, q_cmd=lim.home.copy())
        app._vision_cube()
        assert seen[-1] == pytest.approx(lim.lo[5])
    finally:
        app.runner = old


def test_wave_verdict_and_save_judge_the_K_the_fit_used(app):
    """Fala z nominalnym K, krok 1 (zaufane K z ChArUco) przed "Zapisz" - poza NIE jest zaufana."""
    from lerobot_mp.twin.calib.handeye import CameraFit, Fit
    from lerobot_mp.twin.workspace import CameraRecord, nominal_K

    rec = app.ws.add_camera(CameraRecord("usb1", "7", enabled=False))
    try:
        used = rec.intrinsics()
        T = np.eye(4)
        T[:3, 3] = [0.5, 0.0, 0.3]
        j = app.calib_job
        j.state, j.result = jobs.DONE, Fit({"usb1": CameraFit(T, 0.3, 0.6, 30, 25.0, trusted=True)})
        j.data = {"tag_size": app.tag_mm.value / 1000, "intrinsics": {"usb1": used}}
        rec.K = (nominal_K(640, 480) * np.diag([1.1, 1.1, 1.0])).tolist()
        rec.dist = [-0.2, 0.05, 0.0, 0.0, 0.0]
        rec.intrinsics_from = "szachownica"
        rec.intrinsics_info = {"rms_px": 0.3, "trusted": True, "reason": ""}
        app.calib_apply.visible = False
        app._tick_calibration()
        assert "zmienione od fali" in app.calib_md.content and "zaufana" not in app.calib_md.content
        assert app._apply_calibration() == ["usb1"]
        assert rec.calibrated and not rec.trusted
    finally:
        app.ws.remove_camera("usb1")
        app.calib_job.state, app.calib_job.result, app.calib_job.data = jobs.IDLE, None, {}
        app._refresh_cameras()


def test_status_bar_shows_arm_warnings(app, monkeypatch):
    import copy

    st = copy.copy(app.twin.status)
    st.warnings = ["chwytak: kalibracja LeRobota rozni sie od blizniaka o 60 tikow"]
    monkeypatch.setattr(app.twin, "status", st)
    assert "Uwaga:** chwytak: kalibracja" in app._status_text()
