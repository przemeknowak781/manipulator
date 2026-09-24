"""Caly lancuch sim-2-real naraz: polityka lift na zywym blizniaku, kostka WYLACZNIE z kamer.

Kadry symulowanych kamer -> dopasowanie sylwetki (z maska ramienia) -> sledzenie
kostki w dloni -> runner -> nadzor bezpieczenstwa -> fizyka. Sprawdzamy na
prawdzie z fizyki, czy kostka zostala podniesiona. Na tym tescie `lift-v1`
(uczona na prawdziwej pozycji kostki) podnosila 1 z 8, `lift-v2` (douczona
z modelem percepcji) - 8 z 8.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("torch")

from lerobot_mp.twin import scene as sc  # noqa: E402
from lerobot_mp.twin.kinematics import RobotKinematics, inverse, pose  # noqa: E402
from lerobot_mp.twin.perception import CubeDetector, CubeTracker, TableMapper  # noqa: E402
from lerobot_mp.twin.rl import task as tk  # noqa: E402
from lerobot_mp.twin.rl.policy import Policy, bundled_dir  # noqa: E402
from lerobot_mp.twin.rl.runner import PolicyRunner  # noqa: E402
from lerobot_mp.twin.runtime import Twin  # noqa: E402
from lerobot_mp.twin.ui.watch import arm_mask  # noqa: E402
from lerobot_mp.twin.workspace import CameraRecord, Workspace  # noqa: E402

POLICY = bundled_dir() / "lift-v2" / "policy.pt"


def look(eye, target):
    eye = np.asarray(eye, float)
    z = np.asarray(target, float) - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    return pose(np.column_stack([x, np.cross(z, x), z]), eye)


@pytest.mark.skipif(not POLICY.is_file(), reason="brak bazowej polityki lift-v2 w assets/policies")
def test_lift_policy_lifts_a_cube_seen_only_by_cameras():
    ws = Workspace()
    K = [[560.0, 0, 322.0], [0, 560.0, 236.0], [0, 0, 1]]
    for name, eye in (("a", [0.55, -0.45, 0.45]), ("b", [0.5, 0.5, 0.5])):
        T = look(eye, [0.2, 0, 0]).tolist()
        ws.add_camera(CameraRecord(name, "sim", 640, 480, K=K, sim_pose=T, T_cam2base=T,
                                   calibration={"trusted": True}))
    pol = Policy.load(POLICY)
    task = pol.task
    h = task.cube_half
    twin = Twin(ws)
    try:
        twin.configure(objects=[sc.Box("cube", (h, h, h), (0.2, 0.0), rgba=(0.85, 0.25, 0.2, 1.0),
                                       mass=task.cube_mass)], grasp_sensors=["cube"])
        twin.connect("sim")
        mapper = TableMapper.from_workspace(ws)
        det = CubeDetector()
        kin = RobotKinematics(ws.spec())
        pos, quat = tk.sample_cubes(task, np.random.default_rng(5), 1)
        with twin.lock:
            s = twin.scene
            m, d = s.model, s.data
            b = m.body("cube").id
            a = m.jnt_qposadr[m.body_jntadr[b]]
            T = s.T_base2world
            qb, qw = np.zeros(4), np.zeros(4)
            mujoco.mju_mat2Quat(qb, T[:3, :3].ravel())
            mujoco.mju_mulQuat(qw, qb, quat[0])
            d.qpos[a:a + 3] = T[:3, :3] @ pos[0] + T[:3, 3]
            d.qpos[a + 3:a + 7] = qw
            mujoco.mj_forward(m, d)

        tracker = CubeTracker()
        last = {"det": None}
        stop = threading.Event()

        def vision():
            while not stop.is_set():
                frames = {c: twin.render(c) for c in ("a", "b")}
                occ = twin.render_with(lambda sc_: {c: arm_mask(sc_, c, dilate=5) for c in ("a", "b")})
                last["det"] = det.detect_frames(frames, mapper, occ)
                time.sleep(0.1)

        th = threading.Thread(target=vision, daemon=True)
        th.start()
        runner = None

        def provider():
            joints = dict(twin.status.measured) or twin.joints()
            q = kin.to_q(joints)
            return tracker.update(last["det"], kin.tcp(joints), q[5], runner.q_cmd[5], kin.lo[5], time.monotonic())

        runner = PolicyRunner(twin, pol, cube_provider=provider)
        runner.start()
        deadline = time.monotonic() + 30.0
        while runner.status.running and time.monotonic() < deadline:
            time.sleep(0.1)
        stop.set()
        th.join(timeout=2.0)
        with twin.lock:
            s = twin.scene
            Ti = inverse(s.T_base2world)
            height = (Ti[:3, :3] @ s.data.xpos[s.model.body("cube").id] + Ti[:3, 3])[2] - h
        assert runner.status.stopped_because == "koniec epizodu"
        assert height > task.lift_height, f"kostka tylko {height * 100:.1f} cm nad blatem"
    finally:
        twin.close()
