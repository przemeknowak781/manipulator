"""Panel blizniaka: startuje, odswieza sie i obsluguje kamery symulowane bez przegladarki."""

from __future__ import annotations

import socket

import numpy as np
import pytest

pytest.importorskip("viser")
pytest.importorskip("mujoco")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_panel_starts_adds_a_simulated_camera_and_maps_the_table(tmp_path):
    from lerobot_mp.twin.kinematics import pose
    from lerobot_mp.twin.ui.app import TwinApp

    app = TwinApp(tmp_path / "twin.json", host="127.0.0.1", port=free_port())
    try:
        app._tick_slow()
        eye, target = np.array([0.6, -0.4, 0.45]), np.array([0.2, 0.0, 0.0])
        z = (target - eye) / np.linalg.norm(target - eye)
        x = np.cross(z, [0, 0, 1.0])
        x /= np.linalg.norm(x)
        app._add_sim_camera(app.T_b2w @ pose(np.column_stack([x, np.cross(z, x), z]), eye))
        rec = app.ws.cameras[0]
        assert rec.simulated and rec.sim_pose is not None
        frames = app._tick_slow()
        assert rec.name in frames and frames[rec.name].shape == (480, 640, 3)
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
