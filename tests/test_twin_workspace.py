"""Stanowisko blizniaka: zapis i odczyt, wpisanie wyniku kalibracji, scena z niego.

Bez renderu i bez sprzetu - to jest warstwa danych, na ktorej stoi reszta.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("mujoco")

from lerobot_mp.twin.calib.handeye import CameraFit, Fit  # noqa: E402
from lerobot_mp.twin.cli import main as cli_main  # noqa: E402
from lerobot_mp.twin.workspace import CameraRecord, Workspace, nominal_K  # noqa: E402


def calibrated_camera(name: str = "front") -> CameraRecord:
    T = np.eye(4)
    T[:3, 3] = [0.6, 0.1, 0.3]
    return CameraRecord(name, source="0", T_cam2base=T.tolist(), K=nominal_K(640, 480).tolist())


def test_missing_file_is_a_fresh_workspace_not_an_error(tmp_path):
    ws = Workspace.load(tmp_path / "brak.json")
    assert ws.cameras == [] and ws.robot == "so101"


def test_workspace_round_trips_through_json(tmp_path):
    ws = Workspace(port="COM12", backend="feetech")
    ws.add_camera(calibrated_camera())
    ws.card["tag_size"] = 0.0483                       # zmierzony wydruk, nie nominalny
    path = ws.save(tmp_path / "twin.json")
    back = Workspace.load(path)
    assert back.port == "COM12" and back.backend == "feetech"
    assert back.card_obj().tag_size == pytest.approx(0.0483)
    assert np.allclose(back.camera("front").T_cam2base, ws.camera("front").T_cam2base)


def test_unknown_fields_are_refused_not_silently_dropped(tmp_path):
    path = tmp_path / "twin.json"
    path.write_text(json.dumps({"robot": "so101", "kamerki": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="kamerki"):
        Workspace.load(path)


def test_duplicate_camera_names_are_refused():
    ws = Workspace()
    ws.add_camera(CameraRecord("front"))
    with pytest.raises(ValueError):
        ws.add_camera(CameraRecord("front"))
    assert ws.free_name() == "kamera1"


def test_calibration_result_lands_in_the_camera_with_its_verdict():
    ws = Workspace()
    ws.add_camera(CameraRecord("front"))
    T = np.eye(4)
    T[:3, 3] = [0.5, 0.0, 0.2]
    fit = Fit({"front": CameraFit(T, 0.18, 0.6, 30, 12.5, trusted=True, reason="")})
    assert ws.apply_fit(fit) == ["front"]
    cam = ws.camera("front")
    assert cam.calibrated and cam.trusted
    assert cam.calibration["rms_px"] == pytest.approx(0.18)
    assert cam.calibration["tag_size"] == pytest.approx(ws.card_obj().tag_size)


def test_scene_takes_only_enabled_placed_cameras():
    ws = Workspace()
    ws.add_camera(calibrated_camera("front"))
    ws.add_camera(CameraRecord("nowa"))                # jeszcze nieskalibrowana - nie ma gdzie stac
    off = calibrated_camera("wylaczona")
    off.enabled = False
    ws.add_camera(off)
    names = [v.name for v in ws.scene_config().cameras]
    assert names == ["front"]


def test_cli_prints_a_card_sheet(tmp_path):
    out = tmp_path / "karta.png"
    assert cli_main(["card", "--out", str(out), "--dpi", "100"]) == 0
    assert out.stat().st_size > 1000
