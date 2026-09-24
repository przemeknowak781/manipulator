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


def charuco_camera(name: str = "front", trusted: bool = True) -> CameraRecord:
    return CameraRecord(name, K=nominal_K(640, 480).tolist(), dist=[0.1, -0.05, 0.0, 0.0, 0.0],
                        intrinsics_from="szachownica",
                        intrinsics_info={"rms_px": 0.3, "trusted": trusted, "reason": "" if trusted else "malo kadrow"})


def trusted_fit(name: str = "front") -> Fit:
    T = np.eye(4)
    T[:3, 3] = [0.5, 0.0, 0.2]
    return Fit({name: CameraFit(T, 0.18, 0.6, 30, 12.5, trusted=True, reason="")})


def test_calibration_result_lands_in_the_camera_with_its_verdict():
    ws = Workspace()
    ws.add_camera(charuco_camera())
    K, dist = ws.camera("front").intrinsics()
    assert ws.apply_fit(trusted_fit(), tag_size=0.0491, intrinsics={"front": (K, dist)}) == ["front"]
    cam = ws.camera("front")
    assert cam.calibrated and cam.trusted
    assert cam.calibration["rms_px"] == pytest.approx(0.18)
    # Bok taga z fali (nie z pola w panelu) i K, z ktorym liczono - zapisane przy pozie.
    assert cam.calibration["tag_size"] == pytest.approx(0.0491)
    assert np.allclose(cam.calibration["K"], K) and np.allclose(cam.calibration["dist"], dist)


def test_real_camera_pose_is_not_trusted_without_a_trusted_charuco_K():
    """Dopasowanie z nominalnym K przechodzi prog residuum, a poza jest przesunieta o centymetry."""
    ws = Workspace()
    ws.add_camera(CameraRecord("nominalna"))                       # krok 1 pominiety
    ws.add_camera(charuco_camera("slaba", trusted=False))           # sesja ChArUco niezaufana
    ws.add_camera(CameraRecord("sym", "sim", K=nominal_K(640, 480).tolist(), intrinsics_from="symulacja"))
    fit = Fit({n: trusted_fit(n).cameras[n] for n in ("nominalna", "slaba", "sym")})
    ws.apply_fit(fit)
    assert not ws.camera("nominalna").trusted and "nominalne" in ws.camera("nominalna").calibration["reason"]
    assert not ws.camera("slaba").trusted and "malo kadrow" in ws.camera("slaba").calibration["reason"]
    assert ws.camera("sym").trusted                                 # symulowana zna swoje K


def test_pose_from_a_K_that_changed_before_save_is_not_trusted():
    """Fala z nominalnym K, potem krok 1 zapisal zaufane K z ChArUco, potem "Zapisz".

    Werdykt oceniamy dla K, z ktorym LICZONO poze - wczesniej obecne, zaufane K
    robilo zaufana poze policzona z nominalnego (fx 502 zamiast 552).
    """
    ws = Workspace()
    ws.add_camera(CameraRecord("usb1", "0"))
    rec = ws.camera("usb1")
    used = rec.intrinsics()                                        # K z chwili startu fali: nominalne
    rec.K = (nominal_K(640, 480) * np.diag([1.1, 1.1, 1.0])).tolist()
    rec.dist = [-0.2, 0.05, 0.0, 0.0, 0.0]
    rec.intrinsics_from = "szachownica"
    rec.intrinsics_info = {"rms_px": 0.3, "trusted": True, "reason": ""}
    ws.apply_fit(trusted_fit("usb1"), tag_size=0.05, intrinsics={"usb1": used})
    assert rec.calibrated and not rec.trusted
    assert "zmienione od fali" in rec.calibration["reason"]
    assert "zmienione od fali" in rec.fit_problem(used) and rec.fit_problem(rec.intrinsics()) == ""
    # Kamera, dla ktorej fala nie podala K - niezaufana, a nie oceniana obecnym K.
    ws.add_camera(charuco_camera("bez_K"))
    ws.apply_fit(trusted_fit("bez_K"), intrinsics={"usb1": used})
    assert not ws.camera("bez_K").trusted


def test_trusted_pose_is_revoked_when_the_camera_K_changes_later():
    ws = Workspace()
    ws.add_camera(charuco_camera())
    cam = ws.camera("front")
    ws.apply_fit(trusted_fit(), intrinsics={"front": cam.intrinsics()})
    assert cam.trusted
    # Po zapisie i odczycie (JSON) to wciaz to samo K - zaufanie zostaje.
    back = Workspace(cameras=[CameraRecord(**json.loads(json.dumps(ws.to_dict()))["cameras"][0])])
    assert back.camera("front").trusted
    cam.K[0][0] *= 1.05                                             # K zmienione obok panelu
    assert not cam.trusted
    # Poza bez zapisanego K (sprzed zapisu K przy pozie) - werdykt, jak byl zapisany.
    old = CameraRecord("stara", "sim", T_cam2base=np.eye(4).tolist(), calibration={"trusted": True})
    assert old.trusted


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
