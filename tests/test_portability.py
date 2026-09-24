"""Swiezy klon na innym komputerze: modele w repo, sciezki domyslne, --config, przykladowe stanowisko."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

from lerobot_mp.config import ArmTrackingConfig, TrackerConfig, load_config
from lerobot_mp.paths import CONFIG_ENV, REPO_ROOT, data_path, source_checkout
from lerobot_mp.vision.tracker import model_location


def test_tests_run_from_a_source_checkout():
    assert source_checkout()
    assert (REPO_ROOT / "pyproject.toml").is_file()


def test_default_paths_are_relative_from_the_repo_root(monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    assert data_path("workspace/twin.json") == Path("workspace/twin.json")


def test_default_paths_point_into_the_repo_from_elsewhere(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert data_path("workspace/twin.json") == REPO_ROOT / "workspace" / "twin.json"
    absolute = tmp_path / "x.json"
    assert data_path(absolute) == absolute


def test_only_the_default_model_paths_follow_the_repo(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert model_location(TrackerConfig.model_path) == REPO_ROOT / "models" / "hand_landmarker.task"
    assert model_location(ArmTrackingConfig.model_path) == REPO_ROOT / "models" / "pose_landmarker_lite.task"
    # Sciezka wpisana samodzielnie zostaje wzgledem katalogu biezacego.
    assert model_location("moje/hand.task") == Path("moje/hand.task")


def test_shipped_models_match_the_checksums_in_their_readme():
    readme = (REPO_ROOT / "models" / "README.md").read_text(encoding="utf-8")
    for name in ("hand_landmarker.task", "pose_landmarker_lite.task"):
        path = REPO_ROOT / "models" / name
        assert path.is_file(), f"brak {path} - modele maja byc w repozytorium"
        m = re.search(rf"`{re.escape(name)}`.*?`([0-9a-f]{{64}})`", readme)
        assert m, f"brak sumy SHA-256 dla {name} w models/README.md"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == m.group(1)


def test_config_file_from_environment(monkeypatch, tmp_path):
    cfg = tmp_path / "local.yaml"
    cfg.write_text("robot:\n  gripper_open_ticks: 2700\n", encoding="utf-8")
    monkeypatch.setenv(CONFIG_ENV, str(cfg))
    assert load_config().robot.gripper_open_ticks == 2700
    # Jawna sciezka wygrywa ze zmienna.
    assert load_config(REPO_ROOT / "configs" / "default.yaml").robot.gripper_open_ticks == 2670


def test_missing_config_file_from_environment_is_an_error(monkeypatch, tmp_path):
    monkeypatch.setenv(CONFIG_ENV, str(tmp_path / "brak.yaml"))
    with pytest.raises(FileNotFoundError, match=CONFIG_ENV):
        load_config()


def test_default_yaml_documents_the_gripper_ticks():
    text = (REPO_ROOT / "configs" / "default.yaml").read_text(encoding="utf-8")
    for key in ("center_ticks", "gripper_closed_ticks", "gripper_open_ticks", "baudrate"):
        assert re.search(rf"^\s+{key}:", text, re.M), key


# ------------------------------------------------------------ lerobot-twin


def test_twin_config_option_sets_the_environment_in_both_positions(monkeypatch, tmp_path):
    from lerobot_mp.twin import cli

    cfg = tmp_path / "local.yaml"
    cfg.write_text("robot:\n  center_ticks: 2050\n", encoding="utf-8")
    seen = []
    for argv in (["--config", str(cfg), "policies"], ["policies", "--config", str(cfg)]):
        monkeypatch.delenv(CONFIG_ENV, raising=False)
        monkeypatch.setattr(cli, "_policies", lambda a: seen.append(load_config().robot.center_ticks) or 0)
        # set_defaults(fn=...) trzyma funkcje z chwili budowy parsera, wiec podmieniamy przed main().
        assert cli.main(argv) == 0
    assert seen == [2050, 2050]


def test_twin_config_option_refuses_unknown_keys(monkeypatch, tmp_path):
    from lerobot_mp.twin import cli

    monkeypatch.delenv(CONFIG_ENV, raising=False)
    bad = tmp_path / "bad.yaml"
    bad.write_text("robot:\n  bogus: 1\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="bogus"):
        cli.main(["--config", str(bad), "policies"])


def test_demo_copies_the_example_and_refuses_to_overwrite(tmp_path, capsys):
    pytest.importorskip("mujoco")
    import numpy as np

    from lerobot_mp.twin.cli import main
    from lerobot_mp.twin.workspace import Workspace

    target = tmp_path / "ws" / "twin.json"
    assert main(["demo", "--path", str(target)]) == 0
    ws = Workspace.load(target)
    assert ws.backend == "sim" and len(ws.cameras) == 2
    for cam in ws.cameras:
        # Przyklad nie moze klamac: "zaufana" kalibracja = prawdziwa poza w scenie.
        assert cam.source == "sim" and cam.trusted
        assert np.allclose(cam.T_cam2base, cam.sim_pose)

    target.write_text("{}", encoding="utf-8")
    assert main(["demo", "--path", str(target)]) == 1
    assert target.read_text(encoding="utf-8") == "{}"
    assert "nie nadpisuje" in capsys.readouterr().err
    assert main(["demo", "--path", str(target), "--force"]) == 0
    assert len(Workspace.load(target).cameras) == 2


def test_example_workspace_is_not_shipped_as_the_live_one():
    assert (REPO_ROOT / "examples" / "twin.sim.json").is_file()
    ignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert re.search(r"^workspace/$", ignore, re.M)
    assert re.search(r"^\.claude/$", ignore, re.M)


def _offline(monkeypatch):
    import urllib.request

    def fail(*_a, **_k):
        raise OSError("brak sieci")

    monkeypatch.setattr(urllib.request, "urlopen", fail)


def test_failed_model_download_leaves_no_folder_and_gives_a_working_command(monkeypatch, tmp_path):
    import os

    from lerobot_mp.vision.tracker import ModelDownloadError, download_model

    _offline(monkeypatch)
    target = tmp_path / "nowy" / "hand.task"
    with pytest.raises(ModelDownloadError) as err:
        download_model(str(target), "https://example.invalid/hand.task", "dloni")
    assert not target.parent.exists()
    msg = str(err.value)
    # W PowerShell 5.1 `curl` to alias Invoke-WebRequest i nie zna -L.
    if os.name == "nt":
        assert "curl.exe -L -o" in msg and "Invoke-WebRequest" in msg
    else:
        assert "curl -L -o" in msg


def test_missing_model_is_not_reported_as_missing_mediapipe(monkeypatch):
    pytest.importorskip("mediapipe.tasks.python")
    from lerobot_mp.vision import tracker

    def no_model(_cfg):
        raise tracker.ModelDownloadError("Nie udalo sie pobrac modelu dloni")

    monkeypatch.setattr(tracker, "ensure_model", no_model)
    with pytest.raises(tracker.ModelDownloadError) as err:
        tracker.HandTracker._make_backend(TrackerConfig(backend="auto"))
    assert "pip install" not in str(err.value)


def test_ui_banner_names_the_workspace_file_from_any_directory(monkeypatch, tmp_path):
    from lerobot_mp.twin.cli import _ui_banner

    monkeypatch.chdir(tmp_path)
    text = _ui_banner(None)
    assert str((REPO_ROOT / "workspace" / "twin.json").resolve()) in text
    assert str((REPO_ROOT / "workspace" / "policies").resolve()) in text
    own = tmp_path / "moje.json"
    own.write_text("{}", encoding="utf-8")
    assert f"{own.resolve()} (istnieje)" in _ui_banner("moje.json")
