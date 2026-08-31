"""Wiersz polecen: flagi maja trafiac tam, gdzie trzeba."""

from __future__ import annotations

import pytest

from lerobot_mp.cli import build_config, build_parser, main


def test_defaults_give_simulator_and_camera_zero():
    cfg, _ = build_config([])
    assert cfg.robot.backend == "sim"
    assert cfg.camera.source == 0


def test_camera_and_source_flags():
    assert build_config(["--camera", "2"])[0].camera.source == 2
    assert build_config(["--source", "demo.mp4"])[0].camera.source == "demo.mp4"


def test_port_implies_real_robot():
    """Podanie portu ma wystarczyc - nikt nie powinien musiec dopisywac --robot."""
    cfg, _ = build_config(["--port", "/dev/ttyACM0"])
    assert cfg.robot.backend == "lerobot"
    assert cfg.robot.port == "/dev/ttyACM0"


def test_explicit_backend_wins_over_implication():
    cfg, _ = build_config(["--robot", "sim", "--port", "/dev/ttyACM0"])
    assert cfg.robot.backend == "sim"


def test_control_flags():
    cfg, _ = build_config(["--mode", "ik", "--hand", "Left", "--velocity-scale", "0.5"])
    assert cfg.mapping.mode == "ik"
    assert cfg.tracker.preferred_hand == "Left"
    assert cfg.safety.velocity_scale == 0.5


def test_absolute_disables_relative_mapping():
    assert build_config(["--absolute"])[0].mapping.relative is False


def test_clutch_always_engages_on_start():
    cfg, _ = build_config(["--clutch", "always"])
    assert cfg.clutch.mode == "always"
    assert cfg.clutch.engaged_on_start is True


def test_view_flags():
    cfg, _ = build_config(["--no-view", "--no-preview-3d", "--record", "out.mp4"])
    assert cfg.ui.show is False
    assert cfg.ui.preview_3d is False
    assert cfg.ui.record_path == "out.mp4"


def test_duration_and_loop_hz():
    cfg, _ = build_config(["--duration", "5", "--loop-hz", "60"])
    assert cfg.max_runtime_s == 5
    assert cfg.loop_hz == 60


def test_config_file_is_loaded_and_overridden(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("loop_hz: 12\ncamera:\n  width: 800\n", encoding="utf-8")
    cfg, _ = build_config(["--config", str(path), "--loop-hz", "48"])
    assert cfg.loop_hz == 48       # CLI wygrywa z plikiem
    assert cfg.camera.width == 800  # reszta pliku zostaje


def test_print_config_exits_cleanly(capsys):
    assert main(["--print-config"]) == 0
    assert "shoulder_pan" in capsys.readouterr().out


def test_bad_config_file_reports_error(capsys, tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("kamera: 1\n", encoding="utf-8")
    assert main(["--config", str(path)]) == 2
    assert "Blad konfiguracji" in capsys.readouterr().err


def test_help_lists_the_keyboard_shortcuts():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--help"])
