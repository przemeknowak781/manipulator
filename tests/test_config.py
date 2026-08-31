"""Wczytywanie i scalanie konfiguracji."""

from __future__ import annotations

import pytest

from lerobot_mp.config import AppConfig, JOINT_NAMES, load_config


def test_defaults_cover_all_joints():
    cfg = load_config()
    assert set(cfg.joints) == set(JOINT_NAMES)


def test_yaml_file_matches_defaults():
    """`configs/default.yaml` ma dokumentowac stan domyslny, a nie od niego odbiegac."""
    import dataclasses

    from_file = dataclasses.asdict(load_config("configs/default.yaml"))
    defaults = dataclasses.asdict(load_config())
    assert from_file == defaults


def test_nested_sections_are_typed(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("camera:\n  source: 2\n  mirror: false\n", encoding="utf-8")
    cfg = load_config(path)
    assert cfg.camera.source == 2
    assert cfg.camera.mirror is False
    assert cfg.camera.width == 1280  # nietkniete pole zachowuje wartosc domyslna


def test_unknown_key_is_an_error():
    with pytest.raises(ValueError, match="Nieznane klucze"):
        load_config(overrides={"camera": {"zoom": 3}})


def test_joint_override_merges_field_by_field():
    cfg = load_config(overrides={"joints": {"gripper": {"max_vel": 42.0}}})
    assert cfg.joint("gripper").max_vel == 42.0
    assert cfg.joint("gripper").max == 100.0


def test_unknown_joint_is_an_error():
    with pytest.raises(ValueError, match="Nieznany staw"):
        load_config(overrides={"joints": {"knee": {"min": 0}}})


def test_cli_overrides_win_over_file(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("loop_hz: 10\n", encoding="utf-8")
    assert load_config(path, {"loop_hz": 60}).loop_hz == 60


def test_config_is_a_dataclass():
    assert isinstance(load_config(), AppConfig)
