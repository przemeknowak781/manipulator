"""Backendy robota: symulator, fabryka, adapter LeRobot."""

from __future__ import annotations

import pytest

from lerobot_mp.config import JOINT_NAMES, load_config
from lerobot_mp.robot import create_backend
from lerobot_mp.robot.sim import SimulatedArm

DT = 1.0 / 30.0


def test_simulator_starts_at_home(cfg):
    arm = SimulatedArm(cfg)
    arm.connect()
    assert arm.is_connected
    for name in JOINT_NAMES:
        assert arm.read_joints()[name] == pytest.approx(cfg.safety.home[name], abs=1e-6)


def test_simulator_converges_to_target(cfg):
    arm = SimulatedArm(cfg)
    arm.connect()
    arm.send_joints({"shoulder_pan": 40.0})
    for _ in range(120):
        arm.step(DT)
    assert arm.read_joints()["shoulder_pan"] == pytest.approx(40.0, abs=0.5)


def test_simulator_respects_velocity_limit(cfg):
    arm = SimulatedArm(cfg)
    arm.connect()
    before = arm.read_joints()["shoulder_pan"]
    arm.send_joints({"shoulder_pan": 100.0})
    arm.step(DT)
    step = abs(arm.read_joints()["shoulder_pan"] - before)
    assert step <= cfg.joint("shoulder_pan").max_vel * DT + 1e-9


def test_simulator_clamps_to_limits(cfg):
    arm = SimulatedArm(cfg)
    arm.connect()
    sent = arm.send_joints({"shoulder_pan": 9000.0})
    assert sent["shoulder_pan"] == cfg.joint("shoulder_pan").max


def test_simulator_ignores_unknown_joints(cfg):
    arm = SimulatedArm(cfg)
    arm.connect()
    assert "tentacle" not in arm.send_joints({"tentacle": 10.0})


def test_simulator_zero_dt_is_a_noop(cfg):
    arm = SimulatedArm(cfg)
    arm.connect()
    arm.send_joints({"shoulder_pan": 50.0})
    before = arm.read_joints()
    arm.step(0.0)
    assert arm.read_joints() == before


def test_context_manager_connects_and_disconnects(cfg):
    with SimulatedArm(cfg) as arm:
        assert arm.is_connected
    assert not arm.is_connected


def test_factory_returns_simulator_by_default(cfg):
    assert isinstance(create_backend(cfg), SimulatedArm)


def test_factory_auto_without_port_uses_simulator():
    cfg = load_config(overrides={"robot": {"backend": "auto"}})
    assert isinstance(create_backend(cfg), SimulatedArm)


def test_factory_rejects_unknown_backend():
    cfg = load_config(overrides={"robot": {"backend": "quantum"}})
    with pytest.raises(ValueError, match="Nieznany backend"):
        create_backend(cfg)


def test_lerobot_backend_reports_missing_library_clearly():
    """Bez zainstalowanego LeRobot komunikat ma mowic, co zainstalowac."""
    from lerobot_mp.robot.lerobot_backend import LeRobotArm

    pytest.importorskip  # nie wymagamy lerobot - sprawdzamy sciezke bledu
    try:
        import lerobot  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="pip install"):
            LeRobotArm._resolve_classes("so101")


def test_lerobot_backend_requires_a_port():
    from lerobot_mp.robot.lerobot_backend import LeRobotArm

    cfg = load_config(overrides={"robot": {"backend": "lerobot", "kind": "so101"}})
    arm = LeRobotArm(cfg)
    with pytest.raises(ValueError, match="port"):
        arm._build_config(_FakeConfig)


def test_lerobot_backend_rejects_unknown_arm():
    cfg = load_config(overrides={"robot": {"kind": "so999"}})
    from lerobot_mp.robot.lerobot_backend import LeRobotArm

    with pytest.raises(ValueError, match="Nieznany typ ramienia"):
        LeRobotArm(cfg)


def test_lerobot_backend_skips_fields_the_installed_version_lacks():
    """Starsze wydania LeRobot nie maja `use_degrees` - nie wolno sie wywrocic."""
    import dataclasses

    from lerobot_mp.robot.lerobot_backend import LeRobotArm

    @dataclasses.dataclass
    class OldConfig:
        port: str
        id: str | None = None

    cfg = load_config(overrides={"robot": {"backend": "lerobot", "port": "/dev/null"}})
    built = LeRobotArm(cfg)._build_config(OldConfig)
    assert built.port == "/dev/null"


import dataclasses  # noqa: E402


@dataclasses.dataclass
class _FakeConfig:
    port: str
    id: str | None = None
    use_degrees: bool = True
    max_relative_target: float | None = None
