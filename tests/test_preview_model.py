"""Model 3D prawdziwego zlozenia SO-101.

Najwazniejszy test w tym pliku porownuje NASZA kinematyke z transformacjami
referencyjnymi zapisanymi przez Articulusa przy eksporcie. Bez niego zgodnosc
podgladu ze zrodlem bylaby opinia, a nie faktem.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from lerobot_mp.config import JOINT_NAMES
from lerobot_mp.preview.model import DEFAULT_ASSET, ArmModel, load_model
from lerobot_mp.preview.render import Camera, Renderer3D

ASSET = Path(DEFAULT_ASSET)
pytestmark = pytest.mark.skipif(
    not ASSET.is_file(), reason=f"brak modelu 3D ({ASSET}) - zobacz scripts/import_articulus_model.py"
)


@pytest.fixture(scope="module")
def model() -> ArmModel:
    return ArmModel.load(ASSET)


def test_model_has_the_lerobot_joints(model):
    assert set(model.dof_names) == set(JOINT_NAMES)


def test_meshes_are_consistent(model):
    assert len(model.vertices) > 1000
    assert model.faces.min() >= 0
    assert model.faces.max() < len(model.vertices)
    assert len(model.face_link) == len(model.faces)
    assert model.link_colors.shape == (len(model.link_names), 3)


def test_model_is_in_metres(model):
    """Siatki przychodza z Articulusa w milimetrach - musza byc przeliczone."""
    extent = float(np.abs(model.vertices).max())
    assert 0.02 < extent < 1.0


def test_kinematics_matches_the_articulus_reference(model):
    """Kazdy czlon w kazdej nazwanej pozie - blad ponizej mikrometra.

    Transformacje referencyjne policzyl Articulus przy eksporcie. Jesli ten
    test padnie, to podglad przestal pokazywac to samo, co model zrodlowy.
    """
    assert model.reference, "eksport bez transformacji referencyjnych"

    worst_translation = worst_rotation = 0.0
    for pose in model.reference.values():
        transforms = model.link_transforms(pose["values"])
        for link, flat in pose["links"].items():
            if link not in model.link_names:
                continue
            expected = np.array(flat, dtype=float).reshape(4, 4)
            expected[:3, 3] /= 1000.0  # referencja jest w milimetrach
            actual = transforms[model.link_names.index(link)]
            worst_translation = max(
                worst_translation, float(np.linalg.norm(actual[:3, 3] - expected[:3, 3]))
            )
            worst_rotation = max(worst_rotation, float(np.abs(actual[:3, :3] - expected[:3, :3]).max()))

    assert worst_translation < 1e-6, f"rozjazd polozenia {worst_translation * 1000:.4f} mm"
    assert worst_rotation < 1e-6, f"rozjazd obrotu {worst_rotation:.2e}"


def test_joints_actually_move_the_arm(model):
    zero = {name: 0.0 for name in model.dof_names}
    base = model.posed_vertices(zero)
    for joint in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex"):
        moved = model.posed_vertices(dict(zero, **{joint: 30.0}))
        assert np.abs(moved - base).max() > 0.01, f"{joint} nie rusza modelu"


def test_gripper_is_rescaled_from_lerobot_units(model):
    """LeRobot normalizuje chwytak do 0..100, model ma prawdziwy zakres katowy."""
    low, high = model.dof_min[model.dof_names.index("gripper")], model.dof_max[
        model.dof_names.index("gripper")
    ]
    assert model.from_lerobot({"gripper": 0.0})["gripper"] == pytest.approx(low)
    assert model.from_lerobot({"gripper": 100.0})["gripper"] == pytest.approx(high)
    assert model.from_lerobot({"gripper": 50.0})["gripper"] == pytest.approx((low + high) / 2)


def test_body_joints_pass_through_in_degrees(model):
    assert model.from_lerobot({"shoulder_pan": 42.0})["shoulder_pan"] == pytest.approx(42.0)


def test_from_lerobot_clamps_to_model_limits(model):
    assert model.from_lerobot({"shoulder_pan": 999.0})["shoulder_pan"] == pytest.approx(
        model.dof_max[model.dof_names.index("shoulder_pan")]
    )


def test_from_lerobot_ignores_unknown_joints(model):
    assert model.from_lerobot({"tentacle": 1.0}) == {}


def test_topological_order_is_validated(model):
    """Rodzic musi wystepowac przed dzieckiem - inaczej pozy byly by bledne."""
    broken = ArmModel.load(ASSET)
    broken.chain_parent = broken.chain_parent[::-1].copy()
    broken.chain_link = broken.chain_link[::-1].copy()
    with pytest.raises(ValueError, match="topologicznie"):
        broken._validate()


def test_missing_asset_falls_back_quietly(tmp_path):
    assert load_model(tmp_path / "nie-ma.npz") is None


def test_renderer_draws_something(model):
    renderer = Renderer3D(model, size=(200, 260))
    image = renderer.render({name: 0.0 for name in model.dof_names})
    assert image.shape == (260, 200, 3)
    assert len(np.unique(image.reshape(-1, 3), axis=0)) > 10  # nie samo tlo


def test_renderer_reacts_to_pose(model):
    renderer = Renderer3D(model, size=(200, 260))
    zero = {name: 0.0 for name in model.dof_names}
    folded = dict(zero, shoulder_lift=-90.0, elbow_flex=90.0)
    assert not np.array_equal(renderer.render(zero), renderer.render(folded))


def test_camera_orbit_and_zoom_are_bounded():
    camera = Camera()
    camera.orbit(0.0, 500.0)
    assert camera.elevation_deg <= 85.0
    camera.orbit(0.0, -1000.0)
    assert camera.elevation_deg >= -85.0
    for _ in range(50):
        camera.zoom(0.5)
    assert camera.distance >= 0.18
    for _ in range(50):
        camera.zoom(2.0)
    assert camera.distance <= 2.0


def test_camera_matrix_is_orthonormal():
    view = Camera().view_matrix()
    rotation = view[:3, :3]
    assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-9)


def test_source_records_provenance(model):
    assert "articulus" in json.dumps(model.source).lower()
