#!/usr/bin/env python3
"""Wyprowadza stale geometryczne SO-101 z prawdziwego zlozenia - i je sprawdza.

Uproszczony model plaski z `lerobot_mp/control/kinematics.py` (obrot podstawy
+ trzy ogniwa w plaszczyznie pionowej) jest szybki i odwracalny analitycznie,
ale jest *uproszczeniem*. Ten skrypt robi dwie rzeczy:

1. **Wyprowadza** jego stale z modelu 3D zaimportowanego z Articulusa
   (`assets/so101_preview.npz`) - dlugosci ogniw, wysokosc barku, przesuniecia
   i znaki stawow. Zadna z tych liczb nie jest przepisana z oka.
2. **Mierzy**, jak bardzo model plaski rozjezdza sie z pelna kinematyka na
   siatce poz. Uproszczenie, ktorego bledu nikt nie zmierzyl, jest zalozeniem.

Uzycie:
    python scripts/derive_geometry.py                 # wyprowadz i sprawdz
    python scripts/derive_geometry.py --check         # tylko sprawdz obecne stale
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lerobot_mp.config import ArmGeometryConfig, load_config  # noqa: E402
from lerobot_mp.control.kinematics import ArmKinematics  # noqa: E402
from lerobot_mp.preview.model import ArmModel, DEFAULT_ASSET  # noqa: E402

#: Ile procent najdalszych wierzcholkow chwytaka usredniamy jako punkt narzedzia.
#: Pojedynczy "najdalszy wierzcholek" zalezalby od gestosci uproszczonej siatki.
TIP_PERCENTILE = 99.0


def axis_frames(model: ArmModel, pose: dict[str, float]) -> dict[str, np.ndarray]:
    """Poczatki ramek wszystkich stawow w ukladzie swiata dla zadanej pozy."""
    transforms = model.link_transforms(pose)
    frames: dict[str, np.ndarray] = {}
    for step in range(len(model.chain_link)):
        joint = model.chain_joint[step]
        if not joint:
            continue
        parent = int(model.chain_parent[step])
        base = transforms[parent] if parent >= 0 else np.eye(4)
        frames[joint] = (base @ model.chain_pre[step])[:3, 3]
    return frames


def tool_point_local(model: ArmModel) -> np.ndarray:
    """Punkt narzedzia w ukladzie LOKALNYM czlonu `gripper`.

    Wyznaczamy go raz, w pozie zerowej, jako koniec nieruchomej szczeki
    (srednia najdalszych wierzcholkow od osi nadgarstka), i od tej pory jest
    to *ten sam punkt bryly*. Szukanie "najdalszego wierzcholka" osobno w
    kazdej pozie dawaloby raz jeden naroznik chwytaka, raz inny - i mierzyloby
    wlasna niestabilnosc zamiast bledu modelu.
    """
    zero = {name: 0.0 for name in model.dof_names}
    transforms = model.link_transforms(zero)
    index = model.link_names.index("gripper")
    matrix = transforms[index]

    local = model.vertices[model.vertex_link == index]
    world = local @ matrix[:3, :3].T + matrix[:3, 3]

    origin = axis_frames(model, zero)["wrist_flex"]
    distances = np.linalg.norm(world - origin, axis=1)
    cutoff = np.percentile(distances, TIP_PERCENTILE)
    tip_world = world[distances >= cutoff].mean(axis=0)
    return matrix[:3, :3].T @ (tip_world - matrix[:3, 3])


def tool_point(model: ArmModel, pose: dict[str, float], local: np.ndarray | None = None) -> np.ndarray:
    """Punkt narzedzia w ukladzie swiata dla zadanej pozy."""
    if local is None:
        local = tool_point_local(model)
    matrix = model.link_transforms(pose)[model.link_names.index("gripper")]
    return matrix[:3, :3] @ local + matrix[:3, 3]


def _plane_angle(a: np.ndarray, b: np.ndarray) -> float:
    """Kat odcinka a->b w plaszczyznie pionowej (X-Z), w stopniach."""
    return math.degrees(math.atan2(b[2] - a[2], b[0] - a[0]))


def derive(model: ArmModel) -> ArmGeometryConfig:
    """Wylicza wszystkie stale modelu plaskiego z modelu 3D."""
    zero = {name: 0.0 for name in model.dof_names}
    frames = axis_frames(model, zero)
    shoulder, elbow, wrist = frames["shoulder_lift"], frames["elbow_flex"], frames["wrist_flex"]
    tip = tool_point(model, zero)

    a1 = _plane_angle(shoulder, elbow)
    a2 = _plane_angle(elbow, wrist)
    a3 = _plane_angle(wrist, tip)

    signs = {}
    for joint, label in (("shoulder_lift", 0), ("elbow_flex", 1), ("wrist_flex", 2)):
        pose = dict(zero)
        pose[joint] = 10.0
        moved = axis_frames(model, pose)
        moved_tip = tool_point(model, pose)
        b1 = _plane_angle(moved["shoulder_lift"], moved["elbow_flex"])
        b2 = _plane_angle(moved["elbow_flex"], moved["wrist_flex"])
        b3 = _plane_angle(moved["wrist_flex"], moved_tip)
        delta = [b1 - a1, (b2 - b1) - (a2 - a1), (b3 - b2) - (a3 - a2)][label]
        signs[joint] = 1.0 if delta > 0 else -1.0

    pose = dict(zero)
    pose["shoulder_pan"] = 20.0
    pan_sign = 1.0 if axis_frames(model, pose)["wrist_flex"][1] > wrist[1] else -1.0

    pan_axis = frames["shoulder_pan"]
    return ArmGeometryConfig(
        base_height=round(float(shoulder[2]), 5),
        pan_axis_x=round(float(pan_axis[0]), 5),
        shoulder_offset=round(float(shoulder[0] - pan_axis[0]), 5),
        # Boczne przesuniecie liczymy DLA NARZEDZIA, nie dla barku: to koncowke
        # ustawiamy w zadanym punkcie, a nadgarstek wraca bokiem prawie na os.
        lateral_offset=round(float(tip[1]), 5),
        upper_arm=round(float(np.linalg.norm(elbow - shoulder)), 5),
        forearm=round(float(np.linalg.norm(wrist - elbow)), 5),
        wrist_to_tip=round(float(np.linalg.norm(tip - wrist)), 5),
        lift_offset_deg=round(a1, 3),
        elbow_offset_deg=round(a2 - a1, 3),
        wrist_offset_deg=round(a3 - a2, 3),
        pan_sign=pan_sign,
        lift_sign=signs["shoulder_lift"],
        elbow_sign=signs["elbow_flex"],
        wrist_sign=signs["wrist_flex"],
    )


def check(model: ArmModel, geometry: ArmGeometryConfig) -> float:
    """Porownuje model plaski z pelna kinematyka; zwraca najgorszy blad [m]."""
    kinematics = ArmKinematics(geometry)
    grid = {
        "shoulder_pan": (-90.0, -45.0, 0.0, 45.0, 90.0),
        "shoulder_lift": (-90.0, -45.0, 0.0, 45.0, 90.0),
        "elbow_flex": (-90.0, -45.0, 0.0, 45.0, 90.0),
        "wrist_flex": (-90.0, -45.0, 0.0, 45.0, 90.0),
    }
    local = tool_point_local(model)
    worst = 0.0
    worst_pose: dict[str, float] = {}
    errors = []
    for values in itertools.product(*grid.values()):
        pose = dict(zip(grid.keys(), values))
        full = {name: 0.0 for name in model.dof_names} | pose
        truth = tool_point(model, full, local)
        approx = np.array(kinematics.forward(**pose))
        error = float(np.linalg.norm(truth - approx))
        errors.append(error)
        if error > worst:
            worst, worst_pose = error, pose
    median = float(np.median(errors))
    print(
        f"  sprawdzono {len(errors)} poz; blad mediana {median * 1000:.2f} mm, "
        f"najgorszy {worst * 1000:.2f} mm przy {worst_pose}"
    )
    return worst


def round_trip(geometry: ArmGeometryConfig) -> float:
    """Sprawdza, czy IK odwraca FK (blad domkniecia) [m]."""
    kinematics = ArmKinematics(geometry)
    worst = 0.0
    for pan, lift, elbow, wrist in itertools.product(
        (-60.0, 0.0, 60.0), (-60.0, 0.0, 60.0), (-60.0, 0.0, 60.0), (-45.0, 0.0, 45.0)
    ):
        target = kinematics.forward(pan, lift, elbow, wrist)
        pitch = kinematics.tool_pitch(lift, elbow, wrist)
        result = kinematics.inverse(*target, pitch)
        if result.clamped:
            continue
        worst = max(worst, float(np.linalg.norm(np.array(result.reached) - np.array(target))))
    print(f"  domkniecie FK->IK->FK: najgorszy blad {worst * 1000:.4f} mm")
    return worst


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--asset", type=Path, default=Path(DEFAULT_ASSET))
    parser.add_argument("--check", action="store_true", help="sprawdz stale z konfiguracji")
    args = parser.parse_args()

    if not args.asset.is_file():
        raise SystemExit(
            f"Brak modelu {args.asset}. Wygeneruj go najpierw:\n"
            "  python scripts/import_articulus_model.py --articulus ../articulus"
        )

    model = ArmModel.load(args.asset)
    print(f"Model: {model.title}  ({model.source.get('origin')})\n")

    if args.check:
        geometry = load_config().geometry
        print("Stale z konfiguracji:")
    else:
        geometry = derive(model)
        print("Stale wyprowadzone z modelu 3D:")

    for field in dataclasses.fields(geometry):
        print(f"  {field.name:<18} {getattr(geometry, field.name)}")

    print("\nModel plaski wobec pelnej kinematyki:")
    check(model, geometry)
    print("\nSpojnosc wewnetrzna:")
    round_trip(geometry)

    if not args.check:
        print("\nDo wklejenia w configs/default.yaml (sekcja `geometry`):")
        for field in dataclasses.fields(geometry):
            print(f"  {field.name}: {getattr(geometry, field.name)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
