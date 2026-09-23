"""Kalibracja w symulacji, oceniana wzgledem prawdy - odpowiednik `calibrate.py --sim` z galaxeo.

    python -m lerobot_mp.twin.calib.simulate --n 10 --cameras 2

Kazde ziarno losuje stanowisko: gdzie stoja kamery, jakie maja intrynsyki i jak
krzywo ktos wlozyl karte w szczeki. Sesja widzi tylko to, co widzialaby na
prawdziwym stanowisku - kadry, zmierzone katy stawow, K kamer i NOMINALNA poze
karty. Prawdziwe pozy kamer i karty sluza wylacznie do oceny wyniku.

Ramie i kamery siedza tu za tymi samymi protokolami (`session.Robot`,
`session.Cameras`), co prawdziwy sprzet - wiec to jest test tej samej sesji,
ktora pojedzie na biurku, a nie jej kopii.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass

import mujoco
import numpy as np

from .. import scene as sc
from ..collision import CollisionChecker
from ..kinematics import pose
from ..robots import SO101, RobotSpec
from .card import Card, perturb, pinch_point
from .handeye import pose_error
from .session import Session, WaveConfig


def smoothstep(s: float) -> float:
    s = min(max(s, 0.0), 1.0)
    return s * s * (3 - 2 * s)


class SimRobot:
    """Ramie w scenie: rampa smoothstep na serwa pozycyjne, potem ustalenie - jak w galaxeo."""

    def __init__(self, scene: sc.Scene, settle: float = 0.4):
        self.scene, self.settle = scene, settle

    def joints(self) -> dict[str, float]:
        return self.scene.joints()

    def move(self, joints: Mapping[str, float], duration: float) -> None:
        s = self.scene
        q0 = s.data.ctrl[s.act_ids].copy()
        q1 = s.kin.to_q(joints)
        dt = s.model.opt.timestep
        n = max(1, int(round(duration / dt)))
        for k in range(0, n, 4):
            s.data.ctrl[s.act_ids] = q0 + smoothstep((k + 4) / n) * (q1 - q0)
            mujoco.mj_step(s.model, s.data, nstep=4)
        s.data.ctrl[s.act_ids] = q1
        s.step(self.settle)


class SimCameras:
    def __init__(self, scene: sc.Scene):
        self.scene = scene

    def grab(self) -> dict[str, np.ndarray]:
        return {v.name: self.scene.render(v.name) for v in self.scene.cfg.cameras}


def look_at(eye, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Poza kamery OpenCV (z do przodu, y w dol) patrzacej z `eye` na `target`."""
    z = np.asarray(target, float) - np.asarray(eye, float)
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    return pose(np.column_stack([x, np.cross(z, x), z]), np.asarray(eye, float))


#: Gdzie losujemy kamery - w skali SO-101 na biurku (galaxeo: laptop przed A1X).
CAMERAS = dict(W=640, H=480, fovy=(46.0, 60.0), dist=(0.45, 0.75), bearing=(-1.7, 1.7),
               height=(0.12, 0.40), centre=(0.20, 0.0, 0.06), jitter=0.04, spread=0.9)


def sample_cameras(rng: np.random.Generator, n: int) -> list[sc.CameraView]:
    """N kamer w przedniej polplaszczyznie, rozstawionych co najmniej `spread` rad."""
    cfg = CAMERAS
    centre = np.asarray(cfg["centre"], float)
    bearings: list[float] = []
    while len(bearings) < n:
        b = float(rng.uniform(*cfg["bearing"]))
        if all(abs(b - o) >= cfg["spread"] for o in bearings):
            bearings.append(b)
    views = []
    for k, b in enumerate(bearings):
        dist, h = rng.uniform(*cfg["dist"]), rng.uniform(*cfg["height"])
        eye = centre + np.array([dist * np.cos(b), dist * np.sin(b), 0.0])
        eye[2] = h
        target = centre + rng.uniform(-cfg["jitter"], cfg["jitter"], 3)
        W, H = cfg["W"], cfg["H"]
        f = (H / 2) / np.tan(np.radians(rng.uniform(*cfg["fovy"])) / 2)
        # Prawdziwa kamera nie ma punktu glownego idealnie w srodku ani fx == fy.
        K = np.array([[f * rng.uniform(0.99, 1.01), 0.0, (W - 1) / 2 + rng.uniform(-8, 8)],
                      [0.0, f, (H - 1) / 2 + rng.uniform(-8, 8)], [0.0, 0.0, 1.0]])
        views.append(sc.CameraView(f"cam{k}", K, W, H, look_at(eye, target)))
    return views


@dataclass
class Result:
    seed: int
    poses: int
    trusted: dict[str, bool]
    reasons: dict[str, str]
    rms_px: dict[str, float]
    err_mm: dict[str, float]
    err_deg: dict[str, float]
    card_mm: float
    card_deg: float


def run(seed: int, n_cameras: int = 1, spec: RobotSpec = SO101, card: Card | None = None,
        extreme: bool = False, cfg: WaveConfig | None = None, on_step=None) -> Result:
    rng = np.random.default_rng(seed)
    card = card or Card()
    views = sample_cameras(rng, n_cameras)

    from ..kinematics import RobotKinematics

    kin_free = RobotKinematics(spec)
    nominal = card.nominal(pinch_point(kin_free))
    truth = perturb(nominal, rng, extreme=extreme)     # tam, gdzie karta NAPRAWDE siedzi

    world = sc.build(sc.SceneConfig(spec, cameras=views, card=card, card_pose=truth))
    # Sprawdzacz widzi karte tam, gdzie mysli, ze jest - nominalnie - z zapasem.
    check_scene = sc.build(sc.SceneConfig(spec, card=card, card_pose=nominal, card_collider=0.012))
    try:
        home = dict(spec.home)
        if spec.gripper:
            home[spec.gripper] = 0.0
        world.set_joints(home)
        session = Session(SimRobot(world), SimCameras(world), {v.name: (v.K, None) for v in views},
                          world.kin, card, nominal, CollisionChecker(check_scene), cfg, seed=seed)
        fit = session.run(on_step)
    finally:
        world.close()
        check_scene.close()

    err_mm, err_deg, rms, trusted, reasons = {}, {}, {}, {}, {}
    for v in views:
        cam = fit.cameras.get(v.name)
        if cam is None:
            trusted[v.name], reasons[v.name] = False, "kamera nie zobaczyla karty"
            err_mm[v.name] = err_deg[v.name] = rms[v.name] = float("nan")
            continue
        dt, dr = pose_error(v.T_cam2base, cam.T_cam2base)
        err_mm[v.name], err_deg[v.name], rms[v.name] = dt * 1000, np.degrees(dr), cam.rms_px
        trusted[v.name], reasons[v.name] = cam.trusted, cam.reason
    ct, cr = pose_error(truth, fit.mounts["card"])
    return Result(seed, session.index, trusted, reasons, rms, err_mm, err_deg, ct * 1000, np.degrees(cr))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Kalibracja w symulacji, oceniana wzgledem prawdy.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=5, help="ile losowych stanowisk")
    ap.add_argument("--cameras", type=int, default=1, help="ile kamer na stanowisku")
    ap.add_argument("--extreme", action="store_true", help="karta na granicy przekrzywienia")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)

    rows = []
    for seed in range(a.seed, a.seed + a.n):
        def show(r):
            if a.verbose:
                prog = "  ".join(f"{c}: {o} obs {s:.0f}st" for c, (o, _, s) in r.progress.items())
                print(f"    poza {r.index:2d}  {prog}  {r.note}")
        res = run(seed, a.cameras, extreme=a.extreme, on_step=show)
        rows.append(res)
        for cam in sorted(res.trusted):
            flag = "" if res.trusted[cam] else f"  NIEZAUFANA ({res.reasons[cam]})"
            print(f"ziarno {seed:3d} {cam}: pozy={res.poses:2d} rms={res.rms_px[cam]:.2f}px  "
                  f"blad={res.err_mm[cam]:.2f}mm {res.err_deg[cam]:.3f}st  "
                  f"karta={res.card_mm:.2f}mm {res.card_deg:.2f}st{flag}")

    ok = [(r, c) for r in rows for c in r.trusted if r.trusted[c]]
    total = sum(len(r.trusted) for r in rows)
    print(f"\nzaufane kamery: {len(ok)}/{total}")
    if ok:
        mm = np.array([r.err_mm[c] for r, c in ok])
        deg = np.array([r.err_deg[c] for r, c in ok])
        px = np.array([r.rms_px[c] for r, c in ok])
        for name, v, unit in (("przesuniecie", mm, "mm"), ("obrot", deg, "st"), ("residuum", px, "px")):
            print(f"{name:>13}: mediana {np.median(v):7.3f} {unit}  najgorzej {v.max():7.3f} {unit}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
