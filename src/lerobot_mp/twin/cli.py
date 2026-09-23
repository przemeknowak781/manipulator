"""Wiersz polecen blizniaka: `lerobot-twin <polecenie>`.

    lerobot-twin check                       # czy srodowisko jest gotowe (bez renderu)
    lerobot-twin card --tag-mm 50            # arkusz karty kalibracyjnej do druku
    lerobot-twin calib-sim --n 5 --cameras 2 # kalibracja w symulacji, oceniana wzgledem prawdy
    lerobot-twin workspace                   # co wiadomo o stanowisku (kamery, kalibracja)
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
from pathlib import Path


def _check(_: argparse.Namespace) -> int:
    """Sprawdza srodowisko bez renderowania czegokolwiek."""
    import numpy as np

    ok = True

    def line(label: str, value: str, good: bool = True) -> None:
        nonlocal ok
        ok &= good
        print(f"  {'OK ' if good else 'BLAD'}  {label:<28} {value}")

    print("lerobot-twin: sprawdzenie srodowiska\n")
    line("Python", f"{sys.version.split()[0]} ({platform.machine()}, {platform.system()})")
    for mod in ("mujoco", "gymnasium", "viser", "cv2", "numpy"):
        try:
            m = __import__(mod)
            line(mod, getattr(m, "__version__", "?"))
        except ImportError as exc:
            line(mod, f"brak ({exc}) - pip install -e \".[twin]\"", False)
    line("MUJOCO_GL", os.environ.get("MUJOCO_GL", "(domyslny: glfw; bez ekranu ustaw egl)"))

    try:
        from .kinematics import RobotKinematics
        from .robots import SO101

        kin = RobotKinematics(SO101)
        line("model SO-101 (menagerie)", f"{kin.model.nbody} cial, {int(kin.model.mesh_facenum.sum())} trojkatow")
        from ..preview.model import load_model

        art = load_model()
        if art is not None:
            base = art.link_names.index("base")
            rng = np.random.default_rng(0)
            worst = 0.0
            for _ in range(20):
                j = {n: float(rng.uniform(np.degrees(lo), np.degrees(hi)))
                     for n, lo, hi in zip(SO101.joints, kin.lo, kin.hi)}
                Ta = art.link_transforms(art.from_lerobot(j))
                for name in ("upper_arm", "lower_arm", "wrist", "gripper"):
                    ref = (np.linalg.inv(Ta[base]) @ Ta[art.link_names.index(name)])[:3, 3]
                    worst = max(worst, float(np.linalg.norm(ref - kin.body(name, j)[:3, 3])))
            line("zgodnosc z Articulusem", f"{worst * 1e6:.1f} um (musi byc < 10 um)", worst < 1e-5)
    except Exception as exc:
        line("model SO-101", f"{type(exc).__name__}: {exc}", False)

    try:
        import serial.tools.list_ports as lp

        ports = [p for p in lp.comports() if p.vid is not None]
        text = ", ".join(f"{p.device} ({p.description})" for p in ports) or "brak przejsciowek USB-serial"
        line("porty USB-serial", text)
    except ImportError:
        line("pyserial", "brak - potrzebny do prawdziwego ramienia", False)

    print("\nWszystko gotowe." if ok else "\nSa bledy - patrz wyzej.")
    return 0 if ok else 1


def _card(a: argparse.Namespace) -> int:
    import cv2

    from .calib.card import Card

    card = Card(tag_size=a.tag_mm / 1000.0)
    sheet = card.sheet(dpi=a.dpi)
    out = Path(a.out)
    cv2.imwrite(str(out), sheet[..., ::-1])
    print(f"Zapisano {out} ({sheet.shape[1]} x {sheet.shape[0]} px, {a.dpi} dpi).")
    print(f"Drukuj w skali 100%. Bok czarnego kwadratu ma miec {a.tag_mm:.1f} mm - ZMIERZ go po wydruku.")
    return 0


def _calib_sim(a: argparse.Namespace) -> int:
    from .calib import simulate

    argv = ["--n", str(a.n), "--cameras", str(a.cameras), "--seed", str(a.seed)]
    if a.extreme:
        argv.append("--extreme")
    if a.verbose:
        argv.append("-v")
    return simulate.main(argv)


def _workspace(a: argparse.Namespace) -> int:
    from .workspace import Workspace

    ws = Workspace.load(a.path)
    exists = ws.path is not None and Path(ws.path).is_file()
    print(f"Stanowisko: {ws.path} {'' if exists else '(jeszcze nie zapisane - wartosci domyslne)'}")
    print(f"  ramie: {ws.robot}, backend: {ws.backend}, port: {ws.port or '-'}")
    print(f"  karta: tag {ws.card_obj().tag_size * 1000:.1f} mm")
    if not ws.cameras:
        print("  kamery: brak")
    for cam in ws.cameras:
        cal = cam.calibration
        state = ("zaufana" if cam.trusted else f"NIEZAUFANA ({cal.get('reason', '')})") if cam.calibrated \
            else "nieskalibrowana"
        rms = f", residuum {cal['rms_px']:.2f} px" if "rms_px" in cal else ""
        print(f"  kamera {cam.name}: zrodlo {cam.source}, {cam.width}x{cam.height}, "
              f"intrynsyki {cam.intrinsics_from}, {state}{rms}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="lerobot-twin", description="Cyfrowy blizniak stanowiska SO-101.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("check", help="sprawdz srodowisko (bez renderu)").set_defaults(fn=_check)

    p = sub.add_parser("card", help="arkusz karty kalibracyjnej do druku")
    p.add_argument("--out", default="karta_kalibracyjna.png")
    p.add_argument("--tag-mm", type=float, default=50.0, help="bok czarnego kwadratu taga [mm]")
    p.add_argument("--dpi", type=int, default=300)
    p.set_defaults(fn=_card)

    p = sub.add_parser("calib-sim", help="kalibracja w symulacji wzgledem prawdy (renderuje!)")
    p.add_argument("--n", type=int, default=5, help="ile losowych stanowisk")
    p.add_argument("--cameras", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--extreme", action="store_true", help="karta na granicy przekrzywienia")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(fn=_calib_sim)

    p = sub.add_parser("workspace", help="co wiadomo o stanowisku")
    p.add_argument("--path", default=None)
    p.set_defaults(fn=_workspace)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
