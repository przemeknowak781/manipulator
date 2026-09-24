"""Wiersz polecen blizniaka: `lerobot-twin <polecenie>`.

    lerobot-twin check                       # czy srodowisko jest gotowe (bez renderu)
    lerobot-twin card --tag-mm 50            # arkusz karty kalibracyjnej do druku
    lerobot-twin board --square-mm 28        # tablica ChArUco do intrynsyk kamery
    lerobot-twin calib-sim --n 5 --cameras 2 # kalibracja w symulacji, oceniana wzgledem prawdy
    lerobot-twin workspace                   # co wiadomo o stanowisku (kamery, kalibracja)
    lerobot-twin train --task reach          # polityka PPO na GPU (MuJoCo Warp) + ewaluacja na CPU
    lerobot-twin eval <policy.pt> --rand     # ewaluacja polityki w zwyklym MuJoCo
    lerobot-twin policies                    # zapisane polityki i ich wyniki
    lerobot-twin ui                          # panel w przegladarce: http://localhost:8080
    lerobot-twin ui --host 0.0.0.0           # ... dostepny z sieci (bez hasla - tylko w zaufanej)
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


def _board(a: argparse.Namespace) -> int:
    import cv2

    from .calib.intrinsics import Board

    board = Board(square=a.square_mm / 1000.0, marker=0.75 * a.square_mm / 1000.0)
    sheet = board.image(dpi=a.dpi)
    out = Path(a.out)
    cv2.imwrite(str(out), sheet)
    print(f"Zapisano {out} ({sheet.shape[1]} x {sheet.shape[0]} px, {a.dpi} dpi).")
    print(f"Drukuj w skali 100%. Bok kwadratu ma miec {a.square_mm:.1f} mm - ZMIERZ go po wydruku "
          f"i wpisz w panelu (zakladka Kalibracja).")
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


def _train_plan(dyn, no_rand: bool, spread: float):
    """(randomizacja treningu, [(etykieta, klucz w meta.evals, randomizacja ewaluacji)]).

    "Bez randomizacji" to zmierzona dynamika bez rozrzutu - nie model Menagerie
    (`Randomization.none()` gubil identyfikacje i model percepcji kostki, a log
    i tak pisal "dynamika: identyfikacja"). Ewaluacja "z randomizacja" jest tylko
    wtedy, gdy trening ja mial: wczesniej przy --no-rand druga ewaluacja nominalna
    trafiala do `cpu_rand` i panel pokazywal ja jako odpornosc na randomizacje.
    """
    from .rl.randomize import Randomization

    rand = Randomization.around(dyn, 0.0 if no_rand else spread)
    evals = [("bez randomizacji", "cpu_nominal", Randomization.nominal(dyn))]
    if rand.randomized:
        evals.append(("z randomizacja", "cpu_rand", rand))
    return rand, evals


def _describe(rand) -> str:
    c = rand.centre
    delay = f"{rand.min_delay}" if rand.min_delay == rand.max_delay else f"{rand.min_delay}-{rand.max_delay}"
    return (f"dynamika: {c.source} (kp x{c.kp:.2f}, tlumienie x{c.damping:.2f}, opoznienie {delay} takt.); "
            f"randomizacja: {'tak' if rand.randomized else 'brak'}; "
            f"percepcja kostki: {'jak z kamer' if rand.fold_yaw or rand.cube_delay else 'idealna'}")


def _train(a: argparse.Namespace) -> int:
    import json
    import time

    from .rl.evaluate import evaluate
    from .rl.policy import DEFAULT_DIR
    from .rl.ppo import PPOConfig, train
    from .rl.randomize import Dynamics
    from .workspace import Workspace

    ws = Workspace.load(a.workspace)
    name = a.name or f"{a.task}-{time.strftime('%Y%m%d-%H%M%S')}"
    out = Path(a.out or DEFAULT_DIR) / name
    dyn = Dynamics.from_dict(ws.dynamics)
    rand, evals = _train_plan(dyn, a.no_rand, a.spread)
    cfg = PPOConfig(num_envs=a.envs, iterations=a.iters, seed=a.seed)
    init = None
    if a.init:
        from .rl.policy import Policy

        init = Policy.load(a.init)
        a.task = init.task.name
        cfg.init_std = 0.25                            # douczanie: mniejsza eksploracja na starcie
        print(f"Start z polityki {a.init} ({init.task.name})")
    print(f"Trening {a.task}: {a.envs} swiatow x {a.iters} iteracji -> {out}")
    print(f"  {_describe(rand)}{'' if a.no_rand else f', rozrzut x{a.spread}'}")

    def show(p):
        if p.iteration == 1 or p.iteration % 10 == 0 or p.status != "uczenie":
            print(f"  it {p.iteration:4d}/{p.iterations}  {p.steps / 1e6:7.2f} M krokow  {p.fps / 1e3:5.0f} k/s  "
                  f"sukces {p.success:5.1%}  nagroda {p.reward:7.2f}  {p.elapsed:5.0f} s", flush=True)

    stop = None
    if a.stop_file:
        # Panel zatrzymuje trening, tworzac plik - polityka z tego miejsca zostaje zapisana.
        import threading

        stop = threading.Event()
        flag = Path(a.stop_file)

        def watch():
            while not stop.is_set():
                if flag.exists():
                    stop.set()
                time.sleep(0.5)
        threading.Thread(target=watch, daemon=True).start()
    pol = train(a.task, cfg, workspace=ws, randomization=rand, out_dir=out, on_progress=show, stop=stop,
                init=init)
    pol = pol.to("cpu")
    print("Ewaluacja na CPU (zwykle MuJoCo, inny silnik niz w treningu):")
    for label, key, r in evals:
        res = evaluate(pol, a.eval_episodes, randomization=r, workspace=ws)
        pol.meta.evals[key] = res
        print(f"  {label:17s} sukces {res['success']:5.1%}  ({json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in res.items()})})")
    pol.save(out / "policy.pt")
    print(f"Zapisano {out / 'policy.pt'}")
    return 0


def _eval(a: argparse.Namespace) -> int:
    from .rl.evaluate import evaluate
    from .rl.policy import Policy
    from .rl.randomize import Dynamics, Randomization
    from .workspace import Workspace

    ws = Workspace.load(a.workspace)
    pol = Policy.load(a.policy)
    # Randomizacja wokol zmierzonej dynamiki - jak w treningu i w panelu, nie wokol Menagerie.
    rand = Randomization.around(Dynamics.from_dict(ws.dynamics)) if a.rand else None
    res = evaluate(pol, a.episodes, randomization=rand, workspace=ws)
    print(f"{a.policy}: {res}")
    return 0


def _policies(a: argparse.Namespace) -> int:
    from .rl.policy import DEFAULT_DIR, list_policies

    items = list_policies(a.dir or DEFAULT_DIR)
    if not items:
        print("Brak zapisanych polityk. Naucz pierwsza:  lerobot-twin train --task reach")

    def pct(key):
        v = p["evals"].get(key, {}).get("success")
        return "  -  " if v is None else f"{v:5.1%}"

    for p in items:
        print(f"  {p['name']:32s} {p['task']:6s} GPU {p['success'] or 0:5.1%}  "
              f"CPU bez/z rand. {pct('cpu_nominal')} / {pct('cpu_rand')}  "
              f"{p['steps'] / 1e6:6.1f} M  {p['created']}")
    return 0


def _ui(a: argparse.Namespace) -> int:
    from .ui.app import main as ui_main

    argv = ["--host", a.host, "--port", str(a.port)]
    if a.workspace:
        argv += ["--workspace", a.workspace]
    return ui_main(argv)


def main(argv: list[str] | None = None) -> int:
    a = parser().parse_args(argv)
    return a.fn(a)


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="lerobot-twin", description="Cyfrowy blizniak stanowiska SO-101.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("check", help="sprawdz srodowisko (bez renderu)").set_defaults(fn=_check)

    p = sub.add_parser("card", help="arkusz karty kalibracyjnej do druku")
    p.add_argument("--out", default="karta_kalibracyjna.png")
    p.add_argument("--tag-mm", type=float, default=50.0, help="bok czarnego kwadratu taga [mm]")
    p.add_argument("--dpi", type=int, default=300)
    p.set_defaults(fn=_card)

    p = sub.add_parser("board", help="tablica ChArUco do intrynsyk kamery (PNG, A4)")
    p.add_argument("--out", default="tablica_charuco.png")
    p.add_argument("--square-mm", type=float, default=28.0, help="bok kwadratu szachownicy [mm]")
    p.add_argument("--dpi", type=int, default=300)
    p.set_defaults(fn=_board)

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

    p = sub.add_parser("train", help="naucz polityke PPO na GPU (MuJoCo Warp)")
    p.add_argument("--task", choices=("reach", "lift"), default="reach")
    p.add_argument("--envs", type=int, default=4096, help="ile swiatow naraz")
    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", default=None, help="nazwa przebiegu (domyslnie zadanie + data)")
    p.add_argument("--out", default=None, help="katalog polityk (domyslnie workspace/policies)")
    p.add_argument("--workspace", default=None, help="plik stanowiska (domyslnie workspace/twin.json)")
    p.add_argument("--no-rand", action="store_true", help="bez randomizacji dziedziny")
    p.add_argument("--spread", type=float, default=1.0, help="szerokosc randomizacji wokol zmierzonej dynamiki")
    p.add_argument("--eval-episodes", type=int, default=50)
    p.add_argument("--stop-file", default=None, help="przerwij trening, gdy ten plik sie pojawi")
    p.add_argument("--init", default=None, help="douczanie: start z tej polityki (policy.pt)")
    p.set_defaults(fn=_train)

    p = sub.add_parser("eval", help="ewaluacja polityki na CPU (zwykle MuJoCo)")
    p.add_argument("policy", help="sciezka do policy.pt")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--rand", action="store_true", help="z randomizacja dziedziny")
    p.add_argument("--workspace", default=None)
    p.set_defaults(fn=_eval)

    p = sub.add_parser("policies", help="zapisane polityki i ich wyniki")
    p.add_argument("--dir", default=None)
    p.set_defaults(fn=_policies)

    p = sub.add_parser("ui", help="panel blizniaka w przegladarce (viser)")
    # Panel rusza prawdziwym ramieniem i nie ma hasla, a viser dzieli stan GUI miedzy
    # wszystkich podlaczonych - kazdy, kto dosiegnie portu, moze nacisnac "Polacz" albo
    # uruchomic polityke. Dlatego domyslnie tylko ten komputer; siec na wyrazne zyczenie.
    p.add_argument("--host", default="127.0.0.1",
                   help="adres nasluchu (domyslnie tylko ten komputer; 0.0.0.0 = cala siec, BEZ hasla)")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--workspace", default=None)
    p.set_defaults(fn=lambda a: _ui(a))
    return ap


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
