"""Wiersz polecen blizniaka: `lerobot-twin <polecenie>`.

    lerobot-twin check                       # czy srodowisko jest gotowe (bez renderu), takze GPU
    lerobot-twin demo                        # przykladowe stanowisko w symulacji (2 kamery)
    lerobot-twin card --tag-mm 50            # arkusz karty kalibracyjnej do druku
    lerobot-twin board --square-mm 28        # tablica ChArUco do intrynsyk kamery
    lerobot-twin calib-sim --n 5 --cameras 2 # kalibracja w symulacji, oceniana wzgledem prawdy
    lerobot-twin workspace                   # co wiadomo o stanowisku (kamery, kalibracja)
    lerobot-twin train --task reach          # polityka PPO na GPU (MuJoCo Warp) + ewaluacja na CPU
    lerobot-twin eval <policy.pt> --rand     # ewaluacja polityki w zwyklym MuJoCo
    lerobot-twin policies                    # zapisane polityki i ich wyniki
    lerobot-twin ui                          # panel w przegladarce: http://localhost:8080
    lerobot-twin ui --host 0.0.0.0           # ... dostepny z sieci (bez hasla - tylko w zaufanej)
    lerobot-twin --config configs/local.yaml ui   # tiki chwytaka, baudrate itd. z pliku YAML

`--config` ustawia zmienna LEROBOT_MP_CONFIG, wiec plik widzi tez proces treningu
uruchamiany z panelu.
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
    for mod in ("mujoco", "gymnasium", "viser", "cv2", "numpy", "torch"):
        try:
            m = __import__(mod)
            line(mod, getattr(m, "__version__", "?"))
        except ImportError as exc:
            line(mod, f"brak ({exc}) - pip install -e \".[twin]\"", False)
    line("MUJOCO_GL", os.environ.get("MUJOCO_GL", "(domyslny: glfw; bez ekranu ustaw egl)"))
    from ..paths import CONFIG_ENV, config_from_env

    cfg_file = config_from_env()
    line("konfiguracja", f"{cfg_file} ({CONFIG_ENV})" if cfg_file else "domyslna (bez --config)")

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

    gpu_ok, gpu_lines = _gpu_report()
    print("\nTrening na GPU (lerobot-twin train) - opcjonalny:")
    for label, value, good in gpu_lines:
        # OK gdy gotowe; "--" (nie BLAD) gdy niedostepne - trening jest opcjonalny
        print(f"  {'OK ' if good else '-- '}  {label:<28} {value}")
    print("  " + ("Trening na GPU: gotowy." if gpu_ok else
                  "Trening na GPU: niedostepny. Wymaga karty NVIDIA z CUDA, torcha z CUDA i mujoco-warp\n"
                  "  (pip install -e \".[train]\"). Panel, kalibracja, percepcja i ewaluacja polityk\n"
                  "  na CPU dzialaja bez tego."))

    print("\nWszystko gotowe." if ok else "\nSa bledy - patrz wyzej.")
    return 0 if ok else 1


def _gpu_report() -> tuple[bool, list[tuple[str, str, bool]]]:
    """Co wiadomo o GPU: torch (wersja, CUDA, karta), warp i mujoco_warp. Nic tu nie jest bledem."""
    from importlib.metadata import PackageNotFoundError, version

    out: list[tuple[str, str, bool]] = []
    ready = True
    try:
        import torch

        cuda_build = torch.version.cuda
        avail = bool(torch.cuda.is_available())
        name = torch.cuda.get_device_name(0) if avail else "-"
        out.append(("torch CUDA", f"kompilacja CUDA {cuda_build or 'brak (kolo CPU)'}, "
                                  f"cuda.is_available={avail}, karta: {name}", avail))
        ready &= avail
    except ImportError:
        out.append(("torch", "brak", False))
        ready = False
    try:
        import warp as wp

        try:  # bez bannera przy inicjalizacji
            wp.config.log_level = wp.LOG_WARNING
        except AttributeError:  # pragma: no cover - starszy warp
            wp.config.quiet = True
        wp.init()
        n = wp.get_cuda_device_count()
        names = ", ".join(d.name for d in wp.get_cuda_devices()) if n else "brak karty CUDA"
        out.append(("warp", f"{wp.config.version}, widzi GPU: {names}", n > 0))
        ready &= n > 0
    except ImportError:
        out.append(("warp", "brak - pip install -e \".[train]\"", False))
        ready = False
    except Exception as exc:  # noqa: BLE001 - sterownik, CUDA: tylko raport
        out.append(("warp", f"{type(exc).__name__}: {exc}", False))
        ready = False
    try:
        import mujoco_warp  # noqa: F401

        try:
            ver = version("mujoco-warp")
        except PackageNotFoundError:  # pragma: no cover
            ver = "?"
        out.append(("mujoco_warp", ver, True))
    except ImportError:
        out.append(("mujoco_warp", "brak - pip install -e \".[train]\"", False))
        ready = False
    return ready, out


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
    where = Path(ws.path).resolve() if ws.path is not None else "-"
    print(f"Stanowisko: {where} {'' if exists else '(jeszcze nie zapisane - wartosci domyslne)'}")
    print(f"  ramie: {ws.robot}, backend: {ws.backend}, port: {ws.port or '-'}")
    print(f"  karta: tag {ws.card_obj().tag_size * 1000:.1f} mm")
    if not ws.cameras:
        print("  kamery: brak  (przyklad w symulacji z dwiema kamerami: lerobot-twin demo)")
    for cam in ws.cameras:
        cal = cam.calibration
        state = ("zaufana" if cam.trusted else f"NIEZAUFANA ({cal.get('reason', '')})") if cam.calibrated \
            else "nieskalibrowana"
        rms = f", residuum {cal['rms_px']:.2f} px" if "rms_px" in cal else ""
        print(f"  kamera {cam.name}: zrodlo {cam.source}, {cam.width}x{cam.height}, "
              f"intrynsyki {cam.intrinsics_from}, {state}{rms}")
    return 0


#: Przykladowe stanowisko w symulacji - dwie kamery, ktorych skalibrowana poza
#: jest ich prawdziwa poza w scenie (zaufane), wiec "kostka z kamer" dziala od razu.
DEMO_WORKSPACE = Path("examples") / "twin.sim.json"


def _demo(a: argparse.Namespace) -> int:
    import shutil

    from ..paths import data_path
    from .robots import resolve_asset

    src = resolve_asset(DEMO_WORKSPACE)
    if not src.is_file():
        print(f"Brak {DEMO_WORKSPACE} - to polecenie dziala z klonu repozytorium.", file=sys.stderr)
        return 1
    # To samo co workspace.default_path(), ale bez importu `workspace` (ciagnie mujoco).
    dst = Path(a.path) if a.path else data_path(Path("workspace") / "twin.json")
    if dst.exists() and not a.force:
        print(f"{dst} juz istnieje - nie nadpisuje (to moze byc Twoja kalibracja).\n"
              f"Nadpisz: lerobot-twin demo --force   albo inny plik: lerobot-twin demo --path <plik>",
              file=sys.stderr)
        return 1
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    print(f"Skopiowano {src} -> {dst}")
    print("Stanowisko w symulacji: ramie sim, dwie kamery symulowane (sym-lewa, sym-prawa) z zaufana kalibracja.")
    ws_flag = f" --workspace {dst}" if a.path else ""
    print(f"Dalej: lerobot-twin ui{ws_flag}  ->  Ramie: sim, Polacz  ->  Polityka: lift-v3,"
          f" 'lift: skad polozenie kostki' = kamery.")
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
        cfg.critic_warmup = 30                         # dla polityk bez zapisanego krytyka (patrz ppo)
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


def _ui_banner(workspace: str | None) -> str:
    """Ktory plik stanowiska i katalog polityk wezmie panel - widac bez przegladarki.

    Te same reguly co `Workspace.load` i `rl.policy.DEFAULT_DIR`, ale bez ich
    importu (mujoco, torch), zeby napis byl przed wolnym startem panelu.
    """
    from ..paths import data_path

    ws = (Path(workspace) if workspace else data_path(Path("workspace") / "twin.json")).resolve()
    state = "istnieje" if ws.is_file() else "nowe, puste - przyklad: lerobot-twin demo"
    policies = data_path(Path("workspace") / "policies").resolve()
    return f"Stanowisko: {ws} ({state})\nPolityki:   {policies}"


def _ui(a: argparse.Namespace) -> int:
    from .ui.app import main as ui_main

    print(_ui_banner(a.workspace), flush=True)
    argv = ["--host", a.host, "--port", str(a.port)]
    if a.workspace:
        argv += ["--workspace", a.workspace]
    return ui_main(argv)


def _use_config(path: str) -> None:
    """Ustawia LEROBOT_MP_CONFIG (sciezka bezwzgledna) i od razu sprawdza plik.

    Zmienna, a nie argument, bo blizniak wola `load_config()` w wielu miejscach,
    a trening z panelu idzie w osobnym procesie - dziedziczy ja ze srodowiska.
    """
    import yaml

    from ..config import load_config
    from ..paths import CONFIG_ENV

    cfg = Path(path).expanduser().resolve()
    if not cfg.is_file():
        raise SystemExit(f"lerobot-twin: --config {path}: nie ma takiego pliku")
    os.environ[CONFIG_ENV] = str(cfg)
    try:
        load_config()                                  # nieznany klucz = blad teraz, nie w polowie sesji
    except (ValueError, TypeError, yaml.YAMLError) as exc:
        raise SystemExit(f"lerobot-twin: --config {path}: {exc}") from exc


def main(argv: list[str] | None = None) -> int:
    a = parser().parse_args(argv)
    if getattr(a, "config", None):
        _use_config(a.config)
    return a.fn(a)


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="lerobot-twin", description="Cyfrowy blizniak stanowiska SO-101.")
    config_help = ("plik YAML konfiguracji (np. configs/local.yaml: robot.center_ticks, "
                   "gripper_*_ticks, baudrate); ustawia LEROBOT_MP_CONFIG")
    ap.add_argument("--config", default=None, help=config_help)
    # To samo po nazwie polecenia (`lerobot-twin ui --config ...`); SUPPRESS nie nadpisuje wartosci sprzed niej.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=argparse.SUPPRESS, help=config_help)
    sub = ap.add_subparsers(dest="cmd", required=True)
    _add_parser = sub.add_parser

    def add_parser(*args, **kw):
        return _add_parser(*args, parents=[common], **kw)
    sub.add_parser = add_parser  # type: ignore[method-assign]

    sub.add_parser("check", help="sprawdz srodowisko (bez renderu), takze GPU do treningu").set_defaults(fn=_check)

    p = sub.add_parser("demo", help="przykladowe stanowisko w symulacji -> workspace/twin.json")
    p.add_argument("--path", default=None, help="plik docelowy (domyslnie workspace/twin.json)")
    p.add_argument("--force", action="store_true", help="nadpisz istniejacy plik stanowiska")
    p.set_defaults(fn=_demo)

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
    p.add_argument("--port", type=int, default=8080,
                   help="port panelu (domyslnie 8080; zajety przez inna usluge? np. --port 8765)")
    p.add_argument("--workspace", default=None)
    p.set_defaults(fn=lambda a: _ui(a))
    return ap


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
