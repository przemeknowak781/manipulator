"""Wiersz polecen aplikacji."""

from __future__ import annotations

import argparse
import sys
from typing import Any

from .config import AppConfig, load_config
from .utils.logging import setup_logging

DESCRIPTION = """\
Sterowanie ramieniem LeRobot 101 (SO-101) gestami dloni (MediaPipe).

Przyklady:
  lerobot-mp                                  # symulator + kamera 0
  lerobot-mp --camera 1 --mode ik             # inna kamera, sterowanie kartezjanskie
  lerobot-mp --robot lerobot --port /dev/ttyACM0
  lerobot-mp --source demo.mp4 --no-view      # bez kamery i bez okna
"""

EPILOG = """\
Klawisze w oknie podgladu:
  SPACJA  wlacz/wylacz sterowanie          H  powrot do pozycji domowej
  X       stop awaryjny (i skasowanie)     C  nowe zaczepienie dloni
  O / P   kalibracja chwytaka (otwarty/zamkniety)
  M       przelacz tryb mapowania          V  wlacz/wylacz podglad ramienia
  J L I K obrot kamery podgladu            , .  przyblizenie
  - / =   limit predkosci                  Q / ESC  wyjscie

Gest pauzy: zwin trzy ostatnie palce (srodkowy, serdeczny, maly) - robot stanie
w miejscu, a dlon mozna przelozyc, jak przy podnoszeniu myszy z podkladki.
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lerobot-mp",
        description=DESCRIPTION,
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", help="plik YAML z konfiguracja")

    source = parser.add_argument_group("obraz")
    source.add_argument("--camera", type=int, help="indeks kamery (0, 1, ...)")
    source.add_argument("--source", help="plik wideo lub adres strumienia zamiast kamery")
    source.add_argument("--width", type=int, help="szerokosc obrazu z kamery")
    source.add_argument("--height", type=int, help="wysokosc obrazu z kamery")
    source.add_argument(
        "--no-mirror", action="store_true", help="nie odbijaj obrazu lustrzanie"
    )

    robot = parser.add_argument_group("robot")
    robot.add_argument("--robot", choices=("sim", "lerobot", "auto"), help="backend robota")
    robot.add_argument("--port", help="port szeregowy ramienia, np. /dev/ttyACM0")
    robot.add_argument("--robot-id", help="identyfikator ramienia w LeRobot (plik kalibracji)")
    robot.add_argument("--arm", choices=("so101", "so100"), help="typ ramienia")

    control = parser.add_argument_group("sterowanie")
    control.add_argument("--mode", choices=("direct", "ik"), help="sposob mapowania dloni")
    control.add_argument(
        "--hand", choices=("any", "Left", "Right"), help="ktora dlon steruje ramieniem"
    )
    control.add_argument(
        "--clutch", choices=("gesture", "key", "always"), help="sposob zalaczania sterowania"
    )
    control.add_argument("--loop-hz", type=float, help="czestotliwosc petli sterowania")
    control.add_argument(
        "--velocity-scale", type=float, help="mnoznik limitow predkosci (0.5 = wolniej)"
    )
    control.add_argument(
        "--absolute",
        action="store_true",
        help="mapowanie bezwzgledne zamiast wzglednego (bez zaczepiania)",
    )

    view = parser.add_argument_group("podglad")
    view.add_argument("--no-view", action="store_true", help="bez okna podgladu")
    view.add_argument("--no-preview-3d", action="store_true", help="rysunek schematyczny zamiast 3D")
    view.add_argument("--preview-asset", help="sciezka do modelu 3D (.npz)")
    view.add_argument("--scale", type=float, help="skalowanie okna podgladu")
    view.add_argument("--record", help="zapisz podglad do pliku wideo (np. demo.mp4)")

    parser.add_argument("--duration", type=float, help="zakoncz po tylu sekundach")
    parser.add_argument("--print-config", action="store_true", help="wypisz konfiguracje i zakoncz")
    parser.add_argument("-v", "--verbose", action="store_true", help="szczegolowe logi")
    return parser


def overrides_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """Zamienia flagi CLI na strukture nadpisan konfiguracji."""
    camera: dict[str, Any] = {}
    if args.camera is not None:
        camera["source"] = args.camera
    if args.source:
        camera["source"] = args.source
    if args.width:
        camera["width"] = args.width
    if args.height:
        camera["height"] = args.height
    if args.no_mirror:
        camera["mirror"] = False

    robot: dict[str, Any] = {}
    if args.robot:
        robot["backend"] = args.robot
    if args.port:
        robot["port"] = args.port
        robot.setdefault("backend", "lerobot")
    if args.robot_id:
        robot["robot_id"] = args.robot_id
    if args.arm:
        robot["kind"] = args.arm

    mapping: dict[str, Any] = {}
    if args.mode:
        mapping["mode"] = args.mode
    if args.absolute:
        mapping["relative"] = False

    tracker: dict[str, Any] = {}
    if args.hand:
        tracker["preferred_hand"] = args.hand

    clutch: dict[str, Any] = {}
    if args.clutch:
        clutch["mode"] = args.clutch
        if args.clutch == "always":
            clutch["engaged_on_start"] = True

    safety: dict[str, Any] = {}
    if args.velocity_scale is not None:
        safety["velocity_scale"] = args.velocity_scale

    ui: dict[str, Any] = {}
    if args.no_view:
        ui["show"] = False
    if args.no_preview_3d:
        ui["preview_3d"] = False
    if args.preview_asset:
        ui["preview_asset"] = args.preview_asset
    if args.scale:
        ui["display_scale"] = args.scale
    if args.record:
        ui["record_path"] = args.record

    root: dict[str, Any] = {}
    if args.loop_hz:
        root["loop_hz"] = args.loop_hz
    if args.duration:
        root["max_runtime_s"] = args.duration

    for key, value in (
        ("camera", camera),
        ("robot", robot),
        ("mapping", mapping),
        ("tracker", tracker),
        ("clutch", clutch),
        ("safety", safety),
        ("ui", ui),
    ):
        if value:
            root[key] = value
    return root


def build_config(argv: list[str] | None = None) -> tuple[AppConfig, argparse.Namespace]:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config, overrides_from_args(args))
    return cfg, args


def main(argv: list[str] | None = None) -> int:
    try:
        cfg, args = build_config(argv)
    except (ValueError, TypeError, OSError) as exc:
        print(f"Blad konfiguracji: {exc}", file=sys.stderr)
        return 2

    setup_logging(args.verbose)

    if args.print_config:
        import yaml

        from .config import config_to_dict

        print(yaml.safe_dump(config_to_dict(cfg), allow_unicode=True, sort_keys=False))
        return 0

    from .app import run_app

    try:
        return run_app(cfg)
    except RuntimeError as exc:
        print(f"Blad: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
