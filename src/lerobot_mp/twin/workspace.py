"""Stanowisko blizniaka: wszystko, co wiadomo o biurku, w jednym pliku JSON.

Ramie i jego port, stol i to, gdzie stoi na nim podstawa, karta kalibracyjna
(ze ZMIERZONYM bokiem taga) i kazda kamera z intrynsykami, poza wzgledem
podstawy i wynikiem kalibracji. Z tego jednego opisu powstaje scena MuJoCo,
srodowisko RL i widok w UI - wiec kamera skalibrowana na biurku trafia do
symulacji dokladnie tam, gdzie stoi, bez przepisywania liczb recznie.

Domyslnie `workspace/twin.json` wzgledem katalogu uruchomienia; katalog jest
w `.gitignore`, bo stanowisko jest per biurko, nie per repozytorium.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .calib.card import Card
from .robots import RobotSpec, get_spec
from .scene import CameraView, SceneConfig, Table

DEFAULT_PATH = Path("workspace") / "twin.json"
#: Poziome pole widzenia typowej kamerki internetowej - tylko do nominalnych
#: intrynsyk, zanim kamera zobaczy szachownice.
NOMINAL_HFOV_DEG = 65.0


def nominal_K(width: int, height: int, hfov_deg: float = NOMINAL_HFOV_DEG) -> np.ndarray:
    f = (width / 2) / np.tan(np.radians(hfov_deg) / 2)
    return np.array([[f, 0.0, (width - 1) / 2], [0.0, f, (height - 1) / 2], [0.0, 0.0, 1.0]])


@dataclass
class CameraRecord:
    name: str
    #: Indeks kamery ("0", "1"), plik wideo, adres strumienia albo "sim".
    source: str = "0"
    width: int = 640
    height: int = 480
    fps: int = 30
    K: list[list[float]] | None = None
    dist: list[float] | None = None
    T_cam2base: list[list[float]] | None = None
    #: Skad sa intrynsyki: "szachownica", "nominalne", "symulacja".
    intrinsics_from: str = "nominalne"
    #: Wynik ostatniej kalibracji polozenia: rms_px, spread_deg, n_obs, trusted, reason, time.
    calibration: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    #: Tylko kamery symulowane: gdzie kamera NAPRAWDE stoi w blizniaku (ustawiana
    #: w panelu). Kalibracja jej nie zna - wyznacza `T_cam2base` z kadrow, tak
    #: jak dla prawdziwej kamery - wiec wynik da sie ocenic wzgledem prawdy.
    sim_pose: list[list[float]] | None = None
    #: Wynik kalibracji intrynsyk: rms_px, n_views, coverage, trusted, reason, time.
    intrinsics_info: dict[str, Any] = field(default_factory=dict)

    @property
    def calibrated(self) -> bool:
        return self.T_cam2base is not None

    @property
    def simulated(self) -> bool:
        return self.source == "sim"

    def true_pose(self) -> np.ndarray | None:
        """Poza, z ktorej kamera widzi scene: prawda dla symulowanej, kalibracja dla prawdziwej."""
        T = self.sim_pose if self.simulated and self.sim_pose is not None else self.T_cam2base
        return None if T is None else np.asarray(T, float)

    @property
    def trusted(self) -> bool:
        return bool(self.calibration.get("trusted", False))

    def intrinsics(self) -> tuple[np.ndarray, np.ndarray | None]:
        K = np.asarray(self.K, float) if self.K is not None else nominal_K(self.width, self.height)
        dist = None if self.dist is None else np.asarray(self.dist, float)
        return K, dist

    def view(self) -> CameraView | None:
        """Kamera jako czesc sceny - tylko, gdy wiadomo, gdzie stoi.

        Kamera symulowana renderuje z prawdziwej pozy (`sim_pose`) i z prawdziwym
        K - jej "prawdziwe" intrynsyki to te zapisane w `K`, a nominalne sa tylko
        dla kamer, ktore jeszcze nie widzialy szachownicy.
        """
        T = self.true_pose()
        if T is None:
            return None
        K, _ = self.intrinsics()
        return CameraView(self.name, K, self.width, self.height, T)


@dataclass
class Workspace:
    robot: str = "so101"
    #: "sim" - blizniak bez sprzetu, "feetech" - serwa wprost, "lerobot" - przez LeRobota.
    backend: str = "sim"
    port: str | None = None
    table: dict[str, Any] = field(default_factory=lambda: asdict(Table()))
    card: dict[str, Any] = field(default_factory=lambda: asdict(Card()))
    cameras: list[CameraRecord] = field(default_factory=list)
    #: Dynamika serw zidentyfikowana na tym ramieniu (`rl.randomize.Dynamics`);
    #: pusta = model Menagerie. Srodek randomizacji w treningu.
    dynamics: dict[str, Any] = field(default_factory=dict)
    path: Path | None = field(default=None, compare=False, repr=False)

    # --------------------------------------------------------------- obiekty
    def spec(self) -> RobotSpec:
        return get_spec(self.robot)

    def card_obj(self) -> Card:
        data = dict(self.card)
        data["ids"] = tuple(data.get("ids", (0, 1)))
        return Card(**data)

    def table_obj(self) -> Table:
        data = dict(self.table)
        for key in ("size", "base_xy", "rgba"):
            if key in data:
                data[key] = tuple(data[key])
        return Table(**data)

    def camera(self, name: str) -> CameraRecord:
        for cam in self.cameras:
            if cam.name == name:
                return cam
        raise KeyError(f"brak kamery {name!r} w stanowisku")

    def add_camera(self, record: CameraRecord) -> CameraRecord:
        if any(c.name == record.name for c in self.cameras):
            raise ValueError(f"kamera {record.name!r} juz jest w stanowisku")
        self.cameras.append(record)
        return record

    def remove_camera(self, name: str) -> None:
        self.cameras = [c for c in self.cameras if c.name != name]

    def free_name(self, stem: str = "kamera") -> str:
        taken = {c.name for c in self.cameras}
        k = 1
        while f"{stem}{k}" in taken:
            k += 1
        return f"{stem}{k}"

    def scene_config(self, *, with_card: bool = False, **extra: Any) -> SceneConfig:
        """Scena z tego stanowiska: stol, ramie i kazda WLACZONA, umiejscowiona kamera."""
        views = [v for c in self.cameras if c.enabled for v in [c.view()] if v is not None]
        cfg = SceneConfig(self.spec(), table=self.table_obj(), cameras=views,
                          card=self.card_obj() if with_card else None)
        for key, value in extra.items():
            setattr(cfg, key, value)
        return cfg

    def apply_fit(self, fit) -> list[str]:
        """Wpisuje wynik kalibracji do kamer. Zwraca nazwy kamer, ktore dostaly poze.

        Poza jest zapisywana takze dla kamery niezaufanej - z powodem obok -
        bo lepiej widziec w UI, jak bardzo sie rozjechala, niz nie widziec nic.
        """
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        updated = []
        for name, cam_fit in fit.cameras.items():
            try:
                rec = self.camera(name)
            except KeyError:
                continue
            rec.T_cam2base = np.asarray(cam_fit.T_cam2base, float).tolist()
            rec.calibration = dict(rms_px=float(cam_fit.rms_px), max_px=float(cam_fit.max_px),
                                   n_obs=int(cam_fit.n_obs), spread_deg=float(cam_fit.spread_deg),
                                   trusted=bool(cam_fit.trusted), reason=cam_fit.reason, time=now,
                                   tag_size=float(self.card_obj().tag_size))
            updated.append(name)
        return updated

    # ------------------------------------------------------------------ dysk
    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("path", None)
        return data

    def save(self, path: str | Path | None = None) -> Path:
        target = Path(path) if path else (self.path or DEFAULT_PATH)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".tmp")
        # Najpierw do pliku obok, potem podmiana - przerwany zapis nie zniszczy
        # kalibracji, na ktora ktos poswiecil pol godziny.
        tmp.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(target)
        self.path = target
        return target

    @classmethod
    def load(cls, path: str | Path | None = None) -> Workspace:
        """Wczytuje stanowisko; brak pliku to nowe, puste stanowisko (nie blad)."""
        target = Path(path) if path else DEFAULT_PATH
        if not target.is_file():
            ws = cls()
            ws.path = target
            return ws
        data = json.loads(target.read_text(encoding="utf-8"))
        cams = [CameraRecord(**c) for c in data.pop("cameras", [])]
        known = {f for f in cls.__dataclass_fields__ if f not in ("cameras", "path")}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"nieznane pola w {target}: {sorted(unknown)}")
        ws = cls(cameras=cams, **data)
        ws.path = target
        return ws
