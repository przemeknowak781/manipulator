"""Zadania w tle dla panelu: trening, kalibracje, identyfikacja, ewaluacja.

Kazde zadanie ma stan (`state`), postep 0..1, komunikat i wynik - panel tylko
je czyta w swojej petli, wiec zaden przycisk nie blokuje interfejsu. Trening
idzie w OSOBNYM PROCESIE (`lerobot-twin train`): zajmuje GPU na kilkanascie
minut, a padniecie sterownika CUDA nie moze zabrac ze soba panelu ani petli
sterowania ramieniem.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

IDLE, RUNNING, DONE, FAILED, CANCELLED = "bezczynne", "trwa", "gotowe", "blad", "przerwane"


class Job:
    """Jedno zadanie w watku: `start(fn)`, gdzie `fn(job)` czyta `job.cancel` i ustawia postep."""

    def __init__(self, name: str):
        self.name = name
        self.state = IDLE
        self.progress = 0.0
        self.message = ""
        self.result: Any = None
        self.error = ""
        self.cancel = threading.Event()
        self.data: dict[str, Any] = {}
        self._thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        return self.state == RUNNING

    def start(self, fn: Callable[[Job], Any]) -> None:
        if self.running:
            raise RuntimeError(f"{self.name} juz trwa")
        self.cancel.clear()
        self.state, self.progress, self.message, self.result, self.error = RUNNING, 0.0, "start", None, ""
        self.data = {}

        def body():
            try:
                self.result = fn(self)
                self.state = CANCELLED if self.cancel.is_set() else DONE
            except Exception as exc:
                logger.exception("Zadanie %s padlo", self.name)
                self.error = f"{type(exc).__name__}: {exc}"
                self.state = FAILED
        self._thread = threading.Thread(target=body, name=self.name, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.cancel.set()


class TrainingJob:
    """`lerobot-twin train` w osobnym procesie; postep z `progress.json` przebiegu."""

    def __init__(self, policies_dir: Path, workspace_path: Path | None):
        self.policies_dir = Path(policies_dir).resolve()
        self.workspace_path = workspace_path
        self.proc: subprocess.Popen | None = None
        self.run_dir: Path | None = None
        self.log_path: Path | None = None
        self._log = None
        self.started = 0.0

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, task: str, envs: int, iters: int, spread: float, randomize: bool, name: str,
              init: str | None = None) -> Path:
        if self.running:
            raise RuntimeError("trening juz trwa")
        self.run_dir = self.policies_dir / name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "STOP").unlink(missing_ok=True)
        (self.run_dir / "progress.json").unlink(missing_ok=True)
        cmd = [sys.executable, "-m", "lerobot_mp.twin.cli", "train", "--task", task, "--envs", str(envs),
               "--iters", str(iters), "--spread", str(spread), "--name", name, "--out", str(self.policies_dir),
               "--stop-file", str(self.run_dir / "STOP")]
        if not randomize:
            cmd.append("--no-rand")
        if init:
            cmd += ["--init", str(init)]
        if self.workspace_path is not None:
            cmd += ["--workspace", str(self.workspace_path)]
        self.log_path = self.run_dir / "train.log"
        if self._log is not None:
            self._log.close()
        self._log = open(self.log_path, "w", encoding="utf-8")
        flags = subprocess.CREATE_NO_WINDOW if sys.platform.startswith("win") else 0
        self.proc = subprocess.Popen(cmd, stdout=self._log, stderr=subprocess.STDOUT, creationflags=flags)
        self.started = time.time()
        return self.run_dir

    def stop(self) -> None:
        if self.run_dir is not None:
            (self.run_dir / "STOP").touch()

    def progress(self) -> dict[str, Any] | None:
        if self.run_dir is None:
            return None
        try:
            return json.loads((self.run_dir / "progress.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def tail(self, n: int = 6) -> str:
        if self.log_path is None or not self.log_path.exists():
            return ""
        lines = [ln for ln in self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                 if ln.strip() and "Module " not in ln and "Kernel cache" not in ln]
        return "\n".join(lines[-n:])

    @property
    def exit_code(self) -> int | None:
        return None if self.proc is None else self.proc.poll()


# ------------------------------------------------------------ kalibracja
class _CameraSubset:
    def __init__(self, hub, names: list[str]):
        self.hub, self.names = hub, names

    def grab(self) -> dict[str, np.ndarray]:
        return {n: img for n in self.names for img in [self.hub.frame(n)] if img is not None}


def run_card_calibration(job: Job, twin, cameras: list[str], quick: bool = False) -> Any:
    """Fala kalibracyjna z karta w chwytaku, na blizniaku (sim albo prawdziwe ramie).

    W symulacji karta trafia do sceny w pozie przekrzywionej wzgledem nominalnej
    (jak wlozona reka); sesja zna tylko nominalna - dokladnie jak na biurku.
    """
    from ..calib.card import perturb, pinch_point
    from ..calib.session import Session, WaveConfig
    from ..collision import CollisionChecker
    from ..kinematics import RobotKinematics
    from .. import scene as sc

    ws = twin.workspace
    card = ws.card_obj()
    nominal = card.nominal(pinch_point(RobotKinematics(ws.spec())))
    simulated = twin.status.simulated
    before = dict(twin.extras)                           # np. kostka polozona wczesniej - wroci po fali
    if simulated:
        truth = perturb(nominal, np.random.default_rng(int(time.time())))
        twin.configure(**before, with_card=True, card_pose=truth)
        job.data["card_truth"] = truth
    job.message = "przygotowanie sprawdzacza kolizji"
    checker = CollisionChecker(sc.build(ws.scene_config(with_card=True, card_pose=nominal, card_collider=0.012)))
    intr = {n: ws.camera(n).intrinsics() for n in cameras}
    cfg = WaveConfig(min_obs=10, min_poses=6, max_poses=36) if quick else WaveConfig()
    session = Session(twin, _CameraSubset(twin.cameras, cameras), intr, twin.scene.kin, card, nominal, checker, cfg,
                      seed=int(time.time()) % 10_000)
    try:
        while not session.done:
            if job.cancel.is_set():
                return None
            rep = session.step()
            need = [min(o / cfg.min_obs, p / cfg.min_poses, s / cfg.min_spread) for o, p, s in rep.progress.values()]
            job.progress = min(0.99, max(min(need) if need else 0.0, rep.index / cfg.max_poses))
            job.data["progress"] = rep.progress
            job.message = f"poza {rep.index}: " + ", ".join(
                f"{c} widzi {len(t)} tag." for c, t in rep.seen.items()) + (f" - {rep.note}" if rep.note else "")
        job.message = "dopasowanie"
        return session.solve()
    finally:
        twin.home()
        if simulated:
            twin.configure(**before)


def run_intrinsics(job: Job, hub, camera: str, board, size: tuple[int, int], period: float = 0.25) -> Any:
    """Zbiera kadry tablicy ChArUco z kamery, dopoki panel nie powie `stop`, potem liczy K."""
    from ..calib.intrinsics import Collector

    col = Collector(board, size)
    job.data["collector"] = col
    while not job.cancel.is_set():
        img = hub.frame(camera)
        if img is not None:
            ok, why = col.add(img)
            job.data["last"] = img
            job.data["last_det"] = col.detect(img)
            job.message = why
            job.progress = min(1.0, len(col.views) / 12)
        time.sleep(period)
    job.cancel.clear()
    job.message = "liczenie K"
    return col.solve()


def run_sysid(job: Job, twin) -> Any:
    from ..rl.sysid import excitation, fit, record

    job.message = "ruch pobudzajacy (ramie sie rusza)"
    home = dict(twin.workspace.spec().home)
    rec = record(twin, excitation(home), on_tick=lambda p: setattr(job, "progress", 0.4 * p))
    job.data["recording"] = rec
    job.message = "dopasowanie symulacji do nagrania"

    def prog(n, c):
        job.progress = min(0.99, 0.4 + 0.6 * n / 700)
        job.message = f"dopasowanie: {n} symulacji, blad {c:.3f} st."
    dyn, base = fit(rec, on_progress=prog)
    job.data["base"] = base
    return dyn


def run_eval(job: Job, policy_path: str, workspace, episodes: int = 50) -> Any:
    from ..rl.evaluate import evaluate
    from ..rl.policy import Policy
    from ..rl.randomize import Dynamics, Randomization

    pol = Policy.load(policy_path)
    out = {}
    for k, (label, rand) in enumerate((("nominal", None), ("rand", Randomization.around(
            Dynamics.from_dict(workspace.dynamics))))):
        job.message = f"ewaluacja {label}"
        job.progress = 0.5 * k
        out[f"cpu_{label}"] = evaluate(pol, episodes, randomization=rand, workspace=workspace)
    pol.meta.evals.update(out)
    from ..rl.policy import is_bundled
    if not is_bundled(policy_path):                   # bazowych z repozytorium nie nadpisujemy
        pol.save(policy_path)
    job.data["evals"] = out
    return out
