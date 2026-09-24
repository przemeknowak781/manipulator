"""Zadania RL blizniaka: co polityka widzi, co robi i za co dostaje nagrode.

Jedna definicja dla trzech miejsc, w ktorych polityka zyje:

* srodowisko Gymnasium na CPU (`env.TwinEnv`, numpy, jedno srodowisko),
* srodowisko wsadowe na GPU (`batch.BatchEnv`, torch, tysiace swiatow MuJoCo Warp),
* petla na prawdziwym ramieniu (`runner.PolicyRunner`, numpy, katy z serw).

Funkcje ponizej biora tablice z wymiarem wsadu `(N, ...)` i modul `xp` - numpy
albo torch - wiec wszystkie trzy licza DOKLADNIE te same wzory. Dwie kopie
"prawie tej samej" obserwacji to klasyczne ciche zrodlo sim-2-real: polityka
uczy sie na jednej, a jezdzi na drugiej.

Wszystkie polozenia sa w ukladzie PODSTAWY ramienia (blat na z = 0), katy
stawow w radianach MJCF, w kolejnosci `RobotSpec.joints` (chwytak ostatni).

Akcja ma 6 liczb w [-1, 1]:

* stawy ramienia - PRZYROST celu o najwyzej `arm_step` rad na takt, liczony od
  poprzedniego celu (jak `runtime.Twin.set_target`), przyciety do tych samych
  limitow, ktore stosuje `SafetySupervisor` na prawdziwym ramieniu;
* chwytak - cel BEZWZGLEDNY (-1 zamkniety, +1 otwarty), ograniczony co do
  predkosci tak, jak ogranicza go nadzor (`grip_step` zakresu na takt).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ...config import load_config
from ..kinematics import RobotKinematics

TASKS = ("reach", "lift")


@dataclass(frozen=True)
class TaskConfig:
    name: str = "reach"
    #: Czestotliwosc polityki i krok fizyki [s]. 20 Hz przy 5 ms to 10 krokow na akcje.
    control_hz: float = 20.0
    timestep: float = 0.005
    episode_steps: int = 100
    #: Najwiekszy przyrost celu stawu ramienia na takt [rad] (0,05 przy 20 Hz = 57 st/s).
    arm_step: float = 0.05
    #: Najwiekszy ruch celu chwytaka na takt, jako czesc zakresu (0,15 przy 20 Hz =
    #: 300 jednostek/s - tyle, ile przepuszcza nadzor bezpieczenstwa).
    grip_step: float = 0.15
    #: reach: sukces, gdy TCP jest blizej celu niz tyle [m].
    success_dist: float = 0.02
    #: reach: skad losowany jest cel - promien od osi podstawy i wysokosc nad blatem [m].
    goal_radius: tuple[float, float] = (0.13, 0.32)
    goal_height: tuple[float, float] = (0.03, 0.22)
    #: lift: kostka (polowa boku [m], masa [kg]) i gdzie lezy na blacie.
    cube_half: float = 0.015
    cube_mass: float = 0.03
    cube_radius: tuple[float, float] = (0.15, 0.25)
    cube_bearing: tuple[float, float] = (-0.8, 0.8)
    #: lift: sukces, gdy spod kostki jest wyzej nad blatem niz tyle [m].
    lift_height: float = 0.06
    #: Start ramienia: poza domowa plus/minus tyle stopni na kazdym stawie ramienia.
    start_noise_deg: float = 8.0
    #: Kara za szarpanie: waga ||a - a_poprzednie||^2.
    action_rate: float = 0.02

    @property
    def obs_dim(self) -> int:
        return OBS_DIMS[self.name]

    @property
    def act_dim(self) -> int:
        return 6

    @property
    def substeps(self) -> int:
        return int(round(1.0 / self.control_hz / self.timestep))

    @property
    def cube(self) -> str | None:
        return "cube" if self.name == "lift" else None


#: q(6) + cel stawow(6) + TCP(3) + [cel(3) + cel-TCP(3)] + [kostka(3) + kostka-TCP(3) + obrot(6)] + akcja(6)
OBS_DIMS = {"reach": 6 + 6 + 3 + 3 + 3 + 6, "lift": 6 + 6 + 3 + 3 + 3 + 6 + 6}


def make_task(name: str = "reach", **overrides) -> TaskConfig:
    if name not in TASKS:
        raise ValueError(f"nieznane zadanie {name!r}; dostepne: {', '.join(TASKS)}")
    base = TaskConfig(name=name, episode_steps=100 if name == "reach" else 200)
    return replace(base, **overrides)


# --------------------------------------------------------------- limity
@dataclass(frozen=True)
class Limits:
    """Zakres celow stawow [rad]: przeciecie limitow nadzoru z zakresami MJCF."""

    lo: np.ndarray
    hi: np.ndarray
    home: np.ndarray

    @staticmethod
    def of(kin: RobotKinematics) -> Limits:
        spec = kin.spec
        if spec.gripper != spec.joints[-1]:
            raise ValueError("zadania RL zakladaja chwytak jako ostatni staw")
        cfg = load_config()
        lo = kin.to_q({n: cfg.joint(n).min for n in spec.joints})
        hi = kin.to_q({n: cfg.joint(n).max for n in spec.joints})
        lo, hi = np.maximum(lo, kin.lo), np.minimum(hi, kin.hi)
        return Limits(lo, hi, np.clip(kin.to_q(spec.home), lo, hi))


# -------------------------------------------------------- numpy / torch
def _cat(xp, parts):
    return xp.concatenate(parts, axis=-1) if xp is np else xp.cat(parts, dim=-1)


def _clip(xp, x, lo, hi):
    return np.clip(x, lo, hi) if xp is np else xp.clamp(x, lo, hi)


def _norm(x):
    return ((x * x).sum(-1)) ** 0.5


def _as(xp, a, like):
    """Stala numpy jako tablica tego samego rodzaju i urzadzenia co `like`."""
    return np.asarray(a, like.dtype) if xp is np else xp.as_tensor(a, dtype=like.dtype, device=like.device)


# ---------------------------------------------------------------- akcja
def apply_action(xp, task: TaskConfig, limits: Limits, q_cmd, action):
    """Nowy cel stawow (N, 6) z poprzedniego celu i akcji w [-1, 1]."""
    lo, hi = _as(xp, limits.lo, q_cmd), _as(xp, limits.hi, q_cmd)
    a = _clip(xp, action, -1.0, 1.0)
    arm = _clip(xp, q_cmd[..., :5] + a[..., :5] * task.arm_step, lo[:5], hi[:5])
    span = hi[5:] - lo[5:]
    want = lo[5:] + (a[..., 5:] + 1.0) * 0.5 * span
    step = task.grip_step * span
    grip = q_cmd[..., 5:] + _clip(xp, want - q_cmd[..., 5:], -step, step)
    return _cat(xp, [arm, grip])


# ------------------------------------------------------------ obserwacja
def normalize_q(xp, limits: Limits, q):
    lo, hi = _as(xp, limits.lo, q), _as(xp, limits.hi, q)
    return 2.0 * (q - lo) / (hi - lo) - 1.0


def observe(xp, task: TaskConfig, limits: Limits, q, q_cmd, tcp, prev_action, goal=None,
            cube_pos=None, cube_rot=None):
    """Wektor obserwacji (N, obs_dim). `cube_rot` to macierze obrotu (N, 3, 3)."""
    parts = [normalize_q(xp, limits, q), normalize_q(xp, limits, q_cmd), tcp]
    if task.name == "reach":
        parts += [goal, goal - tcp]
    else:
        rot6 = _cat(xp, [cube_rot[..., :, 0], cube_rot[..., :, 1]])
        parts += [cube_pos, cube_pos - tcp, rot6]
    parts.append(prev_action)
    return _cat(xp, parts)


def fold_yaw(xp, R):
    """Obrot wokol z zlozony do +-45 st. - tak widzi kostke detektor (symetria 90 st.)."""
    yaw = xp.arctan2(R[..., 1, 0], R[..., 0, 0]) if xp is np else xp.atan2(R[..., 1, 0], R[..., 0, 0])
    yaw = (yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4
    c, s = xp.cos(yaw), xp.sin(yaw)
    z, o = xp.zeros_like(c), xp.ones_like(c)
    rows = [xp.stack([c, -s, z], -1), xp.stack([s, c, z], -1), xp.stack([z, z, o], -1)]
    return xp.stack(rows, -2)


# ---------------------------------------------------------------- nagroda
def reward(xp, task: TaskConfig, tcp, action, prev_action, goal=None, cube_pos=None, jaw_contacts=None):
    """(nagroda, sukces, porazka) dla kazdego swiata.

    `jaw_contacts` (N, 2): czy kostka dotyka szczeki stalej i ruchomej.
    Porazka = kostka spadla ze stolu; epizod sie wtedy konczy.
    """
    rate = task.action_rate * ((action - prev_action) ** 2).sum(-1)
    if task.name == "reach":
        d = _norm(goal - tcp)
        # Dwie skale: zgrubna ciagnie z daleka, dokladna nagradza ostatnie milimetry.
        r = 0.5 * (1.0 - xp.tanh(d / 0.10)) + 0.5 * (1.0 - xp.tanh(d / 0.01))
        success = d < task.success_dist
        failure = d < -1.0                                  # w reach nie ma porazki
        return r - rate, success, failure
    d = _norm(cube_pos - tcp)
    grasped = (jaw_contacts[..., 0] > 0.5) & (jaw_contacts[..., 1] > 0.5)
    height = cube_pos[..., 2] - task.cube_half
    lifted = _clip(xp, height / task.lift_height, 0.0, 1.0)
    success = grasped & (height > task.lift_height)
    g = grasped.to(tcp.dtype) if xp is not np else grasped.astype(tcp.dtype)
    s = success.to(tcp.dtype) if xp is not np else success.astype(tcp.dtype)
    r = (1.0 - xp.tanh(d / 0.05)) + g + 4.0 * g * lifted + 2.0 * s
    failure = height < -0.05
    return r - rate, success, failure


# ---------------------------------------------------------------- starty
def sample_goals(kin: RobotKinematics, task: TaskConfig, rng: np.random.Generator, n: int) -> np.ndarray:
    """Cele `reach` (n, 3) w ukladzie podstawy - na pewno osiagalne.

    Losujemy w przestrzeni STAWOW i bierzemy TCP z kinematyki prostej, jak fala
    kalibracyjna: kazdy cel jest osiagalny z definicji, a filtr trzyma go nad
    stolem i przed ramieniem.
    """
    spec = kin.spec
    out = []
    ranges = {**{j: (np.degrees(lo), np.degrees(hi)) for j, lo, hi in zip(spec.joints, kin.lo, kin.hi)},
              **spec.wave_ranges}
    while len(out) < n:
        j = {name: float(rng.uniform(*ranges[name])) for name in spec.arm_joints}
        p = kin.tcp(j)[:3, 3]
        r = float(np.hypot(p[0], p[1]))
        if task.goal_radius[0] <= r <= task.goal_radius[1] and task.goal_height[0] <= p[2] <= task.goal_height[1] \
                and p[0] > 0.05:
            out.append(p)
    return np.array(out)


def sample_start(task: TaskConfig, limits: Limits, rng: np.random.Generator, n: int) -> np.ndarray:
    """Katy startowe (n, 6): poza domowa z szumem, chwytak otwarty w lift."""
    q = np.repeat(limits.home[None], n, axis=0)
    q[:, :5] += np.radians(rng.uniform(-task.start_noise_deg, task.start_noise_deg, (n, 5)))
    if task.name == "lift":
        span = limits.hi[5] - limits.lo[5]
        q[:, 5] = limits.lo[5] + rng.uniform(0.6, 1.0, n) * span
    return np.clip(q, limits.lo, limits.hi)


def sample_cubes(task: TaskConfig, rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Kostki na blacie: polozenia (n, 3) w ukladzie podstawy i kwaterniony (n, 4) obrotu wokol z."""
    r = rng.uniform(*task.cube_radius, n)
    b = rng.uniform(*task.cube_bearing, n)
    pos = np.stack([r * np.cos(b), r * np.sin(b), np.full(n, task.cube_half)], axis=1)
    yaw = rng.uniform(-np.pi, np.pi, n)
    quat = np.stack([np.cos(yaw / 2), np.zeros(n), np.zeros(n), np.sin(yaw / 2)], axis=1)
    return pos, quat
