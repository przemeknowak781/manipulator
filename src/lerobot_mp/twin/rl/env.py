"""Srodowisko Gymnasium blizniaka na CPU: `LeRobotMP/TwinReach-v0`, `LeRobotMP/TwinLift-v0`.

    import gymnasium as gym, lerobot_mp.twin.rl
    env = gym.make("LeRobotMP/TwinReach-v0")
    obs, info = env.reset(seed=0)

To jest wersja "jedno srodowisko, zwykly MuJoCo" - do sprawdzania polityk,
debugowania nagrody, `check_env` i bibliotek w stylu SB3. Do uczenia na
tysiacach swiatow naraz sluzy `batch.BatchEnv` na MuJoCo Warp; obie licza
obserwacje i nagrode tymi samymi funkcjami z `task`.

Scena jest budowana RAZ (293 ms na laptopie, 111 ms na Shadow); reset zmienia
tylko stan i pola skompilowanego modelu.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import mujoco
import numpy as np

from .. import scene as sc
from ..kinematics import inverse, pose
from ..workspace import Workspace
from . import task as tk
from .randomize import Randomization

#: Kamera podgladu, gdy stanowisko nie ma zadnej skalibrowanej.
VIEW = "widok"


def look_at(eye, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    z = np.asarray(target, float) - np.asarray(eye, float)
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    return pose(np.column_stack([x, np.cross(z, x), z]), np.asarray(eye, float))


def scene_config(task: tk.TaskConfig, workspace: Workspace | None = None, view: bool = False) -> sc.SceneConfig:
    """Scena zadania: stanowisko (stol, kamery) plus obiekty zadania."""
    ws = workspace or Workspace()
    cfg = ws.scene_config()
    if task.cube:
        h = task.cube_half
        cfg.objects = [*cfg.objects, sc.Box(task.cube, (h, h, h), (0.2, 0.0), rgba=(0.85, 0.25, 0.2, 1.0),
                                            mass=task.cube_mass)]
        cfg.grasp_sensors = [*cfg.grasp_sensors, task.cube]
    if view and not cfg.cameras:
        cfg.cameras = [sc.CameraView.from_fov(VIEW, 480, 360, 50.0, look_at([0.62, -0.42, 0.42], [0.18, 0.0, 0.05]))]
    return cfg


class ModelFields:
    """Nominalne pola modelu, ktore randomizacja mnozy - zapamietane raz."""

    def __init__(self, scene: sc.Scene, cube: str | None):
        m = scene.model
        self.act = scene.act_ids
        self.dof = scene.kin.dadr
        self.kp = m.actuator_gainprm[self.act, 0].copy()
        self.bias = m.actuator_biasprm[self.act, 1:3].copy()
        self.damping = m.dof_damping[self.dof].copy()
        self.armature = m.dof_armature[self.dof].copy()
        self.frictionloss = m.dof_frictionloss[self.dof].copy()
        self.cube_body = m.body(cube).id if cube else None
        self.cube_geom = m.geom(cube).id if cube else None
        self.cube_mass = m.body_mass[self.cube_body] if cube else 0.0
        self.cube_inertia = m.body_inertia[self.cube_body].copy() if cube else None
        self.cube_friction = m.geom_friction[self.cube_geom].copy() if cube else None

    def apply(self, m: mujoco.MjModel, d: mujoco.MjData, s: dict[str, float]) -> None:
        """Mnozniki `s` na pola modelu, liczone zawsze od nominalu (nie kumuluja sie)."""
        m.actuator_gainprm[self.act, 0] = self.kp * s["kp"]
        m.actuator_biasprm[self.act, 1] = self.bias[:, 0] * s["kp"]
        m.dof_damping[self.dof] = self.damping * s["damping"]
        m.dof_armature[self.dof] = self.armature * s["armature"]
        m.dof_frictionloss[self.dof] = self.frictionloss * s["frictionloss"]
        if self.cube_body is not None:
            # Masa razem z bezwladnoscia - to ten sam obiekt, tylko ciezszy.
            m.body_mass[self.cube_body] = self.cube_mass * s["cube_mass"]
            m.body_inertia[self.cube_body] = self.cube_inertia * s["cube_mass"]
            m.geom_friction[self.cube_geom] = self.cube_friction * np.array([s["cube_friction"], 1.0, 1.0])
        # Masa i armatura wchodza w stale liczone przy kompilacji (body_invweight0,
        # dof_invweight0), a od nich zalezy miekkosc kontaktow. Bez przeliczenia
        # kostka o zmienionej masie miala kontakty "od innej kostki" i przy
        # zakleszczeniu miedzy szczeka a blatem wylatywala z predkoscia 10-180 m/s.
        mujoco.mj_setConst(m, d)


class TwinEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, task: str | tk.TaskConfig = "reach", *, workspace: Workspace | None = None,
                 randomization: Randomization | None = None, render_mode: str | None = None,
                 render_camera: str | None = None):
        self.task = task if isinstance(task, tk.TaskConfig) else tk.make_task(task)
        self.render_mode = render_mode
        self.scene = sc.build(scene_config(self.task, workspace, view=render_mode is not None))
        self.render_camera = render_camera or (self.scene.cfg.cameras[0].name if self.scene.cfg.cameras else None)
        self.kin = self.scene.kin
        self.limits = tk.Limits.of(self.kin)
        self.rand = randomization if randomization is not None else Randomization()
        self.fields = ModelFields(self.scene, self.task.cube)
        m = self.scene.model
        m.opt.timestep = self.task.timestep
        self.T_world2base = inverse(self.scene.T_base2world)
        self.site = self.kin.site_id
        if self.task.cube:
            self.cube_qadr = m.jnt_qposadr[m.body_jntadr[self.fields.cube_body]]
            self.cube_dadr = m.jnt_dofadr[m.body_jntadr[self.fields.cube_body]]
            self.jaw_adr = [m.sensor(f"{self.task.cube}_jaw{k}").adr[0] for k in range(2)]

        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (self.task.obs_dim,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (self.task.act_dim,), np.float32)
        self.goal = np.zeros(3)
        self.q_cmd = self.limits.home.copy()
        self.prev_action = np.zeros(6)
        self.t = 0
        self._delay: list[np.ndarray] = []
        self._noise = 0.0
        self._cube_delay, self._cube_period = 0, 1
        self._cube_hist: list[tuple[np.ndarray, np.ndarray]] = []
        self._cube_seen = (np.zeros(3), np.eye(3))

    # ------------------------------------------------------------ stan
    def state(self) -> dict[str, np.ndarray]:
        """Wszystko, z czego licza sie obserwacja i nagroda - w ukladzie podstawy."""
        d = self.scene.data
        R, t = self.T_world2base[:3, :3], self.T_world2base[:3, 3]
        out = {"q": d.qpos[self.kin.qadr].copy(), "tcp": R @ d.site_xpos[self.site] + t}
        if self.task.cube:
            b = self.fields.cube_body
            out["cube_pos"] = R @ d.xpos[b] + t
            out["cube_rot"] = R @ d.xmat[b].reshape(3, 3)
            out["jaws"] = np.array([d.sensordata[a] for a in self.jaw_adr])
        return out

    def _obs(self, s: dict[str, np.ndarray]) -> np.ndarray:
        q = s["q"] + np.radians(self.np_random.normal(0.0, self._noise, 6)) if self._noise > 0 else s["q"]
        cube_pos, cube_rot = self._cube_seen if self.task.cube else (np.zeros(3), np.eye(3))
        obs = tk.observe(np, self.task, self.limits, q[None], self.q_cmd[None], s["tcp"][None],
                         self.prev_action[None], goal=self.goal[None],
                         cube_pos=cube_pos[None], cube_rot=cube_rot[None])
        return obs[0].astype(np.float32)

    def _perceive(self, s: dict[str, np.ndarray], force: bool = False) -> None:
        """Kostka w obserwacji tak, jak ja widza kamery: z opoznieniem, rzadziej, z szumem.

        Nagroda liczy sie z prawdy; polityka widzi to, co da jej `perception` na biurku.
        """
        if not self.task.cube:
            return
        self._cube_hist.append((s["cube_pos"].copy(), s["cube_rot"].copy()))
        del self._cube_hist[:-(self._cube_delay + 1)]
        if not force and self.t % self._cube_period:
            return
        pos, rot = self._cube_hist[max(0, len(self._cube_hist) - 1 - self._cube_delay)]
        pos = pos.copy()
        if self.rand.cube_noise > 0:
            pos[:2] += self.np_random.normal(0.0, self.rand.cube_noise, 2)
        if self.rand.fold_yaw:
            rot = tk.fold_yaw(np, rot)
        self._cube_seen = (pos, rot)

    # ------------------------------------------------------------ gym
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        options = options or {}
        rng = self.np_random
        m, d = self.scene.model, self.scene.data
        s = {k: float(v[0]) for k, v in self.rand.sample(rng, 1).items()}
        self.fields.apply(m, d, s)
        self._noise = self.rand.obs_noise_deg
        self._delay = [np.zeros(6)] * int(s["delay"])
        self._cube_delay, self._cube_period = int(s["cube_delay"]), int(s["cube_period"])
        self._cube_hist = []

        mujoco.mj_resetData(m, d)
        q0 = np.asarray(options["q"], float) if "q" in options else tk.sample_start(self.task, self.limits, rng, 1)[0]
        d.qpos[self.kin.qadr] = q0
        d.ctrl[self.scene.act_ids] = q0
        if self.task.name == "reach":
            self.goal = np.asarray(options["goal"], float) if "goal" in options else \
                tk.sample_goals(self.kin, self.task, rng, 1)[0]
        else:
            if "cube" in options:
                pos, quat = np.asarray(options["cube"][0], float), np.asarray(options["cube"][1], float)
            else:
                p, qt = tk.sample_cubes(self.task, rng, 1)
                pos, quat = p[0], qt[0]
            self.set_cube(pos, quat)
        mujoco.mj_forward(m, d)
        self.q_cmd = q0.copy()
        self.prev_action = np.zeros(6)
        self.t = 0
        s0 = self.state()
        self._perceive(s0, force=True)
        return self._obs(s0), {"success": False}

    def set_cube(self, pos_base: np.ndarray, quat_base: np.ndarray) -> None:
        """Kostka w pozie podanej w ukladzie podstawy (kwaternion w, x, y, z)."""
        d = self.scene.data
        T = self.scene.T_base2world
        q_base = np.zeros(4)
        mujoco.mju_mat2Quat(q_base, T[:3, :3].ravel())
        q_world = np.zeros(4)
        mujoco.mju_mulQuat(q_world, q_base, np.asarray(quat_base, float))
        a = self.cube_qadr
        d.qpos[a:a + 3] = T[:3, :3] @ pos_base + T[:3, 3]
        d.qpos[a + 3:a + 7] = q_world
        d.qvel[self.cube_dadr:self.cube_dadr + 6] = 0.0

    def step(self, action):
        action = np.clip(np.asarray(action, float), -1.0, 1.0)
        executed = action
        if self._delay:
            self._delay.append(action)
            executed = self._delay.pop(0)
        self.q_cmd = tk.apply_action(np, self.task, self.limits, self.q_cmd[None], executed[None])[0]
        d = self.scene.data
        d.ctrl[self.scene.act_ids] = self.q_cmd
        mujoco.mj_step(self.scene.model, d, nstep=self.task.substeps)
        self.t += 1

        s = self.state()
        r, success, failure = tk.reward(np, self.task, s["tcp"][None], action[None], self.prev_action[None],
                                        goal=self.goal[None], cube_pos=s.get("cube_pos", np.zeros(3))[None],
                                        jaw_contacts=s.get("jaws", np.zeros(2))[None])
        self.prev_action = action
        info = {"success": bool(success[0])}
        if self.task.name == "reach":
            info["distance"] = float(np.linalg.norm(self.goal - s["tcp"]))
        else:
            info["height"] = float(s["cube_pos"][2] - self.task.cube_half)
            info["grasped"] = bool(s["jaws"].min() > 0.5)
        terminated = bool(failure[0])
        truncated = self.t >= self.task.episode_steps
        self._perceive(s)
        return self._obs(s), float(r[0]), terminated, truncated, info

    def render(self):
        if self.render_mode != "rgb_array" or self.render_camera is None:
            return None
        return self.scene.render(self.render_camera)

    def close(self):
        self.scene.close()
