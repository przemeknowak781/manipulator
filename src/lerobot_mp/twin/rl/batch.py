"""Srodowisko wsadowe na GPU: tysiace swiatow MuJoCo Warp, obserwacje jako tensory torch.

    env = BatchEnv("reach", num_envs=4096)
    obs = env.reset()                                  # (N, obs_dim) na cuda
    obs, rew, done, info = env.step(actions)           # actions (N, 6) w [-1, 1]

Ta sama scena, te same wzory (`task`) co `env.TwinEnv` na CPU - roznica to
tylko, gdzie liczy sie fizyka. Na RTX A4500 w Shadow scena z kostka robi ok.
1,1 mln krokow fizyki na sekunde przy 8192 swiatach, a trajektoria jednego
swiata zgadza sie z MuJoCo na CPU do 0,001 st. po 2 s.

Konwencja jak w legged_gym / rsl_rl: srodowiska resetuja sie same, `done`
oznacza koniec epizodu w tym kroku, a zwracana obserwacja jest juz z nowego
epizodu. `info["time_outs"]` mowi, ktore skonczyly sie limitem czasu - PPO
dolicza im wartosc stanu zamiast zera.

Stan MuJoCo Warp jest wystawiony jako widoki torch (`wp.to_torch`) - bez kopii.
Fizyka to jeden przechwycony graf CUDA z `substeps` krokami; torch i Warp
synchronizujemy jawnie przed i po nim (dwa synchronizacje na krok polityki,
nieistotne przy kilkudziesieciu milisekundach fizyki).
"""

from __future__ import annotations

import mujoco
import numpy as np
import torch

from .. import scene as sc
from ..workspace import Workspace
from . import task as tk
from .env import scene_config
from .randomize import Randomization


def _warp():
    import warp as wp

    if hasattr(wp, "LOG_WARNING"):
        wp.config.log_level = wp.LOG_WARNING
    import mujoco_warp as mjw

    return wp, mjw


class BatchEnv:
    def __init__(self, task: str | tk.TaskConfig = "reach", num_envs: int = 4096, *,
                 workspace: Workspace | None = None, randomization: Randomization | None = None,
                 device: str = "cuda:0", seed: int = 0):
        wp, mjw = _warp()
        self.wp, self.mjw = wp, mjw
        self.task = task if isinstance(task, tk.TaskConfig) else tk.make_task(task)
        self.num_envs = N = int(num_envs)
        self.device = torch.device(device)
        self.rand = randomization if randomization is not None else Randomization()
        self.gen = torch.Generator(device=self.device).manual_seed(seed)
        self.np_rng = np.random.default_rng(seed)

        scene = sc.build(scene_config(self.task, workspace))
        self.scene = scene
        mjm = scene.model
        mjm.opt.timestep = self.task.timestep
        self.kin = scene.kin
        self.limits = tk.Limits.of(self.kin)
        mujoco.mj_forward(mjm, scene.data)

        batch = {k: N for k in ("actuator_gainprm", "actuator_biasprm", "dof_damping", "dof_armature",
                                "dof_frictionloss", "body_mass", "body_inertia", "geom_friction",
                                "body_invweight0", "dof_invweight0")}
        self.m = mjw.put_model(mjm, batch_sizes=batch)
        if hasattr(self.m.opt, "warn_overflow"):
            # Przepelnienie iteracji solvera to w MuJoCo zwykle "zatrzymaj sie tu",
            # nie blad - na CPU nikt o tym nie krzyczy. Tu drukowaloby sie co krok.
            self.m.opt.warn_overflow = 0
        cube = self.task.cube is not None
        self.d = mjw.put_data(mjm, scene.data, nworld=N, nconmax=32 if cube else 16,
                              njmax=192 if cube else 64)

        t = lambda a: wp.to_torch(a)                                            # noqa: E731
        self.qpos, self.qvel, self.ctrl = t(self.d.qpos), t(self.d.qvel), t(self.d.ctrl)
        self.qacc_ws = t(self.d.qacc_warmstart)
        self.site_xpos, self.xpos, self.xmat = t(self.d.site_xpos), t(self.d.xpos), t(self.d.xmat)
        self.sensordata = t(self.d.sensordata) if cube else None
        self.f = {k: t(getattr(self.m, k)) for k in batch}
        self.nominal = {k: v[0].clone() for k, v in self.f.items()}

        dev = self.device
        f32 = dict(dtype=torch.float32, device=dev)
        self.qadr = torch.as_tensor(self.kin.qadr, device=dev)
        self.dadr = torch.as_tensor(self.kin.dadr, device=dev)
        self.act = torch.as_tensor(scene.act_ids, device=dev)
        self.site = int(self.kin.site_id)
        T = scene.T_base2world
        self.R_b = torch.as_tensor(T[:3, :3], **f32)
        self.t_b = torch.as_tensor(T[:3, 3], **f32)
        self.qpos0 = torch.as_tensor(mjm.qpos0, **f32)
        self.lo = torch.as_tensor(self.limits.lo, **f32)
        self.hi = torch.as_tensor(self.limits.hi, **f32)
        self.home = torch.as_tensor(self.limits.home, **f32)
        if cube:
            b = mjm.body(self.task.cube).id
            self.cube_body, self.cube_geom = b, mjm.geom(self.task.cube).id
            self.cube_qadr = int(mjm.jnt_qposadr[mjm.body_jntadr[b]])
            self.cube_dadr = int(mjm.jnt_dofadr[mjm.body_jntadr[b]])
            self.jaw_adr = torch.as_tensor([mjm.sensor(f"{self.task.cube}_jaw{k}").adr[0] for k in range(2)],
                                           device=dev)
            qb = np.zeros(4)
            mujoco.mju_mat2Quat(qb, T[:3, :3].ravel())
            self.q_base = torch.as_tensor(qb, **f32)
        else:
            pool = tk.sample_goals(self.kin, self.task, self.np_rng, 20000)
            self.goal_pool = torch.as_tensor(pool, **f32)

        self.goal = torch.zeros(N, 3, **f32)
        self.q_cmd = self.home.repeat(N, 1)
        self.prev_action = torch.zeros(N, 6, **f32)
        self.pending = torch.zeros(N, 6, **f32)          # akcja czekajaca na opoznienie
        self.delay = torch.zeros(N, dtype=torch.bool, device=dev)
        self.t = torch.zeros(N, dtype=torch.long, device=dev)
        self.noise = float(np.radians(self.rand.obs_noise_deg))
        # Percepcja kostki (patrz Randomization.cube_*): historia w buforze pierscieniowym,
        # opoznienie i okres odswiezania osobno dla kazdego swiata.
        self.H = int(self.rand.cube_delay) + 1
        self.hist_pos = torch.zeros(self.H, N, 3, **f32)
        self.hist_rot = torch.eye(3, **f32).repeat(self.H, N, 1, 1)
        self.hist_i = 0
        self.cube_d = torch.zeros(N, dtype=torch.long, device=dev)
        self.cube_p = torch.ones(N, dtype=torch.long, device=dev)
        self.seen_pos = torch.zeros(N, 3, **f32)
        self.seen_rot = torch.eye(3, **f32).repeat(N, 1, 1)

        mjw.step(self.m, self.d)                          # kompilacja kerneli (raz, potem z cache)
        wp.synchronize_device()
        with wp.ScopedCapture() as cap:
            for _ in range(self.task.substeps):
                mjw.step(self.m, self.d)
        self.graph = cap.graph

    # ------------------------------------------------------------ pomocnicze
    def _u(self, *shape, lo=0.0, hi=1.0):
        return lo + (hi - lo) * torch.rand(*shape, generator=self.gen, device=self.device)

    def _state(self) -> dict[str, torch.Tensor]:
        R, t = self.R_b, self.t_b
        out = {"q": self.qpos[:, self.qadr], "tcp": (self.site_xpos[:, self.site] - t) @ R}
        if self.task.cube:
            out["cube_pos"] = (self.xpos[:, self.cube_body] - t) @ R
            out["cube_rot"] = R.T @ self.xmat[:, self.cube_body]
            out["jaws"] = self.sensordata[:, self.jaw_adr]
        return out

    def _obs(self, s: dict[str, torch.Tensor]) -> torch.Tensor:
        q = s["q"]
        if self.noise > 0:
            q = q + self.noise * torch.randn(q.shape, generator=self.gen, device=self.device)
        cube = self.task.cube is not None
        return tk.observe(torch, self.task, self.limits, q, self.q_cmd, s["tcp"], self.prev_action,
                          goal=self.goal, cube_pos=self.seen_pos if cube else None,
                          cube_rot=self.seen_rot if cube else None)

    def _perceive(self, s: dict[str, torch.Tensor]) -> None:
        """Nowy wpis historii kostki i odswiezenie tego, co widzi polityka (co `cube_p` taktow)."""
        self.hist_i = (self.hist_i + 1) % self.H
        self.hist_pos[self.hist_i] = s["cube_pos"]
        self.hist_rot[self.hist_i] = s["cube_rot"]
        upd = (self.t % self.cube_p) == 0
        if not upd.any():
            return
        idx = (self.hist_i - self.cube_d) % self.H
        ar = torch.arange(self.num_envs, device=self.device)
        pos = self.hist_pos[idx, ar].clone()
        rot = self.hist_rot[idx, ar]
        if self.rand.cube_noise > 0:
            pos[:, :2] += self.rand.cube_noise * torch.randn(self.num_envs, 2, generator=self.gen,
                                                             device=self.device)
        if self.rand.fold_yaw:
            rot = tk.fold_yaw(torch, rot)
        self.seen_pos = torch.where(upd[:, None], pos, self.seen_pos)
        self.seen_rot = torch.where(upd[:, None, None], rot, self.seen_rot)

    def _randomize(self, ids: torch.Tensor) -> None:
        """Nowa dynamika dla swiatow `ids` - mnozniki od nominalu, potem `set_const`."""
        s = {k: torch.as_tensor(v, dtype=torch.float32, device=self.device)
             for k, v in self.rand.sample(self.np_rng, len(ids)).items()}
        f, nom, act, dof = self.f, self.nominal, self.act, self.dadr
        f["actuator_gainprm"][ids[:, None], act, 0] = nom["actuator_gainprm"][act, 0] * s["kp"][:, None]
        f["actuator_biasprm"][ids[:, None], act, 1] = nom["actuator_biasprm"][act, 1] * s["kp"][:, None]
        f["dof_damping"][ids[:, None], dof] = nom["dof_damping"][dof] * s["damping"][:, None]
        f["dof_armature"][ids[:, None], dof] = nom["dof_armature"][dof] * s["armature"][:, None]
        f["dof_frictionloss"][ids[:, None], dof] = nom["dof_frictionloss"][dof] * s["frictionloss"][:, None]
        if self.task.cube:
            b, g = self.cube_body, self.cube_geom
            f["body_mass"][ids, b] = nom["body_mass"][b] * s["cube_mass"]
            f["body_inertia"][ids, b] = nom["body_inertia"][b] * s["cube_mass"][:, None]
            f["geom_friction"][ids, g, 0] = nom["geom_friction"][g, 0] * s["cube_friction"]
        self.delay[ids] = s["delay"] > 0.5
        self.cube_d[ids] = s["cube_delay"].long()
        self.cube_p[ids] = s["cube_period"].long()
        torch.cuda.synchronize(self.device)
        # Bez tego kontakty kostki o zmienionej masie licza sie "od innej kostki"
        # (body_invweight0) - na CPU konczylo sie to wystrzeleniem kostki.
        self.mjw.set_const(self.m, self.d)

    def _reset(self, ids: torch.Tensor) -> None:
        n = len(ids)
        if n == 0:
            return
        self._randomize(ids)
        tsk = self.task
        q0 = self.home.repeat(n, 1)
        noise = np.radians(tsk.start_noise_deg)
        q0[:, :5] += self._u(n, 5, lo=-noise, hi=noise)
        if tsk.cube:
            q0[:, 5] = self.lo[5] + self._u(n, lo=0.6, hi=1.0) * (self.hi[5] - self.lo[5])
        q0 = torch.clamp(q0, self.lo, self.hi)

        qpos = self.qpos0.repeat(n, 1)
        qpos[:, self.qadr] = q0
        if tsk.cube:
            r = self._u(n, lo=tsk.cube_radius[0], hi=tsk.cube_radius[1])
            b = self._u(n, lo=tsk.cube_bearing[0], hi=tsk.cube_bearing[1])
            p = torch.stack([r * torch.cos(b), r * torch.sin(b), torch.full_like(r, tsk.cube_half)], 1)
            yaw = self._u(n, lo=-np.pi, hi=np.pi)
            qb = self.q_base
            # obrot wokol z w ukladzie podstawy, potem do swiata: q_base * q_yaw
            cw, sw = torch.cos(yaw / 2), torch.sin(yaw / 2)
            quat = torch.stack([qb[0] * cw - qb[3] * sw, qb[1] * cw + qb[2] * sw,
                                qb[2] * cw - qb[1] * sw, qb[0] * sw + qb[3] * cw], 1)
            a = self.cube_qadr
            qpos[:, a:a + 3] = p @ self.R_b.T + self.t_b
            qpos[:, a + 3:a + 7] = quat
            # Pierwsza detekcja na starcie: kostka lezy, wiec cala historia to ta sama poza.
            z0, o1 = torch.zeros_like(yaw), torch.ones_like(yaw)
            R = torch.stack([torch.stack([torch.cos(yaw), -torch.sin(yaw), z0], -1),
                             torch.stack([torch.sin(yaw), torch.cos(yaw), z0], -1),
                             torch.stack([z0, z0, o1], -1)], -2)
            self.hist_pos[:, ids] = p
            self.hist_rot[:, ids] = R
            self.seen_pos[ids] = p
            self.seen_rot[ids] = tk.fold_yaw(torch, R) if self.rand.fold_yaw else R
        else:
            k = torch.randint(0, len(self.goal_pool), (n,), generator=self.gen, device=self.device)
            self.goal[ids] = self.goal_pool[k]
        self.qpos[ids] = qpos
        self.qvel[ids] = 0.0
        self.qacc_ws[ids] = 0.0
        self.ctrl[ids[:, None], self.act] = q0
        self.q_cmd[ids] = q0
        self.prev_action[ids] = 0.0
        self.pending[ids] = 0.0
        self.t[ids] = 0

    # ------------------------------------------------------------------ api
    def reset(self) -> torch.Tensor:
        self._reset(torch.arange(self.num_envs, device=self.device))
        torch.cuda.synchronize(self.device)
        self.mjw.kinematics(self.m, self.d)
        self.wp.synchronize_device()
        return self._obs(self._state())

    @torch.no_grad()
    def step(self, action: torch.Tensor):
        action = torch.clamp(action.to(self.device, torch.float32), -1.0, 1.0)
        executed = torch.where(self.delay[:, None], self.pending, action)
        self.pending = action
        self.q_cmd = tk.apply_action(torch, self.task, self.limits, self.q_cmd, executed)
        self.ctrl[:, self.act] = self.q_cmd
        torch.cuda.synchronize(self.device)
        self.wp.capture_launch(self.graph)
        self.wp.synchronize_device()
        self.t += 1

        s = self._state()
        rew, success, failure = tk.reward(torch, self.task, s["tcp"], action, self.prev_action, goal=self.goal,
                                          cube_pos=s.get("cube_pos"), jaw_contacts=s.get("jaws"))
        if self.task.cube:
            self._perceive(s)
        # Straznik: swiat, ktory mimo wszystko wybuchl, konczy epizod bez nagrody,
        # zanim jego liczby zatruja gradient.
        broken = ~torch.isfinite(self.qvel).all(1) | (self.qvel.abs().amax(1) > 100.0)
        rew = torch.where(broken, torch.zeros_like(rew), rew)
        self.prev_action = action
        time_out = self.t >= self.task.episode_steps
        terminated = failure | broken
        done = terminated | time_out
        info = {"success": success, "time_outs": time_out & ~terminated, "broken": broken}
        if self.task.name == "reach":
            info["distance"] = ((self.goal - s["tcp"]) ** 2).sum(1).sqrt()
        else:
            info["height"] = s["cube_pos"][:, 2] - self.task.cube_half

        ids = done.nonzero().flatten()
        if len(ids):
            self._reset(ids)
            torch.cuda.synchronize(self.device)
            self.mjw.kinematics(self.m, self.d)
            self.wp.synchronize_device()
            s = self._state()
        return self._obs(s), rew, done, info

    @property
    def obs_dim(self) -> int:
        return self.task.obs_dim

    @property
    def act_dim(self) -> int:
        return self.task.act_dim
