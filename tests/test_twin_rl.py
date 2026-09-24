"""Srodowiska RL blizniaka: poprawnosc API, wykonalnosc zadan, zgodnosc CPU/GPU, stabilnosc fizyki."""

from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
gym = pytest.importorskip("gymnasium")
torch = pytest.importorskip("torch")

import lerobot_mp.twin.rl  # noqa: E402,F401
from lerobot_mp.twin.rl import task as tk  # noqa: E402
from lerobot_mp.twin.rl.env import TwinEnv  # noqa: E402
from lerobot_mp.twin.rl.policy import Policy, PolicyMeta  # noqa: E402
from lerobot_mp.twin.rl.randomize import Dynamics, Randomization  # noqa: E402

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="MuJoCo Warp potrzebuje GPU z CUDA")


@pytest.mark.parametrize("name", ["LeRobotMP/TwinReach-v0", "LeRobotMP/TwinLift-v0"])
def test_env_passes_the_gymnasium_checker(name):
    from gymnasium.utils.env_checker import check_env

    env = gym.make(name)
    try:
        check_env(env.unwrapped, skip_render_check=True)
    finally:
        env.close()


def test_scripted_ik_expert_solves_reach():
    """Dowod, ze zadanie jest wykonalne w granicach akcji - zanim cokolwiek sie uczy."""
    env = TwinEnv("reach", randomization=Randomization.none())
    try:
        for ep in range(5):
            env.reset(seed=ep)
            q_goal = env.kin.to_q(env.kin.ik(env.goal, seed=env.kin.from_q(env.q_cmd)).joints)
            for _ in range(env.task.episode_steps):
                a = np.zeros(6)
                a[:5] = np.clip((q_goal[:5] - env.q_cmd[:5]) / env.task.arm_step, -1, 1)
                _, _, term, trunc, info = env.step(a)
                if term or trunc:
                    break
            assert info["success"], f"epizod {ep}: {info['distance'] * 1000:.1f} mm od celu"
    finally:
        env.close()


def test_randomized_cube_stays_on_the_table_under_random_actions():
    """Losowe akcje wciskaja kostke w blat; kiedys wylatywala z predkoscia 10-180 m/s,
    bo zmiana masy nie przeliczala stalych kontaktu (`mj_setConst`)."""
    env = TwinEnv("lift", randomization=Randomization())
    d = env.scene.data
    try:
        for ep in range(12):
            env.reset(seed=100 + ep)
            for _ in range(env.task.episode_steps):
                _, _, term, trunc, info = env.step(env.action_space.sample())
                v = float(np.linalg.norm(d.qvel[env.cube_dadr:env.cube_dadr + 3]))
                assert v < 3.0 and abs(info["height"]) < 0.5, f"epizod {ep}: kostka {v:.1f} m/s"
                if term or trunc:
                    break
    finally:
        env.close()


def test_randomization_is_relative_to_the_nominal_model():
    env = TwinEnv("lift", randomization=Randomization())
    try:
        m = env.scene.model
        kp0 = env.fields.kp.copy()
        for seed in range(5):
            env.reset(seed=seed)
        env.fields.apply(m, env.scene.data, {k: 1.0 for k in ("kp", "damping", "armature", "frictionloss",
                                                              "cube_mass", "cube_friction")})
        assert np.allclose(m.actuator_gainprm[env.scene.act_ids, 0], kp0)
        assert m.body_mass[env.fields.cube_body] == pytest.approx(env.fields.cube_mass)
    finally:
        env.close()


@pytest.mark.parametrize("name", ["reach", "lift"])
def test_observation_and_reward_are_the_same_in_numpy_and_torch(name):
    """Jedna definicja zadania dla CPU, GPU i ramienia - wzory nie moga sie rozjechac."""
    task = tk.make_task(name)
    env = TwinEnv(task, randomization=Randomization.none())
    lim = env.limits
    env.close()
    rng = np.random.default_rng(0)
    n = 16
    q = rng.uniform(lim.lo, lim.hi, (n, 6))
    q_cmd = rng.uniform(lim.lo, lim.hi, (n, 6))
    tcp, goal, cube = rng.normal(0, 0.2, (n, 3)), rng.normal(0, 0.2, (n, 3)), rng.normal(0, 0.2, (n, 3))
    rot = np.array([np.linalg.qr(rng.normal(size=(3, 3)))[0] for _ in range(n)])
    act, prev = rng.uniform(-1, 1, (n, 6)), rng.uniform(-1, 1, (n, 6))
    jaws = rng.integers(0, 2, (n, 2)).astype(float)

    def run(xp, conv):
        obs = tk.observe(xp, task, lim, conv(q), conv(q_cmd), conv(tcp), conv(prev), goal=conv(goal),
                         cube_pos=conv(cube), cube_rot=conv(rot))
        r, s, f = tk.reward(xp, task, conv(tcp), conv(act), conv(prev), goal=conv(goal), cube_pos=conv(cube),
                            jaw_contacts=conv(jaws))
        nxt = tk.apply_action(xp, task, lim, conv(q_cmd), conv(act))
        return [np.asarray(x) for x in (obs, r, s, f, nxt)]

    a = run(np, lambda x: x)
    b = run(torch, lambda x: torch.as_tensor(x, dtype=torch.float64))
    for x, y in zip(a, b):
        assert np.allclose(x, y, atol=1e-9)


def test_actions_respect_the_supervisor_limits():
    env = TwinEnv("reach", randomization=Randomization.none())
    try:
        env.reset(seed=0)
        for _ in range(60):
            env.step(np.ones(6))
        assert np.all(env.q_cmd <= env.limits.hi + 1e-9)
        # chwytak: najwyzej `grip_step` zakresu na takt
        env.reset(seed=0)
        g0 = env.q_cmd[5]
        env.step(-np.ones(6))
        span = env.limits.hi[5] - env.limits.lo[5]
        assert g0 - env.q_cmd[5] == pytest.approx(min(env.task.grip_step * span, g0 - env.limits.lo[5]))
    finally:
        env.close()


def test_policy_file_roundtrip(tmp_path):
    task = tk.make_task("reach")
    from dataclasses import asdict

    pol = Policy(PolicyMeta(task=asdict(task), hidden=[32, 32], obs_dim=task.obs_dim, act_dim=6))
    pol.norm.update(torch.randn(100, task.obs_dim) * 3 + 1)
    obs = np.random.default_rng(0).normal(size=(4, task.obs_dim)).astype(np.float32)
    path = pol.save(tmp_path / "p" / "policy.pt")
    back = Policy.load(path)
    assert np.allclose(pol.act(obs), back.act(obs))
    assert back.task == task
    assert (tmp_path / "p" / "meta.json").is_file()


def test_randomization_around_measured_dynamics_centres_on_it():
    dyn = Dynamics(kp=1.3, damping=0.7, delay=1.2, source="pomiar")
    r = Randomization.around(dyn, spread=0.5)
    s = r.sample(np.random.default_rng(0), 2000)
    assert np.median(s["kp"]) == pytest.approx(1.3, rel=0.02)
    assert r.max_delay == 3
    assert s["kp"].min() >= 1.3 * r.kp[0] - 1e-9


def test_policy_sees_the_cube_as_the_cameras_would():
    """Obserwacja kostki spozniona i ze zlozonym obrotem - nagroda dalej z prawdy."""
    from dataclasses import replace

    rand = replace(Randomization.none(), cube_delay=2, cube_period=1, fold_yaw=True)
    env = TwinEnv("lift", randomization=rand)
    try:
        env.reset(seed=0)
        env._cube_delay = 2                          # losowane z {0..2}; tu wymuszamy najgorsze
        start = env.state()["cube_pos"].copy()
        moved = start + np.array([0.03, 0.0, 0.0])
        env.set_cube(moved, np.array([np.cos(0.8), 0, 0, np.sin(0.8)]))   # obrot 91,7 st.
        mujoco.mj_forward(env.scene.model, env.scene.data)
        seen = []
        for _ in range(3):
            obs, *_ = env.step(np.zeros(6))
            seen.append(obs[15:18].copy())
        assert np.allclose(seen[0], start, atol=2e-3)           # jeszcze stara poza (opoznienie)
        assert np.allclose(seen[2], env.state()["cube_pos"], atol=2e-3)
        yaw = np.arctan2(obs[22], obs[21])
        assert abs(np.degrees(yaw)) <= 45.0 + 1e-6               # 91,7 st. widziane jak 1,7 st.
    finally:
        env.close()


@cuda
def test_gpu_batch_env_matches_the_cpu_env():
    """Ta sama scena, start i akcje: MuJoCo Warp (float32) i MuJoCo CPU daja te same obserwacje."""
    from lerobot_mp.twin.rl.batch import BatchEnv

    cpu = TwinEnv("reach", randomization=Randomization.none())
    cpu.reset(seed=3)
    gpu = BatchEnv("reach", num_envs=2, randomization=Randomization.none())
    gpu.reset()
    dev = gpu.device
    gpu.qpos[:] = torch.as_tensor(cpu.scene.data.qpos, dtype=torch.float32, device=dev)
    gpu.qvel[:] = 0
    q_cmd = torch.as_tensor(cpu.q_cmd, dtype=torch.float32, device=dev)
    gpu.ctrl[:, gpu.act] = q_cmd
    gpu.q_cmd[:] = q_cmd
    gpu.goal[:] = torch.as_tensor(cpu.goal, dtype=torch.float32, device=dev)
    rng = np.random.default_rng(0)
    for _ in range(20):
        a = rng.uniform(-1, 1, 6)
        oc, rc, *_ = cpu.step(a)
        og, rg, _, _ = gpu.step(torch.as_tensor(a, device=dev).repeat(2, 1))
        assert np.abs(og[0].cpu().numpy() - oc).max() < 1e-3
        assert float(rg[0]) == pytest.approx(rc, abs=1e-3)
    cpu.close()


def test_runner_drives_the_simulated_arm_through_the_supervisor():
    """Polityka na zywym blizniaku (sim): ruch idzie przez nadzor, a epizod konczy sie limitem."""
    import time
    from dataclasses import asdict

    from lerobot_mp.twin.rl.runner import PolicyRunner
    from lerobot_mp.twin.runtime import Twin
    from lerobot_mp.twin.workspace import Workspace

    task = tk.make_task("reach", episode_steps=12)
    pol = Policy(PolicyMeta(task=asdict(task), hidden=[8], obs_dim=task.obs_dim, act_dim=6))
    with torch.no_grad():
        for p in pol.actor.parameters():
            p.zero_()
        pol.actor[-1].bias[0] = 1.0                     # stale "obracaj podstawe w lewo"
    twin = Twin(Workspace())
    try:
        twin.connect("sim")
        start = twin.joints()["shoulder_pan"]
        runner = PolicyRunner(twin, pol)
        runner.start()
        deadline = time.monotonic() + 5.0
        while runner.status.running and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not runner.status.running
        assert runner.status.stopped_because == "koniec epizodu"
        assert twin.joints()["shoulder_pan"] > start + 5.0
    finally:
        twin.close()
