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

    task = tk.make_task(name, limit_penalty=0.5, hold_still=0.5, end_on_success=2)
    cube[: n // 2, 2] = 0.09                                   # polowa "podniesiona" - sukces i kara za ruch
    streak = rng.integers(0, 3, n)

    def run(xp, conv):
        obs = tk.observe(xp, task, lim, conv(q), conv(q_cmd), conv(tcp), conv(prev), goal=conv(goal),
                         cube_pos=conv(cube), cube_rot=conv(rot))
        r, s, f = tk.reward(xp, task, conv(tcp), conv(act), conv(prev), goal=conv(goal), cube_pos=conv(cube),
                            jaw_contacts=conv(jaws), q_cmd=conv(q_cmd), limits=lim)
        nxt = tk.apply_action(xp, task, lim, conv(q_cmd), conv(act))
        st, done = tk.success_streak(xp, task, streak if xp is np else torch.as_tensor(streak), s)
        return [np.asarray(x) for x in (obs, r, s, f, nxt, st, done)]

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


def test_no_randomization_keeps_the_measured_arm_and_the_camera_model():
    """--no-rand = zmierzona dynamika bez rozrzutu (a nie Menagerie), percepcja kostki jak z kamer."""
    dyn = Dynamics(kp=0.8, damping=1.4, delay=1.6, source="identyfikacja")
    r = Randomization.around(dyn, spread=0.0)
    s = r.sample(np.random.default_rng(0), 500)
    assert np.allclose(s["kp"], 0.8) and np.allclose(s["damping"], 1.4) and np.all(s["delay"] == 2)
    assert np.allclose(s["cube_mass"], 1.0)
    assert r.centre.source == "identyfikacja" and not r.randomized
    assert r.fold_yaw and r.cube_delay > 0 and r.cube_period > 1
    nom = Randomization.nominal(dyn)
    assert not nom.randomized and nom.centre is dyn and nom.min_delay == nom.max_delay == 2
    assert not nom.fold_yaw and nom.cube_delay == 0 and nom.obs_noise_deg == 0.0
    assert Randomization.around(dyn, 1.0).randomized and Randomization().randomized
    assert not Randomization.none().randomized


def test_train_plan_does_not_label_a_nominal_run_as_randomized():
    from lerobot_mp.twin.cli import _train_plan, parser

    dyn = Dynamics(kp=0.8, delay=1.2, source="identyfikacja")
    rand, evals = _train_plan(dyn, no_rand=True, spread=1.0)
    assert rand.centre is dyn and [e[1] for e in evals] == ["cpu_nominal"]
    assert evals[0][2].centre is dyn
    rand, evals = _train_plan(dyn, no_rand=False, spread=1.0)
    assert [e[1] for e in evals] == ["cpu_nominal", "cpu_rand"] and evals[1][2] is rand and rand.randomized
    # panel ruszajacy ramieniem nie wystawia sie domyslnie na siec
    assert parser().parse_args(["ui"]).host == "127.0.0.1"
    assert parser().parse_args(["ui", "--host", "0.0.0.0"]).host == "0.0.0.0"


def test_nominal_evaluation_runs_on_the_identified_arm():
    from dataclasses import asdict

    from lerobot_mp.twin.rl.evaluate import evaluate
    from lerobot_mp.twin.workspace import Workspace

    task = tk.make_task("reach", episode_steps=5)
    pol = Policy(PolicyMeta(task=asdict(task), hidden=[8], obs_dim=task.obs_dim, act_dim=6))
    ws = Workspace()
    ws.dynamics = Dynamics(kp=0.8, delay=1.0, source="identyfikacja").to_dict()
    res = evaluate(pol, 1, workspace=ws)
    assert res["centre"] == "identyfikacja" and res["randomized"] is False
    res = evaluate(pol, 1, randomization=Randomization.none(), workspace=ws)
    assert res["centre"] == "menagerie" and res["randomized"] is False


def test_lift_episode_ends_after_the_success_streak(monkeypatch):
    """Po sukcesie lift-v2 wymachiwal kostka do limitow przez 170 taktow - teraz epizod sie konczy."""
    real = tk.reward

    def always_success(xp, task, *a, **kw):
        r, s, f = real(xp, task, *a, **kw)
        return r, s | True, f

    monkeypatch.setattr(tk, "reward", always_success)
    task = tk.make_task("lift")
    assert task.end_on_success == 10 and tk.make_task("reach").end_on_success == 0
    env = TwinEnv(task, randomization=Randomization.none())
    try:
        env.reset(seed=0)
        for k in range(1, 11):
            _, _, term, trunc, info = env.step(np.zeros(6))
            assert not term and trunc == (k == 10) and info["finished"] == (k == 10)
    finally:
        env.close()
    env = TwinEnv(tk.make_task("reach", episode_steps=15), randomization=Randomization.none())
    try:
        env.reset(seed=0)
        ends = [env.step(np.zeros(6))[3] for _ in range(15)]
        assert ends == [False] * 14 + [True]                  # reach: tylko limit czasu
    finally:
        env.close()


def test_old_lift_policy_file_stops_on_success_too():
    from dataclasses import asdict

    old = asdict(tk.make_task("lift"))
    for k in ("end_on_success", "limit_penalty", "limit_margin", "hold_still"):
        old.pop(k)
    meta = PolicyMeta(task=old, hidden=[8], obs_dim=tk.OBS_DIMS["lift"], act_dim=6)
    assert meta.task_config().end_on_success == 10


def test_lift_reward_pushes_away_from_joint_limits_and_holds_still_after_success():
    task = tk.make_task("lift")
    env = TwinEnv("reach", randomization=Randomization.none())
    lim = env.limits
    env.close()
    tcp = np.array([[0.2, 0.0, 0.1]])
    cube = tcp.copy()
    jaws = np.ones((1, 2))
    a0 = np.zeros((1, 6))

    def r(q_cmd, action=a0, cube_pos=cube):
        return tk.reward(np, task, tcp, action, action, cube_pos=cube_pos, jaw_contacts=jaws,
                         q_cmd=q_cmd[None], limits=lim)[0][0]

    home = lim.home.copy()
    at_limits = home.copy()
    at_limits[[0, 2, 3, 4]] = [lim.lo[0], lim.lo[2], lim.hi[3], lim.lo[4]]     # poza z wymachu lift-v2
    assert r(home) - r(at_limits) == pytest.approx(4 * task.limit_penalty)
    near = home.copy()
    near[0] = lim.lo[0] + task.limit_margin / 2
    assert 0 < r(home) - r(near) < task.limit_penalty
    moving = np.array([[1.0, -1.0, 0.0, 0.0, 0.0, 0.0]])
    assert r(home) - r(home, moving) == pytest.approx(2 * task.hold_still)           # kostka w gorze
    low = cube.copy()
    low[0, 2] = 0.03
    assert r(home, moving, low) == pytest.approx(r(home, a0, low))                   # jeszcze nie sukces


def test_observed_tcp_matches_the_joint_angles():
    """Runner liczy TCP z FK katow; w treningu TCP bylo sprzed ostatniego kroku fizyki (4-8 mm w ruchu)."""
    env = TwinEnv("reach", randomization=Randomization.none())
    try:
        env.reset(seed=0)
        rng = np.random.default_rng(0)
        worst = 0.0
        for _ in range(40):
            env.step(np.sign(rng.uniform(-1, 1, 6)))
            s = env.state()
            fk = env.kin.tcp(env.kin.from_q(s["q"]))[:3, 3]
            worst = max(worst, float(np.linalg.norm(fk - s["tcp"])))
        assert worst < 1e-4, f"TCP {worst * 1000:.2f} mm od FK(q)"
    finally:
        env.close()


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


@cuda
def test_fine_tuning_waits_for_the_critic_before_moving_the_actor(tmp_path):
    """Douczanie z krytykiem od zera psulo reach-v1 (98% -> 9%): aktor ma stac, dopoki
    krytyk sie nie rozgrzeje, a polityka zapisana z krytykiem rozgrzewki nie potrzebuje."""
    from dataclasses import replace

    from lerobot_mp.twin.rl.ppo import PPOConfig, train

    cfg = PPOConfig(num_envs=64, horizon=8, iterations=2, hidden=(32, 32))
    train("reach", cfg, randomization=Randomization.none(), out_dir=tmp_path / "a")
    base = Policy.load(tmp_path / "a" / "policy.pt")
    assert base.critic_state is not None                  # krytyk jedzie w pliku do douczania

    obs = np.random.default_rng(0).normal(size=(16, base.meta.obs_dim)).astype(np.float32)

    def actor_moved(init):
        before = init.act(obs)
        tuned = train("reach", replace(cfg, critic_warmup=2), randomization=Randomization.none(), init=init)
        return not np.array_equal(before, tuned.to("cpu").act(obs))

    assert actor_moved(base)                              # krytyk z pliku - aktor uczy sie od razu
    base.critic_state = None                              # jak polityka sprzed zapisu krytyka
    assert not actor_moved(base)


@cuda
def test_gpu_applies_the_same_multi_tick_action_delay_as_the_cpu():
    """Opoznienie z identyfikacji (np. 2 takty) - GPU scinalo je do 1 taktu."""
    from dataclasses import replace

    from lerobot_mp.twin.rl.batch import BatchEnv

    rand = replace(Randomization.none(), min_delay=2, max_delay=2)
    cpu = TwinEnv("reach", randomization=rand)
    cpu.reset(seed=3)
    assert len(cpu._delay) == 2
    gpu = BatchEnv("reach", num_envs=2, randomization=rand)
    gpu.reset()
    assert gpu.delay_n.tolist() == [2, 2]
    dev = gpu.device
    gpu.qpos[:] = torch.as_tensor(cpu.scene.data.qpos, dtype=torch.float32, device=dev)
    gpu.qvel[:] = 0
    q_cmd = torch.as_tensor(cpu.q_cmd, dtype=torch.float32, device=dev)
    gpu.ctrl[:, gpu.act] = q_cmd
    gpu.q_cmd[:] = q_cmd
    gpu.goal[:] = torch.as_tensor(cpu.goal, dtype=torch.float32, device=dev)
    rng = np.random.default_rng(1)
    for _ in range(12):
        a = rng.uniform(-1, 1, 6)
        oc, *_ = cpu.step(a)
        og, *_ = gpu.step(torch.as_tensor(a, device=dev).repeat(2, 1))
        assert np.abs(gpu.q_cmd[0].cpu().numpy() - cpu.q_cmd).max() < 1e-5
        assert np.abs(og[0].cpu().numpy() - oc).max() < 1e-3
    cpu.close()


@cuda
def test_gpu_ends_episodes_after_the_success_streak(monkeypatch):
    from lerobot_mp.twin.rl.batch import BatchEnv

    real = tk.reward

    def always_success(xp, task, *a, **kw):
        r, s, f = real(xp, task, *a, **kw)
        return r, s | True, f

    monkeypatch.setattr(tk, "reward", always_success)
    gpu = BatchEnv(tk.make_task("reach", end_on_success=3), num_envs=4, randomization=Randomization.none())
    gpu.reset()
    for k in range(1, 4):
        _, _, done, info = gpu.step(torch.zeros(4, 6, device=gpu.device))
        assert bool(done.all()) == (k == 3) and bool(info["finished"].all()) == (k == 3)
    assert bool(info["time_outs"].all())                     # PPO dolicza wartosc stanu
    assert gpu.t.tolist() == [0] * 4 and gpu.streak.tolist() == [0] * 4


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
