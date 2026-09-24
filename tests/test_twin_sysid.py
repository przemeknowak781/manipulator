"""Identyfikacja dynamiki serw: nagranie z symulacji o ZNANEJ dynamice ma dac ja z powrotem."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mujoco")

from lerobot_mp.twin.rl.randomize import Dynamics  # noqa: E402
from lerobot_mp.twin.rl.sysid import JOINTS, Recording, Replayer, excitation, fit, plan_target  # noqa: E402
from lerobot_mp.twin.robots import SO101  # noqa: E402


def synthetic_recording(truth: Dynamics, delay: float, joints=("shoulder_lift", "elbow_flex"), hold=0.7):
    plan = excitation(SO101.home, joints=joints, hold=hold)
    hz = 50.0
    t = np.arange(0, plan[-1][0] + 0.5, 1 / hz)
    cur = np.array([SO101.home[j] for j in JOINTS])
    cmd = []
    for tk in t:                                   # jak nadzor: cel z planu, najwyzej 120 st/s
        want = np.array([plan_target(plan, tk)[j] for j in JOINTS])
        cur = cur + np.clip(want - cur, -120 / hz, 120 / hz)
        cmd.append(cur.copy())
    cmd = np.array(cmd)
    q = Replayer().run(Recording(t, cmd, np.full_like(cmd, np.nan)), truth, delay)
    meas = q + np.random.default_rng(0).normal(0, 0.05, q.shape)
    meas[1::2] = np.nan                             # serwa czytane co drugi takt
    return Recording(t, cmd, meas, {"backend": "test", "time": "test"})


def test_identification_recovers_known_damping_and_delay():
    truth = Dynamics(kp=1.0, damping=1.4, armature=1.0, frictionloss=1.0)
    rec = synthetic_recording(truth, delay=0.04)
    dyn, base = fit(rec)
    assert dyn.fit_deg < 0.5 * base                 # symulacja po dopasowaniu wyraznie blizej nagrania
    assert dyn.fit_deg < 0.12                       # na poziomie szumu (0,05 st.) i kwantyzacji
    assert dyn.damping == pytest.approx(1.4, rel=0.2)
    assert dyn.delay / 20 == pytest.approx(0.04, abs=0.012)


def test_recording_roundtrip(tmp_path):
    rec = Recording(np.arange(3.0), np.ones((3, 6)), np.full((3, 6), np.nan), {"a": 1})
    back = Recording.load(rec.save(tmp_path / "r.npz"))
    assert np.allclose(back.t, rec.t) and back.meta == {"a": 1}
