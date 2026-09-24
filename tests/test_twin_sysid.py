"""Identyfikacja dynamiki serw: nagranie z symulacji o ZNANEJ dynamice ma dac ja z powrotem.

Prawdy sa celowo "krzywe": poza siatka startowa dopasowania i z kp oraz tarciem
innymi niz w modelu. Dopasowanie ma oddac to, co z tego ruchu da sie wyznaczyc
(tlumienie, armatura, opoznienie), a kp i tarcia NIE udawac - stary test
sprawdzal tylko tlumienie i opoznienie na prawdzie z siatki startowej i nie
zauwazyl kp 0,72 przy prawdzie 1,0.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mujoco")

from lerobot_mp.twin.rl.randomize import Dynamics, Randomization  # noqa: E402
from lerobot_mp.twin.rl.sysid import (  # noqa: E402
    JOINTS, Recording, Replayer, check_recording, excitation, fit, identify, plan_target, record,
)
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


@pytest.mark.parametrize("truth, delay", [
    (Dynamics(kp=0.8, damping=1.4, armature=1.3, frictionloss=0.7), 0.055),
    (Dynamics(kp=0.75, damping=1.1, armature=1.15, frictionloss=0.6), 0.07),
])
def test_identification_recovers_what_the_motion_determines(truth, delay):
    rec = synthetic_recording(truth, delay)
    out = identify(rec)
    dyn = out.dyn
    assert dyn.fit_deg < 0.5 * out.base              # symulacja po dopasowaniu wyraznie blizej nagrania
    assert dyn.fit_deg < 0.08                        # blisko szumu (0,05 st.)
    assert dyn.damping == pytest.approx(truth.damping, rel=0.1)
    assert dyn.armature == pytest.approx(truth.armature, rel=0.1)
    assert dyn.delay / 20 == pytest.approx(delay, abs=0.005)
    # kp i tarcie nie sa wyznaczalne z tego ruchu - zostaja z modelu i opis to mowi.
    assert dyn.kp == 1.0 and dyn.frictionloss == 1.0
    assert "kp" in dyn.source and "frictionloss" in dyn.source
    assert np.isinf(out.band["kp"]) and out.band["armature"] <= 0.2
    # Obietnica randomizacji: prawdziwe ramie lezy w zakresie wokol wyniku (dla tego, co zmierzono).
    rand = Randomization.around(dyn)
    for name in ("damping", "armature"):
        lo, hi = getattr(rand, name)
        assert lo * getattr(dyn, name) <= getattr(truth, name) <= hi * getattr(dyn, name), name


def test_fit_keeps_the_old_shape():
    truth = Dynamics(kp=1.0, damping=1.4, armature=1.0, frictionloss=1.0)
    dyn, base = fit(synthetic_recording(truth, 0.04))
    assert isinstance(dyn, Dynamics) and base > dyn.fit_deg


def test_truncated_recording_is_rejected():
    """Petla ramienia padla w trakcie: reszta nagrania to NaN - nie wolno z tego dopasowac dynamiki."""
    rec = synthetic_recording(Dynamics(), 0.04)
    cut = len(rec.t) // 4
    rec.measured[cut:] = np.nan
    with pytest.raises(ValueError, match="bez pomiaru"):
        check_recording(rec)
    rec.command[cut:] = np.nan
    with pytest.raises(ValueError, match="bez rozkazu"):
        identify(rec)


def test_recording_roundtrip(tmp_path):
    rec = Recording(np.arange(3.0), np.ones((3, 6)), np.full((3, 6), np.nan), {"a": 1})
    back = Recording.load(rec.save(tmp_path / "r.npz"))
    assert np.allclose(back.t, rec.t) and back.meta == {"a": 1}


# ------------------------------------------------------------ nagranie na blizniaku
@pytest.fixture
def twin():
    from lerobot_mp.twin.runtime import Twin
    from lerobot_mp.twin.workspace import Workspace

    tw = Twin(Workspace())
    try:
        yield tw
    finally:
        tw.close()


def short_plan():
    home = dict(SO101.home)
    return [(0.0, home), (0.3, dict(home, shoulder_pan=5.0)), (0.6, home)]


def test_record_approaches_the_start_smoothly_before_recording(twin):
    """Ramie stoi 40 st. od domu: nagranie zaczyna sie w domu, a dojazd idzie rampa, nie skokiem."""
    with twin.lock:
        twin.scene.set_joints(dict(SO101.home, shoulder_pan=40.0))
    twin.connect("sim")
    seen = []
    orig = twin.set_target

    def spy(joints, owner=None):
        seen.append(float(joints.get("shoulder_pan", np.nan)))
        orig(joints, owner=owner)
    twin.set_target = spy
    rec = record(twin, short_plan(), settle=0.2)
    assert abs(rec.command[0, 0] - SO101.home["shoulder_pan"]) < 1.0      # dojazd PRZED nagraniem
    assert twin.owner is None and not twin._engaged                       # ramie oddane
    # Nagrywane rozkazy nie skacza: najwyzej ~5 st. (skok planu) miedzy taktami.
    assert np.abs(np.diff(rec.command[:, 0])).max() < 6.0


def test_record_stops_when_the_arm_loop_dies(twin):
    """Kabel wypadl w trakcie: `record` ma rzucic, a nie oddac nagrania z samymi NaN."""
    twin.connect("sim")
    b = twin._backend
    calls = [0]
    real = b.read_joints

    def failing():
        calls[0] += 1
        if calls[0] > 150:
            raise OSError("kabel wypadl")
        return real()
    b.read_joints = failing
    with pytest.raises(RuntimeError, match="petla ramienia|odebrane|przerwana"):
        record(twin, short_plan() + [(3.0, dict(SO101.home))], settle=0.2)


def test_record_is_preempted_by_home(twin):
    import threading

    twin.connect("sim")
    threading.Timer(1.8, twin.home).start()
    with pytest.raises(RuntimeError, match="przerwan"):
        record(twin, short_plan() + [(3.0, dict(SO101.home))], settle=0.2)
    assert twin.owner is None


def test_record_refuses_while_another_owner_drives(twin):
    twin.connect("sim")
    twin.claim("polityka")
    with pytest.raises(RuntimeError, match="ramie zajete: polityka"):
        record(twin, short_plan())
