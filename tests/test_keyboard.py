"""Tryb `keys`: prowadzenie koncowki chwytaka klawiszami.

Testy sprawdzaja SKUTEK FIZYCZNY (gdzie pojechala koncowka), a nie wartosc
konkretnego stawu - w kalibracji SO-101 znaki stawow bywaja odwrotne do
intuicji, wiec test "wartosc rosnie" przepuscilby odwrocony kierunek.
"""

from __future__ import annotations

import math

import pytest

from lerobot_mp.config import JOINT_NAMES
from lerobot_mp.control.keyboard import (
    ARROW_DOWN,
    ARROW_LEFT,
    ARROW_RIGHT,
    ARROW_UP,
    KeyboardPilot,
)

DT = 1.0 / 30.0
W, S, A, D, Q, E = (ord(c) for c in "wsadqe")


def measured(cfg) -> dict[str, float]:
    return dict(cfg.safety.home)


def drive(pilot: KeyboardPilot, cfg, key: int | None, steps: int = 10):
    """Trzyma klawisz przez `steps` krokow petli i zwraca ostatni wynik."""
    out = None
    for _ in range(steps):
        if key is not None:
            pilot.press(key)
        out = pilot.update(DT, engaged=True, measured=measured(cfg))
    assert out is not None
    return out


def tip(pilot: KeyboardPilot, targets: dict[str, float]) -> tuple[float, float, float]:
    return pilot.kinematics.forward(
        targets["shoulder_pan"],
        targets["shoulder_lift"],
        targets["elbow_flex"],
        targets["wrist_flex"],
    )


def reach_of(pilot: KeyboardPilot, point: tuple[float, float, float]) -> float:
    """Odleglosc koncowki od osi obrotu podstawy, rzutowana na poziom."""
    return math.hypot(point[0] - pilot.cfg.geometry.pan_axis_x, point[1])


def test_first_step_reproduces_the_measured_pose(cfg):
    """Zalaczenie nie moze ruszyc ramieniem: cel startowy to poza zmierzona."""
    pilot = KeyboardPilot(cfg)
    out = pilot.update(DT, engaged=True, measured=measured(cfg))
    assert out.targets is not None
    for name in JOINT_NAMES:
        assert out.targets[name] == pytest.approx(cfg.safety.home[name], abs=0.5)


def test_e_extends_the_reach_and_q_pulls_it_back(cfg):
    pilot = KeyboardPilot(cfg)
    start = tip(pilot, drive(pilot, cfg, None, steps=1).targets)
    out_e = drive(pilot, cfg, E)
    assert reach_of(pilot, tip(pilot, out_e.targets)) > reach_of(pilot, start) + 0.01

    pilot.release()
    start = tip(pilot, drive(pilot, cfg, None, steps=1).targets)
    out_q = drive(pilot, cfg, Q)
    assert reach_of(pilot, tip(pilot, out_q.targets)) < reach_of(pilot, start) - 0.01


def test_a_and_d_move_sideways_in_opposite_directions(cfg):
    pilot = KeyboardPilot(cfg)
    start = tip(pilot, drive(pilot, cfg, None, steps=1).targets)
    left = tip(pilot, drive(pilot, cfg, A).targets)

    pilot.release()
    drive(pilot, cfg, None, steps=1)
    right = tip(pilot, drive(pilot, cfg, D).targets)

    assert (left[1] - start[1]) * (right[1] - start[1]) < 0
    assert abs(left[1] - start[1]) > 0.01


def test_a_and_d_keep_the_reach_while_moving_sideways(cfg):
    """Ruch w bok ma isc w bok, a nie wysuwac ramienia przy okazji."""
    pilot = KeyboardPilot(cfg)
    start = tip(pilot, drive(pilot, cfg, None, steps=1).targets)
    moved = tip(pilot, drive(pilot, cfg, D).targets)
    sideways = abs(moved[1] - start[1])
    assert abs(reach_of(pilot, moved) - reach_of(pilot, start)) < 0.25 * sideways


def test_w_lifts_the_tip_and_s_lowers_it(cfg):
    pilot = KeyboardPilot(cfg)
    start = tip(pilot, drive(pilot, cfg, None, steps=1).targets)
    up = tip(pilot, drive(pilot, cfg, W).targets)

    pilot.release()
    drive(pilot, cfg, None, steps=1)
    down = tip(pilot, drive(pilot, cfg, S).targets)

    assert up[2] > start[2] + 0.01
    assert down[2] < start[2] - 0.01


def test_side_arrows_roll_the_wrist_both_ways(cfg):
    pilot = KeyboardPilot(cfg)
    start = drive(pilot, cfg, None, steps=1).targets["wrist_roll"]
    left = drive(pilot, cfg, ARROW_LEFT[0]).targets["wrist_roll"]

    pilot.release()
    drive(pilot, cfg, None, steps=1)
    right = drive(pilot, cfg, ARROW_RIGHT[0]).targets["wrist_roll"]

    assert left < start < right


def test_vertical_arrows_open_and_close_the_gripper(cfg):
    pilot = KeyboardPilot(cfg)
    start = drive(pilot, cfg, None, steps=1).targets["gripper"]
    closing = drive(pilot, cfg, ARROW_DOWN[0]).targets["gripper"]

    pilot.release()
    drive(pilot, cfg, None, steps=1)
    opening = drive(pilot, cfg, ARROW_UP[0]).targets["gripper"]

    assert closing < start < opening


@pytest.mark.parametrize("arrow", [ARROW_LEFT, ARROW_UP, ARROW_RIGHT, ARROW_DOWN])
def test_every_known_arrow_encoding_is_accepted(cfg, arrow):
    """Windows, GTK i Cocoa koduja strzalki inaczej - wszystkie maja dzialac."""
    pilot = KeyboardPilot(cfg)
    for code in arrow:
        assert pilot.press(code) is True


def test_unrelated_keys_are_left_to_the_application(cfg):
    pilot = KeyboardPilot(cfg)
    for key in (ord("m"), ord("h"), 27, -1, 255):
        assert pilot.press(key) is False


def test_motion_stops_when_repeats_stop_arriving(cfg):
    """Bez zdarzenia "puszczono" ruch ma wygasnac sam - i potem stac."""
    pilot = KeyboardPilot(cfg)
    drive(pilot, cfg, W)

    # Ruch trwa jeszcze `hold_timeout` po ostatnim powtorzeniu - to jest cel
    # tego progu. Sprawdzamy dopiero to, co dzieje sie PO jego uplynieciu.
    coasting = int(cfg.keyboard.hold_timeout / DT) + 2
    for _ in range(coasting):
        pilot.update(DT, engaged=True, measured=measured(cfg))
    settled = tip(pilot, pilot.update(DT, engaged=True, measured=measured(cfg)).targets)

    for _ in range(30):
        out = pilot.update(DT, engaged=True, measured=measured(cfg))
    assert tip(pilot, out.targets) == pytest.approx(settled, abs=1e-9)


def test_the_key_keeps_the_arm_moving_while_held(cfg):
    """Autopowtarzanie ma dawac jazde ciagla, a nie skok na wcisniecie."""
    pilot = KeyboardPilot(cfg)
    start = reach_of(pilot, tip(pilot, drive(pilot, cfg, None, steps=1).targets))
    first = reach_of(pilot, tip(pilot, drive(pilot, cfg, E, steps=5).targets))
    second = reach_of(pilot, tip(pilot, drive(pilot, cfg, E, steps=5).targets))
    assert second > first > start


def test_holding_past_the_reach_does_not_wind_up(cfg):
    """Punkt zadany nie moze uciekac poza zasieg, bo powrot trwalby tyle, co wyjscie."""
    pilot = KeyboardPilot(cfg)
    out = drive(pilot, cfg, E, steps=300)
    far = reach_of(pilot, tip(pilot, out.targets))

    back = drive(pilot, cfg, Q, steps=3)
    assert reach_of(pilot, tip(pilot, back.targets)) < far - 0.005


def test_target_stays_inside_the_workspace_ring(cfg):
    """Cel ma sie zatrzymac na granicy pierscienia, a nie dojezdzac do krawedzi
    zasiegu, gdzie IK zle sie warunkuje i ramie miota stawami."""
    pilot = KeyboardPilot(cfg)
    for key in (E, Q):
        out = drive(pilot, cfg, key, steps=400)
        radius = reach_of(pilot, out.ee_target)
        assert cfg.workspace.radius_min - 1e-6 <= radius <= cfg.workspace.radius_max + 1e-6
        pilot.release()


def test_reaching_out_stops_at_the_edge_instead_of_drifting(cfg):
    """Wysuw ma dojechac do granicy i tam ZOSTAC.

    Z pozycji domowej tego wysuwu jest tylko okolo 14 cm - to cecha geometrii
    przy nieruchomym kacie narzedzia, nie usterka. Wazne jest, ze dalsze
    trzymanie klawisza nic juz nie psuje: cel nie ucieka poza zasieg.
    """
    pilot = KeyboardPilot(cfg)
    start = reach_of(pilot, tip(pilot, drive(pilot, cfg, None, steps=1).targets))
    far = reach_of(pilot, tip(pilot, drive(pilot, cfg, E, steps=120).targets))
    farther = reach_of(pilot, tip(pilot, drive(pilot, cfg, E, steps=120).targets))

    assert far > start + 0.10
    assert farther == pytest.approx(far, abs=1e-3)


def test_tool_pitch_starts_inside_the_workspace_range(cfg):
    pilot = KeyboardPilot(cfg)
    pilot.update(DT, engaged=True, measured=measured(cfg))
    assert cfg.workspace.pitch_min_deg <= pilot._pitch <= cfg.workspace.pitch_max_deg


def test_disengaging_stops_the_arm_and_re_engaging_does_not_jump(cfg):
    pilot = KeyboardPilot(cfg)
    drive(pilot, cfg, W, steps=20)

    idle = pilot.update(DT, engaged=False, measured=measured(cfg))
    assert idle.targets is None
    assert idle.engaged is False
    assert pilot.seeded is False

    # Po ponownym zalaczeniu cel siada na pozie ZMIERZONEJ, a nie na zapamietanej.
    again = pilot.update(DT, engaged=True, measured=measured(cfg))
    for name in JOINT_NAMES:
        assert again.targets[name] == pytest.approx(cfg.safety.home[name], abs=0.5)


def test_gripper_and_roll_stay_inside_joint_limits(cfg):
    pilot = KeyboardPilot(cfg)
    for key in (ARROW_UP[0], ARROW_RIGHT[0]):
        out = drive(pilot, cfg, key, steps=500)
        pilot.release()
    for name in ("gripper", "wrist_roll"):
        jc = cfg.joint(name)
        assert jc.min <= out.targets[name] <= jc.max

    pilot = KeyboardPilot(cfg)
    for key in (ARROW_DOWN[0], ARROW_LEFT[0]):
        out = drive(pilot, cfg, key, steps=500)
        pilot.release()
    for name in ("gripper", "wrist_roll"):
        jc = cfg.joint(name)
        assert jc.min <= out.targets[name] <= jc.max


def test_keyboard_output_keeps_the_watchdog_satisfied(cfg):
    """Bez kamery nie ma dloni - nadzor nie moze przez to zamrozic ruchu."""
    out = KeyboardPilot(cfg).update(DT, engaged=True, measured=measured(cfg))
    assert out.hand_present is True
