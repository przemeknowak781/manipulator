"""Watek podgladu 3D: najnowsza poza wygrywa, petla sterowania nie czeka."""

from __future__ import annotations

import time

import numpy as np
import pytest

from lerobot_mp.preview.stream import PreviewStream


class FakeCamera:
    def __init__(self) -> None:
        self.orbits: list[tuple[float, float]] = []
        self.zooms: list[float] = []

    def orbit(self, d_azimuth: float, d_elevation: float) -> None:
        self.orbits.append((d_azimuth, d_elevation))

    def zoom(self, factor: float) -> None:
        self.zooms.append(factor)


class FakeRenderer:
    """Renderer, ktory rysuje wolno i zapisuje, co dostal."""

    def __init__(self, delay: float = 0.0) -> None:
        self.camera = FakeCamera()
        self.delay = delay
        self.poses: list[dict[str, float]] = []

    def render(self, joints_deg, ee_target=None):
        if self.delay:
            time.sleep(self.delay)
        self.poses.append(dict(joints_deg))
        value = int(joints_deg.get("shoulder_pan", 0.0)) % 256
        return np.full((8, 6, 3), value, dtype=np.uint8)


def wait_until(predicate, timeout: float = 4.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


@pytest.fixture
def stream():
    made: list[PreviewStream] = []

    def build(delay: float = 0.0, hz: float = 1000.0) -> PreviewStream:
        s = PreviewStream(FakeRenderer(delay), model=None, hz=hz)
        made.append(s)
        return s

    yield build
    for s in made:
        s.close()


def test_first_frame_is_ready_before_the_loop_starts(stream):
    """Panel musi istniec od pierwszego kadru - inaczej okno zmienia szerokosc."""
    s = stream()
    s.start({"shoulder_pan": 7.0})
    assert s.image is not None
    assert s.image[0, 0, 0] == 7


def test_submit_does_not_wait_for_the_render(stream):
    s = stream(delay=0.25)
    s.start({"shoulder_pan": 0.0})

    started = time.monotonic()
    for angle in range(1, 6):
        s.submit({"shoulder_pan": float(angle)})
    assert time.monotonic() - started < 0.05


def test_latest_pose_wins_without_a_backlog(stream):
    """Poz posrednich nie odrabiamy - inaczej podglad zostawalby w tyle."""
    s = stream(delay=0.05)
    s.start({"shoulder_pan": 0.0})

    submitted = 40
    for angle in range(1, submitted + 1):
        s.submit({"shoulder_pan": float(angle)})

    assert wait_until(lambda: s.image[0, 0, 0] == submitted)
    # 40 zadan przy 50 ms na render to 2 s zaleglosci, gdyby kolejkowac.
    assert len(s.renderer.poses) < 5


def test_an_unchanged_pose_is_not_drawn_a_second_time(stream):
    """Nieruchome ramie nie ma po co rysowac ponownie.

    Ten watek konkuruje o rdzen z detekcja dloni, ktora siedzi na sciezce
    opoznienia - a robot stoi przez wiekszosc czasu pracy (sprzeglo
    rozlaczone, pauza, dojechany cel).
    """
    s = stream()
    s.start({"shoulder_pan": 3.0})
    drawn = len(s.renderer.poses)

    for _ in range(20):
        s.submit({"shoulder_pan": 3.0})
    time.sleep(0.2)

    assert len(s.renderer.poses) == drawn


def test_a_moved_joint_is_drawn_again(stream):
    s = stream()
    s.start({"shoulder_pan": 3.0})
    s.submit({"shoulder_pan": 9.0})
    assert wait_until(lambda: s.image[0, 0, 0] == 9)


def test_moving_the_target_marker_alone_is_enough_to_redraw(stream):
    """Sam znacznik celu tez zmienia obraz, choc poza stawow stoi w miejscu."""
    s = stream()
    s.start({"shoulder_pan": 3.0})
    drawn = len(s.renderer.poses)

    s.submit({"shoulder_pan": 3.0}, ee_target=(0.3, 0.0, 0.1))
    assert wait_until(lambda: len(s.renderer.poses) > drawn)


def test_camera_moves_are_applied_in_the_render_thread(stream):
    s = stream()
    s.start({"shoulder_pan": 0.0})

    s.orbit(-8.0, 0.0)
    s.zoom(1.1)

    assert wait_until(lambda: s.renderer.camera.orbits and s.renderer.camera.zooms)
    assert s.renderer.camera.orbits == [(-8.0, 0.0)]
    assert s.renderer.camera.zooms == [1.1]


def test_render_error_does_not_kill_the_thread(stream):
    s = stream()
    s.start({"shoulder_pan": 0.0})

    boom = {"shoulder_pan": 1.0}
    original = s.renderer.render

    def failing(joints_deg, ee_target=None):
        if joints_deg == boom:
            raise RuntimeError("render padl")
        return original(joints_deg, ee_target)

    s.renderer.render = failing
    s.submit(boom)
    s.submit({"shoulder_pan": 44.0})

    assert wait_until(lambda: s.image[0, 0, 0] == 44)


def test_close_joins_the_thread(stream):
    s = stream()
    s.start({"shoulder_pan": 0.0})
    thread = s._thread
    s.close()
    assert thread is not None and not thread.is_alive()
