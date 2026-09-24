"""Intrynsyki z ChArUco: rogi w konwencji `projectPoints` i K odzyskane z renderu blizniaka."""

from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
cv2 = pytest.importorskip("cv2")

from lerobot_mp.twin.calib.intrinsics import Board, Collector, simulate  # noqa: E402

K = np.array([[600.0, 0, 330.0], [0, 590.0, 235.0], [0, 0, 1]])


def test_calibration_recovers_the_camera_matrix_of_the_simulated_camera():
    res, _ = simulate(K, n=14, seed=1)
    assert res.trusted, res.reason
    assert abs(res.K[0, 0] - K[0, 0]) < 2.0 and abs(res.K[1, 1] - K[1, 1]) < 2.0
    assert np.abs(res.K[:2, 2] - K[:2, 2]).max() < 2.0
    assert res.rms_px < 0.3


def test_the_same_view_twice_is_refused():
    """Dwadziescia kadrow z jednego miejsca daje pewne siebie i zle K."""
    _, col = simulate(K, n=5, seed=2)
    first = col.views[0]
    fake = np.zeros((480, 640), np.uint8)
    assert col.novelty(first) is not None
    assert col.add(fake)[0] is False


def _synthetic_session(max_tilt_deg: float, seed: int, n: int = 14) -> Collector:
    """Rogi tablicy rzutowane znanym K (bez renderu): przesuwana po kadrze, pochylona do `max_tilt_deg`."""
    W, H = 640, 480
    Kt = np.array([[610.0, 0, 322], [0, 610.0, 238], [0, 0, 1]])
    dist = np.array([0.08, -0.15, 0, 0, 0.0])
    rng = np.random.default_rng(seed)
    board = Board()
    col = Collector(board, (W, H))
    obj = np.asarray(board.cv().getChessboardCorners(), float)
    for _ in range(3000):
        if len(col.views) >= n:
            break
        axis = np.r_[rng.normal(size=2), 0.0]
        axis /= np.linalg.norm(axis)
        R = cv2.Rodrigues(axis * np.radians(rng.uniform(0, max_tilt_deg)))[0]
        z = rng.uniform(0.25, 0.55)
        c = np.linalg.inv(Kt) @ np.array([rng.uniform(0.1, 0.9) * W, rng.uniform(0.1, 0.9) * H, 1.0]) * z
        px, _ = cv2.projectPoints(obj, cv2.Rodrigues(R)[0], c - R @ obj.mean(0), Kt, dist)
        px = px.reshape(-1, 2) + rng.normal(0, 0.15, (len(obj), 2))
        ok = (px[:, 0] >= 0) & (px[:, 0] < W) & (px[:, 1] >= 0) & (px[:, 1] < H)
        if ok.sum() >= 12:
            col.detect = lambda img, o=obj[ok], p=px[ok]: (o, p)
            col.add(None)
    return col


def test_board_held_parallel_to_the_camera_is_not_trusted():
    """Tablica tylko przesuwana (pochylenie do 2 st.): fx wychodzilo 424-27340 zamiast 610 - i "zaufane"."""
    for seed in range(3):
        res = _synthetic_session(2.0, seed).solve()
        assert not res.trusted and "rownolegla" in res.reason
    # Ta sama sesja z pochylaniem o 30 st.: K w 1%.
    res = _synthetic_session(30.0, 0).solve()
    assert res.trusted, res.reason
    assert abs(res.K[0, 0] - 610.0) < 6.0


def test_once_the_frame_is_covered_only_tilted_views_are_taken():
    col = _synthetic_session(2.0, 0, n=8)
    assert col.coverage >= col.prefer_tilt_coverage and np.hypot(*col.tilt_spread()) < col.min_tilt_spread
    flat = col.views[0]
    moved = type(flat)(flat.obj, flat.img, flat.centre + [0.3, 0.0], flat.scale, flat.normal)
    assert "pochyl" in col.novelty(moved)
    a = np.radians(25.0)
    tilted = type(flat)(flat.obj, flat.img, flat.centre + [0.3, 0.0], flat.scale,
                        np.array([0.0, np.sin(a), np.cos(a)]))
    assert col.novelty(tilted) is None


def test_printable_sheet_is_detectable():
    board = Board()
    sheet = board.image(dpi=150)
    det = Collector(board, sheet.shape[::-1]).detect(sheet)
    assert det is not None
    assert len(det[0]) == (board.squares[0] - 1) * (board.squares[1] - 1)
