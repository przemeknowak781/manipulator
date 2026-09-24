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


def test_printable_sheet_is_detectable():
    board = Board()
    sheet = board.image(dpi=150)
    det = Collector(board, sheet.shape[::-1]).detect(sheet)
    assert det is not None
    assert len(det[0]) == (board.squares[0] - 1) * (board.squares[1] - 1)
