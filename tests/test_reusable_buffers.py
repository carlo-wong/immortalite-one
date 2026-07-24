"""Regression tests for evaluator staging-buffer reuse."""

from __future__ import annotations

import chess
import numpy as np

from engine.encoding import POLICY_SIZE, board_to_planes


def _boards(count: int) -> list[chess.Board]:
    board = chess.Board()
    boards: list[chess.Board] = []
    for _ in range(count):
        boards.append(board.copy())
        board.push(next(iter(board.legal_moves)))
    return boards


def test_repeated_batch_and_plane_evaluation_preserves_outputs(tiny_evaluator) -> None:
    boards = _boards(3)
    planes = np.stack([board_to_planes(board) for board in boards])

    batch_a = tiny_evaluator.evaluate_batch(boards)
    plane_a = tiny_evaluator.evaluate_planes(planes)
    batch_b = tiny_evaluator.evaluate_batch(boards)
    plane_b = tiny_evaluator.evaluate_planes(planes)

    for first, second in ((batch_a, batch_b), (plane_a, plane_b)):
        np.testing.assert_allclose(first[0], second[0], rtol=0, atol=0)
        np.testing.assert_allclose(first[1], second[1], rtol=0, atol=0)
    np.testing.assert_allclose(batch_a[0], plane_a[0], rtol=0, atol=1e-6)
    np.testing.assert_allclose(batch_a[1], plane_a[1], rtol=0, atol=1e-6)


def test_batch_capacity_grows_once_then_reuses_staging(tiny_evaluator) -> None:
    tiny_evaluator.evaluate_batch(_boards(1))
    initial_capacity = tiny_evaluator._batch_cap
    initial_buffer = tiny_evaluator._planes_buf

    tiny_evaluator.evaluate_batch(_boards(initial_capacity + 1))
    grown_capacity = tiny_evaluator._batch_cap
    grown_buffer = tiny_evaluator._planes_buf
    tiny_evaluator.evaluate_batch(_boards(2))

    assert initial_capacity >= 1
    assert grown_capacity >= initial_capacity + 1
    assert grown_buffer is tiny_evaluator._planes_buf
    assert grown_buffer is not initial_buffer


def test_empty_inputs_have_stable_contract(tiny_evaluator) -> None:
    batch_logits, batch_values = tiny_evaluator.evaluate_batch([])
    planes_logits, planes_values = tiny_evaluator.evaluate_planes(
        np.empty((0, 20, 8, 8), dtype=np.float32)
    )

    assert batch_logits.shape == planes_logits.shape == (0, POLICY_SIZE)
    assert batch_values.shape == planes_values.shape == (0,)
