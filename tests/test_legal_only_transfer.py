"""Acceptance coverage for legal-only policy transfer into native MCTS."""

from __future__ import annotations

import chess
import numpy as np
import pytest

from engine.config import MCTSConfig
from engine.encoding import board_to_planes, legal_move_indices
from engine.mcts import MCTS, _load_native


def _native_session(native, simulations: int = 8):
    cfg = MCTSConfig(simulations=simulations, dirichlet_epsilon=0.0)
    return native.MctsSession(
        chess.STARTING_FEN,
        simulations,
        {
            "simulations": cfg.simulations,
            "c_puct": cfg.c_puct,
            "dirichlet_alpha": cfg.dirichlet_alpha,
            "dirichlet_epsilon": cfg.dirichlet_epsilon,
            "draw_contempt": cfg.draw_contempt,
            "claim_draw": cfg.claim_draw,
        },
        False,
        None,
    )


def test_evaluate_legal_matches_full_policy_gather(tiny_evaluator) -> None:
    if not hasattr(tiny_evaluator, "evaluate_legal"):
        pytest.skip("NetEvaluator.evaluate_legal has not been restored")

    boards = [chess.Board(), chess.Board()]
    boards[1].push_uci("e2e4")
    planes = np.stack([board_to_planes(board) for board in boards])
    legal = [np.fromiter(legal_move_indices(board), dtype=np.int64) for board in boards]
    offsets = np.asarray([0, len(legal[0]), len(legal[0]) + len(legal[1])], dtype=np.int64)
    indices = np.concatenate(legal)

    full_logits, full_values = tiny_evaluator.evaluate_planes(planes)
    legal_logits, legal_values = tiny_evaluator.evaluate_legal(planes, indices, offsets)
    expected = np.concatenate(
        [full_logits[row, row_indices] for row, row_indices in enumerate(legal)]
    )
    np.testing.assert_allclose(legal_logits, expected, rtol=0, atol=1e-6)
    np.testing.assert_allclose(legal_values, full_values, rtol=0, atol=1e-6)


def test_evaluate_legal_avoids_dense_policy_d2h(tiny_evaluator, monkeypatch) -> None:
    """Sparse gather must not materialize a full (N, 4672) host policy array."""
    if not hasattr(tiny_evaluator, "evaluate_legal"):
        pytest.skip("NetEvaluator.evaluate_legal has not been restored")

    board = chess.Board()
    planes = board_to_planes(board)[None, ...]
    indices = np.fromiter(legal_move_indices(board), dtype=np.int64)
    offsets = np.asarray([0, len(indices)], dtype=np.int64)

    def _boom(*_args, **_kwargs):
        raise AssertionError("evaluate_legal must not call evaluate_planes")

    monkeypatch.setattr(tiny_evaluator, "evaluate_planes", _boom)
    legal_logits, legal_values = tiny_evaluator.evaluate_legal(planes, indices, offsets)
    assert legal_logits.shape == (len(indices),)
    assert legal_values.shape == (1,)


def test_evaluate_legal_preserves_csr_order_with_empty_rows(tiny_evaluator) -> None:
    boards = [chess.Board(), chess.Board(), chess.Board()]
    planes = np.stack([board_to_planes(board) for board in boards])
    legal = np.fromiter(legal_move_indices(boards[1]), dtype=np.int32)
    offsets = np.asarray([0, 0, len(legal), len(legal)], dtype=np.int32)

    full_logits, full_values = tiny_evaluator.evaluate_planes(planes)
    packed, values = tiny_evaluator.evaluate_legal(planes, legal, offsets)

    np.testing.assert_allclose(packed, full_logits[1, legal], rtol=0, atol=1e-6)
    np.testing.assert_allclose(values, full_values, rtol=0, atol=1e-6)


def test_pending_legal_indices_follow_python_encoding_order() -> None:
    native = _load_native()
    if native is None or not hasattr(native, "MctsSession"):
        pytest.skip("native MctsSession is unavailable")
    session = _native_session(native)
    if not hasattr(session, "pending_legal_indices"):
        pytest.skip("native pending_legal_indices API is unavailable")

    expected = list(legal_move_indices(chess.Board()))
    assert list(session.pending_legal_indices()) == expected


def test_legal_apply_matches_full_policy_search(fixed_evaluator) -> None:
    native = _load_native()
    if native is None or not hasattr(native, "MctsSession"):
        pytest.skip("native MctsSession is unavailable")
    legal_session = _native_session(native)
    if not all(
        hasattr(legal_session, name)
        for name in ("pending_legal_indices", "apply_eval_legal")
    ):
        pytest.skip("native legal-only MctsSession APIs are unavailable")

    full = MCTS(fixed_evaluator, MCTSConfig(simulations=8, dirichlet_epsilon=0.0)).run(
        chess.Board(), simulations=8, add_noise=False
    )
    while not legal_session.done():
        planes = np.asarray(legal_session.positions_needing_eval(), dtype=np.float32)
        if planes.shape[0] == 0:
            break
        indices = np.asarray(legal_session.pending_legal_indices(), dtype=np.int64)
        offsets = np.asarray([0, len(indices)], dtype=np.int64)
        logits, values = fixed_evaluator.evaluate_legal(planes, indices, offsets)
        # A session advances one pending position at a time; the native API
        # therefore accepts its scalar value rather than a length-one batch.
        legal_session.apply_eval_legal(logits, float(values[0]))
    legal = legal_session.result()

    assert list(legal["moves"]) == [move.uci() for move in full.moves]
    np.testing.assert_array_equal(np.asarray(legal["visits"]), full.visits)
    np.testing.assert_allclose(np.asarray(legal["q_values"]), full.q_values, atol=1e-6)
