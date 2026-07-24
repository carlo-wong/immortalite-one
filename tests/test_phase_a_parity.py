"""Acceptance checks that protect Phase A's quality-neutral behavior."""

from __future__ import annotations

import chess
import numpy as np
import pytest

from engine.config import Config, MCTSConfig
from engine.encoding import board_to_planes, legal_move_indices
from engine.mcts import MCTS, _board_root_fen_and_moves, _load_native, _native_search_ready
from engine.selfplay import _config_from_dict, _config_to_dict, _termination_reason


def test_config_recipe_round_trip_preserves_fractional_fields() -> None:
    """Compare values after serialization; source text may contain ``1 / 3``."""
    cfg = Config()
    cfg.mcts.draw_contempt = 1 / 3
    cfg.train.draw_penalty = 1 / 3

    restored = _config_from_dict(_config_to_dict(cfg))

    assert restored.mcts == cfg.mcts
    assert restored.train == cfg.train


@pytest.mark.parametrize(
    ("fen", "expected"),
    [
        ("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1", "checkmate"),
        ("7k/8/8/8/8/8/8/K7 w - - 0 1", "insufficient_material"),
    ],
)
def test_terminal_routing_is_stable(fen: str, expected: str) -> None:
    board = chess.Board(fen)
    assert _termination_reason(
        board.outcome(claim_draw=True),
        hit_max_moves=False,
        no_legal_moves=False,
    ) == expected


def test_native_legal_order_matches_encoding_mapping() -> None:
    native = _load_native()
    if native is None or not hasattr(native, "legal_move_indices_fen"):
        pytest.skip("native legal-move API is unavailable")

    board = chess.Board(
        "r1b2bnr/pp2k1qp/n1pp2p1/4PpQ1/2P1P3/2NP3N/PP4PP/R1B1KB1R b KQ - 2 10"
    )
    expected = [(index, move.uci()) for index, move in legal_move_indices(board).items()]
    assert list(native.legal_move_indices_fen(board.fen())) == expected


def test_native_search_result_fingerprint_is_repeatable(fixed_evaluator) -> None:
    native = _load_native()
    if native is None or not _native_search_ready(native):
        pytest.skip("native MctsSession is unavailable")

    cfg = MCTSConfig(simulations=12, dirichlet_epsilon=0.0)
    results = [
        MCTS(fixed_evaluator, cfg).run(chess.Board(), simulations=12, add_noise=False)
        for _ in range(2)
    ]
    assert [tuple(move.uci() for move in result.moves) for result in results] == [
        tuple(move.uci() for move in results[0].moves)
    ] * 2
    np.testing.assert_array_equal(results[0].visits, results[1].visits)
    np.testing.assert_allclose(results[0].q_values, results[1].q_values, atol=0, rtol=0)


def test_native_history_routing_preserves_repetition_context() -> None:
    native = _load_native()
    if native is None or not hasattr(native, "fill_planes_fen"):
        pytest.skip("native encoding API is unavailable")

    board = chess.Board()
    for uci in ("g1f3", "g8f6", "f3g1", "f6g8") * 2:
        board.push_uci(uci)
    root_fen, moves = _board_root_fen_and_moves(board)
    np.testing.assert_array_equal(
        np.asarray(native.fill_planes_fen(root_fen, moves), dtype=np.float32),
        board_to_planes(board),
    )
