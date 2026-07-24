"""Broad deterministic parity checks for the native core and Zero's Python semantics."""

from __future__ import annotations

import random

import chess
import numpy as np
import pytest

from engine import _native
from engine.config import MCTSConfig
from engine.encoding import POLICY_SIZE, board_to_planes, legal_move_indices, move_to_index
from engine.mcts import MCTS
from engine._python_mcts import PythonMCTS


class FixedEvaluator:
    """Position-independent, non-uniform policy with a fixed value."""

    def __init__(self, seed: int, value: float) -> None:
        self.logits = np.random.default_rng(seed).normal(
            0.0, 1.0, POLICY_SIZE
        ).astype(np.float32)
        self.value = float(value)

    def evaluate(self, board: chess.Board) -> tuple[np.ndarray, float]:
        del board
        return self.logits.copy(), self.value

    def evaluate_planes(self, planes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = int(planes.shape[0])
        return (
            np.repeat(self.logits[None, :], n, axis=0),
            np.full(n, self.value, dtype=np.float32),
        )


def _native_position(board: chess.Board) -> tuple[np.ndarray, dict[int, str]]:
    root = board.root()
    history = [move.uci() for move in board.move_stack]
    planes = np.asarray(
        _native.fill_planes_fen(root.fen(), history if history else None),
        dtype=np.float32,
    )
    moves = {
        int(index): uci
        for index, uci in _native.legal_move_indices_fen(board.fen())
    }
    return planes, moves


def _assert_position_parity(board: chess.Board) -> None:
    native_planes, native_moves = _native_position(board)
    np.testing.assert_array_equal(native_planes, board_to_planes(board))

    expected = {index: move.uci() for index, move in legal_move_indices(board).items()}
    assert native_moves == expected, board.fen()
    assert len(native_moves) == board.legal_moves.count()
    for index, uci in native_moves.items():
        move = chess.Move.from_uci(uci)
        assert int(_native.move_to_index_fen(board.fen(), uci)) == index
        assert move_to_index(move, board) == index


def test_random_walk_board_movegen_and_encoding_parity() -> None:
    """Exercise hundreds of reachable positions with reproducible random games."""
    rng = random.Random(20260724)
    positions = 0
    for _ in range(16):
        board = chess.Board()
        for _ in range(80):
            _assert_position_parity(board)
            positions += 1
            moves = list(board.legal_moves)
            if not moves or board.is_game_over(claim_draw=True):
                break
            board.push(rng.choice(moves))
    assert positions >= 500


@pytest.mark.parametrize(
    "fen",
    [
        # Both sides may castle.
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
        # Legal en passant.
        "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
        # White and black promotions, including underpromotions.
        "4k3/P7/8/8/8/8/7p/4K3 w - - 0 1",
        "4k3/P7/8/8/8/8/7p/4K3 b - - 0 1",
        # Double check and pinned-piece-heavy positions.
        "4k3/8/8/8/1b6/8/4r3/4K3 w - - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        # Terminal positions.
        "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",
        "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1",
    ],
)
def test_special_position_parity(fen: str) -> None:
    _assert_position_parity(chess.Board(fen))


@pytest.mark.parametrize(
    ("fen", "seed", "value"),
    [
        (chess.STARTING_FEN, 1, 0.0),
        ("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", 2, 0.35),
        ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", 3, -0.4),
        (
            "r1b2bnr/pp2k1qp/n1pp2p1/4PpQ1/2P1P3/2NP3N/PP4PP/R1B1KB1R b KQ - 2 10",
            4,
            0.2,
        ),
    ],
)
def test_native_mcts_matches_zero_semantics(
    fen: str, seed: int, value: float
) -> None:
    board = chess.Board(fen)
    cfg = MCTSConfig(
        simulations=48,
        dirichlet_epsilon=0.0,
        claim_draw=True,
        draw_contempt=1 / 3,
    )
    evaluator = FixedEvaluator(seed, value)
    expected = PythonMCTS(evaluator, cfg).run(
        board.copy(), simulations=48, add_noise=False
    )
    actual_mcts = MCTS(evaluator, cfg)
    assert actual_mcts.using_native
    actual = actual_mcts.run(board.copy(), simulations=48, add_noise=False)

    expected_by_move = {
        move.uci(): (visits, q, prior, clean)
        for move, visits, q, prior, clean in zip(
            expected.moves,
            expected.visits,
            expected.q_values,
            expected.priors,
            expected.clean_priors,
        )
    }
    actual_by_move = {
        move.uci(): (visits, q, prior, clean)
        for move, visits, q, prior, clean in zip(
            actual.moves,
            actual.visits,
            actual.q_values,
            actual.priors,
            actual.clean_priors,
        )
    }
    assert actual_by_move.keys() == expected_by_move.keys()
    for uci in expected_by_move:
        np.testing.assert_allclose(
            actual_by_move[uci],
            expected_by_move[uci],
            rtol=1e-5,
            atol=1e-6,
            err_msg=uci,
        )
    assert actual.root_value == pytest.approx(expected.root_value, abs=1e-6)
    assert actual.searched_root_q == pytest.approx(expected.searched_root_q, abs=1e-6)
    np.testing.assert_allclose(
        actual.improved_policy(), expected.improved_policy(), rtol=1e-5, atol=1e-6
    )


@pytest.mark.parametrize(
    ("board", "expected_value"),
    [
        (
            chess.Board(
                "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
            ),
            -1.0,
        ),
        (chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"), -1 / 3),
        (chess.Board("7k/8/8/8/8/8/8/K7 w - - 0 1"), -1 / 3),
        (chess.Board("7k/8/8/8/8/8/8/KR6 w - - 100 1"), -1 / 3),
    ],
)
def test_native_terminal_roots_match_zero_semantics(
    board: chess.Board, expected_value: float
) -> None:
    cfg = MCTSConfig(simulations=16, claim_draw=True, draw_contempt=1 / 3)
    evaluator = FixedEvaluator(9, 0.9)
    expected = PythonMCTS(evaluator, cfg).run(board.copy(), add_noise=False)
    actual = MCTS(evaluator, cfg).run(board.copy(), add_noise=False)
    assert actual.moves == expected.moves == []
    assert actual.visits.size == expected.visits.size == 0
    assert actual.root_value == pytest.approx(expected.root_value, abs=1e-6)
    assert actual.root_value == pytest.approx(expected_value, abs=1e-6)


def test_native_claims_threefold_from_move_history() -> None:
    board = chess.Board()
    for uci in ("g1f3", "g8f6", "f3g1", "f6g8") * 2:
        board.push_uci(uci)
    assert board.can_claim_threefold_repetition()

    cfg = MCTSConfig(simulations=16, claim_draw=True, draw_contempt=1 / 3)
    evaluator = FixedEvaluator(10, 0.9)
    expected = PythonMCTS(evaluator, cfg).run(board.copy(), add_noise=False)
    actual = MCTS(evaluator, cfg).run(board.copy(), add_noise=False)
    assert actual.moves == expected.moves == []
    assert actual.root_value == pytest.approx(expected.root_value, abs=1e-6)
    assert actual.root_value == pytest.approx(-cfg.draw_contempt, abs=1e-6)
