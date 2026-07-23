"""Encoding parity: native C++ vs Immortalite Zero / local python-chess encoding."""

from __future__ import annotations

import chess
import numpy as np
import pytest

from engine import _native
from engine.encoding import (
    ENCODING_VERSION,
    board_to_planes,
    legal_move_indices,
    move_to_index,
)


def test_native_constants() -> None:
    assert int(_native.ENCODING_VERSION) == ENCODING_VERSION
    assert int(_native.NUM_INPUT_PLANES) == 20
    assert int(_native.POLICY_SIZE) == 4672


@pytest.mark.parametrize(
    "fen",
    [
        chess.STARTING_FEN,
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    ],
)
def test_planes_match_python(fen: str) -> None:
    board = chess.Board(fen)
    py = board_to_planes(board)
    cpp = np.asarray(_native.fill_planes_fen(fen), dtype=np.float32)
    np.testing.assert_allclose(cpp, py, atol=0, rtol=0)


def test_planes_with_move_history_repetition() -> None:
    board = chess.Board()
    for uci in ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1", "f6g8"]:
        board.push_uci(uci)
    root = board.root()
    moves = [m.uci() for m in board.move_stack]
    py = board_to_planes(board)
    cpp = np.asarray(_native.fill_planes_fen(root.fen(), moves), dtype=np.float32)
    np.testing.assert_allclose(cpp, py, atol=0, rtol=0)
    assert py[17].max() == 1.0  # repetition >= 2


@pytest.mark.parametrize(
    "fen",
    [
        chess.STARTING_FEN,
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    ],
)
def test_legal_indices_match(fen: str) -> None:
    board = chess.Board(fen)
    py_map = legal_move_indices(board)
    cpp_pairs = list(_native.legal_move_indices_fen(fen))
    cpp_map = {int(i): chess.Move.from_uci(u) for i, u in cpp_pairs}
    assert set(cpp_map) == set(py_map)
    for idx, mv in py_map.items():
        assert cpp_map[idx] == mv
        assert int(_native.move_to_index_fen(fen, mv.uci())) == move_to_index(mv, board)


@pytest.mark.parametrize(
    "fen",
    [
        # In-check roots where python-chess uses evasion order (king first).
        "r1b2bnr/pp2k1qp/n1pp2p1/4PpQ1/2P1P3/2NP3N/PP4PP/R1B1KB1R b KQ - 2 10",
        "rqB5/p3k1br/8/1pP1p1pp/2P2p2/2N1Pb1P/P2P1PP1/R1BK2NR w - - 5 25",
    ],
)
def test_legal_move_order_in_check(fen: str) -> None:
    board = chess.Board(fen)
    assert board.is_check()
    py = [m.uci() for m in board.legal_moves]
    cpp = [u for _, u in _native.legal_move_indices_fen(fen)]
    assert cpp == py
    assert set(cpp) == set(py)


def test_random_walk_planes(seed: int = 0) -> None:
    """Use root FEN + UCI history: python-chess may keep ep_square when FEN omits it."""
    rng = np.random.default_rng(seed)
    board = chess.Board()
    for _ in range(40):
        if board.is_game_over(claim_draw=True):
            break
        root_fen = board.root().fen()
        history = [m.uci() for m in board.move_stack]
        py = board_to_planes(board)
        cpp = np.asarray(
            _native.fill_planes_fen(root_fen, history if history else None),
            dtype=np.float32,
        )
        np.testing.assert_allclose(cpp, py, atol=0, rtol=0)
        moves = list(board.legal_moves)
        board.push(moves[int(rng.integers(0, len(moves)))])
