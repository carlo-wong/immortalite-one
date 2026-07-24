"""Recursive native move-generation parity against standard perft positions."""

from __future__ import annotations

import chess
import pytest

from engine import _native


def _native_perft(board: chess.Board, depth: int) -> int:
    if depth == 0:
        return 1

    native_moves = [
        chess.Move.from_uci(uci)
        for _, uci in _native.legal_move_indices_fen(board.fen())
    ]
    assert len(native_moves) == len(set(native_moves))
    assert set(native_moves) == set(board.legal_moves), board.fen()
    if depth == 1:
        return len(native_moves)

    nodes = 0
    for move in native_moves:
        board.push(move)
        nodes += _native_perft(board, depth - 1)
        board.pop()
    return nodes


@pytest.mark.parametrize(
    ("fen", "depth", "expected_nodes"),
    [
        (chess.STARTING_FEN, 3, 8_902),
        (
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            2,
            2_039,
        ),
        ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 3, 2_812),
    ],
)
def test_native_perft(fen: str, depth: int, expected_nodes: int) -> None:
    board = chess.Board(fen)
    original_fen = board.fen()
    assert _native_perft(board, depth) == expected_nodes
    assert board.fen() == original_fen
