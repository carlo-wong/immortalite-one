"""Incremental native Zobrist make/unmake parity across move classes."""

from __future__ import annotations

import chess
import pytest

from engine import _native


def _assert_trace(fen: str, moves: list[str]) -> None:
    board = chess.Board(fen)
    expected = [int(_native.transposition_key_fen(board.fen(en_passant="fen")))]
    for uci in moves:
        board.push_uci(uci)
        expected.append(int(_native.transposition_key_fen(board.fen(en_passant="fen"))))

    forward, backward = _native.zobrist_trace_fen(fen, moves)
    assert [int(key) for key in forward] == expected
    assert [int(key) for key in backward] == list(reversed(expected))


@pytest.mark.parametrize(
    ("fen", "moves"),
    [
        (
            chess.STARTING_FEN,
            ["g1f3", "g8f6", "e2e4", "d7d5", "e4d5", "f6d5"],
        ),
        (
            "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
            ["e1g1", "e8c8"],
        ),
        (
            "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
            ["a1a8"],
        ),
        ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", ["e5d6"]),
        ("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", ["a7a8q"]),
        ("4k2r/6P1/8/8/8/8/8/4K3 w - - 0 1", ["g7h8n"]),
    ],
)
def test_incremental_zobrist_matches_fresh_fen_and_unmake(
    fen: str, moves: list[str]
) -> None:
    _assert_trace(fen, moves)


def test_ep_key_is_present_only_for_legal_en_passant() -> None:
    legal = "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1"
    pinned = "4r1k1/8/8/3pP3/8/8/8/4K3 w - d6 0 1"

    assert _native.transposition_key_fen(legal) != _native.transposition_key_fen(
        legal.replace(" d6 ", " - ")
    )
    assert _native.transposition_key_fen(pinned) == _native.transposition_key_fen(
        pinned.replace(" d6 ", " - ")
    )
    assert "e5d6" not in {
        uci for _, uci in _native.legal_move_indices_fen(pinned)
    }
