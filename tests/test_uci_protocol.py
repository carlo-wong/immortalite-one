"""Deterministic UCI protocol acceptance tests without neural search."""

from __future__ import annotations

import io
from types import SimpleNamespace

import chess
import pytest

import uci.uci_engine as uci_engine


class _FakeAnalyzer:
    def __init__(self, checkpoint, cfg) -> None:
        self.checkpoint = checkpoint
        self.cfg = cfg
        self.calls: list[tuple[str, int]] = []

    def analyze(self, board: chess.Board, multipv: int):
        self.calls.append((board.fen(), multipv))
        if board.is_game_over(claim_draw=True):
            return SimpleNamespace(lines=[], best_move=None)
        lines = [
            {"eval_cp": 23, "pv": ["e2e4", "e7e5"]},
            {"eval_cp": 11, "pv": ["d2d4", "d7d5"]},
        ][:multipv]
        return SimpleNamespace(lines=lines, best_move="e2e4")


@pytest.fixture
def engine(monkeypatch: pytest.MonkeyPatch) -> uci_engine.UCIEngine:
    monkeypatch.setattr(uci_engine, "Analyzer", _FakeAnalyzer)
    return uci_engine.UCIEngine(None)


def test_uci_handshake_and_isready(
    engine: uci_engine.UCIEngine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stdin = io.StringIO("uci\nisready\nquit\n")
    stdout = io.StringIO()
    monkeypatch.setattr(uci_engine.sys, "stdin", stdin)
    monkeypatch.setattr(uci_engine.sys, "stdout", stdout)

    engine.run()

    assert stdout.getvalue().splitlines() == [
        "id name Immortalite Zero",
        "id author self-play",
        "option name Simulations type spin default 100 min 1 max 100000",
        "option name MultiPV type spin default 1 min 1 max 5",
        "uciok",
        "readyok",
    ]


def test_position_parses_startpos_and_fen(engine: uci_engine.UCIEngine) -> None:
    engine._set_position("position startpos moves e2e4 e7e5 g1f3")
    expected = chess.Board()
    for move in ("e2e4", "e7e5", "g1f3"):
        expected.push_uci(move)
    assert engine.board.fen() == expected.fen()

    fen = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 4 17"
    engine._set_position(f"position fen {fen} moves e1g1")
    expected = chess.Board(fen)
    expected.push_uci("e1g1")
    assert engine.board.fen() == expected.fen()


def test_go_emits_multipv_info_and_bestmove(
    engine: uci_engine.UCIEngine,
    capsys: pytest.CaptureFixture[str],
) -> None:
    engine.multipv = 2

    engine._go()

    assert capsys.readouterr().out.splitlines() == [
        "info multipv 1 depth 1 score cp 23 pv e2e4 e7e5",
        "info multipv 2 depth 1 score cp 11 pv d2d4 d7d5",
        "bestmove e2e4",
    ]
    assert engine.analyzer.calls == [(chess.STARTING_FEN, 2)]


def test_go_emits_null_move_for_terminal_position(
    engine: uci_engine.UCIEngine,
    capsys: pytest.CaptureFixture[str],
) -> None:
    engine.board = chess.Board(
        "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
    )

    engine._go()

    assert capsys.readouterr().out.splitlines() == ["bestmove 0000"]
