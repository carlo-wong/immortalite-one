"""Uniform-random self-play opening prefixes (tabula rasa, no human book)."""

from __future__ import annotations

import chess
import numpy as np
import pytest

from engine.config import Config
from engine.encoding import POLICY_SIZE
from engine.network import ChessNet, NetEvaluator
from engine.openings import (
    diversity_move_uci,
    random_legal_opening_prefixes,
)
from engine.selfplay import (
    _prefer_native_selfplay,
    _selfplay_start_moves,
    play_game_gen,
    play_games_batched,
)


def test_random_legal_opening_prefixes_k1_covers_startpos_moves() -> None:
    rng = np.random.default_rng(0)
    prefixes = random_legal_opening_prefixes(200, 1, rng=rng)
    assert len(prefixes) == 200
    start = chess.Board()
    legal = {m.uci() for m in start.legal_moves}
    seen = {row[0] for row in prefixes}
    assert all(len(row) == 1 for row in prefixes)
    assert seen <= legal
    # With 200 draws, expect broad coverage (not a single mode lock).
    assert len(seen) >= 10


def test_diversity_move_skips_prefix() -> None:
    assert diversity_move_uci(["c2c4", "e7e5", "g1f3"], 0) == "c2c4"
    assert diversity_move_uci(["c2c4", "e7e5", "g1f3"], 1) == "e7e5"
    assert diversity_move_uci(["c2c4"], 1) is None


def test_white_first_move_is_prefix_when_k1() -> None:
    """metrics_first_moves should keep White ply-0; free reply is separate."""
    prefixes = random_legal_opening_prefixes(8, 1, rng=np.random.default_rng(1))
    for row in prefixes:
        assert len(row) == 1
        # White first-move UCI from startpos always has from-rank 1/2.
        assert row[0][1] in "12"


def test_random_opening_mixture_preserves_start_position_games() -> None:
    cfg = Config()
    cfg.train.random_opening_plies = 1
    cfg.train.random_opening_probability = 0.3
    prefixes = _selfplay_start_moves(cfg, 1000, rng=np.random.default_rng(7))

    assert prefixes is not None
    selected = [row for row in prefixes if row]
    assert 250 <= len(selected) <= 350
    assert all(len(row) == 1 for row in selected)
    start_legal = {move.uci() for move in chess.Board().legal_moves}
    assert {row[0] for row in selected} <= start_legal


def test_random_opening_probability_keeps_historical_all_games_behavior() -> None:
    cfg = Config()
    cfg.train.random_opening_plies = 1
    cfg.train.random_opening_probability = 1.0
    prefixes = _selfplay_start_moves(cfg, 16, rng=np.random.default_rng(3))

    assert prefixes is not None
    assert all(len(row) == 1 for row in prefixes)


def test_random_opening_probability_zero_disables_prefixes() -> None:
    cfg = Config()
    cfg.train.random_opening_plies = 1
    cfg.train.random_opening_probability = 0.0
    assert _selfplay_start_moves(cfg, 16, rng=np.random.default_rng(3)) is None


def test_python_play_game_gen_prefix_writes_no_samples() -> None:
    cfg = Config()
    cfg.train.max_game_moves = 4
    cfg.train.random_opening_plies = 0
    cfg.mcts.claim_draw = True
    gen = play_game_gen(cfg, simulations=4, add_noise=False, start_moves=["e2e4"])
    request = next(gen)
    # First eval is after the forced ply — Black to move.
    assert request.board.turn == chess.BLACK
    assert request.board.peek().uci() == "e2e4"
    # Drive a short game to completion.
    while True:
        logits = np.zeros(POLICY_SIZE, dtype=np.float32)
        try:
            request = gen.send((logits, 0.0))
        except StopIteration as stop:
            game = stop.value
            break
    assert game.moves[0] == "e2e4"
    assert len(game.samples) == len(game.moves) - 1
    assert game.opening_prefix_plies == 1


@pytest.mark.skipif(not _prefer_native_selfplay(), reason="native GameActorBatch required")
def test_native_actors_random_opening_plies() -> None:
    cfg = Config()
    cfg.train.max_game_moves = 6
    cfg.train.random_opening_plies = 1
    cfg.train.move_temperature = 1.0
    cfg.train.move_temperature_plies = 0
    ev = NetEvaluator(ChessNet(cfg.net), device="cpu")
    games = play_games_batched(
        ev, cfg, simulations=8, num_games=4, concurrency=4,
    )
    assert len(games) == 4
    start_legal = {m.uci() for m in chess.Board().legal_moves}
    for g in games:
        assert g.moves[0] in start_legal
        assert len(g.samples) == len(g.moves) - 1
        assert diversity_move_uci(g.moves, 1) == g.moves[1]
