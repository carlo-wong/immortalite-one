"""Smoke: native batched self-play produces samples."""

from __future__ import annotations

from engine.config import Config
from engine.network import ChessNet, NetEvaluator
from engine.selfplay import _prefer_native_selfplay, play_games_batched


def test_native_batched_selfplay_short_games() -> None:
    assert _prefer_native_selfplay()
    cfg = Config()
    cfg.train.max_game_moves = 6
    ev = NetEvaluator(ChessNet(cfg.net), device="cpu")
    games = play_games_batched(
        ev, cfg, simulations=8, num_games=2, concurrency=2,
    )
    assert len(games) == 2
    for g in games:
        assert len(g.samples) == len(g.moves) == 6
        assert abs(float(g.samples[0].policy.sum()) - 1.0) < 1e-3
