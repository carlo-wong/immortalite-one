"""Smoke: native batched self-play produces samples."""

from __future__ import annotations

import chess
import numpy as np
import pytest

from engine.config import Config
from engine.encoding import POLICY_SIZE
from engine.network import ChessNet, NetEvaluator
from engine.selfplay import (
    _prefer_native_selfplay,
    play_game_gen,
    play_games_batched,
    play_games_batched_native,
)


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


class _DeterministicEvaluator:
    def __init__(self) -> None:
        self.logits = np.random.default_rng(42).normal(
            size=POLICY_SIZE
        ).astype(np.float32)

    def evaluate(self, board: chess.Board) -> tuple[np.ndarray, float]:
        del board
        return self.logits.copy(), 0.15

    def evaluate_planes(self, planes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = int(planes.shape[0])
        values = np.full(n, 0.15, dtype=np.float32)
        return np.repeat(self.logits[None, :], n, axis=0), values


def _play_python_deterministic(evaluator, cfg: Config, simulations: int):
    gen = play_game_gen(
        cfg,
        simulations,
        add_noise=False,
        exploration_moves=0,
    )
    request = next(gen)
    while True:
        logits, value = evaluator.evaluate(request.board)
        try:
            request = gen.send((logits, value))
        except StopIteration as stop:
            return stop.value


def test_native_selfplay_matches_python_zero_semantics(monkeypatch) -> None:
    cfg = Config()
    cfg.train.max_game_moves = 10
    cfg.train.value_target = "root_q"
    cfg.mcts.dirichlet_epsilon = 0.0
    evaluator = _DeterministicEvaluator()

    native = play_games_batched_native(
        evaluator,
        cfg,
        simulations=16,
        num_games=1,
        concurrency=1,
        add_noise=False,
        exploration_moves=0,
    )[0]

    monkeypatch.setenv("IMMORTALITE_ONE_FORCE_PYTHON", "1")
    python = _play_python_deterministic(evaluator, cfg, simulations=16)

    assert native.moves == python.moves
    assert native.termination == python.termination
    assert native.winner == python.winner
    assert len(native.samples) == len(python.samples)
    for actual, expected in zip(native.samples, python.samples):
        assert actual.player == expected.player
        np.testing.assert_array_equal(actual.planes, expected.planes)
        np.testing.assert_allclose(actual.policy, expected.policy, rtol=0, atol=1e-3)
        assert actual.root_q == pytest.approx(expected.root_q, abs=1e-6)
        assert actual.value == pytest.approx(expected.value, abs=1e-6)
