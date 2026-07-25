"""Persistent native game actor coverage."""

from __future__ import annotations

import numpy as np
import pytest

from engine.config import Config
from engine.encoding import POLICY_SIZE
from engine.selfplay import play_games_batched_native, play_games_batched_native_actors


class _DeterministicEvaluator:
    def __init__(self) -> None:
        self.logits = np.random.default_rng(42).normal(size=POLICY_SIZE).astype(np.float32)

    def evaluate_planes(self, planes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.repeat(self.logits[None, :], planes.shape[0], axis=0), np.full(
            planes.shape[0], 0.15, dtype=np.float32
        )

    def evaluate_legal(
        self,
        planes: np.ndarray,
        legal_indices: np.ndarray,
        legal_offsets: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        n = int(planes.shape[0])
        indices = np.asarray(legal_indices)
        offsets = np.asarray(legal_offsets)
        gathered = np.empty(len(indices), dtype=np.float32)
        for row in range(n):
            start, end = int(offsets[row]), int(offsets[row + 1])
            gathered[start:end] = self.logits[indices[start:end]]
        return gathered, np.full(n, 0.15, dtype=np.float32)



def _cfg() -> Config:
    cfg = Config()
    cfg.mcts.dirichlet_epsilon = 0.0
    cfg.train.tb_max_pieces = 0
    return cfg


def test_actor_games_match_native_sessions_without_noise() -> None:
    cfg = _cfg()
    cfg.train.max_game_moves = 6
    evaluator = _DeterministicEvaluator()
    expected = play_games_batched_native(
        evaluator, cfg, simulations=8, num_games=1, concurrency=1,
        add_noise=False, exploration_moves=0,
    )[0]
    actual = play_games_batched_native_actors(
        evaluator, cfg, simulations=8, num_games=1, concurrency=1,
        add_noise=False, exploration_moves=0,
    )[0]
    assert actual.moves == expected.moves
    assert actual.termination == expected.termination
    for got, want in zip(actual.samples, expected.samples):
        assert got.player == want.player
        np.testing.assert_array_equal(got.planes, want.planes)
        np.testing.assert_allclose(got.policy, want.policy, rtol=0, atol=1e-3)
        assert got.root_q == want.root_q


def test_actor_max_moves_terminates() -> None:
    cfg = _cfg()
    cfg.train.max_game_moves = 3
    game = play_games_batched_native_actors(
        _DeterministicEvaluator(), cfg, simulations=4, num_games=1, concurrency=1,
        add_noise=False, exploration_moves=0,
    )[0]
    assert game.termination == "max_moves"
    assert len(game.moves) == len(game.samples) == 3


def test_actor_root_q_values_match_sample_root_q() -> None:
    cfg = _cfg()
    cfg.train.max_game_moves = 4
    cfg.train.value_target = "root_q"
    game = play_games_batched_native_actors(
        _DeterministicEvaluator(), cfg, simulations=4, num_games=1, concurrency=1,
        add_noise=False, exploration_moves=0,
    )[0]
    assert all(sample.value == sample.root_q for sample in game.samples)


def test_actor_q_z_blends_root_q_and_outcome_z() -> None:
    cfg = _cfg()
    cfg.train.max_game_moves = 4
    cfg.train.value_target = "q_z"
    cfg.train.value_q_ratio = 0.5
    game = play_games_batched_native_actors(
        _DeterministicEvaluator(), cfg, simulations=4, num_games=1, concurrency=1,
        add_noise=False, exploration_moves=0,
    )[0]
    assert game.samples
    # Reconstruct outcome z from the same rules as GameActorBatch::complete.
    if game.termination == "max_moves":
        last = game.samples[-1].player
        bootstrap = game.samples[-1].root_q
        z_values = [
            bootstrap if s.player == last else -bootstrap for s in game.samples
        ]
    elif game.termination in {"checkmate", "resign", "tablebase_win"} and game.winner is not None:
        z_values = [
            1.0 if s.player == game.winner else -1.0 for s in game.samples
        ]
    else:
        z_values = [-cfg.train.draw_penalty] * len(game.samples)
    alpha = cfg.train.value_q_ratio
    for sample, z in zip(game.samples, z_values):
        expected = alpha * sample.root_q + (1.0 - alpha) * z
        assert sample.value == pytest.approx(expected, abs=1e-5)
