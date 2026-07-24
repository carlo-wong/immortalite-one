"""Native dual-net match path (strength gates)."""

from __future__ import annotations

import numpy as np
import torch

import engine.selfplay as selfplay
from engine.config import Config, NetConfig
from engine.encoding import POLICY_SIZE, board_to_planes
from engine.network import ChessNet
from engine.selfplay import (
    _run_match_games,
    _subset_legal_csr,
    play_match_batched_native_actors,
)
from engine.train import play_match


class _DeterministicEvaluator:
    def __init__(self, bias: float) -> None:
        rng = np.random.default_rng(42)
        ramp = np.linspace(-1.0, 1.0, POLICY_SIZE, dtype=np.float32)
        self.logits = rng.normal(size=POLICY_SIZE).astype(np.float32) + bias * ramp
        self.legal_calls = 0

    def evaluate_planes(self, planes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.repeat(self.logits[None, :], planes.shape[0], axis=0), np.full(
            planes.shape[0], 0.1, dtype=np.float32
        )

    def evaluate_batch(self, boards: list) -> tuple[np.ndarray, np.ndarray]:
        planes = np.stack([board_to_planes(board) for board in boards])
        return self.evaluate_planes(planes)

    def evaluate_legal(
        self,
        planes: np.ndarray,
        legal_indices: np.ndarray,
        legal_offsets: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        self.legal_calls += 1
        n = int(planes.shape[0])
        indices = np.asarray(legal_indices)
        offsets = np.asarray(legal_offsets)
        gathered = np.empty(len(indices), dtype=np.float32)
        for row in range(n):
            start, end = int(offsets[row]), int(offsets[row + 1])
            gathered[start:end] = self.logits[indices[start:end]]
        return gathered, np.full(n, 0.1, dtype=np.float32)


def _tiny_cfg() -> Config:
    cfg = Config()
    cfg.mcts.dirichlet_epsilon = 0.0
    cfg.train.tb_max_pieces = 0
    cfg.train.max_game_moves = 4
    cfg.train.selfplay_concurrency = 4
    cfg.mcts.draw_contempt = 0.0
    return cfg


def test_stale_native_extension_does_not_select_match_fast_path(monkeypatch) -> None:
    class _OldActorBatch:
        pass

    class _OldNative:
        MctsSession = object()
        GameActorBatch = _OldActorBatch

    monkeypatch.setattr(selfplay, "_load_native", lambda: _OldNative)
    assert not selfplay._native_match_actors_ready()


def test_match_native_runs_with_openings_and_colors() -> None:
    cfg = _tiny_cfg()
    openings = [["e2e4", "e7e5"], ["d2d4", "d7d5"]]
    eval_a = _DeterministicEvaluator(0.0)
    eval_b = _DeterministicEvaluator(0.5)
    stats = play_match_batched_native_actors(
        eval_a,
        eval_b,
        cfg,
        simulations=4,
        num_games=4,
        concurrency=4,
        exploration_moves=0,
        openings=openings,
        base_seed=123,
    )
    assert len(stats.game_lengths) == 4
    assert sum(stats.termination_counts.values()) == 4
    assert len(stats.openings) == 4
    assert [row["a_is_white"] for row in stats.openings] == [1, 0, 1, 0]
    assert stats.openings[0]["opening_uci"].startswith("e2e4 e7e5")
    assert stats.openings[1]["opening_uci"].startswith("e2e4 e7e5")
    assert stats.openings[2]["opening_uci"].startswith("d2d4 d7d5")
    assert eval_a.legal_calls > 0
    assert eval_b.legal_calls > 0


def test_native_match_preserves_decisive_winner_colors() -> None:
    cfg = _tiny_cfg()
    fools_mate = [["f2f3", "e7e5", "g2g4", "d8h4"]]
    stats = play_match_batched_native_actors(
        _DeterministicEvaluator(0.0),
        _DeterministicEvaluator(0.5),
        cfg,
        simulations=1,
        num_games=2,
        concurrency=2,
        exploration_moves=0,
        openings=fools_mate,
        base_seed=123,
    )

    assert stats.wins_w == 0
    assert stats.wins_b == 1
    assert stats.losses_w == 1
    assert stats.losses_b == 0
    assert [row["result"] for row in stats.openings] == ["L", "W"]


def test_legal_csr_subset_preserves_requested_rows() -> None:
    indices = np.asarray([10, 11, 20, 30, 31, 32], dtype=np.int32)
    offsets = np.asarray([0, 2, 3, 6], dtype=np.int32)
    got_indices, got_offsets = _subset_legal_csr(
        indices, offsets, np.asarray([0, 2], dtype=np.int32)
    )
    np.testing.assert_array_equal(got_indices, [10, 11, 30, 31, 32])
    np.testing.assert_array_equal(got_offsets, [0, 2, 5])


def test_native_match_matches_legacy_driver_without_sampling() -> None:
    cfg = _tiny_cfg()
    cfg.train.max_game_moves = 6
    openings = [["e2e4", "e7e5"], ["d2d4", "d7d5"]]

    native = play_match_batched_native_actors(
        _DeterministicEvaluator(0.0),
        _DeterministicEvaluator(0.5),
        cfg,
        simulations=4,
        num_games=4,
        concurrency=2,
        exploration_moves=0,
        openings=openings,
        base_seed=123,
    )
    legacy = _run_match_games(
        _DeterministicEvaluator(0.0),
        _DeterministicEvaluator(0.5),
        cfg,
        sims=4,
        n_games=4,
        exploration_moves=0,
        tablebase=None,
        openings=openings,
    )

    assert native.score == legacy.score
    assert native.game_lengths == legacy.game_lengths
    assert native.termination_counts == legacy.termination_counts
    assert [
        (row["game_idx"], row["a_is_white"], row["opening_uci"], row["result"])
        for row in native.openings
    ] == [
        (row["game_idx"], row["a_is_white"], row["opening_uci"], row["result"])
        for row in legacy.openings
    ]


def test_play_match_uses_native_path() -> None:
    cfg = _tiny_cfg()
    net_cfg = NetConfig(blocks=1, filters=8, value_bins=3)
    net_a = ChessNet(net_cfg).eval()
    net_b = ChessNet(net_cfg).eval()
    with torch.no_grad():
        for p in net_b.parameters():
            p.add_(0.01)
    metrics = play_match(
        net_a,
        net_b,
        cfg,
        n_games=2,
        sims=4,
        device="cpu",
        exploration_moves=0,
        workers=2,  # ignored on native path
        concurrency=2,
        openings=[["e2e4"]],
        sprt=False,
    )
    assert metrics["games_played"] == 2
    assert len(metrics["openings"]) == 2
    assert sum(
        int(part.rsplit(":", 1)[1]) for part in metrics["terminations"].split(";")
    ) == 2
    assert metrics["mean_game_len"] > 0
