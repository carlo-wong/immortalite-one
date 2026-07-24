"""Shared deterministic fixtures for acceptance tests."""

from __future__ import annotations

import random
from collections.abc import Callable

import chess
import numpy as np
import pytest
import torch

from engine.config import NetConfig
from engine.encoding import POLICY_SIZE
from engine.network import ChessNet, NetEvaluator


@pytest.fixture
def seed_all() -> Callable[[int], None]:
    def _seed(seed: int = 20260724) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.set_num_threads(1)

    _seed()
    return _seed


@pytest.fixture
def tiny_net(seed_all: Callable[[int], None]) -> ChessNet:
    seed_all()
    return ChessNet(NetConfig(blocks=1, filters=8, value_bins=11))


@pytest.fixture
def tiny_evaluator(tiny_net: ChessNet) -> NetEvaluator:
    return NetEvaluator(tiny_net, device="cpu")


class FixedEvaluator:
    """Deterministic evaluator implementing the full and legal-only contracts."""

    def __init__(self, seed: int = 20260724, value: float = 0.125) -> None:
        self.logits = np.random.default_rng(seed).normal(
            size=POLICY_SIZE
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

    def evaluate_legal(
        self,
        planes: np.ndarray,
        legal_indices: np.ndarray,
        legal_offsets: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        n = int(planes.shape[0])
        indices = np.asarray(legal_indices)
        offsets = np.asarray(legal_offsets)
        assert offsets.shape == (n + 1,)
        assert int(offsets[0]) == 0 and int(offsets[-1]) == len(indices)
        # Position-independent logits: gather matches NetEvaluator's per-row CSR contract.
        gathered = np.empty(len(indices), dtype=np.float32)
        for row in range(n):
            start, end = int(offsets[row]), int(offsets[row + 1])
            gathered[start:end] = self.logits[indices[start:end]]
        return gathered, np.full(n, self.value, dtype=np.float32)


@pytest.fixture
def fixed_evaluator() -> FixedEvaluator:
    return FixedEvaluator()


def result_fingerprint(result: object) -> tuple[tuple[str, ...], tuple[int, ...], tuple[float, ...]]:
    return (
        tuple(move.uci() for move in result.moves),
        tuple(int(v) for v in result.visits),
        tuple(np.round(np.asarray(result.q_values), 7)),
    )
