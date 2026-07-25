"""Soft Q+Z value_target assignment."""

from __future__ import annotations

import chess
import numpy as np
import pytest

from engine.config import Config
from engine.encoding import POLICY_SIZE
from engine.selfplay import Sample, _assign_values


def _sample(player: chess.Color, root_q: float) -> Sample:
    return Sample(
        planes=np.zeros((20, 8, 8), dtype=np.float32),
        policy=np.zeros(POLICY_SIZE, dtype=np.float32),
        player=player,
        root_q=root_q,
    )


def test_assign_values_q_z_checkmate_blend() -> None:
    cfg = Config()
    cfg.train.value_target = "q_z"
    cfg.train.value_q_ratio = 0.5
    samples = [
        _sample(chess.WHITE, 0.4),
        _sample(chess.BLACK, -0.2),
    ]
    outcome = chess.Outcome(termination=chess.Termination.CHECKMATE, winner=chess.WHITE)
    _assign_values(samples, outcome, "checkmate", cfg, move_count=10)
    assert samples[0].value == pytest.approx(0.5 * 0.4 + 0.5 * 1.0)
    assert samples[1].value == pytest.approx(0.5 * (-0.2) + 0.5 * (-1.0))


def test_assign_values_q_z_draw_uses_draw_penalty() -> None:
    cfg = Config()
    cfg.train.value_target = "q_z"
    cfg.train.value_q_ratio = 0.25
    cfg.train.draw_penalty = 1 / 3
    samples = [_sample(chess.WHITE, 0.8), _sample(chess.BLACK, -0.8)]
    _assign_values(samples, None, "threefold_repetition", cfg, move_count=40)
    z = -cfg.train.draw_penalty
    assert samples[0].value == pytest.approx(0.25 * 0.8 + 0.75 * z)
    assert samples[1].value == pytest.approx(0.25 * (-0.8) + 0.75 * z)


def test_assign_values_q_z_max_moves_bootstrap() -> None:
    cfg = Config()
    cfg.train.value_target = "q_z"
    cfg.train.value_q_ratio = 0.5
    samples = [
        _sample(chess.WHITE, 0.1),
        _sample(chess.BLACK, -0.3),
    ]
    _assign_values(
        samples, None, "max_moves", cfg, move_count=4, truncation_bootstrap=0.6,
    )
    # z: +0.6 for final player (BLACK), -0.6 for WHITE
    assert samples[0].value == pytest.approx(0.5 * 0.1 + 0.5 * (-0.6))
    assert samples[1].value == pytest.approx(0.5 * (-0.3) + 0.5 * 0.6)
