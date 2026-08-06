"""WDL aux-head label contract (STM POV), independent of value_target / draw_penalty."""

from __future__ import annotations

import chess
import numpy as np
import pytest

from engine.config import Config
from engine.encoding import POLICY_SIZE
from engine.selfplay import (
    WDL_DRAW,
    WDL_LOSS,
    WDL_UNLABELED,
    WDL_WIN,
    Sample,
    _DRAW_TERMINATION_SET,
    _assign_values,
    _clone_sample,
    _config_from_dict,
    _config_to_dict,
    _net_cfg_dict,
    assign_sample_wdl,
    wdl_label_from_result,
)


def _sample(player: chess.Color, root_q: float = 0.25) -> Sample:
    return Sample(
        planes=np.zeros((20, 8, 8), dtype=np.float32),
        policy=np.zeros(POLICY_SIZE, dtype=np.float32),
        player=player,
        root_q=root_q,
    )


@pytest.mark.parametrize(
    "player,winner,expected",
    [
        (chess.WHITE, chess.WHITE, WDL_WIN),
        (chess.BLACK, chess.WHITE, WDL_LOSS),
        (chess.WHITE, chess.BLACK, WDL_LOSS),
        (chess.BLACK, chess.BLACK, WDL_WIN),
    ],
)
@pytest.mark.parametrize("termination", ["checkmate", "resign", "tablebase_win"])
def test_wdl_decisive_both_colors(player, winner, expected, termination) -> None:
    assert wdl_label_from_result(player, termination, winner) == expected


@pytest.mark.parametrize("termination", sorted(_DRAW_TERMINATION_SET) + ["tablebase_draw"])
def test_wdl_draws_are_class_d(termination: str) -> None:
    assert wdl_label_from_result(chess.WHITE, termination, None) == WDL_DRAW
    assert wdl_label_from_result(chess.BLACK, termination, None) == WDL_DRAW


@pytest.mark.parametrize("termination", ["max_moves", "no_legal_moves", "unknown"])
def test_wdl_truncation_and_unknown_unlabeled(termination: str) -> None:
    assert wdl_label_from_result(chess.WHITE, termination, chess.WHITE) == WDL_UNLABELED
    assert wdl_label_from_result(chess.BLACK, termination, None) == WDL_UNLABELED


def test_wdl_draw_penalty_does_not_affect_labels() -> None:
    cfg = Config()
    cfg.net.wdl_head = True
    cfg.train.value_target = "outcome"
    cfg.train.draw_penalty = 0.99
    samples = [_sample(chess.WHITE, 0.5), _sample(chess.BLACK, -0.5)]
    _assign_values(samples, None, "threefold_repetition", cfg, move_count=40)
    assign_sample_wdl(
        samples,
        termination="threefold_repetition",
        winner=None,
        enabled=True,
    )
    assert samples[0].value == pytest.approx(-0.99)
    assert samples[1].value == pytest.approx(-0.99)
    assert samples[0].wdl == WDL_DRAW
    assert samples[1].wdl == WDL_DRAW


def test_wdl_root_q_value_target_unchanged() -> None:
    cfg = Config()
    cfg.net.wdl_head = True
    cfg.train.value_target = "root_q"
    samples = [_sample(chess.WHITE, 0.42), _sample(chess.BLACK, -0.17)]
    outcome = chess.Outcome(termination=chess.Termination.CHECKMATE, winner=chess.WHITE)
    _assign_values(samples, outcome, "checkmate", cfg, move_count=12)
    assign_sample_wdl(
        samples,
        termination="checkmate",
        winner=chess.WHITE,
        enabled=True,
    )
    assert samples[0].value == pytest.approx(0.42)
    assert samples[1].value == pytest.approx(-0.17)
    assert samples[0].wdl == WDL_WIN
    assert samples[1].wdl == WDL_LOSS


def test_wdl_disabled_leaves_unlabeled() -> None:
    samples = [_sample(chess.WHITE), _sample(chess.BLACK)]
    samples[0].wdl = WDL_WIN  # stale; disabled path must clear
    assign_sample_wdl(
        samples,
        termination="checkmate",
        winner=chess.WHITE,
        enabled=False,
    )
    assert samples[0].wdl == WDL_UNLABELED
    assert samples[1].wdl == WDL_UNLABELED
    assert Sample(
        planes=np.zeros((20, 8, 8), dtype=np.float32),
        policy=np.zeros(POLICY_SIZE, dtype=np.float32),
        player=chess.WHITE,
    ).wdl == WDL_UNLABELED


def test_native_wrapper_helper_matches_python_semantics() -> None:
    """Same (termination, winner, players) → identical labels for actor wrapping."""
    # Mimic GameActorBatch meta fields used by play_games_batched_native_actors.
    meta = {"termination": "resign", "winner": 1}  # native BLACK=1
    winner = chess.BLACK
    termination = str(meta["termination"])
    samples = [_sample(chess.WHITE, 0.1), _sample(chess.BLACK, -0.2)]
    values_before = [float(s.root_q) for s in samples]

    assign_sample_wdl(
        samples,
        termination=termination,
        winner=winner,
        enabled=True,
    )
    assert [s.wdl for s in samples] == [WDL_LOSS, WDL_WIN]
    assert [float(s.root_q) for s in samples] == values_before

    # Python completion path uses the same helper with GameResult fields.
    py_samples = [_sample(chess.WHITE, 0.1), _sample(chess.BLACK, -0.2)]
    assign_sample_wdl(
        py_samples,
        termination="resign",
        winner=chess.BLACK,
        enabled=True,
    )
    assert [s.wdl for s in py_samples] == [s.wdl for s in samples]


def test_wdl_clone_and_net_cfg_serialization() -> None:
    s = _sample(chess.WHITE, 0.3)
    s.wdl = WDL_LOSS
    cloned = _clone_sample(s)
    assert cloned.wdl == WDL_LOSS
    assert cloned is not s

    cfg = Config()
    cfg.net.wdl_head = True
    cfg.train.wdl_coef = 0.35
    assert _net_cfg_dict(cfg.net)["wdl_head"] is True
    assert _config_to_dict(cfg)["train"]["wdl_coef"] == pytest.approx(0.35)
    roundtrip = _config_from_dict(_config_to_dict(cfg))
    assert roundtrip.net.wdl_head is True
    assert roundtrip.train.wdl_coef == pytest.approx(0.35)
