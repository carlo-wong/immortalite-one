"""Production evaluator paths must produce equivalent policy and value outputs."""

from __future__ import annotations

import chess
import numpy as np
import pytest
import torch

from engine.config import NetConfig
from engine.encoding import POLICY_SIZE, board_to_planes
from engine.network import ChessNet, NetEvaluator


@pytest.fixture
def evaluator() -> NetEvaluator:
    torch.manual_seed(20260724)
    net = ChessNet(NetConfig(blocks=2, filters=8, value_bins=51))
    return NetEvaluator(net, device="cpu")


def _boards() -> list[chess.Board]:
    repeated = chess.Board()
    for uci in ("g1f3", "g8f6", "f3g1", "f6g8"):
        repeated.push_uci(uci)
    return [
        chess.Board(),
        chess.Board(
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
        ),
        chess.Board(
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
        ),
        repeated,
    ]


def test_batch_and_preencoded_paths_match_single_evaluation(
    evaluator: NetEvaluator,
) -> None:
    boards = _boards()
    batch_logits, batch_values = evaluator.evaluate_batch(boards)
    planes = np.stack([board_to_planes(board) for board in boards])
    plane_logits, plane_values = evaluator.evaluate_planes(planes)

    assert batch_logits.shape == plane_logits.shape == (len(boards), POLICY_SIZE)
    assert batch_values.shape == plane_values.shape == (len(boards),)
    np.testing.assert_allclose(batch_logits, plane_logits, rtol=0, atol=1e-6)
    np.testing.assert_allclose(batch_values, plane_values, rtol=0, atol=1e-6)

    for index, board in enumerate(boards):
        logits, value = evaluator.evaluate(board)
        np.testing.assert_allclose(batch_logits[index], logits, rtol=0, atol=1e-6)
        assert batch_values[index] == pytest.approx(value, abs=1e-6)


def test_empty_batch_and_empty_planes_match(evaluator: NetEvaluator) -> None:
    batch_logits, batch_values = evaluator.evaluate_batch([])
    plane_logits, plane_values = evaluator.evaluate_planes(
        np.empty((0, 20, 8, 8), dtype=np.float32)
    )
    assert batch_logits.shape == plane_logits.shape == (0, POLICY_SIZE)
    assert batch_values.shape == plane_values.shape == (0,)


def test_legal_evaluation_matches_full_policy_gather(
    evaluator: NetEvaluator,
) -> None:
    boards = _boards()
    planes = np.stack([board_to_planes(board) for board in boards]).astype(np.float32)
    full_logits, full_values = evaluator.evaluate_planes(planes)
    per_row = (
        np.array([0, 13, POLICY_SIZE - 1], dtype=np.int32),
        np.array([7], dtype=np.int32),
        np.array([], dtype=np.int32),
        np.array([3, 9], dtype=np.int32),
    )
    offsets = np.zeros(len(per_row) + 1, dtype=np.int32)
    offsets[1:] = np.cumsum([indices.size for indices in per_row], dtype=np.int32)
    legal_indices = np.concatenate(per_row)

    legal_logits, legal_values = evaluator.evaluate_legal(
        planes, legal_indices, offsets
    )

    expected = np.concatenate(
        [full_logits[row, indices] for row, indices in enumerate(per_row)]
    )
    assert legal_logits.shape == (legal_indices.size,)
    np.testing.assert_allclose(legal_logits, expected, rtol=0, atol=1e-6)
    np.testing.assert_allclose(legal_values, full_values, rtol=0, atol=1e-6)
