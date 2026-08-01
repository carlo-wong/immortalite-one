"""Validation coverage for the persistent native game actor."""

from __future__ import annotations

import numpy as np
import pytest

from engine import _native as native
from engine.encoding import POLICY_SIZE


def test_actor_batch_rejects_malformed_or_stale_eval_rows() -> None:
    batch = native.GameActorBatch(
        2,
        {"simulations": 2, "max_game_moves": 4, "tb_max_pieces": 0},
        {},
        123,
        start_moves=[["e2e4"], ["d2d4"]],
        a_is_white=[1, 0],
    )
    actor_ids, _ = batch.positions_needing_eval(
        np.empty((2, 20, 8, 8), dtype=np.float32)
    )
    assert actor_ids.tolist() == [0, 1]
    assert batch.pending_net_ids().tolist() == [1, 0]

    legal_indices, legal_offsets = batch.pending_legal_csr()
    malformed_offsets = np.asarray(legal_offsets).copy()
    malformed_offsets[-1] -= 1
    with pytest.raises(ValueError, match="offsets"):
        batch.apply_eval_legal(
            actor_ids,
            np.zeros(len(legal_indices), dtype=np.float32),
            malformed_offsets,
            np.zeros(2, dtype=np.float32),
        )

    logits = np.zeros((1, POLICY_SIZE), dtype=np.float32)
    values = np.zeros(1, dtype=np.float32)
    batch.apply_eval(actor_ids[:1], logits, values)
    assert batch.pending_net_ids().tolist() == [0]
    with pytest.raises(ValueError, match="not pending"):
        batch.apply_eval(actor_ids[:1], logits, values)


def test_actor_batch_dense_pending_markers_preserve_partial_order() -> None:
    batch = native.GameActorBatch(
        3,
        {"simulations": 2, "max_game_moves": 4, "tb_max_pieces": 0},
        {},
        456,
        a_is_white=[1, 0, 1],
    )
    actor_ids, _ = batch.positions_needing_eval(
        np.empty((3, 20, 8, 8), dtype=np.float32)
    )
    assert actor_ids.tolist() == [0, 1, 2]

    logits = np.zeros((2, POLICY_SIZE), dtype=np.float32)
    values = np.zeros(2, dtype=np.float32)
    batch.apply_eval(actor_ids[[2, 0]], logits, values)
    assert batch.pending_net_ids().tolist() == [1]

    with pytest.raises(ValueError, match="duplicate actor id"):
        batch.apply_eval(actor_ids[[1, 1]], logits, values)

    batch.apply_eval(
        actor_ids[1:2],
        np.zeros((1, POLICY_SIZE), dtype=np.float32),
        np.zeros(1, dtype=np.float32),
    )
    assert batch.pending_net_ids().shape == (0,)
