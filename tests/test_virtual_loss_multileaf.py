"""Flag-gated multi-leaf MCTS with virtual loss (native)."""

from __future__ import annotations

import numpy as np
import pytest

from engine.encoding import POLICY_SIZE
from engine.mcts import _load_native


def _native():
    native = _load_native()
    if native is None or not hasattr(native, "MctsSession"):
        pytest.skip("native MctsSession unavailable")
    return native


def _run_session(session, value: float = 0.1) -> None:
    while not session.done():
        planes = np.asarray(session.positions_needing_eval(), dtype=np.float32)
        if planes.shape[0] == 0:
            break
        n = int(planes.shape[0])
        logits = np.zeros((n, POLICY_SIZE), dtype=np.float32)
        values = np.full(n, value, dtype=np.float32)
        session.apply_eval(logits, values)


def test_defaults_keep_single_leaf_wave() -> None:
    native = _native()
    session = native.MctsSession(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        8,
        {"simulations": 8, "dirichlet_epsilon": 0.0},
        False,
    )
    planes = np.asarray(session.positions_needing_eval(), dtype=np.float32)
    assert planes.shape[0] == 1  # root
    session.apply_eval(
        np.zeros((1, POLICY_SIZE), dtype=np.float32),
        np.zeros(1, dtype=np.float32),
    )
    if not session.done():
        planes = np.asarray(session.positions_needing_eval(), dtype=np.float32)
        assert planes.shape[0] == 1  # single leaf when VL off


def test_virtual_loss_off_ignores_max_leaves() -> None:
    native = _native()
    session = native.MctsSession(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        16,
        {
            "simulations": 16,
            "dirichlet_epsilon": 0.0,
            "virtual_loss": 0,
            "max_leaves_per_eval": 8,
        },
        False,
    )
    _run_session(session)
    assert session.done()
    assert session.total_virtual_loss() == 0
    result = session.result()
    assert int(np.asarray(result["visits"]).sum()) == 16


def test_multileaf_requests_k_planes_and_progresses() -> None:
    native = _native()
    max_leaves = 4
    session = native.MctsSession(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        32,
        {
            "simulations": 32,
            "dirichlet_epsilon": 0.0,
            "virtual_loss": 1,
            "max_leaves_per_eval": max_leaves,
        },
        False,
    )
    # Root eval is still a single plane.
    planes = np.asarray(session.positions_needing_eval(), dtype=np.float32)
    assert planes.shape[0] == 1
    session.apply_eval(
        np.zeros((1, POLICY_SIZE), dtype=np.float32),
        np.full(1, 0.05, dtype=np.float32),
    )
    assert not session.done()

    saw_multi = False
    waves = 0
    while not session.done() and waves < 64:
        planes = np.asarray(session.positions_needing_eval(), dtype=np.float32)
        if planes.shape[0] == 0:
            break
        k = int(planes.shape[0])
        assert 1 <= k <= max_leaves
        if k > 1:
            saw_multi = True
            assert session.pending_eval_count() >= k
            assert session.total_virtual_loss() > 0
        logits = np.zeros((k, POLICY_SIZE), dtype=np.float32)
        values = np.full(k, 0.1, dtype=np.float32)
        session.apply_eval(logits, values)
        assert session.total_virtual_loss() == 0 or not session.done()
        waves += 1

    assert saw_multi, "expected at least one multi-leaf eval wave"
    assert session.done()
    assert session.total_virtual_loss() == 0
    result = session.result()
    assert int(np.asarray(result["visits"]).sum()) == 32


def test_game_actor_multileaf_csr_and_apply() -> None:
    native = _native()
    if not hasattr(native, "GameActorBatch"):
        pytest.skip("native GameActorBatch unavailable")
    batch = native.GameActorBatch(
        2,
        {"simulations": 16, "tb_max_pieces": 0, "add_noise": False, "max_game_moves": 2},
        {
            "simulations": 16,
            "dirichlet_epsilon": 0.0,
            "virtual_loss": 3,
            "max_leaves_per_eval": 4,
        },
        7,
    )
    saw_multi = False
    completed_games = 0
    for _ in range(400):
        actor_ids, _planes = batch.positions_needing_eval()
        if actor_ids.shape[0] == 0:
            completed_games += len(batch.take_completed()["games"])
            if completed_games >= 1:
                break
            continue
        if int(actor_ids.shape[0]) > 2:
            saw_multi = True
        legal_indices, legal_offsets = batch.pending_legal_csr()
        assert legal_offsets.shape[0] == actor_ids.shape[0] + 1
        assert int(legal_offsets[-1]) == len(legal_indices)
        gathered = np.zeros(len(legal_indices), dtype=np.float32)
        values = np.full(actor_ids.shape[0], 0.12, dtype=np.float32)
        batch.apply_eval_legal(actor_ids, gathered, legal_offsets, values)
        completed_games += len(batch.take_completed()["games"])
    assert saw_multi
    assert completed_games >= 1
