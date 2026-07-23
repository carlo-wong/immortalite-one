"""MCTS visit parity: native session vs Python MCTS under uniform logits."""

from __future__ import annotations

import chess
import numpy as np
import pytest

from engine.config import MCTSConfig
from engine.encoding import POLICY_SIZE
from engine.mcts import MCTS
from engine._python_mcts import PythonMCTS


class UniformEvaluator:
    """Constant policy logits + fixed value (no torch)."""

    def __init__(self, value: float = 0.0):
        self.value = value

    def evaluate(self, board: chess.Board) -> tuple[np.ndarray, float]:
        return np.zeros(POLICY_SIZE, dtype=np.float32), self.value

    def evaluate_planes(self, planes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = int(planes.shape[0])
        return (
            np.zeros((n, POLICY_SIZE), dtype=np.float32),
            np.full(n, self.value, dtype=np.float32),
        )


def _visit_map(result) -> dict[str, int]:
    return {m.uci(): int(n) for m, n in zip(result.moves, result.visits)}


@pytest.mark.parametrize("sims", [8, 32])
def test_native_vs_python_visits_startpos(sims: int) -> None:
    cfg = MCTSConfig(simulations=sims, claim_draw=True, draw_contempt=1 / 3)
    board = chess.Board()
    ev = UniformEvaluator(0.0)

    py = PythonMCTS(ev, cfg).run(board.copy(), simulations=sims, add_noise=False)

    native_mcts = MCTS(ev, cfg)
    assert native_mcts.using_native
    cpp = native_mcts.run(board.copy(), simulations=sims, add_noise=False)

    assert _visit_map(cpp) == _visit_map(py)
    assert abs(cpp.root_value - py.root_value) < 1e-5
    py_imp = {m.uci(): float(p) for m, p in zip(py.moves, py.improved_policy())}
    cpp_imp = {m.uci(): float(p) for m, p in zip(cpp.moves, cpp.improved_policy())}
    assert set(py_imp) == set(cpp_imp)
    for uci in py_imp:
        assert abs(py_imp[uci] - cpp_imp[uci]) < 1e-5


@pytest.mark.parametrize("sims", [8, 32, 64])
def test_native_vs_python_visits_in_check(sims: int) -> None:
    fen = "r1b2bnr/pp2k1qp/n1pp2p1/4PpQ1/2P1P3/2NP3N/PP4PP/R1B1KB1R b KQ - 2 10"
    cfg = MCTSConfig(simulations=sims, claim_draw=True, draw_contempt=1 / 3)
    board = chess.Board(fen)
    assert board.is_check()
    ev = UniformEvaluator(0.0)

    py = PythonMCTS(ev, cfg).run(board.copy(), simulations=sims, add_noise=False)
    native_mcts = MCTS(ev, cfg)
    assert native_mcts.using_native
    cpp = native_mcts.run(board.copy(), simulations=sims, add_noise=False)

    assert _visit_map(cpp) == _visit_map(py)
    assert abs(cpp.root_value - py.root_value) < 1e-5
    py_imp = {m.uci(): float(p) for m, p in zip(py.moves, py.improved_policy())}
    cpp_imp = {m.uci(): float(p) for m, p in zip(cpp.moves, cpp.improved_policy())}
    assert set(py_imp) == set(cpp_imp)
    for uci in py_imp:
        assert abs(py_imp[uci] - cpp_imp[uci]) < 1e-5


def test_native_best_move_defined() -> None:
    cfg = MCTSConfig(simulations=16)
    mcts = MCTS(UniformEvaluator(0.1), cfg)
    assert mcts.using_native
    result = mcts.run(chess.Board(), simulations=16, add_noise=False)
    assert result.best_move() in chess.Board().legal_moves
    assert result.visits.sum() == 16
