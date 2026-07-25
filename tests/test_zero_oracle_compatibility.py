"""Optional cross-repository checks against the archived pure-Python Zero checkout."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import chess
import numpy as np
import pytest
import torch

from engine import _native
from engine._oracle_path import find_zero_root
from engine.config import MCTSConfig, NetConfig
from engine.mcts import MCTS
from engine.network import ChessNet


ZERO_ROOT = find_zero_root()
pytestmark = pytest.mark.skipif(
    ZERO_ROOT is None,
    reason="Archived pure-Python Zero sibling checkout not found",
)


def _run_zero(script: str, payload: object) -> object:
    assert ZERO_ROOT is not None
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ZERO_ROOT)
    env["IMMORTALITE_ZERO_ORACLE_ROOT"] = str(ZERO_ROOT)
    bootstrap = r"""
import importlib.util, os, site, sys
sys.path.extend(path for path in site.getsitepackages() if path not in sys.path)
for _name in list(sys.modules):
    if _name == "engine" or _name.startswith("engine."):
        del sys.modules[_name]
_engine_dir = os.path.join(os.environ["IMMORTALITE_ZERO_ORACLE_ROOT"], "engine")
_spec = importlib.util.spec_from_file_location(
    "engine",
    os.path.join(_engine_dir, "__init__.py"),
    submodule_search_locations=[_engine_dir],
)
_engine = importlib.util.module_from_spec(_spec)
sys.modules["engine"] = _engine
_spec.loader.exec_module(_engine)
"""
    completed = subprocess.run(
        [sys.executable, "-S", "-c", bootstrap + script],
        cwd=ZERO_ROOT,
        env=env,
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"Immortalite Zero oracle failed ({completed.returncode}):\n"
            f"{completed.stderr}"
        )
    return json.loads(completed.stdout)


def _histories() -> list[list[str]]:
    rng = np.random.default_rng(20260724)
    histories: list[list[str]] = []
    for _ in range(8):
        board = chess.Board()
        history: list[str] = []
        for ply in range(64):
            if ply % 4 == 0:
                histories.append(history.copy())
            moves = list(board.legal_moves)
            if not moves or board.is_game_over(claim_draw=True):
                break
            move = moves[int(rng.integers(len(moves)))]
            board.push(move)
            history.append(move.uci())
    histories.extend(
        [
            ["e2e4", "a7a6", "e4e5", "d7d5"],
            list(("g1f3", "g8f6", "f3g1", "f6g8") * 2),
        ]
    )
    return histories


def test_native_positions_match_actual_zero_oracle() -> None:
    histories = _histories()
    script = r"""
import hashlib, json, sys
import chess
from engine.encoding import board_to_planes, legal_move_indices

out = []
for history in json.load(sys.stdin):
    board = chess.Board()
    for uci in history:
        board.push_uci(uci)
    out.append({
        "fen": board.fen(),
        "planes": hashlib.sha256(board_to_planes(board).tobytes()).hexdigest(),
        "moves": sorted((int(i), m.uci()) for i, m in legal_move_indices(board).items()),
    })
print(json.dumps(out))
"""
    expected = _run_zero(script, histories)

    assert isinstance(expected, list)
    for history, oracle in zip(histories, expected):
        board = chess.Board()
        for uci in history:
            board.push_uci(uci)
        planes = np.asarray(
            _native.fill_planes_fen(chess.STARTING_FEN, history if history else None),
            dtype=np.float32,
        )
        moves = sorted(
            (int(index), uci)
            for index, uci in _native.legal_move_indices_fen(board.fen())
        )
        assert board.fen() == oracle["fen"]
        assert hashlib.sha256(planes.tobytes()).hexdigest() == oracle["planes"]
        assert [list(item) for item in moves] == oracle["moves"]


class _FixedEvaluator:
    def __init__(self, seed: int, value: float) -> None:
        self.logits = np.random.default_rng(seed).normal(
            size=4672
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


def test_native_mcts_matches_actual_zero_oracle() -> None:
    cases = [
        [chess.STARTING_FEN, 11, 0.0],
        ["r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", 12, 0.3],
        ["4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", 13, -0.25],
    ]
    script = r"""
import json, sys
import chess
import numpy as np
from engine.config import MCTSConfig
from engine.mcts import MCTS

class Evaluator:
    def __init__(self, seed, value):
        self.logits = np.random.default_rng(seed).normal(size=4672).astype(np.float32)
        self.value = float(value)
    def evaluate(self, board):
        return self.logits.copy(), self.value

out = []
for fen, seed, value in json.load(sys.stdin):
    cfg = MCTSConfig(simulations=48, dirichlet_epsilon=0.0, claim_draw=True)
    result = MCTS(Evaluator(seed, value), cfg).run(
        chess.Board(fen), simulations=48, add_noise=False
    )
    out.append({
        move.uci(): [int(n), float(q), float(p)]
        for move, n, q, p in zip(
            result.moves, result.visits, result.q_values, result.clean_priors
        )
    })
print(json.dumps(out))
"""
    expected = _run_zero(script, cases)

    assert isinstance(expected, list)
    for (fen, seed, value), oracle in zip(cases, expected):
        cfg = MCTSConfig(simulations=48, dirichlet_epsilon=0.0, claim_draw=True)
        result = MCTS(_FixedEvaluator(seed, value), cfg).run(
            chess.Board(fen), simulations=48, add_noise=False
        )
        actual = {
            move.uci(): [int(n), float(q), float(p)]
            for move, n, q, p in zip(
                result.moves, result.visits, result.q_values, result.clean_priors
            )
        }
        assert actual.keys() == oracle.keys()
        for uci in oracle:
            np.testing.assert_allclose(actual[uci], oracle[uci], rtol=1e-5, atol=1e-6)


def test_zero_loads_one_network_weights(tmp_path: Path) -> None:
    cfg = NetConfig(blocks=2, filters=8, value_bins=51)
    path = tmp_path / "one_state_dict.pt"
    torch.save(ChessNet(cfg).state_dict(), path)
    script = r"""
import json, sys, torch
from engine.config import NetConfig
from engine.network import ChessNet

path = json.load(sys.stdin)
net = ChessNet(NetConfig(blocks=2, filters=8, value_bins=51))
state = torch.load(path, map_location="cpu", weights_only=True)
result = net.load_state_dict(state)
print(json.dumps({
    "missing": list(result.missing_keys),
    "unexpected": list(result.unexpected_keys),
    "keys": len(state),
}))
"""
    result = _run_zero(script, str(path))
    assert result == {
        "missing": [],
        "unexpected": [],
        "keys": len(ChessNet(cfg).state_dict()),
    }
