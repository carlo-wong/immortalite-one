"""MCTS facade: prefer native C++ search, fall back to Python for bootstrap.

Environment:
  USE_NATIVE=1 (default) — try ``engine._native`` first.
  IMMORTALITE_ZERO_FORCE_PYTHON=1 — skip native and use the python-chess port
  (alias: IMMORTALITE_ONE_FORCE_PYTHON).
"""

from __future__ import annotations

import importlib
import logging
import warnings
from typing import Any, Generator

import chess
import numpy as np

from .config import MCTSConfig
from .network import NetEvaluator
from ._env import env_flag
from ._python_mcts import PythonMCTS, SearchResult, _Node

__all__ = [
    "MCTS",
    "SearchResult",
    "PythonMCTS",
    "_board_root_fen_and_moves",
    "_load_native",
    "_native_search_ready",
]

_LOG = logging.getLogger(__name__)
_PYTHON_FALLBACK_WARNED = False


def _attr(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj[name]
    return getattr(obj, name)


def _force_python() -> bool:
    return env_flag(
        "IMMORTALITE_ZERO_FORCE_PYTHON",
        "IMMORTALITE_ONE_FORCE_PYTHON",
        default=False,
    )


def _use_native() -> bool:
    return env_flag("USE_NATIVE", default=True)


def _load_native() -> Any | None:
    try:
        return importlib.import_module("engine._native")
    except ImportError:
        return None


def _native_search_ready(native: Any) -> bool:
    return hasattr(native, "MctsSession")


def _warn_python_fallback(reason: str) -> None:
    global _PYTHON_FALLBACK_WARNED
    if _PYTHON_FALLBACK_WARNED:
        return
    _PYTHON_FALLBACK_WARNED = True
    msg = (
        f"Immortalite Zero: using Python MCTS fallback ({reason}). "
        "Install the native extension with `pip install -e .` for production search. "
        "Set IMMORTALITE_ZERO_FORCE_PYTHON=1 to silence this during bootstrap."
    )
    warnings.warn(msg, RuntimeWarning, stacklevel=3)
    _LOG.warning(msg)


def _board_root_fen_and_moves(board: chess.Board) -> tuple[str, list[str]]:
    """Split board into root FEN + UCI moves so native search keeps repetition history."""
    root = board.root()
    moves = [m.uci() for m in board.move_stack]
    return root.fen(), moves


def _node_from_export(data: Any) -> _Node:
    """Rebuild a Python ``_Node`` from a native ``export_tree`` dict."""
    move_uci = _attr(data, "move")
    move = chess.Move.from_uci(move_uci) if move_uci else None
    node = _Node(float(_attr(data, "prior")), move=move)
    node.N = int(_attr(data, "N"))
    node.W = float(_attr(data, "W"))
    for child in _attr(data, "children") or ():
        node.children[int(_attr(child, "index"))] = _node_from_export(child)
    return node


def _root_from_native_tree(raw: Any) -> _Node:
    """Build SearchResult ``_root`` from native result ``tree`` (list of root children)."""
    root = _Node(0.0)
    if isinstance(raw, dict):
        tree = raw.get("tree")
    else:
        tree = getattr(raw, "tree", None)
    if not tree:
        return root
    for child in tree:
        root.children[int(_attr(child, "index"))] = _node_from_export(child)
    if root.children:
        root.N = sum(c.N for c in root.children.values())
    return root


class _NativeMCTSAdapter:
    """Drive ``engine._native.MctsSession`` step API with PyTorch eval."""

    def __init__(self, native: Any, evaluator: NetEvaluator | None, cfg: MCTSConfig):
        self._native = native
        self.evaluator = evaluator
        self.cfg = cfg
        self._python = PythonMCTS(evaluator, cfg)

    def _cfg_dict(self) -> dict[str, Any]:
        return {
            "simulations": self.cfg.simulations,
            "c_puct": self.cfg.c_puct,
            "dirichlet_alpha": self.cfg.dirichlet_alpha,
            "dirichlet_epsilon": self.cfg.dirichlet_epsilon,
            "gumbel_c_visit": self.cfg.gumbel_c_visit,
            "gumbel_c_scale": self.cfg.gumbel_c_scale,
            "draw_contempt": self.cfg.draw_contempt,
            "claim_draw": self.cfg.claim_draw,
            "virtual_loss": int(self.cfg.virtual_loss),
            "max_leaves_per_eval": int(self.cfg.max_leaves_per_eval),
        }

    def run(self, board: chess.Board, simulations: int | None = None,
            add_noise: bool = False) -> SearchResult:
        if self.evaluator is None:
            raise ValueError("native MCTS requires an evaluator")

        sims = simulations if simulations is not None else self.cfg.simulations
        fen, moves = _board_root_fen_and_moves(board)
        session = self._native.MctsSession(
            fen, sims, self._cfg_dict(), add_noise, moves if moves else None
        )
        while not session.done():
            planes = session.positions_needing_eval()
            if planes.shape[0] == 0:
                break
            logits, values = self.evaluator.evaluate_planes(np.asarray(planes, dtype=np.float32))
            session.apply_eval(logits, values)
        return self._coerce_result(session.result(), board)

    def search_gen(self, board: chess.Board, simulations: int | None = None,
                   add_noise: bool = False
                   ) -> Generator[chess.Board, tuple[np.ndarray, float], SearchResult]:
        # Native session is run-oriented; Python generator kept for batching tests.
        return self._python.search_gen(
            board, simulations=simulations, add_noise=add_noise
        )

    def _terminal_value(self, board: chess.Board,
                        root_turn: chess.Color | None = None) -> float:
        return self._python._terminal_value(board, root_turn)

    def _coerce_result(self, raw: Any, board: chess.Board) -> SearchResult:
        if isinstance(raw, SearchResult):
            return raw
        moves = list(_attr(raw, "moves"))
        if moves and isinstance(moves[0], str):
            moves = [chess.Move.from_uci(m) for m in moves]
        return SearchResult(
            moves=moves,
            indices=list(_attr(raw, "indices")),
            visits=np.asarray(_attr(raw, "visits"), dtype=np.float64),
            q_values=np.asarray(_attr(raw, "q_values"), dtype=np.float64),
            priors=np.asarray(_attr(raw, "priors"), dtype=np.float64),
            clean_priors=np.asarray(_attr(raw, "clean_priors"), dtype=np.float64),
            root_value=float(_attr(raw, "root_value")),
            _root=_root_from_native_tree(raw),
            _board=board,
            _cfg=self.cfg,
        )


class MCTS:
    """Public MCTS entry point used by analyze / UCI / tests."""

    def __init__(self, evaluator: NetEvaluator | None, cfg: MCTSConfig | None = None):
        self.evaluator = evaluator
        self.cfg = cfg or MCTSConfig()
        self._backend: PythonMCTS | _NativeMCTSAdapter = self._select_backend()

    def _select_backend(self) -> PythonMCTS | _NativeMCTSAdapter:
        if _force_python():
            return PythonMCTS(self.evaluator, self.cfg)

        if not _use_native():
            _warn_python_fallback("USE_NATIVE=0")
            return PythonMCTS(self.evaluator, self.cfg)

        native = _load_native()
        if native is None:
            _warn_python_fallback("native extension not installed")
            return PythonMCTS(self.evaluator, self.cfg)

        if not _native_search_ready(native):
            _warn_python_fallback("native module has no MCTS search API yet")
            return PythonMCTS(self.evaluator, self.cfg)

        return _NativeMCTSAdapter(native, self.evaluator, self.cfg)

    @property
    def using_native(self) -> bool:
        return isinstance(self._backend, _NativeMCTSAdapter)

    def run(self, board: chess.Board, simulations: int | None = None,
            add_noise: bool = False) -> SearchResult:
        return self._backend.run(board, simulations=simulations, add_noise=add_noise)

    def search_gen(self, board: chess.Board, simulations: int | None = None,
                   add_noise: bool = False
                   ) -> Generator[chess.Board, tuple[np.ndarray, float], SearchResult]:
        return self._backend.search_gen(
            board, simulations=simulations, add_noise=add_noise
        )

    def _terminal_value(self, board: chess.Board,
                        root_turn: chess.Color | None = None) -> float:
        return self._backend._terminal_value(board, root_turn)
