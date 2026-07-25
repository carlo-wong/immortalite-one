#!/usr/bin/env python3
"""Compare native vs Python MCTS wall time (CPU, uniform / random-init net).

Example:
  python scripts/bench_mcts.py --sims 32 --searches 50
"""

from __future__ import annotations

import argparse
import os
import time

import chess
import numpy as np

# Ensure package import works from repo root.
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _run_loop(label: str, mcts, board_fen: str, sims: int, searches: int) -> float:
    board = chess.Board(board_fen)
    # Warmup
    mcts.run(board.copy(), simulations=min(4, sims), add_noise=False)
    t0 = time.perf_counter()
    for _ in range(searches):
        mcts.run(board.copy(), simulations=sims, add_noise=False)
    elapsed = time.perf_counter() - t0
    per = elapsed / searches
    print(f"{label:10s}  {searches} searches × {sims} sims  "
          f"total={elapsed:.3f}s  per_search={per*1000:.1f}ms")
    return per


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sims", type=int, default=32)
    parser.add_argument("--searches", type=int, default=30)
    parser.add_argument("--fen", type=str, default=chess.STARTING_FEN)
    args = parser.parse_args()

    from engine.config import Config, MCTSConfig
    from engine.encoding import POLICY_SIZE
    from engine.mcts import MCTS
    from engine.network import ChessNet, NetEvaluator
    from engine._python_mcts import PythonMCTS

    class UniformEvaluator:
        def evaluate(self, board: chess.Board):
            return np.zeros(POLICY_SIZE, dtype=np.float32), 0.0

        def evaluate_planes(self, planes: np.ndarray):
            n = int(planes.shape[0])
            return (
                np.zeros((n, POLICY_SIZE), dtype=np.float32),
                np.zeros(n, dtype=np.float32),
            )

    cfg = MCTSConfig(simulations=args.sims, claim_draw=True)
    ev = UniformEvaluator()

    # Native
    os.environ.pop("IMMORTALITE_ZERO_FORCE_PYTHON", None)
    os.environ.pop("IMMORTALITE_ONE_FORCE_PYTHON", None)
    os.environ["USE_NATIVE"] = "1"
    native_mcts = MCTS(ev, cfg)
    if not native_mcts.using_native:
        raise SystemExit("native MCTS unavailable — run `pip install -e .` first")
    t_native = _run_loop("native", native_mcts, args.fen, args.sims, args.searches)

    # Python
    py_mcts = PythonMCTS(ev, cfg)
    t_py = _run_loop("python", py_mcts, args.fen, args.sims, args.searches)

    speedup = t_py / t_native if t_native > 0 else float("inf")
    print(f"speedup    native is {speedup:.2f}× faster than Python "
          f"(uniform logits, CPU, no torch forward)")

    # Optional: with real net forward (still CPU)
    net_ev = NetEvaluator(ChessNet(Config().net), device="cpu")
    os.environ["USE_NATIVE"] = "1"
    native_net = MCTS(net_ev, cfg)
    t_nn = _run_loop("native+nn", native_net, args.fen, args.sims, max(5, args.searches // 3))
    print(f"note       native+nn includes ChessNet forward on CPU "
          f"({t_nn*1000:.1f} ms/search)")


if __name__ == "__main__":
    main()
