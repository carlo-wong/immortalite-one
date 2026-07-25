#!/usr/bin/env python3
"""Benchmark parity and self-play throughput for native One versus Zero's Python path."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time

import chess
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


class UniformEvaluator:
    """Cheap evaluator that isolates board, encoding, and MCTS throughput."""

    def __init__(self, policy_size: int) -> None:
        self.logits = np.zeros(policy_size, dtype=np.float32)
        self.positions = 0

    def reset(self) -> None:
        self.positions = 0

    def evaluate(self, board: chess.Board) -> tuple[np.ndarray, float]:
        del board
        self.positions += 1
        return self.logits.copy(), 0.0

    def evaluate_batch(
        self, boards: list[chess.Board]
    ) -> tuple[np.ndarray, np.ndarray]:
        n = len(boards)
        self.positions += n
        return (
            np.repeat(self.logits[None, :], n, axis=0),
            np.zeros(n, dtype=np.float32),
        )

    def evaluate_planes(
        self, planes: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        n = int(planes.shape[0])
        self.positions += n
        return (
            np.repeat(self.logits[None, :], n, axis=0),
            np.zeros(n, dtype=np.float32),
        )


class CountingNetEvaluator:
    """Count positions while delegating to the production PyTorch evaluator."""

    def __init__(self, net, device: str) -> None:
        from engine.network import NetEvaluator

        self.inner = NetEvaluator(net, device=device)
        self.positions = 0

    def reset(self) -> None:
        self.positions = 0

    def evaluate(self, board: chess.Board) -> tuple[np.ndarray, float]:
        self.positions += 1
        return self.inner.evaluate(board)

    def evaluate_batch(
        self, boards: list[chess.Board]
    ) -> tuple[np.ndarray, np.ndarray]:
        self.positions += len(boards)
        return self.inner.evaluate_batch(boards)

    def evaluate_planes(
        self, planes: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        self.positions += int(planes.shape[0])
        return self.inner.evaluate_planes(planes)


def _play_python_batched(
    evaluator,
    cfg,
    *,
    simulations: int,
    games: int,
    concurrency: int,
) -> list:
    from engine.selfplay import play_game_gen

    active: list[tuple[object, object]] = []
    completed: list = []
    launched = 0
    while len(completed) < games:
        while launched < games and len(active) < concurrency:
            gen = play_game_gen(
                cfg,
                simulations,
                add_noise=False,
            )
            active.append((gen, next(gen)))
            launched += 1

        boards = [request.board for _, request in active]
        logits_batch, values_batch = evaluator.evaluate_batch(boards)
        next_active: list[tuple[object, object]] = []
        for (gen, _), logits, value in zip(active, logits_batch, values_batch):
            try:
                request = gen.send((logits, float(value)))
                next_active.append((gen, request))
            except StopIteration as stop:
                completed.append(stop.value)
        active = next_active
    return completed


def _run(
    backend: str,
    evaluator,
    cfg,
    *,
    games: int,
    concurrency: int,
    simulations: int,
    seed: int,
) -> dict:
    from engine.selfplay import _prefer_native_selfplay, play_games_batched_native

    if backend == "native":
        os.environ.pop("IMMORTALITE_ZERO_FORCE_PYTHON", None)
        os.environ.pop("IMMORTALITE_ONE_FORCE_PYTHON", None)
        os.environ["USE_NATIVE"] = "1"
        if not _prefer_native_selfplay():
            raise RuntimeError("native MctsSession unavailable; run `pip install -e .`")
    else:
        os.environ["IMMORTALITE_ZERO_FORCE_PYTHON"] = "1"

    np.random.seed(seed)
    evaluator.reset()
    started = time.perf_counter()
    if backend == "native":
        results = play_games_batched_native(
            evaluator,
            cfg,
            simulations=simulations,
            num_games=games,
            concurrency=concurrency,
            add_noise=False,
        )
    else:
        results = _play_python_batched(
            evaluator,
            cfg,
            simulations=simulations,
            games=games,
            concurrency=concurrency,
        )
    elapsed = time.perf_counter() - started
    samples = sum(len(game.samples) for game in results)
    return {
        "backend": backend,
        "seconds": elapsed,
        "games": len(results),
        "samples": samples,
        "eval_positions": evaluator.positions,
        "games_per_second": len(results) / elapsed,
        "samples_per_second": samples / elapsed,
        "moves": [game.moves for game in results],
        "terminations": [game.termination for game in results],
    }


def _summary(rows: list[dict]) -> dict:
    seconds = [float(row["seconds"]) for row in rows]
    games_per_second = [float(row["games_per_second"]) for row in rows]
    samples_per_second = [float(row["samples_per_second"]) for row in rows]
    return {
        "median_seconds": statistics.median(seconds),
        "median_games_per_second": statistics.median(games_per_second),
        "median_samples_per_second": statistics.median(samples_per_second),
        "trials": [
            {
                key: row[key]
                for key in (
                    "seconds",
                    "games",
                    "samples",
                    "eval_positions",
                    "games_per_second",
                    "samples_per_second",
                )
            }
            for row in rows
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--sims", type=int, default=32)
    parser.add_argument("--max-moves", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--evaluator",
        choices=("uniform", "network"),
        default="uniform",
        help="Uniform isolates search; network measures the production inference path.",
    )
    parser.add_argument("--device", default="cpu", help="PyTorch device for --evaluator network.")
    parser.add_argument("--net-blocks", type=int, default=2)
    parser.add_argument("--net-filters", type=int, default=16)
    parser.add_argument(
        "--min-speedup",
        type=float,
        default=0.0,
        help="Exit nonzero if median native speedup is below this value.",
    )
    parser.add_argument("--json", type=Path, help="Optional JSON result path.")
    args = parser.parse_args()
    if min(args.games, args.concurrency, args.sims, args.max_moves, args.repeats) <= 0:
        parser.error("games, concurrency, sims, max-moves, and repeats must be positive")

    from engine.config import Config
    from engine.encoding import POLICY_SIZE

    cfg = Config()
    cfg.mcts.simulations = args.sims
    cfg.mcts.dirichlet_epsilon = 0.0
    cfg.train.max_game_moves = args.max_moves
    cfg.train.move_temperature = 1.0
    cfg.train.move_temperature_plies = 0
    if args.evaluator == "uniform":
        evaluator = UniformEvaluator(POLICY_SIZE)
        evaluator_label = "uniform (search/self-play pipeline only)"
    else:
        from engine.network import ChessNet

        cfg.net.blocks = args.net_blocks
        cfg.net.filters = args.net_filters
        evaluator = CountingNetEvaluator(ChessNet(cfg.net), device=args.device)
        evaluator_label = (
            f"ChessNet blocks={args.net_blocks} filters={args.net_filters} "
            f"device={args.device}"
        )

    previous_force = os.environ.get("IMMORTALITE_ZERO_FORCE_PYTHON")
    previous_force_alias = os.environ.get("IMMORTALITE_ONE_FORCE_PYTHON")
    previous_native = os.environ.get("USE_NATIVE")
    native_rows: list[dict] = []
    python_rows: list[dict] = []
    try:
        # Warm both code paths so imports, allocation, and one-time initialization
        # do not inflate a measured trial.
        _run(
            "native",
            evaluator,
            cfg,
            games=1,
            concurrency=1,
            simulations=min(args.sims, 8),
            seed=0,
        )
        _run(
            "python",
            evaluator,
            cfg,
            games=1,
            concurrency=1,
            simulations=min(args.sims, 8),
            seed=0,
        )

        for trial in range(args.repeats):
            order = ("native", "python") if trial % 2 == 0 else ("python", "native")
            rows: dict[str, dict] = {}
            for backend in order:
                rows[backend] = _run(
                    backend,
                    evaluator,
                    cfg,
                    games=args.games,
                    concurrency=min(args.concurrency, args.games),
                    simulations=args.sims,
                    seed=trial + 1,
                )
            native = rows["native"]
            python = rows["python"]
            if (
                native["moves"] != python["moves"]
                or native["terminations"] != python["terminations"]
                or native["samples"] != python["samples"]
                or native["eval_positions"] != python["eval_positions"]
            ):
                mismatches = [
                    key
                    for key in ("moves", "terminations", "samples", "eval_positions")
                    if native[key] != python[key]
                ]
                raise RuntimeError(
                    f"functional parity failed in benchmark trial {trial + 1}: "
                    + ", ".join(
                        f"{key} native={native[key]!r} python={python[key]!r}"
                        for key in mismatches
                    )
                )
            native_rows.append(native)
            python_rows.append(python)
    finally:
        if previous_force is None:
            os.environ.pop("IMMORTALITE_ZERO_FORCE_PYTHON", None)
        else:
            os.environ["IMMORTALITE_ZERO_FORCE_PYTHON"] = previous_force
        if previous_force_alias is None:
            os.environ.pop("IMMORTALITE_ONE_FORCE_PYTHON", None)
        else:
            os.environ["IMMORTALITE_ONE_FORCE_PYTHON"] = previous_force_alias
        if previous_native is None:
            os.environ.pop("USE_NATIVE", None)
        else:
            os.environ["USE_NATIVE"] = previous_native

    native_summary = _summary(native_rows)
    python_summary = _summary(python_rows)
    speedup = (
        python_summary["median_seconds"] / native_summary["median_seconds"]
    )
    report = {
        "workload": {
            "games": args.games,
            "concurrency": min(args.concurrency, args.games),
            "simulations": args.sims,
            "max_moves": args.max_moves,
            "repeats": args.repeats,
            "evaluator": evaluator_label,
        },
        "native": native_summary,
        "python_zero_path": python_summary,
        "native_speedup": speedup,
        "functional_parity": "passed",
    }
    print(json.dumps(report, indent=2))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.min_speedup > 0.0 and speedup < args.min_speedup:
        raise SystemExit(
            f"native speedup {speedup:.2f}x is below required {args.min_speedup:.2f}x"
        )


if __name__ == "__main__":
    main()
