#!/usr/bin/env python3
"""Checkpoint-aware Phase A self-play throughput and profiling benchmark."""

from __future__ import annotations

import argparse
from collections import Counter
from functools import lru_cache
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import threading
import time
from typing import Any
import uuid

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from engine.config import Config, NetConfig
from engine.encoding import ENCODING_VERSION
from engine.inference import InferenceSettings
from engine.network import ChessNet, NetEvaluator
from engine.profile import ProfileCounters
from engine.selfplay import SelfplayWorkerPool, play_games_batched_native_actors

_WORKER: dict[str, Any] = {}


@lru_cache(maxsize=None)
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_ROOT, text=True
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _checkpoint_net_config(
    checkpoint: Path | None, blocks: int, filters: int
) -> NetConfig:
    if checkpoint is None:
        return NetConfig(blocks=blocks, filters=filters)
    state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, dict):
        checkpoint_encoding = int(state.get("encoding_version", 1))
        if checkpoint_encoding != ENCODING_VERSION:
            raise ValueError(
                f"checkpoint encoding version {checkpoint_encoding} does not match "
                f"engine encoding version {ENCODING_VERSION}"
            )
    if isinstance(state, dict) and "net" in state:
        return NetConfig(**state["net"])
    return NetConfig(blocks=blocks, filters=filters)


def _worker_init(
    checkpoint: str | None,
    net_cfg: dict[str, Any],
    device: str,
    syzygy_path: str | None,
    compile_net: bool,
) -> None:
    import chess.syzygy

    torch.set_num_threads(1)
    lifecycle: dict[str, float] = {}
    started = time.perf_counter()
    net = ChessNet(NetConfig(**net_cfg))
    lifecycle["network_build"] = time.perf_counter() - started
    if checkpoint is not None:
        started = time.perf_counter()
        state = torch.load(checkpoint, map_location=device)
        model_state = state["model"] if isinstance(state, dict) and "model" in state else state
        net.load_state_dict(model_state, strict=True)
        lifecycle["checkpoint_load"] = time.perf_counter() - started
    started = time.perf_counter()
    net.to(device).eval()
    if compile_net and device.startswith("cuda") and hasattr(torch, "compile"):
        net = torch.compile(net, dynamic=True)
    lifecycle["device_and_compile_setup"] = time.perf_counter() - started
    tablebase = chess.syzygy.open_tablebase(syzygy_path) if syzygy_path else None
    _WORKER.clear()
    _WORKER.update(
        evaluator=NetEvaluator(net, device=device),
        tablebase=tablebase,
        lifecycle=lifecycle,
    )


def _fingerprint(games: list[Any]) -> str:
    digest = hashlib.sha256()
    for game in games:
        digest.update(game.termination.encode())
        digest.update(str(game.winner).encode())
        digest.update("\0".join(game.moves).encode())
        for sample in game.samples:
            digest.update(sample.planes.tobytes())
            digest.update(sample.policy.tobytes())
            digest.update(bytes((int(sample.player),)))
            digest.update(np.float64(sample.value).tobytes())
            digest.update(np.float64(sample.root_q).tobytes())
    return digest.hexdigest()


def _worker_run(payload: dict[str, Any]) -> dict[str, Any]:
    random.seed(payload["seed"])
    np.random.seed(payload["seed"] % (2**32 - 1))
    torch.manual_seed(payload["seed"])
    cfg = Config()
    cfg.net = NetConfig(**payload["net_cfg"])
    cfg.mcts.simulations = payload["simulations"]
    cfg.mcts.draw_contempt = 1 / 3
    cfg.mcts.claim_draw = True
    cfg.train.sims_per_move = payload["simulations"]
    cfg.train.max_game_moves = payload["max_moves"]
    cfg.train.draw_penalty = 1 / 3
    cfg.train.value_target = "root_q"
    cfg.train.resign_threshold = -1.1
    cfg.train.resign_plies = 0
    cfg.train.move_temperature = 4.0
    cfg.train.move_temperature_plies = 10

    profile = ProfileCounters() if payload["profile"] else None
    evaluator: NetEvaluator = _WORKER["evaluator"]
    evaluator.profile = profile
    cpu_started = time.process_time()
    started = time.perf_counter()
    # Match production self-play's persistent GameActorBatch path. Its profile
    # exposes network, batch-width, game, and total timing counters, but not the
    # legacy per-session native node counters because sessions stay inside C++.
    games = play_games_batched_native_actors(
        evaluator,
        cfg,
        simulations=payload["simulations"],
        num_games=payload["games"],
        concurrency=payload["concurrency"],
        tablebase=_WORKER["tablebase"],
        add_noise=payload["add_noise"],
        exploration_moves=payload["exploration_moves"],
        profile=profile,
    )
    elapsed = time.perf_counter() - started
    max_rss_kib = None
    try:
        import resource

        max_rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except (ImportError, OSError):
        pass
    terminations = Counter(game.termination for game in games)
    return {
        "seconds": elapsed,
        "cpu_seconds": time.process_time() - cpu_started,
        "max_rss_kib": max_rss_kib,
        "games": len(games),
        "plies": sum(len(game.samples) for game in games),
        "samples": sum(len(game.samples) for game in games),
        "terminations": dict(terminations),
        "fingerprint": _fingerprint(games) if payload["fingerprint"] else None,
        "profile": profile.snapshot() if profile is not None else None,
        "lifecycle": _WORKER["lifecycle"],
    }


class ResourceMonitor:
    def __init__(self, interval: float) -> None:
        self.interval = interval
        self.samples: list[dict[str, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> ResourceMonitor:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    def _run(self) -> None:
        query = (
            "utilization.gpu,utilization.memory,clocks.sm,clocks.mem,"
            "power.draw,temperature.gpu,memory.used,memory.total"
        )
        while not self._stop.is_set():
            try:
                output = subprocess.check_output(
                    [
                        "nvidia-smi",
                        f"--query-gpu={query}",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    stderr=subprocess.DEVNULL,
                    timeout=max(1.0, self.interval),
                ).splitlines()[0]
                values = [float(value.strip()) for value in output.split(",")]
                self.samples.append(
                    dict(
                        zip(
                            (
                                "gpu_util_pct",
                                "memory_util_pct",
                                "sm_clock_mhz",
                                "memory_clock_mhz",
                                "power_w",
                                "temperature_c",
                                "vram_used_mib",
                                "vram_total_mib",
                            ),
                            values,
                        )
                    )
                )
            except (OSError, subprocess.SubprocessError, ValueError, IndexError):
                pass
            self._stop.wait(self.interval)


def _resource_summary(samples: list[dict[str, float]]) -> dict[str, Any]:
    if not samples:
        return {"samples": 0}
    return {
        "samples": len(samples),
        "mean": {
            key: statistics.fmean(sample[key] for sample in samples)
            for key in samples[0]
        },
        "max": {key: max(sample[key] for sample in samples) for key in samples[0]},
    }


def _split(total: int, parts: int) -> list[int]:
    return [total // parts + (1 if i < total % parts else 0) for i in range(parts)]


def _run_trial(
    pool: Any,
    *,
    workers: int,
    games: int,
    concurrency: int,
    simulations: int,
    max_moves: int,
    net_cfg: NetConfig,
    profile: bool,
    deterministic: bool,
    seed: int,
    resource_interval: float,
) -> dict[str, Any]:
    game_counts = _split(games, workers)
    concurrency_counts = _split(concurrency, workers)
    payloads = [
        {
            "games": game_counts[i],
            "concurrency": min(game_counts[i], max(1, concurrency_counts[i])),
            "simulations": simulations,
            "max_moves": max_moves,
            "net_cfg": vars(net_cfg),
            "profile": profile,
            "add_noise": not deterministic,
            "exploration_moves": 0 if deterministic else 20,
            "fingerprint": deterministic,
            "seed": seed + i * 1_000_003,
        }
        for i in range(workers)
        if game_counts[i] > 0
    ]
    process_started = time.process_time()
    with ResourceMonitor(resource_interval) as monitor:
        started = time.perf_counter()
        chunks = pool.map(_worker_run, payloads)
        wall_seconds = time.perf_counter() - started
    aggregate = ProfileCounters()
    for chunk in chunks:
        if chunk["profile"] is not None:
            aggregate.merge(chunk["profile"])
    games_done = sum(chunk["games"] for chunk in chunks)
    samples = sum(chunk["samples"] for chunk in chunks)
    plies = sum(chunk["plies"] for chunk in chunks)
    terminations: Counter[str] = Counter()
    for chunk in chunks:
        terminations.update(chunk["terminations"])
    return {
        "wall_seconds": wall_seconds,
        "process_cpu_seconds": time.process_time() - process_started,
        "games": games_done,
        "plies": plies,
        "samples": samples,
        "games_per_hour": games_done / wall_seconds * 3600,
        "seconds_per_game": wall_seconds / games_done,
        "seconds_per_evaluated_position": (
            wall_seconds / aggregate.counts["selfplay.evaluated_positions"]
            if aggregate.counts["selfplay.evaluated_positions"]
            else None
        ),
        "terminations": dict(terminations),
        "fingerprints": [chunk["fingerprint"] for chunk in chunks],
        "profile": aggregate.snapshot() if profile else None,
        "worker_lifecycle": [chunk["lifecycle"] for chunk in chunks],
        "resources": {
            **_resource_summary(monitor.samples),
            "worker_cpu_seconds": [chunk["cpu_seconds"] for chunk in chunks],
            "worker_max_rss_kib": [chunk["max_rss_kib"] for chunk in chunks],
        },
    }


def _environment() -> dict[str, Any]:
    gpu = None
    try:
        gpu = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        pass
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "python_compiler": platform.python_compiler(),
        "cpu": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": gpu,
    }


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _write_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _make_pool(args: argparse.Namespace, workers: int, net_cfg: NetConfig) -> Any:
    ctx = mp.get_context("spawn")
    return ctx.Pool(
        processes=workers,
        initializer=_worker_init,
        initargs=(
            str(args.checkpoint) if args.checkpoint else None,
            vars(net_cfg),
            args.device,
            args.syzygy_path,
            not args.no_compile,
        ),
    )


def _make_central_pool(
    args: argparse.Namespace, workers: int, net_cfg: NetConfig,
) -> tuple[SelfplayWorkerPool, NetEvaluator]:
    net = ChessNet(net_cfg)
    if args.checkpoint is not None:
        state = torch.load(args.checkpoint, map_location=args.device)
        model_state = state["model"] if isinstance(state, dict) and "model" in state else state
        net.load_state_dict(model_state, strict=True)
    net.to(args.device).eval()
    if not args.no_compile and args.device.startswith("cuda") and hasattr(torch, "compile"):
        net = torch.compile(net, dynamic=True)
    evaluator = NetEvaluator(net, device=args.device)
    pool = SelfplayWorkerPool(
        workers=workers,
        net_cfg=net_cfg,
        device=args.device,
        syzygy_path=args.syzygy_path,
        inference=InferenceSettings(enabled=True),
    )
    return pool, evaluator


def _central_config(args: argparse.Namespace, net_cfg: NetConfig) -> Config:
    cfg = Config()
    cfg.net = net_cfg
    cfg.mcts.simulations = args.sims
    cfg.mcts.draw_contempt = 1 / 3
    cfg.mcts.claim_draw = True
    cfg.train.sims_per_move = args.sims
    cfg.train.max_game_moves = args.max_moves
    cfg.train.draw_penalty = 1 / 3
    cfg.train.value_target = "root_q"
    cfg.train.resign_threshold = -1.1
    cfg.train.resign_plies = 0
    cfg.train.move_temperature = 4.0
    cfg.train.move_temperature_plies = 10
    return cfg


def _fingerprint_samples(samples: list[Any]) -> str:
    digest = hashlib.sha256()
    for sample in samples:
        digest.update(sample.planes.tobytes())
        digest.update(sample.policy.tobytes())
        digest.update(bytes((int(sample.player),)))
        digest.update(np.float64(sample.value).tobytes())
        digest.update(np.float64(sample.root_q).tobytes())
    return digest.hexdigest()


def _run_central_trial(
    pool: SelfplayWorkerPool,
    evaluator: NetEvaluator,
    *,
    args: argparse.Namespace,
    net_cfg: NetConfig,
    workers: int,
    profile: bool,
    deterministic: bool,
    seed: int,
) -> dict[str, Any]:
    # SelfplayWorkerPool distributes games evenly and uses native actors when available.
    # Its production seed/noise behavior matches engine.train; A0 only records a
    # sample fingerprint because it cannot disable worker-local root noise.
    del deterministic, seed
    counters = ProfileCounters() if profile else None
    evaluator.profile = counters
    process_started = time.process_time()
    with ResourceMonitor(args.resource_interval) as monitor:
        started = time.perf_counter()
        samples, terminations, lengths, _, _, _ = pool.run(
            _central_config(args, net_cfg),
            weights_path=None,
            simulations=args.sims,
            num_games=args.games,
            evaluator=evaluator,
        )
        wall_seconds = time.perf_counter() - started
    games_done = len(lengths)
    return {
        "wall_seconds": wall_seconds,
        "process_cpu_seconds": time.process_time() - process_started,
        "games": games_done,
        "plies": sum(lengths),
        "samples": len(samples),
        "games_per_hour": games_done / wall_seconds * 3600,
        "seconds_per_game": wall_seconds / games_done,
        "seconds_per_evaluated_position": None,
        "terminations": dict(terminations),
        "fingerprints": [_fingerprint_samples(samples)],
        "profile": counters.snapshot() if counters is not None else None,
        "worker_lifecycle": [{"central_inference": True, "workers": workers}],
        "resources": {
            **_resource_summary(monitor.samples),
            "worker_cpu_seconds": [],
            "worker_max_rss_kib": [],
        },
    }


def _identity(args: argparse.Namespace, variant: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_id": str(uuid.uuid4()),
        "variant": variant,
        "git_sha": _git_sha(),
        "checkpoint_sha256": _sha256(args.checkpoint) if args.checkpoint else "random-init",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _recipe(
    args: argparse.Namespace, net_cfg: NetConfig, workers: int, concurrency: int
) -> dict[str, Any]:
    return {
        "network": vars(net_cfg),
        "simulations": args.sims,
        "games": args.games,
        "workers": workers,
        "total_concurrency": concurrency,
        "value_target": "root_q",
        "move_temperature": 4.0,
        "move_temperature_plies": 10,
        "claim_draw": True,
        "max_game_moves": args.max_moves,
        "resign": False,
        "draw_penalty": 1 / 3,
        "draw_contempt": 1 / 3,
        "syzygy_path": args.syzygy_path,
        "central_inference": args.central_inference,
    }


def _central_enabled(args: argparse.Namespace, workers: int) -> bool:
    return (
        args.central_inference == "on"
        or (
            args.central_inference == "auto"
            and args.device.startswith("cuda")
            and workers > 1
        )
    ) and args.device.startswith("cuda") and workers > 1


def _production_variant(args: argparse.Namespace, workers: int) -> str:
    return "production-central-inference" if _central_enabled(args, workers) else "production-direct-batching"


def _run_a0(
    args: argparse.Namespace,
    net_cfg: NetConfig,
    environment: dict[str, Any],
    workers: int,
    concurrency: int,
) -> bool:
    rows: dict[bool, list[dict[str, Any]]] = {False: [], True: []}
    central = _central_enabled(args, workers)
    if central:
        pool, evaluator = _make_central_pool(args, workers, net_cfg)
        run_trial = lambda profile, seed: _run_central_trial(
            pool, evaluator, args=args, net_cfg=net_cfg, workers=workers,
            profile=profile, deterministic=True, seed=seed,
        )
    else:
        pool = _make_pool(args, workers, net_cfg)
        run_trial = lambda profile, seed: _run_trial(
            pool, workers=workers, games=args.games, concurrency=concurrency,
            simulations=args.sims, max_moves=args.max_moves, net_cfg=net_cfg,
            profile=profile, deterministic=True, seed=seed,
            resource_interval=args.resource_interval,
        )
    try:
        # Warm each mode at the measured shape so lazy compilation, allocator
        # growth, and first-use profile setup stay outside paired A0 trials.
        for enabled in (False, True):
            run_trial(enabled, args.seed - 1)
        for repeat in range(args.repeats):
            order = [False, True]
            random.Random(args.seed + repeat).shuffle(order)
            paired: dict[bool, dict[str, Any]] = {}
            for enabled in order:
                result = run_trial(enabled, args.seed + repeat * 10_000)
                paired[enabled] = result
                rows[enabled].append(result)
                artifact = {
                    **_identity(args, f"a0-profile-{'on' if enabled else 'off'}"),
                    "kind": "trial",
                    "environment": environment,
                    "recipe": _recipe(args, net_cfg, workers, concurrency),
                    "repeat": repeat,
                    "profile_enabled": enabled,
                    "deterministic": True,
                    "result": result,
                }
                _write_jsonl(args.output, artifact)
            if paired[False]["fingerprints"] != paired[True]["fingerprints"]:
                raise RuntimeError(f"A0 correctness fingerprint mismatch at repeat {repeat}")
    finally:
        pool.close()
    overheads = [
        (enabled["seconds_per_game"] / disabled["seconds_per_game"] - 1) * 100
        for disabled, enabled in zip(rows[False], rows[True])
    ]
    median_overhead = statistics.median(overheads)
    p90_overhead = _percentile(overheads, 0.9)
    passed = median_overhead < 2.0 and p90_overhead < 3.0
    _write_jsonl(
        args.output,
        {
            **_identity(args, "a0-summary"),
            "kind": "a0_summary",
            "environment": environment,
            "recipe": _recipe(args, net_cfg, workers, concurrency),
            "correctness": "passed",
            "overhead_pct": {
                "paired": overheads,
                "median": median_overhead,
                "p90": p90_overhead,
            },
            "thresholds_pct": {"median_lt": 2.0, "p90_lt": 3.0},
            "verdict": "passed" if passed else "failed",
        },
    )
    print(
        f"A0 correctness passed; profiling overhead median={median_overhead:.2f}% "
        f"p90={p90_overhead:.2f}% ({'PASS' if passed else 'FAIL'})"
    )
    return passed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--random-init", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("benchmark_throughput.jsonl"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", default="1,2,4")
    parser.add_argument("--concurrency", default="32,64,96,128,160")
    parser.add_argument("--games", type=int, default=128)
    parser.add_argument("--sims", type=int, default=150)
    parser.add_argument("--max-moves", type=int, default=200)
    parser.add_argument("--cold-observations", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--syzygy-path")
    parser.add_argument("--resource-interval", type=float, default=0.5)
    parser.add_argument("--net-blocks", type=int, default=8)
    parser.add_argument("--net-filters", type=int, default=96)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument(
        "--central-inference", choices=("auto", "on", "off"), default="auto",
        help="central CUDA evaluator mode for multi-worker lanes (default: auto)",
    )
    parser.add_argument("--a0", action="store_true")
    parser.add_argument("--a0-only", action="store_true")
    parser.add_argument("--allow-a0-overhead-failure", action="store_true")
    args = parser.parse_args()
    if (args.checkpoint is None) == (not args.random_init):
        parser.error("provide exactly one of --checkpoint or --random-init")
    if args.checkpoint is not None and not args.checkpoint.is_file():
        parser.error(f"checkpoint not found: {args.checkpoint}")
    if min(args.games, args.sims, args.max_moves, args.repeats) <= 0:
        parser.error("games, sims, max-moves, and repeats must be positive")
    if min(args.cold_observations, args.warmups) < 0 or args.resource_interval <= 0:
        parser.error(
            "cold-observations and warmups must be nonnegative; "
            "resource-interval must be positive"
        )

    workers_values = [int(value) for value in args.workers.split(",")]
    concurrency_values = [int(value) for value in args.concurrency.split(",")]
    if min(workers_values + concurrency_values) <= 0:
        parser.error("workers and concurrency values must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA device requested but torch.cuda.is_available() is false")

    net_cfg = _checkpoint_net_config(args.checkpoint, args.net_blocks, args.net_filters)
    environment = _environment()
    a0_passed = True
    if args.a0 or args.a0_only:
        a0_passed = _run_a0(
            args,
            net_cfg,
            environment,
            workers_values[0],
            concurrency_values[0],
        )
    if not args.a0_only:
        cells = [
            (workers, concurrency)
            for workers in workers_values
            for concurrency in concurrency_values
            if concurrency >= workers
        ]
        random.Random(args.seed).shuffle(cells)
        for workers, concurrency in cells:
            central = _central_enabled(args, workers)
            if central:
                pool, evaluator = _make_central_pool(args, workers, net_cfg)
                run_trial = lambda seed: _run_central_trial(
                    pool, evaluator, args=args, net_cfg=net_cfg, workers=workers,
                    profile=True, deterministic=False, seed=seed,
                )
            else:
                pool = _make_pool(args, workers, net_cfg)
                run_trial = lambda seed: _run_trial(
                    pool, workers=workers, games=args.games, concurrency=concurrency,
                    simulations=args.sims, max_moves=args.max_moves, net_cfg=net_cfg,
                    profile=True, deterministic=False, seed=seed,
                    resource_interval=args.resource_interval,
                )
            try:
                for cold in range(args.cold_observations):
                    result = run_trial(args.seed - 10_000 - cold)
                    _write_jsonl(
                        args.output,
                        {
                            **_identity(args, _production_variant(args, workers)),
                            "kind": "cold_observation",
                            "environment": environment,
                            "recipe": _recipe(args, net_cfg, workers, concurrency),
                            "repeat": cold,
                            "profile_enabled": True,
                            "deterministic": False,
                            "result": result,
                        },
                    )
                for warmup in range(args.warmups):
                    result = run_trial(args.seed + warmup)
                    _write_jsonl(
                        args.output,
                        {
                            **_identity(args, _production_variant(args, workers)),
                            "kind": "warmup",
                            "environment": environment,
                            "recipe": _recipe(args, net_cfg, workers, concurrency),
                            "repeat": warmup,
                            "profile_enabled": True,
                            "deterministic": False,
                            "result": result,
                        },
                    )
                for repeat in range(args.repeats):
                    result = run_trial(args.seed + repeat * 10_000)
                    _write_jsonl(
                        args.output,
                        {
                            **_identity(args, _production_variant(args, workers)),
                            "kind": "trial",
                            "environment": environment,
                            "recipe": _recipe(args, net_cfg, workers, concurrency),
                            "repeat": repeat,
                            "profile_enabled": True,
                            "deterministic": False,
                            "result": result,
                        },
                    )
                    print(
                        f"workers={workers} concurrency={concurrency} repeat={repeat + 1} "
                        f"{result['seconds_per_game']:.3f}s/game "
                        f"{result['games_per_hour']:.1f} games/hour"
                    )
            finally:
                pool.close()
    if not a0_passed and not args.allow_a0_overhead_failure:
        raise SystemExit("A0 profiling overhead exceeded its budget")


if __name__ == "__main__":
    main()
