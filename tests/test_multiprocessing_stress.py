"""Short spawn-safety acceptance checks for central inference transport."""

from __future__ import annotations

import importlib
import inspect
import multiprocessing as mp
import queue

import pytest


def _inference_api():
    module = importlib.import_module("engine.inference")
    required = (
        "CentralInferenceBroker",
        "RemoteEvaluator",
        "SharedInferenceArena",
        "InferenceSettings",
    )
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        pytest.skip(f"central inference API is unavailable: {', '.join(missing)}")
    return module


def test_remote_evaluator_uses_shared_request_and_worker_response_queues() -> None:
    inference = _inference_api()
    parameters = list(inspect.signature(inference.RemoteEvaluator).parameters)

    assert parameters[:4] == ["arena", "worker_id", "requests", "responses"]
    assert "settings" in parameters
    assert "run_id" in parameters


def test_spawn_context_can_create_shared_and_per_worker_queues() -> None:
    """The queue topology required by RemoteEvaluator stays spawn-safe."""
    _inference_api()
    ctx = mp.get_context("spawn")
    requests = ctx.Queue(maxsize=1)
    responses = [ctx.Queue(maxsize=1) for _ in range(2)]
    try:
        requests.put_nowait(("request", 0))
        assert requests.get(timeout=1) == ("request", 0)
        responses[1].put_nowait(("response", 1))
        assert responses[1].get(timeout=1) == ("response", 1)
        with pytest.raises(queue.Full):
            requests.put_nowait(1)
            requests.put_nowait(2)
    finally:
        for channel in [requests, *responses]:
            channel.close()
            channel.join_thread()


def test_broker_exposes_generation_and_shutdown_lifecycle() -> None:
    inference = _inference_api()
    broker = inference.CentralInferenceBroker
    lifecycle_names = set(dir(broker))
    # Weight generations are carried by requests; the broker owns cancellation
    # and teardown so blocked workers cannot survive a generation change.
    assert "abort" in lifecycle_names
    assert "close" in lifecycle_names


def test_central_worker_init_does_not_build_local_evaluator() -> None:
    """Central workers must attach shared slots only — never create a local net/CUDA ctx."""
    inference = _inference_api()
    selfplay = importlib.import_module("engine.selfplay")

    arena = inference.SharedInferenceArena.create([4, 4])
    requests: queue.Queue = queue.Queue()
    responses = (queue.Queue(), queue.Queue())
    settings = inference.InferenceSettings(enabled=True, max_batch_size=4)
    central = (arena, requests, responses, settings)
    try:
        selfplay._WORKER_STATE.clear()
        selfplay._selfplay_worker_init(
            {"blocks": 1, "filters": 8, "value_bins": 11},
            "cpu",
            None,
            central,
        )
        state = selfplay._WORKER_STATE
        assert "central" in state
        assert "net" not in state
        assert "evaluator" not in state
        assert "device" not in state
    finally:
        selfplay._WORKER_STATE.clear()
        arena.close()
        arena.unlink()


def test_central_worker_concurrency_capped_to_arena_capacity() -> None:
    """A worker with more games than its shared slot must not oversize batches."""
    selfplay = importlib.import_module("engine.selfplay")
    src = inspect.getsource(selfplay._selfplay_worker_run)
    assert "min(n_games, int(arena.capacities[worker_id]))" in src
