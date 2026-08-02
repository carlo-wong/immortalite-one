"""Tests for centralized shared-memory inference."""

from __future__ import annotations

import queue
import threading
import time

import numpy as np
import pytest
import torch

from engine.config import NetConfig
from engine.inference import (
    CentralInferenceBroker,
    InferenceResponse,
    InferenceSettings,
    RemoteEvaluator,
    SharedInferenceArena,
)
from engine.network import ChessNet, NetEvaluator


class _FakeEvaluator:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def evaluate_planes(self, planes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.batch_sizes.append(int(planes.shape[0]))
        tags = planes[:, 0, 0, 0].astype(np.float32)
        return np.repeat(tags[:, None], 4672, axis=1), tags + 0.5

    def evaluate_legal(
        self, planes: np.ndarray, indices: np.ndarray, offsets: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        tags = planes[:, 0, 0, 0].astype(np.float32)
        rows = np.repeat(tags, np.diff(offsets))
        return rows + indices.astype(np.float32), tags + 0.5


def _broker_parts(workers: int = 2, timeout: float = 1.0):
    arena = SharedInferenceArena.create([8] * workers)
    requests: queue.Queue = queue.Queue()
    responses = tuple(queue.Queue() for _ in range(workers))
    settings = InferenceSettings(max_batch_size=8, max_wait_us=10_000, response_timeout_s=timeout)
    return arena, requests, responses, settings


def test_remote_routing_ignores_out_of_order_response() -> None:
    arena, requests, responses, settings = _broker_parts()
    try:
        remote = RemoteEvaluator(arena, 0, requests, responses[0], settings, run_id=5)
        # A stale response must not be mistaken for the request being awaited.
        responses[0].put(InferenceResponse(run_id=4, request_id=0, count=1))
        result: list[tuple[np.ndarray, np.ndarray]] = []
        thread = threading.Thread(
            target=lambda: result.append(remote.evaluate_planes(np.ones((1, 20, 8, 8), np.float32))),
        )
        thread.start()
        broker = CentralInferenceBroker(_FakeEvaluator(), arena, requests, responses, settings)
        while thread.is_alive():
            broker.service_once(0.01)
        thread.join(timeout=5.0)
        assert not thread.is_alive()
        logits, values = result[0]
        assert logits[0, 0] == 1.0
        assert values[0] == 1.5
    finally:
        arena.close()
        arena.unlink()


def test_broker_batches_multiple_remote_evaluators() -> None:
    arena, requests, responses, settings = _broker_parts()
    try:
        remotes = [
            RemoteEvaluator(arena, worker_id, requests, responses[worker_id], settings, run_id=7)
            for worker_id in range(2)
        ]
        results: list[tuple[np.ndarray, np.ndarray] | None] = [None, None]
        threads = [
            threading.Thread(
                target=lambda i=i: results.__setitem__(
                    i, remotes[i].evaluate_planes(
                        np.full((2, 20, 8, 8), i + 2, dtype=np.float32)
                    ),
                ),
            )
            for i in range(2)
        ]
        for thread in threads:
            thread.start()
        while requests.qsize() < 2:
            time.sleep(0.001)
        fake = _FakeEvaluator()
        broker = CentralInferenceBroker(fake, arena, requests, responses, settings)
        while any(thread.is_alive() for thread in threads):
            broker.service_once(0.01)
        for thread in threads:
            thread.join(timeout=5.0)
            assert not thread.is_alive()
        assert results[0] is not None and results[1] is not None
        assert results[0][0].shape == (2, 4672)
        np.testing.assert_allclose(results[0][1], [2.5, 2.5])
        np.testing.assert_allclose(results[1][1], [3.5, 3.5])
        assert fake.batch_sizes == [4]
    finally:
        arena.close()
        arena.unlink()


def test_remote_timeout_and_broker_abort() -> None:
    arena, requests, responses, settings = _broker_parts(workers=1, timeout=0.03)
    try:
        remote = RemoteEvaluator(arena, 0, requests, responses[0], settings, run_id=1)
        with pytest.raises(TimeoutError):
            remote.evaluate_planes(np.zeros((1, 20, 8, 8), dtype=np.float32))
        while not requests.empty():
            requests.get_nowait()

        remote = RemoteEvaluator(arena, 0, requests, responses[0], settings, run_id=2)
        failure: list[BaseException] = []
        thread = threading.Thread(
            target=lambda: _capture(
                failure, lambda: remote.evaluate_planes(np.zeros((1, 20, 8, 8), dtype=np.float32))
            ),
        )
        thread.start()
        while requests.empty():
            time.sleep(0.001)
        broker = CentralInferenceBroker(_FakeEvaluator(), arena, requests, responses, settings)
        broker.abort("stopped")
        thread.join(timeout=5.0)
        assert not thread.is_alive()
        assert isinstance(failure[0], RuntimeError)
        assert "stopped" in str(failure[0])
    finally:
        arena.close()
        arena.unlink()


def _capture(errors: list[BaseException], fn) -> None:
    try:
        fn()
    except BaseException as exc:
        errors.append(exc)


def test_remote_planes_match_direct_evaluator() -> None:
    torch.manual_seed(42)
    direct = NetEvaluator(ChessNet(NetConfig(blocks=1, filters=8, value_bins=11)), device="cpu")
    arena, requests, responses, settings = _broker_parts(workers=1)
    try:
        remote = RemoteEvaluator(arena, 0, requests, responses[0], settings, run_id=9)
        planes = np.random.default_rng(3).normal(size=(3, 20, 8, 8)).astype(np.float32)
        result: list[tuple[np.ndarray, np.ndarray]] = []
        thread = threading.Thread(target=lambda: result.append(remote.evaluate_planes(planes)))
        thread.start()
        broker = CentralInferenceBroker(direct, arena, requests, responses, settings)
        while thread.is_alive():
            broker.service_once(0.01)
        thread.join(timeout=5.0)
        assert not thread.is_alive()
        expected = direct.evaluate_planes(planes)
        np.testing.assert_allclose(result[0][0], expected[0], rtol=0, atol=1e-6)
        np.testing.assert_allclose(result[0][1], expected[1], rtol=0, atol=1e-6)
    finally:
        arena.close()
        arena.unlink()


def test_remote_legal_matches_direct_evaluator() -> None:
    torch.manual_seed(7)
    direct = NetEvaluator(ChessNet(NetConfig(blocks=1, filters=8, value_bins=11)), device="cpu")
    arena, requests, responses, settings = _broker_parts(workers=1)
    try:
        remote = RemoteEvaluator(arena, 0, requests, responses[0], settings, run_id=11)
        planes = np.random.default_rng(4).normal(size=(2, 20, 8, 8)).astype(np.float32)
        indices = np.asarray([0, 13, 7, 9], dtype=np.int32)
        offsets = np.asarray([0, 2, 4], dtype=np.int32)
        result: list[tuple[np.ndarray, np.ndarray]] = []
        thread = threading.Thread(
            target=lambda: result.append(remote.evaluate_legal(planes, indices, offsets)),
        )
        thread.start()
        broker = CentralInferenceBroker(direct, arena, requests, responses, settings)
        while thread.is_alive():
            broker.service_once(0.01)
        thread.join(timeout=5.0)
        assert not thread.is_alive()
        expected = direct.evaluate_legal(planes, indices, offsets)
        np.testing.assert_allclose(result[0][0], expected[0], rtol=0, atol=1e-6)
        np.testing.assert_allclose(result[0][1], expected[1], rtol=0, atol=1e-6)
    finally:
        arena.close()
        arena.unlink()


def test_remote_legal_batches_match_direct_evaluator() -> None:
    torch.manual_seed(7)
    direct = NetEvaluator(ChessNet(NetConfig(blocks=1, filters=8, value_bins=11)), device="cpu")
    arena, requests, responses, settings = _broker_parts(workers=2)
    try:
        remotes = [
            RemoteEvaluator(arena, worker_id, requests, responses[worker_id], settings, run_id=3)
            for worker_id in range(2)
        ]
        payloads = []
        for worker_id in range(2):
            planes = np.full((2, 20, 8, 8), worker_id + 1, dtype=np.float32)
            indices = np.asarray([0, 1, 2, 3], dtype=np.int32)
            offsets = np.asarray([0, 2, 4], dtype=np.int32)
            payloads.append((planes, indices, offsets))
        results: list[tuple[np.ndarray, np.ndarray] | None] = [None, None]
        threads = [
            threading.Thread(
                target=lambda i=i: results.__setitem__(
                    i, remotes[i].evaluate_legal(*payloads[i]),
                ),
            )
            for i in range(2)
        ]
        for thread in threads:
            thread.start()
        while requests.qsize() < 2:
            time.sleep(0.001)
        broker = CentralInferenceBroker(direct, arena, requests, responses, settings)
        while any(thread.is_alive() for thread in threads):
            broker.service_once(0.01)
        for thread in threads:
            thread.join()
        assert results[0] is not None and results[1] is not None
        for worker_id, (planes, indices, offsets) in enumerate(payloads):
            expected = direct.evaluate_legal(planes, indices, offsets)
            np.testing.assert_allclose(results[worker_id][0], expected[0], rtol=0, atol=1e-6)
            np.testing.assert_allclose(results[worker_id][1], expected[1], rtol=0, atol=1e-6)
    finally:
        arena.close()
        arena.unlink()

