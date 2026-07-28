"""Single-owner batched inference for multiprocessing self-play.

Workers only exchange small request descriptors; all tensors live in named shared
memory segments so spawned workers never create CUDA contexts.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Any, Literal, Sequence

import numpy as np

from .encoding import NUM_INPUT_PLANES, POLICY_SIZE


@dataclass(frozen=True)
class InferenceSettings:
    enabled: bool = True
    max_batch_size: int = 128
    max_wait_us: int = 250
    queue_depth: int = 1
    pinned_buffers: int = 2
    use_cuda_streams: bool = True
    cuda_graphs: Literal["auto", "on", "off"] = "auto"
    graph_buckets: tuple[int, ...] = (8, 16, 32, 64, 128)
    response_timeout_s: float = 120.0


@dataclass(frozen=True)
class InferenceRequest:
    run_id: int
    worker_id: int
    request_id: int
    count: int
    legal: bool = False
    legal_count: int = 0


@dataclass(frozen=True)
class InferenceResponse:
    run_id: int
    request_id: int
    count: int
    error: str | None = None
    legal: bool = False


@dataclass(frozen=True)
class _Segment:
    name: str
    shape: tuple[int, ...]
    dtype: str


class SharedInferenceArena:
    """Per-worker fixed shared slots for input and inference output."""

    def __init__(self, capacities: Sequence[int], segments: list[dict[str, _Segment]]) -> None:
        self.capacities = tuple(int(cap) for cap in capacities)
        self._segments = segments
        self._handles: list[dict[str, shared_memory.SharedMemory]] = []
        self._arrays: list[dict[str, np.ndarray]] = []
        self._attach()

    @classmethod
    def create(cls, worker_capacities: Sequence[int]) -> SharedInferenceArena:
        if not worker_capacities or any(int(cap) <= 0 for cap in worker_capacities):
            raise ValueError("worker_capacities must contain positive capacities")
        segment_specs: list[dict[str, _Segment]] = []
        handles: list[dict[str, shared_memory.SharedMemory]] = []
        arrays: list[dict[str, np.ndarray]] = []
        try:
            for cap_raw in worker_capacities:
                cap = int(cap_raw)
                shapes = {
                    "planes": (cap, NUM_INPUT_PLANES, 8, 8),
                    "policy": (cap, POLICY_SIZE),
                    "values": (cap,),
                    "legal_indices": (cap * POLICY_SIZE,),
                    "legal_logits": (cap * POLICY_SIZE,),
                    "legal_offsets": (cap + 1,),
                }
                dtypes = {
                    "planes": np.float32, "policy": np.float32, "values": np.float32,
                    "legal_indices": np.int32, "legal_logits": np.float32,
                    "legal_offsets": np.int32,
                }
                specs: dict[str, _Segment] = {}
                worker_handles: dict[str, shared_memory.SharedMemory] = {}
                worker_arrays: dict[str, np.ndarray] = {}
                for key, shape in shapes.items():
                    dtype = np.dtype(dtypes[key])
                    shm = shared_memory.SharedMemory(create=True, size=int(np.prod(shape)) * dtype.itemsize)
                    specs[key] = _Segment(shm.name, shape, dtype.str)
                    worker_handles[key] = shm
                    worker_arrays[key] = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
                segment_specs.append(specs)
                handles.append(worker_handles)
                arrays.append(worker_arrays)
        except BaseException:
            for worker_handles in handles:
                for shm in worker_handles.values():
                    shm.close()
                    shm.unlink()
            raise
        arena = cls.__new__(cls)
        arena.capacities = tuple(int(cap) for cap in worker_capacities)
        arena._segments = segment_specs
        arena._handles = handles
        arena._arrays = arrays
        return arena

    def __getstate__(self) -> dict[str, Any]:
        return {"capacities": self.capacities, "segments": self._segments}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.capacities = state["capacities"]
        self._segments = state["segments"]
        self._handles = []
        self._arrays = []
        self._attach()

    def _attach(self) -> None:
        if self._handles:
            return
        for specs in self._segments:
            worker_handles: dict[str, shared_memory.SharedMemory] = {}
            worker_arrays: dict[str, np.ndarray] = {}
            for key, segment in specs.items():
                shm = shared_memory.SharedMemory(name=segment.name)
                worker_handles[key] = shm
                worker_arrays[key] = np.ndarray(
                    segment.shape, dtype=np.dtype(segment.dtype), buffer=shm.buf
                )
            self._handles.append(worker_handles)
            self._arrays.append(worker_arrays)

    def worker(self, worker_id: int) -> dict[str, np.ndarray]:
        return self._arrays[worker_id]

    def close(self) -> None:
        for worker_handles in self._handles:
            for shm in worker_handles.values():
                shm.close()
        self._handles = []
        self._arrays = []

    def unlink(self) -> None:
        for specs in self._segments:
            for segment in specs.values():
                try:
                    shared_memory.SharedMemory(name=segment.name).unlink()
                except FileNotFoundError:
                    pass


class RemoteEvaluator:
    """NetEvaluator-compatible facade used by CPU-only self-play workers."""

    def __init__(
        self, arena: SharedInferenceArena, worker_id: int, requests: Any,
        responses: Any, settings: InferenceSettings, run_id: int,
    ) -> None:
        self._arena = arena
        self._worker_id = worker_id
        self._requests = requests
        self._responses = responses
        self._settings = settings
        self._run_id = run_id
        self._next_request_id = 0
        self._lock = threading.Lock()
        self._pending: dict[int, InferenceResponse] = {}

    def evaluate_planes(self, planes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        planes = np.asarray(planes, dtype=np.float32)
        return self._submit(planes, None, None)

    def evaluate_legal(
        self, planes: np.ndarray, legal_indices: np.ndarray, legal_offsets: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        planes = np.asarray(planes, dtype=np.float32)
        indices = np.asarray(legal_indices, dtype=np.int32)
        offsets = np.asarray(legal_offsets, dtype=np.int32)
        return self._submit(planes, indices, offsets)

    def _submit(
        self, planes: np.ndarray, legal_indices: np.ndarray | None, legal_offsets: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if planes.ndim != 4 or planes.shape[1:] != (NUM_INPUT_PLANES, 8, 8):
            raise ValueError(f"planes must have shape (N, {NUM_INPUT_PLANES}, 8, 8)")
        n = int(planes.shape[0])
        if n > self._arena.capacities[self._worker_id]:
            raise ValueError("request exceeds this worker's shared slot capacity")
        if (legal_indices is None) != (legal_offsets is None):
            raise ValueError("legal_indices and legal_offsets must be supplied together")
        with self._lock:
            slot = self._arena.worker(self._worker_id)
            np.copyto(slot["planes"][:n], planes)
            legal = legal_indices is not None
            legal_count = 0
            if legal:
                assert legal_indices is not None and legal_offsets is not None
                if legal_offsets.shape != (n + 1,) or legal_offsets[0] != 0:
                    raise ValueError("legal_offsets must have shape (N + 1) and start at zero")
                legal_count = int(legal_indices.size)
                if legal_count > slot["legal_indices"].size or legal_offsets[-1] != legal_count:
                    raise ValueError("invalid legal CSR data for shared slot")
                slot["legal_indices"][:legal_count] = legal_indices
                slot["legal_offsets"][:n + 1] = legal_offsets
            request_id = self._next_request_id
            self._next_request_id += 1
            request = InferenceRequest(
                self._run_id, self._worker_id, request_id, n, legal, legal_count,
            )
            self._requests.put(request)
            response = self._wait(request_id)
            if response.error is not None:
                raise RuntimeError(response.error)
            values = slot["values"][:n].copy()
            if response.legal:
                return slot["legal_logits"][:legal_count].copy(), values
            return slot["policy"][:n].copy(), values

    def _wait(self, request_id: int) -> InferenceResponse:
        deadline = time.monotonic() + self._settings.response_timeout_s
        while True:
            cached = self._pending.pop(request_id, None)
            if cached is not None:
                return cached
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"inference request {request_id} timed out")
            try:
                response = self._responses.get(timeout=remaining)
            except queue.Empty as exc:
                raise TimeoutError(f"inference request {request_id} timed out") from exc
            if response.run_id != self._run_id:
                continue
            if response.request_id == request_id:
                return response
            self._pending[response.request_id] = response


class CentralInferenceBroker:
    """Owns the evaluator and is the sole process allowed to touch CUDA."""

    def __init__(
        self, evaluator: Any, arena: SharedInferenceArena, requests: Any,
        responses: Sequence[Any], settings: InferenceSettings,
    ) -> None:
        self.evaluator = evaluator
        self.arena = arena
        self.requests = requests
        self.responses = tuple(responses)
        self.settings = settings
        self._pending: dict[int, list[InferenceRequest]] = {
            worker_id: [] for worker_id in range(len(self.responses))
        }
        self._next_worker = 0
        self._aborted: str | None = None
        configure_graphs = getattr(evaluator, "configure_cuda_graphs", None)
        if callable(configure_graphs):
            configure_graphs(settings.cuda_graphs, settings.graph_buckets)

    def service_once(self, timeout_s: float = 0.0) -> int:
        self._fill(timeout_s)
        selected = self._select()
        if not selected:
            return 0
        try:
            self._execute(selected)
        except BaseException as exc:
            message = f"central inference failed: {exc}"
            for request in selected:
                self.responses[request.worker_id].put(
                    InferenceResponse(request.run_id, request.request_id, request.count, message)
                )
        return len(selected)

    def drain_until(self, done: Any, poll_s: float = 0.001) -> None:
        while not done():
            self.service_once(poll_s)
        while any(self._pending.values()):
            self.service_once(0.0)

    def abort(self, error: str = "central inference broker aborted") -> None:
        self._aborted = error
        while True:
            try:
                request = self.requests.get_nowait()
            except queue.Empty:
                break
            self._pending[request.worker_id].append(request)
        for pending in self._pending.values():
            for request in pending:
                self.responses[request.worker_id].put(
                    InferenceResponse(request.run_id, request.request_id, request.count, error)
                )
            pending.clear()

    def close(self) -> None:
        self.abort("central inference broker closed")

    def _fill(self, timeout_s: float) -> None:
        if self._aborted is not None:
            return
        deadline = time.monotonic() + max(0.0, timeout_s)
        wait = max(0.0, deadline - time.monotonic())
        try:
            request = self.requests.get(timeout=wait) if wait else self.requests.get_nowait()
            self._pending[request.worker_id].append(request)
        except queue.Empty:
            return
        max_wait = self.settings.max_wait_us / 1_000_000
        batch_deadline = time.monotonic() + max_wait
        while time.monotonic() < batch_deadline:
            try:
                request = self.requests.get_nowait()
            except queue.Empty:
                break
            self._pending[request.worker_id].append(request)

    def _select(self) -> list[InferenceRequest]:
        selected: list[InferenceRequest] = []
        total = 0
        legal: bool | None = None
        worker_count = len(self.responses)
        while total < self.settings.max_batch_size:
            found = False
            for offset in range(worker_count):
                worker_id = (self._next_worker + offset) % worker_count
                pending = self._pending[worker_id]
                if not pending:
                    continue
                request = pending[0]
                if legal is not None and request.legal != legal:
                    continue
                if request.count + total > self.settings.max_batch_size and selected:
                    continue
                pending.pop(0)
                selected.append(request)
                total += request.count
                legal = request.legal
                self._next_worker = (worker_id + 1) % worker_count
                found = True
                break
            if not found:
                break
        return selected

    def _execute(self, requests: list[InferenceRequest]) -> None:
        planes = np.concatenate(
            [self.arena.worker(req.worker_id)["planes"][:req.count] for req in requests], axis=0
        )
        legal = all(req.legal for req in requests)
        if legal:
            index_parts = [
                self.arena.worker(req.worker_id)["legal_indices"][:req.legal_count] for req in requests
            ]
            offsets = np.zeros(planes.shape[0] + 1, dtype=np.int32)
            cursor = 0
            index_cursor = 0
            for req in requests:
                local = self.arena.worker(req.worker_id)["legal_offsets"][:req.count + 1]
                offsets[cursor:cursor + req.count + 1] = local + index_cursor
                cursor += req.count
                index_cursor += req.legal_count
            indices = np.concatenate(index_parts) if index_parts else np.empty(0, dtype=np.int32)
            logits, values = self.evaluator.evaluate_legal(planes, indices, offsets)
        else:
            logits, values = self.evaluator.evaluate_planes(planes)
        row = 0
        legal_row = 0
        for req in requests:
            slot = self.arena.worker(req.worker_id)
            slot["values"][:req.count] = values[row:row + req.count]
            if legal:
                slot["legal_logits"][:req.legal_count] = logits[legal_row:legal_row + req.legal_count]
                legal_row += req.legal_count
            else:
                slot["policy"][:req.count] = logits[row:row + req.count]
            row += req.count
            self.responses[req.worker_id].put(
                InferenceResponse(req.run_id, req.request_id, req.count, legal=legal)
            )
