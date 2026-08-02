"""Lightweight ResNet with policy and value heads (AlphaZero-style)."""

from __future__ import annotations

import time

import chess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import NetConfig
from .encoding import NUM_INPUT_PLANES, POLICY_SIZE, board_to_planes, fill_planes_batch
from .profile import ProfileCounters


class ResidualBlock(nn.Module):
    def __init__(self, filters: int):
        super().__init__()
        self.conv1 = nn.Conv2d(filters, filters, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class ChessNet(nn.Module):
    def __init__(self, cfg: NetConfig | None = None):
        super().__init__()
        cfg = cfg or NetConfig()
        f = cfg.filters
        self.value_bins = cfg.value_bins

        self.stem = nn.Sequential(
            nn.Conv2d(NUM_INPUT_PLANES, f, 3, padding=1, bias=False),
            nn.BatchNorm2d(f),
            nn.ReLU(inplace=True),
        )
        self.tower = nn.Sequential(*[ResidualBlock(f) for _ in range(cfg.blocks)])

        # Policy head.
        self.policy_conv = nn.Sequential(
            nn.Conv2d(f, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True)
        )
        self.policy_fc = nn.Linear(32 * 8 * 8, POLICY_SIZE)

        # Value head.
        self.value_conv = nn.Sequential(
            nn.Conv2d(f, 8, 1, bias=False), nn.BatchNorm2d(8), nn.ReLU(inplace=True)
        )
        self.value_fc1 = nn.Linear(8 * 8 * 8, 128)
        self.value_fc2 = nn.Linear(128, self.value_bins)
        self.register_buffer("value_support", torch.linspace(-1.0, 1.0, self.value_bins))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.tower(x)

        p = self.policy_conv(x).flatten(1)
        p = self.policy_fc(p)  # logits

        v = self.value_conv(x).flatten(1)
        v = F.relu(self.value_fc1(v))
        value_logits = self.value_fc2(v)
        return p, value_logits

    def value_from_logits(self, value_logits: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(value_logits.float(), dim=-1)
        return torch.sum(probs * self.value_support, dim=-1)


class CudaBatchExecutor:
    """Run fixed-size CUDA graph buckets without changing model semantics."""

    BUCKETS = (8, 16, 32, 64, 128, 160)

    def __init__(
        self,
        net: ChessNet,
        device: str,
        graph_mode: str = "auto",
        profile: ProfileCounters | None = None,
        graph_buckets: tuple[int, ...] = BUCKETS,
    ):
        if graph_mode not in {"auto", "on", "off"}:
            raise ValueError("graph_mode must be 'auto', 'on', or 'off'")
        if not graph_buckets or any(
            bucket <= 0 or (i and bucket <= graph_buckets[i - 1])
            for i, bucket in enumerate(graph_buckets)
        ):
            raise ValueError("graph_buckets must be strictly increasing positive integers")
        self.net = net
        self.device = torch.device(device)
        self.graph_mode = graph_mode
        self.graph_buckets = tuple(int(bucket) for bucket in graph_buckets)
        self.profile = profile
        self._cuda = self.device.type == "cuda" and torch.cuda.is_available()
        self._graphs: dict[int, tuple[torch.cuda.CUDAGraph, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._graph_weight_versions: dict[int, tuple[int, ...]] = {}
        self._unavailable_buckets: set[int] = set()
        self._fallback_count = 0
        self._lane = 0
        self._host_lanes: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._device_lanes: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        if self._cuda:
            self._h2d_stream = torch.cuda.Stream(device=self.device)
            self._compute_stream = torch.cuda.Stream(device=self.device)
            self._d2h_stream = torch.cuda.Stream(device=self.device)
            self._h2d_done = (torch.cuda.Event(), torch.cuda.Event())
            self._compute_done = (torch.cuda.Event(), torch.cuda.Event())
            self._d2h_done = (torch.cuda.Event(), torch.cuda.Event())
            self._h2d_index_host: torch.Tensor | None = None
            self._index_device: torch.Tensor | None = None
            self._d2h_gather_host: torch.Tensor | None = None
            self._d2h_value_host: torch.Tensor | None = None

    @property
    def graph_fallback_count(self) -> int:
        return self._fallback_count

    def _pinned_float_host(self, attr: str, shape: tuple[int, ...]) -> torch.Tensor:
        """Grow-only pinned float32 buffer viewed as ``shape``."""
        need = 1
        for dim in shape:
            need *= int(dim)
        buf: torch.Tensor | None = getattr(self, attr)
        if buf is None or buf.numel() < need:
            buf = torch.empty(max(need, 1), dtype=torch.float32, pin_memory=True)
            setattr(self, attr, buf)
        return buf[:need].view(shape)

    def _ensure_index_buffers(self, n_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Pinned host + device int64 buffers for legal gather indices."""
        cap = max(int(n_idx), 1)
        if self._h2d_index_host is None or self._h2d_index_host.numel() < cap:
            self._h2d_index_host = torch.empty(cap, dtype=torch.int64, pin_memory=True)
        if self._index_device is None or self._index_device.numel() < cap:
            self._index_device = torch.empty(cap, dtype=torch.int64, device=self.device)
        return self._h2d_index_host[:n_idx], self._index_device[:n_idx]

    @torch.inference_mode()
    def legal_gather_to_numpy(
        self,
        logits: torch.Tensor,
        values: torch.Tensor,
        linear_indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Gather legal logits on-device, then async pinned D2H on ``_d2h_stream``.

        Synchronizes the D2H stream before returning so callers observe completed
        host arrays (numerical parity with sync ``.cpu().numpy()``).
        """
        n_idx = int(linear_indices.shape[0])
        current = torch.cuda.current_stream(self.device)
        if n_idx:
            idx_host, idx_dev = self._ensure_index_buffers(n_idx)
            np.copyto(idx_host.numpy(), np.ascontiguousarray(linear_indices))
            h2d_done = self._h2d_done[0]
            with torch.cuda.stream(self._h2d_stream):
                idx_dev.copy_(idx_host, non_blocking=True)
                h2d_done.record(self._h2d_stream)
            current.wait_event(h2d_done)
            gathered = torch.take(logits, idx_dev).float().contiguous()
        else:
            gathered = torch.empty(0, dtype=torch.float32, device=self.device)

        value_f = values.float().contiguous()
        g_host = self._pinned_float_host("_d2h_gather_host", tuple(gathered.shape))
        v_host = self._pinned_float_host("_d2h_value_host", tuple(value_f.shape))
        d2h_done = self._d2h_done[0]
        with torch.cuda.stream(self._d2h_stream):
            self._d2h_stream.wait_stream(current)
            if n_idx:
                g_host.copy_(gathered, non_blocking=True)
            v_host.copy_(value_f, non_blocking=True)
            d2h_done.record(self._d2h_stream)
        d2h_done.synchronize()
        # Copy out: reusable pinned buffers are overwritten on the next call.
        return g_host.numpy().copy(), v_host.numpy().copy()

    def invalidate_graphs(self) -> None:
        """Discard captured graphs after model weights or architecture change."""
        self._graphs.clear()
        self._graph_weight_versions.clear()
        self._unavailable_buckets.clear()

    def _weight_versions(self) -> tuple[int, ...]:
        return tuple(parameter._version for parameter in self.net.parameters())

    def _bucket_for(self, n: int) -> int | None:
        return next((bucket for bucket in self.graph_buckets if n <= bucket), None)

    def _ensure_lanes(self, bucket: int) -> None:
        if bucket in self._host_lanes:
            return
        shape = (bucket, NUM_INPUT_PLANES, 8, 8)
        self._host_lanes[bucket] = (
            torch.empty(shape, dtype=torch.float32, pin_memory=True),
            torch.empty(shape, dtype=torch.float32, pin_memory=True),
        )
        self._device_lanes[bucket] = (
            torch.empty(shape, dtype=torch.float32, device=self.device),
            torch.empty(shape, dtype=torch.float32, device=self.device),
        )

    def _capture_bucket(
        self, bucket: int
    ) -> tuple[torch.cuda.CUDAGraph, torch.Tensor, torch.Tensor, torch.Tensor]:
        static_input = torch.empty(
            (bucket, NUM_INPUT_PLANES, 8, 8), dtype=torch.float32, device=self.device
        )
        # Warm up allocations outside capture so the graph only contains inference.
        with torch.cuda.stream(self._compute_stream):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                self.net(static_input)
        torch.cuda.current_stream(self.device).wait_stream(self._compute_stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=self._compute_stream):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits, value_logits = self.net(static_input)
            values = self.net.value_from_logits(value_logits)
        return graph, static_input, logits, values

    def _eager(self, planes: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(planes).to(self.device)
        if self._cuda:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits, value_logits = self.net(x)
        else:
            logits, value_logits = self.net(x)
        return logits, self.net.value_from_logits(value_logits)

    @torch.inference_mode()
    def forward(self, planes: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """Return dense policy logits and scalar values for pre-encoded planes."""
        n = int(planes.shape[0])
        if not self._cuda or self.graph_mode == "off":
            return self._eager(planes)
        bucket = self._bucket_for(n)
        if bucket is None or bucket in self._unavailable_buckets:
            return self._eager(planes)

        self._ensure_lanes(bucket)
        lane = self._lane
        self._lane = 1 - lane
        host = self._host_lanes[bucket][lane]
        device_input = self._device_lanes[bucket][lane]
        # Write only the live slice into pinned memory; zero pad for full-bucket graphs.
        np.copyto(host.numpy()[:n], planes)
        if n < bucket:
            host[n:].zero_()
        h2d_done = self._h2d_done[lane]
        compute_done = self._compute_done[lane]
        with torch.cuda.stream(self._h2d_stream):
            device_input.copy_(host, non_blocking=True)
            h2d_done.record(self._h2d_stream)

        try:
            captured = self._graphs.get(bucket)
            if (
                captured is not None
                and self._graph_weight_versions.get(bucket) != self._weight_versions()
            ):
                self.invalidate_graphs()
                captured = None
            if captured is None:
                captured = self._capture_bucket(bucket)
                self._graphs[bucket] = captured
                self._graph_weight_versions[bucket] = self._weight_versions()
            graph, static_input, logits, values = captured
            with torch.cuda.stream(self._compute_stream):
                self._compute_stream.wait_event(h2d_done)
                static_input.copy_(device_input)
                graph.replay()
                compute_done.record(self._compute_stream)
            torch.cuda.current_stream(self.device).wait_event(compute_done)
            return logits[:n], values[:n]
        except (RuntimeError, torch.cuda.OutOfMemoryError):
            self._graphs.pop(bucket, None)
            self._unavailable_buckets.add(bucket)
            self._fallback_count += 1
            return self._eager(planes)


class NetEvaluator:
    """Wraps a ChessNet for single-position inference used by MCTS."""

    def __init__(
        self,
        net: ChessNet,
        device: str = "cpu",
        profile: ProfileCounters | None = None,
        graph_mode: str = "auto",
        graph_buckets: tuple[int, ...] = CudaBatchExecutor.BUCKETS,
    ):
        self.net = net.to(device).eval()
        self.device = device
        self.profile = profile
        self.graph_mode = graph_mode
        self.graph_buckets = tuple(graph_buckets)
        self._use_cuda_autocast = str(device).startswith("cuda")
        self._batch_cap = 0
        self._planes_buf: np.ndarray | None = None
        self._host_input: torch.Tensor | None = None
        self._profile_cuda_events: tuple[torch.cuda.Event, ...] | None = None
        self._cuda_executor: CudaBatchExecutor | None = None

    def configure_cuda_graphs(
        self, graph_mode: str, graph_buckets: tuple[int, ...]
    ) -> None:
        """Apply central-inference graph settings, invalidating stale captures."""
        buckets = tuple(graph_buckets)
        if self.graph_mode == graph_mode and self.graph_buckets == buckets:
            return
        if graph_mode not in {"auto", "on", "off"}:
            raise ValueError("graph_mode must be 'auto', 'on', or 'off'")
        if not buckets or any(
            bucket <= 0 or (i and bucket <= buckets[i - 1])
            for i, bucket in enumerate(buckets)
        ):
            raise ValueError("graph_buckets must be strictly increasing positive integers")
        self.graph_mode = graph_mode
        self.graph_buckets = buckets
        self._cuda_executor = None

    def _ensure_batch_buffers(self, batch_size: int) -> None:
        if batch_size <= self._batch_cap and self._planes_buf is not None:
            return
        new_cap = max(batch_size, self._batch_cap, 128)
        self._batch_cap = new_cap
        self._planes_buf = np.zeros(
            (new_cap, NUM_INPUT_PLANES, 8, 8), dtype=np.float32,
        )
        if self._use_cuda_autocast:
            self._host_input = torch.empty(
                (new_cap, NUM_INPUT_PLANES, 8, 8),
                dtype=torch.float32,
                pin_memory=True,
            )
        else:
            self._host_input = None

    @torch.inference_mode()
    def evaluate(self, board: chess.Board) -> tuple[np.ndarray, float]:
        """Return (policy_logits over POLICY_SIZE, value in [-1, 1])."""
        x = torch.from_numpy(board_to_planes(board)).unsqueeze(0).to(self.device)
        if self._use_cuda_autocast:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits, value_logits = self.net(x)
        else:
            logits, value_logits = self.net(x)
        value = self.net.value_from_logits(value_logits)
        return logits[0].float().cpu().numpy(), float(value[0].float().cpu())

    @torch.inference_mode()
    def evaluate_batch(self, boards: list[chess.Board]) -> tuple[np.ndarray, np.ndarray]:
        n = len(boards)
        if n == 0:
            return np.zeros((0, POLICY_SIZE), dtype=np.float32), np.zeros(0, dtype=np.float32)
        self._ensure_batch_buffers(n)
        assert self._planes_buf is not None
        fill_planes_batch(boards, self._planes_buf[:n])
        if self._host_input is not None:
            self._host_input[:n].copy_(torch.from_numpy(self._planes_buf[:n]))
            x = self._host_input[:n].to(self.device, non_blocking=True)
        else:
            x = torch.from_numpy(self._planes_buf[:n]).to(self.device)
        if self._use_cuda_autocast:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits, value_logits = self.net(x)
        else:
            logits, value_logits = self.net(x)
        value = self.net.value_from_logits(value_logits)
        return logits.float().cpu().numpy(), value.float().cpu().numpy()

    def _validate_planes(self, planes: np.ndarray) -> tuple[np.ndarray, int]:
        if planes.ndim != 4 or planes.shape[1:] != (NUM_INPUT_PLANES, 8, 8):
            raise ValueError(
                f"planes must have shape (N, {NUM_INPUT_PLANES}, 8, 8), got {planes.shape}"
            )
        n = int(planes.shape[0])
        if planes.dtype != np.float32:
            planes = np.asarray(planes, dtype=np.float32)
        return np.ascontiguousarray(planes), n

    @torch.inference_mode()
    def _forward_policy_value(
        self, contiguous: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        """Run the net; return ``(logits, values, used_eager_cuda_events)``.

        Tensors remain on the inference device so callers can gather before D2H.
        """
        n = int(contiguous.shape[0])
        profile = self.profile
        if profile is not None:
            profile.add_count("network.calls")
            profile.add_count("network.positions", n)
            profile.add_bytes("network.input", contiguous.nbytes)

        cuda_events = (
            profile is not None
            and self._use_cuda_autocast
            and self.graph_mode == "off"
        )
        h2d_start = h2d_end = forward_start = forward_end = None
        if cuda_events:
            if self._profile_cuda_events is None:
                self._profile_cuda_events = tuple(
                    torch.cuda.Event(enable_timing=True) for _ in range(4)
                )
            h2d_start, h2d_end, forward_start, forward_end = self._profile_cuda_events
        forward_started = time.perf_counter() if profile is not None else 0.0
        if self._use_cuda_autocast and self.graph_mode != "off":
            if self._cuda_executor is None:
                self._cuda_executor = CudaBatchExecutor(
                    self.net,
                    self.device,
                    graph_mode=self.graph_mode,
                    profile=profile,
                    graph_buckets=self.graph_buckets,
                )
            logits, value = self._cuda_executor.forward(contiguous)
        elif self._use_cuda_autocast:
            assert h2d_start is not None and h2d_end is not None
            assert forward_start is not None and forward_end is not None
            h2d_start.record()
            x = torch.from_numpy(contiguous).to(self.device)
            h2d_end.record()
            forward_start.record()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits, value_logits = self.net(x)
            value = self.net.value_from_logits(value_logits)
            forward_end.record()
        else:
            x = torch.from_numpy(contiguous).to(self.device)
            logits, value_logits = self.net(x)
            value = self.net.value_from_logits(value_logits)
        if profile is not None:
            profile.add_seconds("network.forward_host", time.perf_counter() - forward_started)
        if cuda_events:
            profile.add_seconds("network.h2d_cuda", h2d_start.elapsed_time(h2d_end) / 1000.0)
            profile.add_seconds(
                "network.forward_cuda", forward_start.elapsed_time(forward_end) / 1000.0
            )
        return logits, value, cuda_events

    @torch.inference_mode()
    def evaluate_planes(self, planes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate pre-encoded planes.

        Args:
            planes: float32 array of shape ``(N, 20, 8, 8)``.

        Returns:
            ``(logits, values)`` where ``logits`` is ``(N, 4672)`` and
            ``values`` is ``(N,)`` in ``[-1, 1]``.
        """
        contiguous, n = self._validate_planes(planes)
        if n == 0:
            return np.zeros((0, POLICY_SIZE), dtype=np.float32), np.zeros(0, dtype=np.float32)
        profile = self.profile
        assembly_started = time.perf_counter() if profile is not None else 0.0
        if profile is not None:
            profile.add_seconds("network.input_staging", time.perf_counter() - assembly_started)

        logits, value, _cuda_events = self._forward_policy_value(contiguous)

        policy_started = time.perf_counter() if profile is not None else 0.0
        policy = logits.float().cpu().numpy()
        if profile is not None:
            profile.add_seconds(
                "network.policy_d2h_and_sync_host",
                time.perf_counter() - policy_started,
            )
            profile.add_bytes("network.policy_d2h", policy.nbytes)
        value_started = time.perf_counter() if profile is not None else 0.0
        values = value.float().cpu().numpy()
        if profile is not None:
            profile.add_seconds("network.value_d2h_host", time.perf_counter() - value_started)
            profile.add_bytes("network.value_d2h", values.nbytes)
        return policy, values

    @torch.inference_mode()
    def evaluate_legal(
        self,
        planes: np.ndarray,
        legal_indices: np.ndarray,
        legal_offsets: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate planes and return policy logits packed in CSR legal-move order.

        Gathers legal logits on-device before D2H so the full ``(N, 4672)`` policy
        is not transferred when only legal moves are needed.
        """
        contiguous, n = self._validate_planes(planes)
        indices = np.asarray(legal_indices)
        offsets = np.asarray(legal_offsets)
        if not np.issubdtype(indices.dtype, np.integer):
            raise ValueError("legal_indices must be an integer array")
        if not np.issubdtype(offsets.dtype, np.integer):
            raise ValueError("legal_offsets must be an integer array")
        if offsets.shape != (n + 1,):
            raise ValueError(f"legal_offsets must have shape ({n + 1},), got {offsets.shape}")
        if int(offsets[0]) != 0 or int(offsets[-1]) != len(indices):
            raise ValueError("legal_offsets must span legal_indices")
        if np.any(offsets[1:] < offsets[:-1]):
            raise ValueError("legal_offsets must be nondecreasing")
        if len(indices) and (np.any(indices < 0) or np.any(indices >= POLICY_SIZE)):
            raise ValueError("legal_indices contain an out-of-range policy index")
        if n == 0:
            return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)

        profile = self.profile
        assembly_started = time.perf_counter() if profile is not None else 0.0
        if profile is not None:
            profile.add_seconds("network.input_staging", time.perf_counter() - assembly_started)

        logits, value, _cuda_events = self._forward_policy_value(contiguous)

        started = time.perf_counter() if profile is not None else 0.0
        counts = np.diff(offsets.astype(np.int64, copy=False))
        row_ids = np.repeat(np.arange(n, dtype=np.int64), counts)
        linear_indices = row_ids * POLICY_SIZE + indices.astype(np.int64, copy=False)
        executor = self._cuda_executor
        if executor is not None and executor._cuda and logits.is_cuda:
            gathered, values = executor.legal_gather_to_numpy(
                logits, value, linear_indices
            )
        else:
            device = logits.device
            gather_t = torch.from_numpy(np.ascontiguousarray(linear_indices)).to(device)
            gathered_t = torch.take(logits, gather_t).float()
            gathered = gathered_t.cpu().numpy()
            values = value.float().cpu().numpy()
        if profile is not None:
            profile.add_seconds("network.legal_gather", time.perf_counter() - started)
            profile.add_count("network.legal_gather_calls")
            profile.add_bytes("network.legal_gather", gathered.nbytes)
            profile.add_bytes("network.value_d2h", values.nbytes)
        return gathered, values
