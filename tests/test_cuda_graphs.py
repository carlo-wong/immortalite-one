"""CUDA graph batching must preserve eager inference results."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from engine.config import NetConfig
from engine.encoding import POLICY_SIZE
from engine.network import ChessNet, CudaBatchExecutor, NetEvaluator


def _net() -> ChessNet:
    torch.manual_seed(20260724)
    return ChessNet(NetConfig(blocks=1, filters=8, value_bins=51)).eval()


def _planes(n: int) -> np.ndarray:
    return np.random.default_rng(n).standard_normal((n, 20, 8, 8)).astype(np.float32)


def test_cpu_executor_runs_eagerly_without_graphs() -> None:
    executor = CudaBatchExecutor(_net(), "cpu", graph_mode="on")
    logits, values = executor.forward(_planes(3))

    assert logits.shape == (3, POLICY_SIZE)
    assert values.shape == (3,)
    assert executor.graph_fallback_count == 0
    assert not executor._cuda
    assert executor._graphs == {}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("n", [1, 7, 8, 9, 16, 17, 64])
def test_graph_bucket_padding_matches_eager(n: int) -> None:
    planes = _planes(n)
    eager = NetEvaluator(_net(), "cuda", graph_mode="off")
    graph = NetEvaluator(_net(), "cuda", graph_mode="on")
    graph.net.load_state_dict(eager.net.state_dict())

    eager_logits, eager_values = eager.evaluate_planes(planes)
    graph_logits, graph_values = graph.evaluate_planes(planes)

    assert graph_logits.shape == (n, POLICY_SIZE)
    assert graph_values.shape == (n,)
    np.testing.assert_allclose(graph_logits, eager_logits, rtol=0, atol=1e-3)
    np.testing.assert_allclose(graph_values, eager_values, rtol=0, atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_invalidated_graphs_run_after_weight_reload() -> None:
    evaluator = NetEvaluator(_net(), "cuda", graph_mode="on")
    planes = _planes(8)
    before_logits, before_values = evaluator.evaluate_planes(planes)
    assert evaluator._cuda_executor is not None

    torch.manual_seed(20260725)
    reloaded = ChessNet(NetConfig(blocks=1, filters=8, value_bins=51)).eval().to("cuda")
    evaluator.net.load_state_dict(reloaded.state_dict())
    evaluator._cuda_executor.invalidate_graphs()
    logits, values = evaluator.evaluate_planes(planes)

    eager = NetEvaluator(reloaded, "cuda", graph_mode="off")
    eager_logits, eager_values = eager.evaluate_planes(planes)
    assert logits.shape == (8, POLICY_SIZE)
    assert values.shape == (8,)
    np.testing.assert_allclose(logits, eager_logits, rtol=0, atol=1e-3)
    np.testing.assert_allclose(values, eager_values, rtol=0, atol=1e-3)
    # Reloaded weights must change outputs so this is not a vacuous shape-only check.
    assert not np.allclose(logits, before_logits, rtol=0, atol=1e-3) or not np.allclose(
        values, before_values, rtol=0, atol=1e-3
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_capture_failure_falls_back_to_eager(monkeypatch: pytest.MonkeyPatch) -> None:
    executor = CudaBatchExecutor(_net().to("cuda"), "cuda", graph_mode="on")

    def fail_capture(bucket: int) -> object:
        raise RuntimeError(f"forced capture failure for bucket {bucket}")

    monkeypatch.setattr(executor, "_capture_bucket", fail_capture)
    # Capture must be attempted (not skipped) so the except-path fallback runs.
    assert executor._bucket_for(3) == 8
    logits, values = executor.forward(_planes(3))

    assert logits.shape == (3, POLICY_SIZE)
    assert values.shape == (3,)
    assert executor.graph_fallback_count == 1
    assert 8 in executor._unavailable_buckets
    # Second call must take the unavailable-bucket eager path, not re-capture.
    logits2, values2 = executor.forward(_planes(3))
    assert logits2.shape == (3, POLICY_SIZE)
    assert values2.shape == (3,)
    assert executor.graph_fallback_count == 1
