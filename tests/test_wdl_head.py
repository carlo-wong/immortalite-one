"""Optional WDL head: baseline compatibility and training-only API."""

from __future__ import annotations

import chess
import numpy as np
import torch

from engine.config import NetConfig, TrainConfig
from engine.encoding import ENCODING_VERSION, NUM_INPUT_PLANES, POLICY_SIZE
from engine.network import ChessNet, NetEvaluator


def _tiny_cfg(*, wdl_head: bool = False) -> NetConfig:
    return NetConfig(blocks=1, filters=8, value_bins=11, wdl_head=wdl_head)


def _planes(batch: int = 4, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(batch, NUM_INPUT_PLANES, 8, 8, generator=g)


def test_config_defaults_disable_wdl() -> None:
    assert NetConfig().wdl_head is False
    assert TrainConfig().wdl_coef == 0.0
    # Optional WDL must not bump the plane/policy encoding contract.
    assert ENCODING_VERSION == 2


def test_disabled_state_dict_and_forward_tuple_match_baseline() -> None:
    torch.manual_seed(1)
    baseline = ChessNet(NetConfig(blocks=1, filters=8, value_bins=11))
    torch.manual_seed(1)
    disabled = ChessNet(_tiny_cfg(wdl_head=False))

    assert set(disabled.state_dict()) == set(baseline.state_dict())
    for key, tensor in baseline.state_dict().items():
        assert disabled.state_dict()[key].shape == tensor.shape
    assert not any(name.startswith("wdl") for name, _ in disabled.named_parameters())

    x = _planes(3, seed=7)
    out_b = baseline(x)
    out_d = disabled(x)
    assert isinstance(out_b, tuple) and len(out_b) == 2
    assert isinstance(out_d, tuple) and len(out_d) == 2
    assert out_b[0].shape == out_d[0].shape == (3, POLICY_SIZE)
    assert out_b[1].shape == out_d[1].shape == (3, 11)

    train_out = disabled.forward_train(x)
    assert isinstance(train_out, tuple) and len(train_out) == 2


def test_enabled_wdl_shapes_and_inference_stays_two_output() -> None:
    net = ChessNet(_tiny_cfg(wdl_head=True)).eval()
    x = _planes(5, seed=3)

    p, v = net(x)
    assert p.shape == (5, POLICY_SIZE)
    assert v.shape == (5, 11)

    p_t, v_t, wdl = net.forward_train(x)
    assert p_t.shape == (5, POLICY_SIZE)
    assert v_t.shape == (5, 11)
    assert wdl.shape == (5, 3)
    assert any(name.startswith("wdl") for name, _ in net.named_parameters())


def test_eval_primary_heads_match_between_forward_and_forward_train() -> None:
    torch.manual_seed(42)
    net = ChessNet(_tiny_cfg(wdl_head=True)).eval()
    x = _planes(4, seed=11)

    p0, v0 = net(x)
    p1, v1, wdl = net.forward_train(x)
    assert torch.equal(p0, p1)
    assert torch.equal(v0, v1)
    assert wdl.shape == (4, 3)

    # Inference path must not require WDL module execution for its return value.
    torch.manual_seed(42)
    net2 = ChessNet(_tiny_cfg(wdl_head=True)).eval()
    p2, v2 = net2.forward(x)
    assert torch.equal(p0, p2)
    assert torch.equal(v0, v2)


def test_wdl_gradients_flow_into_shared_tower() -> None:
    net = ChessNet(_tiny_cfg(wdl_head=True)).train()
    x = _planes(2, seed=99)
    _p, _v, wdl = net.forward_train(x)
    loss = wdl.sum()
    loss.backward()

    stem_grad = next(net.stem.parameters()).grad
    tower_grad = next(net.tower.parameters()).grad
    value_fc1_grad = net.value_fc1.weight.grad
    assert stem_grad is not None and stem_grad.abs().sum() > 0
    assert tower_grad is not None and tower_grad.abs().sum() > 0
    assert value_fc1_grad is not None and value_fc1_grad.abs().sum() > 0
    # Primary value head params may also get grads via shared v representation.
    assert net.wdl_fc.weight.grad is not None
    assert net.wdl_fc.weight.grad.abs().sum() > 0


def test_net_evaluator_ignores_wdl_head() -> None:
    """Inference/MCTS evaluator must stay a (policy, value) contract with WDL on."""
    net = ChessNet(_tiny_cfg(wdl_head=True))
    ev = NetEvaluator(net, device="cpu", graph_mode="off")
    board = chess.Board()
    logits, value = ev.evaluate(board)
    assert logits.shape == (POLICY_SIZE,)
    assert isinstance(value, float)

    planes = np.zeros((2, NUM_INPUT_PLANES, 8, 8), dtype=np.float32)
    batch_logits, batch_values = ev.evaluate_planes(planes)
    assert batch_logits.shape == (2, POLICY_SIZE)
    assert batch_values.shape == (2,)
