"""WDL aux-head training path: masked CE, shards, resume guards, metrics keys."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from engine.config import Config, NetConfig
from engine.encoding import POLICY_SIZE
from engine.network import ChessNet
from engine.selfplay import WDL_DRAW, WDL_LOSS, WDL_UNLABELED, WDL_WIN, Sample
from engine.train import (
    _checkpoint_wdl_head,
    _module_has_wdl_head,
    _net_cfg_from_module,
    _require_wdl_upgrade_reset,
    train_step,
)


def _batch(
    n: int = 4,
    *,
    wdl: list[int] | None = None,
    seed: int = 0,
) -> list[Sample]:
    rng = np.random.default_rng(seed)
    labels = wdl if wdl is not None else [WDL_UNLABELED] * n
    assert len(labels) == n
    out: list[Sample] = []
    for i in range(n):
        policy = rng.random(POLICY_SIZE).astype(np.float32)
        policy /= policy.sum()
        out.append(
            Sample(
                planes=rng.random((20, 8, 8)).astype(np.float32),
                policy=policy,
                player=True,
                value=float(rng.uniform(-1, 1)),
                wdl=int(labels[i]),
            )
        )
    return out


def _tiny_net(*, wdl_head: bool) -> ChessNet:
    return ChessNet(NetConfig(blocks=1, filters=8, value_bins=11, wdl_head=wdl_head))


def test_net_cfg_from_module_preserves_wdl_head() -> None:
    disabled = _tiny_net(wdl_head=False)
    enabled = _tiny_net(wdl_head=True)
    assert _net_cfg_from_module(disabled).wdl_head is False
    assert _net_cfg_from_module(enabled).wdl_head is True
    assert _module_has_wdl_head(enabled) is True


class _FakeCompiledModule(torch.nn.Module):
    """Stand-in for torch.compile OptimizedModule: __call__ vs forward_train."""

    def __init__(self, inner: ChessNet) -> None:
        super().__init__()
        self._orig_mod = inner
        self.call_count = 0
        self.forward_train_count = 0

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.call_count += 1
        return self._orig_mod(x)

    def forward_train(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.forward_train_count += 1
        return self._orig_mod.forward_train(x)

    def parameters(self, recurse: bool = True):  # type: ignore[override]
        return self._orig_mod.parameters(recurse=recurse)


def test_train_step_disabled_keeps_legacy_keys_and_zero_wdl_metrics() -> None:
    net = _tiny_net(wdl_head=False)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    result = train_step(net, opt, _batch(), "cpu", value_coef=1.0, wdl_coef=0.0)
    for key in (
        "policy_loss",
        "value_loss",
        "policy_entropy",
        "value_sign_acc",
        "policy_top1_agree",
        "grad_norm",
        "wdl_loss",
        "wdl_accuracy",
        "wdl_labeled_fraction",
    ):
        assert key in result
        assert math.isfinite(result[key])
    assert result["wdl_loss"] == 0.0
    assert result["wdl_accuracy"] == 0.0
    assert result["wdl_labeled_fraction"] == 0.0


def test_train_step_disabled_uses_call_not_forward_train() -> None:
    """WDL-off must hit compiled __call__/forward, not forward_train."""
    inner = _tiny_net(wdl_head=False)
    net = _FakeCompiledModule(inner)
    opt = torch.optim.Adam(inner.parameters(), lr=1e-3)
    train_step(net, opt, _batch(), "cpu", wdl_coef=0.0)
    assert net.call_count == 1
    assert net.forward_train_count == 0


def test_train_step_wdl_active_uses_forward_train() -> None:
    inner = _tiny_net(wdl_head=True)
    net = _FakeCompiledModule(inner)
    opt = torch.optim.Adam(inner.parameters(), lr=1e-3)
    train_step(
        net,
        opt,
        _batch(4, wdl=[WDL_WIN, WDL_DRAW, WDL_LOSS, WDL_WIN], seed=1),
        "cpu",
        wdl_coef=1.0,
    )
    assert net.forward_train_count == 1
    assert net.call_count == 0


def test_train_step_wdl_head_idle_at_zero_coef_uses_call() -> None:
    """wdl_head=True + wdl_coef=0 must keep compiled __call__/forward idle of aux."""
    inner = _tiny_net(wdl_head=True)
    net = _FakeCompiledModule(inner)
    opt = torch.optim.Adam(inner.parameters(), lr=1e-3)
    result = train_step(
        net,
        opt,
        _batch(4, wdl=[WDL_WIN, WDL_DRAW, WDL_LOSS, WDL_WIN], seed=2),
        "cpu",
        wdl_coef=0.0,
    )
    assert net.call_count == 1
    assert net.forward_train_count == 0
    assert result["wdl_loss"] == 0.0
    assert result["wdl_accuracy"] == 0.0
    assert result["wdl_labeled_fraction"] == 0.0


def test_wdl_coef_without_head_fails_clearly() -> None:
    net = _tiny_net(wdl_head=False)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    with pytest.raises(ValueError, match="wdl_coef > 0 requires"):
        train_step(net, opt, _batch(), "cpu", wdl_coef=0.5)


def test_masked_ce_ignores_unlabeled_rows() -> None:
    torch.manual_seed(11)
    net = _tiny_net(wdl_head=True)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    mixed = _batch(
        4,
        wdl=[WDL_WIN, WDL_UNLABELED, WDL_DRAW, WDL_UNLABELED],
        seed=3,
    )
    result = train_step(net, opt, mixed, "cpu", wdl_coef=1.0)
    assert result["wdl_labeled_fraction"] == pytest.approx(0.5)
    assert math.isfinite(result["wdl_loss"])
    assert 0.0 <= result["wdl_accuracy"] <= 1.0


def test_all_unlabeled_batch_has_zero_wdl_contribution() -> None:
    torch.manual_seed(22)
    net = _tiny_net(wdl_head=True)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    batch = _batch(4, wdl=[WDL_UNLABELED] * 4, seed=5)
    w0 = net.wdl_fc.weight.detach().clone()
    b0 = net.wdl_fc.bias.detach().clone()
    result = train_step(net, opt, batch, "cpu", wdl_coef=1.0)
    assert result["wdl_loss"] == 0.0
    assert result["wdl_accuracy"] == 0.0
    assert result["wdl_labeled_fraction"] == 0.0
    assert not math.isnan(result["wdl_loss"])
    # Zero CE term is detached from WDL logits, so the head params stay put.
    assert torch.equal(net.wdl_fc.weight.detach(), w0)
    assert torch.equal(net.wdl_fc.bias.detach(), b0)


def test_labeled_batch_updates_wdl_head() -> None:
    torch.manual_seed(33)
    net = _tiny_net(wdl_head=True)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    batch = _batch(8, wdl=[WDL_WIN, WDL_DRAW, WDL_LOSS, WDL_WIN] * 2, seed=9)
    w0 = net.wdl_fc.weight.detach().clone()
    result = train_step(net, opt, batch, "cpu", wdl_coef=1.0)
    assert result["wdl_labeled_fraction"] == pytest.approx(1.0)
    assert result["wdl_loss"] > 0.0
    assert not torch.equal(net.wdl_fc.weight.detach(), w0)


def test_legacy_upgrade_requires_reset_optimizer() -> None:
    with pytest.raises(ValueError, match="--reset-optimizer"):
        _require_wdl_upgrade_reset(
            cli_wdl_head=True,
            checkpoint_wdl_head=False,
            reset_optimizer=False,
            resuming=True,
        )
    _require_wdl_upgrade_reset(
        cli_wdl_head=True,
        checkpoint_wdl_head=False,
        reset_optimizer=True,
        resuming=True,
    )
    # Resuming a stamped WDL net without CLI flag is fine.
    _require_wdl_upgrade_reset(
        cli_wdl_head=False,
        checkpoint_wdl_head=True,
        reset_optimizer=False,
        resuming=True,
    )


def test_raw_model_optimizer_checkpoint_is_legacy_for_wdl_guard() -> None:
    """No architecture stamp means legacy for the upgrade guard."""
    net = _tiny_net(wdl_head=False)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    raw_state = {"model": net.state_dict(), "optimizer": opt.state_dict()}
    assert "net" not in raw_state
    assert _checkpoint_wdl_head(raw_state) is False
    with pytest.raises(ValueError, match="--reset-optimizer"):
        _require_wdl_upgrade_reset(
            cli_wdl_head=True,
            checkpoint_wdl_head=_checkpoint_wdl_head(raw_state),
            reset_optimizer=False,
            resuming=True,
        )
    _require_wdl_upgrade_reset(
        cli_wdl_head=True,
        checkpoint_wdl_head=_checkpoint_wdl_head(raw_state),
        reset_optimizer=True,
        resuming=True,
    )


def test_config_defaults_keep_wdl_training_off() -> None:
    cfg = Config()
    assert cfg.net.wdl_head is False
    assert cfg.train.wdl_coef == 0.0


def test_colab_and_lightning_recipes_leave_wdl_disabled() -> None:
    """Current cloud recipes must not enable the optional aux head."""
    root = Path(__file__).resolve().parents[1]
    lightning = (root / "lightning-ai" / "run_train.py").read_text(encoding="utf-8")
    colab = (root / "colab" / "train.ipynb").read_text(encoding="utf-8")
    for text in (lightning, colab):
        assert "--wdl-head" not in text
        assert "--wdl-coef" not in text
    assert "wdl_head" not in lightning
    assert "wdl_coef" not in lightning
    # Notebook may mention Syzygy/game WDL prose; must not set train knobs.
    assert "'wdl_head'" not in colab
    assert '"wdl_head"' not in colab
    assert "wdl_coef" not in colab
