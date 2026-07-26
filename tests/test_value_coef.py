"""value_coef scales value_loss in the train objective."""

from __future__ import annotations

import numpy as np
import torch

from engine.config import Config
from engine.encoding import POLICY_SIZE
from engine.network import ChessNet
from engine.selfplay import Sample
from engine.train import train_step


def _batch(n: int = 4) -> list[Sample]:
    rng = np.random.default_rng(0)
    out: list[Sample] = []
    for _ in range(n):
        policy = rng.random(POLICY_SIZE).astype(np.float32)
        policy /= policy.sum()
        out.append(
            Sample(
                planes=rng.random((20, 8, 8)).astype(np.float32),
                policy=policy,
                player=True,
                value=float(rng.uniform(-1, 1)),
            )
        )
    return out


def test_value_coef_increases_grad_norm_vs_equal_weight() -> None:
    cfg = Config()
    cfg.net.blocks = 1
    cfg.net.filters = 8
    torch.manual_seed(1)
    net_a = ChessNet(cfg.net)
    net_b = ChessNet(cfg.net)
    net_b.load_state_dict(net_a.state_dict())
    opt_a = torch.optim.Adam(net_a.parameters(), lr=1e-3)
    opt_b = torch.optim.Adam(net_b.parameters(), lr=1e-3)
    batch = _batch()
    g_eq = train_step(net_a, opt_a, batch, "cpu", value_coef=1.0)["grad_norm"]
    g_up = train_step(net_b, opt_b, batch, "cpu", value_coef=1.5)["grad_norm"]
    assert g_up > g_eq
