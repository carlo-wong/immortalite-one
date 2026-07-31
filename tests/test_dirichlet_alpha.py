import argparse

import pytest

from engine.config import Config
from engine.selfplay import _mcts_cfg_dict
from engine.train import _positive_dirichlet_alpha


def test_dirichlet_alpha_accepts_positive_value() -> None:
    assert _positive_dirichlet_alpha("0.15") == pytest.approx(0.15)


@pytest.mark.parametrize("value", ["0", "-0.1"])
def test_dirichlet_alpha_rejects_nonpositive_value(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="must be > 0"):
        _positive_dirichlet_alpha(value)


def test_dirichlet_alpha_reaches_native_mcts_config() -> None:
    cfg = Config()
    cfg.mcts.dirichlet_alpha = 0.15

    assert _mcts_cfg_dict(cfg)["dirichlet_alpha"] == pytest.approx(0.15)
