"""Checkpoint / shard round-trips and value_target resume guards."""

from __future__ import annotations

from collections import deque

import numpy as np
import pytest
import torch

from engine.analyze import load_evaluator
from engine.config import Config
from engine.encoding import ENCODING_VERSION, POLICY_SIZE
from engine.network import ChessNet
from engine.selfplay import Sample
from engine.train import (
    _load_sample_shard,
    _require_value_target_compat,
    _save_sample_shard,
    _warm_replay_buffer,
    save_checkpoint,
)


def _tiny_net_and_optimizer() -> tuple[ChessNet, torch.optim.Adam, Config]:
    cfg = Config()
    cfg.net.blocks = 1
    cfg.net.filters = 4
    cfg.train.value_target = "root_q"
    net = ChessNet(cfg.net)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    return net, optimizer, cfg


def _fake_sample() -> Sample:
    return Sample(
        planes=np.zeros((20, 8, 8), dtype=np.float32),
        policy=np.zeros(POLICY_SIZE, dtype=np.float32),
        player=True,
        value=0.5,
    )


def test_load_evaluator_accepts_raw_current_state_dict(tmp_path) -> None:
    cfg = Config()
    cfg.net.blocks = 1
    cfg.net.filters = 4
    torch.manual_seed(20260801)
    expected = ChessNet(cfg.net).eval()
    path = tmp_path / "raw_state_dict.pt"
    torch.save(expected.state_dict(), path)

    evaluator = load_evaluator(str(path), cfg, device="cpu")
    actual = evaluator.net.eval()

    expected_state = expected.state_dict()
    actual_state = actual.state_dict()
    assert actual_state.keys() == expected_state.keys()
    for key in expected_state:
        torch.testing.assert_close(actual_state[key], expected_state[key], rtol=0, atol=0)

    inputs = torch.linspace(-1.0, 1.0, steps=2 * 20 * 8 * 8).reshape(2, 20, 8, 8)
    with torch.inference_mode():
        expected_outputs = expected(inputs)
        actual_outputs = actual(inputs)
    for actual_output, expected_output in zip(actual_outputs, expected_outputs):
        torch.testing.assert_close(actual_output, expected_output, rtol=0, atol=0)


def test_save_checkpoint_round_trip_with_optimizer(tmp_path) -> None:
    net, optimizer, cfg = _tiny_net_and_optimizer()
    x = torch.randn(1, 20, 8, 8)
    loss = net(x)[0].sum()
    loss.backward()
    optimizer.step()
    expected_opt_state = optimizer.state_dict()

    path = str(tmp_path / "ckpt.pt")
    save_checkpoint(net, cfg, path, iteration=7, optimizer=optimizer)

    state = torch.load(path, map_location="cpu")
    assert state["iteration"] == 7
    assert state["encoding_version"] == ENCODING_VERSION
    assert state["value_target"] == "root_q"
    assert "optimizer" in state

    net2 = ChessNet(cfg.net)
    net2.load_state_dict(state["model"])
    optimizer2 = torch.optim.Adam(net2.parameters(), lr=1e-3)
    optimizer2.load_state_dict(state["optimizer"])
    loaded_opt_state = optimizer2.state_dict()
    assert loaded_opt_state["param_groups"] == expected_opt_state["param_groups"]
    assert loaded_opt_state["state"].keys() == expected_opt_state["state"].keys()
    for key in loaded_opt_state["state"]:
        for field, tensor in loaded_opt_state["state"][key].items():
            assert torch.allclose(tensor, expected_opt_state["state"][key][field])


def test_save_checkpoint_without_optimizer_backward_compat(tmp_path) -> None:
    net, _, cfg = _tiny_net_and_optimizer()
    path = str(tmp_path / "legacy.pt")
    save_checkpoint(net, cfg, path, iteration=3)

    state = torch.load(path, map_location="cpu")
    assert "optimizer" not in state
    assert state["iteration"] == 3
    assert state["value_target"] == "root_q"

    net2 = ChessNet(cfg.net)
    net2.load_state_dict(state["model"])
    optimizer = torch.optim.Adam(net2.parameters(), lr=1e-3)
    assert optimizer.state_dict()["state"] == {}


def test_warm_replay_buffer_uses_newest_shards_only(tmp_path) -> None:
    ckpt_dir = str(tmp_path)
    for iteration in range(5):
        sample = _fake_sample()
        sample.value = float(iteration)
        sample.source_iter = iteration
        _save_sample_shard(ckpt_dir, iteration, [sample], value_target="root_q")

    buffer: deque[Sample] = deque(maxlen=100)
    loaded = _warm_replay_buffer(
        buffer,
        ckpt_dir,
        replay_window=100,
        expected_value_target="root_q",
        max_shards=2,
    )

    assert loaded == 2
    assert [s.source_iter for s in buffer] == [3, 4]


def test_sample_shard_atomic_round_trip(tmp_path) -> None:
    ckpt_dir = str(tmp_path)
    samples = [_fake_sample(), _fake_sample()]
    _save_sample_shard(ckpt_dir, 12, samples, value_target="root_q")

    path = tmp_path / "samples_iter_0012.npz"
    assert path.exists()
    with np.load(path) as data:
        assert str(np.asarray(data["value_target"]).reshape(-1)[0]) == "root_q"
    loaded = _load_sample_shard(str(path), expected_value_target="root_q")
    assert len(loaded) == 2
    assert loaded[0].value == pytest.approx(0.5)
    assert loaded[0].planes.dtype == np.float16
    assert loaded[0].policy.dtype == np.float16


def test_load_sample_shard_skips_value_target_mismatch(tmp_path) -> None:
    _save_sample_shard(str(tmp_path), 1, [_fake_sample()], value_target="root_q")
    path = str(tmp_path / "samples_iter_0001.npz")
    assert _load_sample_shard(path, expected_value_target="outcome") == []
    assert len(_load_sample_shard(path, expected_value_target="root_q")) == 1


def test_require_value_target_compat_stamped_match() -> None:
    _require_value_target_compat("root_q", "root_q", source="ckpt")


def test_require_value_target_compat_stamped_mismatch() -> None:
    with pytest.raises(ValueError, match="does not match"):
        _require_value_target_compat("outcome", "root_q", source="ckpt")


def test_require_value_target_compat_explicit_cli_allows_cutover(
    capsys: pytest.CaptureFixture[str],
) -> None:
    _require_value_target_compat(
        "q_z",
        "root_q",
        source="ckpt",
        explicit_cli=True,
    )
    captured = capsys.readouterr()
    assert "root_q" in captured.out and "q_z" in captured.out


def test_require_value_target_compat_legacy_requires_explicit_cli() -> None:
    with pytest.raises(ValueError, match="no value_target metadata"):
        _require_value_target_compat(
            "root_q",
            None,
            source="legacy.pt",
            require_explicit_for_legacy=True,
            explicit_cli=False,
        )
    _require_value_target_compat(
        "root_q",
        None,
        source="legacy.pt",
        require_explicit_for_legacy=True,
        explicit_cli=True,
    )


def test_legacy_shard_without_value_target_still_loads(tmp_path) -> None:
    """Zero-era shards lack value_target; load them when expected is set."""
    path = str(tmp_path / "samples_iter_0002.npz")
    sample = _fake_sample()
    np.savez_compressed(
        path,
        planes=np.stack([sample.planes]).astype(np.float16),
        policies=np.stack([sample.policy]).astype(np.float16),
        players=np.array([True], dtype=np.bool_),
        values=np.array([0.5], dtype=np.float32),
        source_iters=np.array([2], dtype=np.int32),
        encoding_version=np.array([ENCODING_VERSION], dtype=np.int16),
    )
    loaded = _load_sample_shard(path, expected_value_target="root_q")
    assert len(loaded) == 1
    assert loaded[0].value == pytest.approx(0.5)


def test_legacy_shard_singular_keys_match_plural_keys(tmp_path) -> None:
    planes = np.arange(20 * 8 * 8, dtype=np.float16).reshape(1, 20, 8, 8)
    policy = np.zeros((1, POLICY_SIZE), dtype=np.float16)
    policy[0, 17] = 1.0
    player = np.array([False], dtype=np.bool_)
    value = np.array([-0.25], dtype=np.float32)
    common = {
        "planes": planes,
        "encoding_version": np.array([ENCODING_VERSION], dtype=np.int16),
    }
    singular_path = tmp_path / "samples_iter_0003.npz"
    plural_path = tmp_path / "samples_iter_0004.npz"
    np.savez_compressed(
        singular_path,
        **common,
        policy=policy,
        player=player,
        value=value,
    )
    np.savez_compressed(
        plural_path,
        **common,
        policies=policy,
        players=player,
        values=value,
    )

    singular = _load_sample_shard(str(singular_path), expected_value_target="root_q")
    plural = _load_sample_shard(str(plural_path), expected_value_target="root_q")

    assert len(singular) == len(plural) == 1
    np.testing.assert_array_equal(singular[0].planes, plural[0].planes)
    np.testing.assert_array_equal(singular[0].policy, plural[0].policy)
    assert singular[0].player is plural[0].player is False
    assert singular[0].value == plural[0].value == pytest.approx(-0.25)
    assert singular[0].source_iter == plural[0].source_iter == 0
