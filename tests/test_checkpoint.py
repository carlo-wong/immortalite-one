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
from engine.selfplay import WDL_DRAW, WDL_LOSS, WDL_UNLABELED, WDL_WIN, Sample
from engine.train import (
    DriveDisconnectedError,
    _checkpoint_wdl_head,
    _load_sample_shard,
    _net_cfg_from_module,
    _require_value_target_compat,
    _require_wdl_upgrade_reset,
    _require_warm_buffer_fill,
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


def test_warm_replay_buffer_ignores_shards_newer_than_resume_checkpoint(tmp_path) -> None:
    ckpt_dir = str(tmp_path)
    for iteration in range(618, 641):
        sample = _fake_sample()
        sample.source_iter = iteration
        _save_sample_shard(ckpt_dir, iteration, [sample], value_target="root_q")

    buffer: deque[Sample] = deque(maxlen=100)
    loaded = _warm_replay_buffer(
        buffer,
        ckpt_dir,
        replay_window=100,
        expected_value_target="root_q",
        max_shards=0,
        max_source_iter=620,
    )

    assert loaded == 3
    assert [sample.source_iter for sample in buffer] == [618, 619, 620]


def test_warm_replay_buffer_fills_beyond_old_twenty_shard_cap(tmp_path) -> None:
    """Default warm must keep loading until the window is full (no silent 20-shard stop)."""
    ckpt_dir = str(tmp_path)
    for iteration in range(30):
        samples = []
        for _ in range(10):
            sample = _fake_sample()
            sample.source_iter = iteration
            samples.append(sample)
        _save_sample_shard(ckpt_dir, iteration, samples, value_target="root_q")

    buffer: deque[Sample] = deque(maxlen=250)
    loaded = _warm_replay_buffer(
        buffer,
        ckpt_dir,
        replay_window=250,
        expected_value_target="root_q",
        max_shards=0,
    )

    assert loaded == 250
    assert len(buffer) == 250
    # Newest shards dominate; oldest of the kept window should be well past iter 0.
    assert min(s.source_iter for s in buffer) >= 5


def test_warm_replay_fails_fast_on_drive_disconnect(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ckpt_dir = str(tmp_path)
    for iteration in range(3):
        sample = _fake_sample()
        sample.source_iter = iteration
        _save_sample_shard(ckpt_dir, iteration, [sample], value_target="root_q")

    real_load = __import__("engine.train", fromlist=["_load_sample_shard"])._load_sample_shard

    def flaky_load(path: str, *, expected_value_target: str | None = None):
        # Newest shard loads; next one kills FUSE — must not walk the rest.
        if path.endswith("samples_iter_0001.npz"):
            raise OSError(107, "Transport endpoint is not connected")
        return real_load(path, expected_value_target=expected_value_target)

    monkeypatch.setattr("engine.train._load_sample_shard", flaky_load)

    buffer: deque[Sample] = deque(maxlen=10)
    with pytest.raises(DriveDisconnectedError, match="Remount Drive"):
        _warm_replay_buffer(
            buffer,
            ckpt_dir,
            replay_window=10,
            expected_value_target="root_q",
            max_shards=0,
        )


def test_require_warm_buffer_fill_blocks_cold_resume(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="cold replay buffer"):
        _require_warm_buffer_fill(
            50_000,
            replay_buffer_size=200_000,
            replay_window=200_000,
            start_iter=621,
            allow_cold=False,
            ckpt_dir=str(tmp_path),
        )


def test_require_warm_buffer_fill_allows_fresh_start_and_opt_out(tmp_path) -> None:
    _require_warm_buffer_fill(
        0,
        replay_buffer_size=200_000,
        replay_window=200_000,
        start_iter=0,
        allow_cold=False,
        ckpt_dir=str(tmp_path),
    )
    _require_warm_buffer_fill(
        1_000,
        replay_buffer_size=200_000,
        replay_window=200_000,
        start_iter=621,
        allow_cold=True,
        ckpt_dir=str(tmp_path),
    )


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


def test_wdl_checkpoint_architecture_round_trip(tmp_path) -> None:
    cfg = Config()
    cfg.net.blocks = 1
    cfg.net.filters = 4
    cfg.net.wdl_head = True
    cfg.train.wdl_coef = 0.35
    cfg.train.value_target = "outcome"
    net = ChessNet(cfg.net)
    path = str(tmp_path / "wdl.pt")
    save_checkpoint(net, cfg, path, iteration=4)

    state = torch.load(path, map_location="cpu")
    assert state["net"]["wdl_head"] is True
    assert state["wdl_coef"] == pytest.approx(0.35)
    assert state["encoding_version"] == ENCODING_VERSION == 2
    restored_cfg = Config()
    restored_cfg.net = type(cfg.net)(**state["net"])
    restored = ChessNet(restored_cfg.net)
    restored.load_state_dict(state["model"])
    assert restored.wdl_head is True
    assert any(name.startswith("wdl") for name, _ in restored.named_parameters())
    assert _net_cfg_from_module(restored).wdl_head is True


def test_wdl_checkpoint_stamp_prevents_silent_downgrade(tmp_path) -> None:
    """Resume applies stamped net cfg even when live defaults keep wdl_head=False."""
    cfg = Config()
    cfg.net.blocks = 1
    cfg.net.filters = 4
    cfg.net.wdl_head = True
    cfg.train.value_target = "outcome"
    net = ChessNet(cfg.net)
    path = str(tmp_path / "wdl_stamp.pt")
    save_checkpoint(net, cfg, path, iteration=2)

    state = torch.load(path, map_location="cpu")
    live = Config()
    assert live.net.wdl_head is False
    # Mirrors engine.train resume: stamped architecture wins over missing CLI flag.
    from engine.config import NetConfig

    ckpt_net = NetConfig(**state["net"])
    _require_wdl_upgrade_reset(
        cli_wdl_head=False,
        checkpoint_wdl_head=bool(ckpt_net.wdl_head),
        reset_optimizer=False,
        resuming=True,
    )
    live.net = ckpt_net
    rebuilt = ChessNet(live.net)
    rebuilt.load_state_dict(state["model"])
    assert live.net.wdl_head is True
    assert rebuilt.wdl_head is True
    assert any(name.startswith("wdl") for name, _ in rebuilt.named_parameters())


def test_legacy_checkpoint_upgrade_guard() -> None:
    with pytest.raises(ValueError, match="--reset-optimizer"):
        _require_wdl_upgrade_reset(
            cli_wdl_head=True,
            checkpoint_wdl_head=False,
            reset_optimizer=False,
            resuming=True,
        )


def test_raw_checkpoint_missing_net_stamp_treated_as_legacy_for_wdl_guard() -> None:
    net, optimizer, _cfg = _tiny_net_and_optimizer()
    state = {"model": net.state_dict(), "optimizer": optimizer.state_dict(), "iteration": 3}
    assert _checkpoint_wdl_head(state) is False
    with pytest.raises(ValueError, match="--reset-optimizer"):
        _require_wdl_upgrade_reset(
            cli_wdl_head=True,
            checkpoint_wdl_head=_checkpoint_wdl_head(state),
            reset_optimizer=False,
            resuming=True,
        )


def test_sample_shard_wdl_round_trip(tmp_path) -> None:
    ckpt_dir = str(tmp_path)
    samples = [
        Sample(
            planes=np.zeros((20, 8, 8), dtype=np.float32),
            policy=np.zeros(POLICY_SIZE, dtype=np.float32),
            player=True,
            value=0.25,
            wdl=WDL_WIN,
        ),
        Sample(
            planes=np.ones((20, 8, 8), dtype=np.float32),
            policy=np.zeros(POLICY_SIZE, dtype=np.float32),
            player=False,
            value=-0.5,
            wdl=WDL_LOSS,
        ),
        Sample(
            planes=np.full((20, 8, 8), 0.5, dtype=np.float32),
            policy=np.zeros(POLICY_SIZE, dtype=np.float32),
            player=True,
            value=0.0,
            wdl=WDL_DRAW,
        ),
    ]
    _save_sample_shard(ckpt_dir, 9, samples, value_target="outcome")
    path = tmp_path / "samples_iter_0009.npz"
    with np.load(path) as data:
        assert "wdl" in data
        assert str(np.asarray(data["wdl_schema"]).reshape(-1)[0]) == "stm_wdl_v1"
        assert data["wdl"].dtype == np.int8
        assert int(np.asarray(data["encoding_version"]).reshape(-1)[0]) == ENCODING_VERSION == 2
    loaded = _load_sample_shard(str(path), expected_value_target="outcome")
    assert [s.wdl for s in loaded] == [WDL_WIN, WDL_LOSS, WDL_DRAW]


def test_legacy_shard_missing_wdl_loads_unlabeled(tmp_path) -> None:
    path = str(tmp_path / "samples_iter_0005.npz")
    sample = _fake_sample()
    np.savez_compressed(
        path,
        planes=np.stack([sample.planes]).astype(np.float16),
        policies=np.stack([sample.policy]).astype(np.float16),
        players=np.array([True], dtype=np.bool_),
        values=np.array([0.5], dtype=np.float32),
        source_iters=np.array([5], dtype=np.int32),
        encoding_version=np.array([ENCODING_VERSION], dtype=np.int16),
        value_target=np.array(["root_q"], dtype=np.str_),
    )
    loaded = _load_sample_shard(path, expected_value_target="root_q")
    assert len(loaded) == 1
    assert loaded[0].wdl == WDL_UNLABELED


def test_unlabeled_only_shard_omits_wdl_key(tmp_path) -> None:
    sample = _fake_sample()
    sample.wdl = WDL_UNLABELED
    _save_sample_shard(str(tmp_path), 6, [sample], value_target="root_q")
    path = tmp_path / "samples_iter_0006.npz"
    with np.load(path) as data:
        assert "wdl" not in data
        assert "wdl_schema" not in data
    loaded = _load_sample_shard(str(path), expected_value_target="root_q")
    assert loaded[0].wdl == WDL_UNLABELED


def test_load_sample_shard_skips_wdl_schema_mismatch(
    tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = str(tmp_path / "samples_iter_0011.npz")
    sample = _fake_sample()
    np.savez_compressed(
        path,
        planes=np.stack([sample.planes]).astype(np.float16),
        policies=np.stack([sample.policy]).astype(np.float16),
        players=np.array([True], dtype=np.bool_),
        values=np.array([0.5], dtype=np.float32),
        source_iters=np.array([11], dtype=np.int32),
        encoding_version=np.array([ENCODING_VERSION], dtype=np.int16),
        value_target=np.array(["root_q"], dtype=np.str_),
        wdl=np.array([WDL_WIN], dtype=np.int8),
        wdl_schema=np.array(["other_wdl_v0"], dtype=np.str_),
    )
    loaded = _load_sample_shard(path, expected_value_target="root_q")
    assert loaded == []
    captured = capsys.readouterr().out
    assert "wdl_schema=other_wdl_v0" in captured
    assert "stm_wdl_v1" in captured


def test_load_sample_shard_skips_wdl_without_schema(
    tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = str(tmp_path / "samples_iter_0012.npz")
    sample = _fake_sample()
    np.savez_compressed(
        path,
        planes=np.stack([sample.planes]).astype(np.float16),
        policies=np.stack([sample.policy]).astype(np.float16),
        players=np.array([True], dtype=np.bool_),
        values=np.array([0.5], dtype=np.float32),
        source_iters=np.array([12], dtype=np.int32),
        encoding_version=np.array([ENCODING_VERSION], dtype=np.int16),
        value_target=np.array(["root_q"], dtype=np.str_),
        wdl=np.array([WDL_WIN], dtype=np.int8),
    )
    loaded = _load_sample_shard(path, expected_value_target="root_q")
    assert loaded == []
    assert "wdl_schema=missing" in capsys.readouterr().out


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
