"""Tests for split training and performance metrics logging."""

from __future__ import annotations

import csv

import pytest

from engine.train import (
    PERFORMANCE_METRICS_COLUMNS,
    TRAINING_METRICS_COLUMNS,
    _log_metrics,
    migrate_legacy_metrics,
)


EXPECTED_TRAINING_COLUMNS = (
    "iter", "games", "sims", "samples", "policy_loss", "value_loss",
    "policy_entropy", "value_sign_acc", "policy_top1_agree", "grad_norm",
    "mean_game_len", "decisive_rate", "white_win_rate", "draw_rate",
    "max_moves_trunc_rate", "value_mean", "value_std", "winrate_vs_prev",
    "winrate_quick", "learning_rate", "buffer_size", "buffer_min_iter",
    "buffer_max_iter", "terminations", "value_target", "value_q_ratio",
    "value_coef", "policy_surprise_data_weight", "c_puct",
    "dirichlet_alpha", "dirichlet_epsilon", "move_temperature",
    "move_temperature_plies", "random_opening_plies",
)

EXPECTED_PERFORMANCE_COLUMNS = (
    "iter", "games", "sims", "samples", "seconds", "selfplay_seconds",
    "train_seconds", "overhead_seconds", "train_steps", "batch_size",
    "buffer_size", "selfplay_concurrency", "selfplay_workers",
    "central_inference", "device", "net_blocks", "net_filters",
    "games_per_hour", "samples_per_second", "selfplay_games_per_hour",
    "selfplay_samples_per_second", "train_steps_per_second", "gpu_util_pct",
)

LEGACY_COLUMNS = (
    "iter", "sims", "samples", "seconds", "selfplay_seconds", "train_seconds",
    "policy_loss", "value_loss", "policy_entropy", "value_sign_acc",
    "policy_top1_agree", "grad_norm", "mean_game_len", "decisive_rate",
    "white_win_rate", "draw_rate", "max_moves_trunc_rate", "value_mean",
    "value_std", "winrate_vs_prev", "learning_rate", "games", "train_steps",
    "batch_size", "buffer_size", "terminations", "winrate_quick",
    "gpu_util_pct", "buffer_min_iter", "buffer_max_iter",
)


def _read_csv(path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def _write_legacy(path) -> None:
    rows = [
        {
            "iter": "7", "games": "1", "sims": "64", "samples": "200",
            "seconds": "100.0", "selfplay_seconds": "60.0",
            "train_seconds": "30.0", "train_steps": "10",
            "batch_size": "32", "buffer_size": "500",
            "policy_loss": "1.250000", "terminations": "checkmate:1",
            "gpu_util_pct": "75.000000", "buffer_min_iter": "2",
            "buffer_max_iter": "7",
        },
        {
            "iter": "8", "games": "2", "sims": "64", "samples": "0",
            "seconds": "0", "selfplay_seconds": "na", "train_seconds": "",
            "train_steps": "0", "batch_size": "32", "buffer_size": "500",
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LEGACY_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _log_fresh_metrics(checkpoint_dir: str) -> None:
    _log_metrics(
        checkpoint_dir,
        9,
        80,
        240,
        120.0,
        selfplay_seconds=90.0,
        train_seconds=20.0,
        policy_loss=1.2,
        value_loss=0.8,
        policy_entropy=2.3,
        value_sign_acc=0.75,
        policy_top1_agree=0.25,
        grad_norm=3.5,
        mean_game_len=60.0,
        decisive_rate=0.6,
        white_win_rate=0.4,
        draw_rate=0.2,
        max_moves_trunc_rate=0.1,
        value_mean=-0.05,
        value_std=0.7,
        winrate_vs_prev=float("nan"),
        learning_rate=5e-4,
        games=12,
        train_steps=40,
        batch_size=128,
        buffer_size=4096,
        termination_counts={"draw": 2, "checkmate": 10},
        value_target="q_z",
        value_q_ratio=0.7,
        value_coef=1.5,
        policy_surprise_data_weight=0.25,
        c_puct=1.8,
        dirichlet_alpha=0.2,
        dirichlet_epsilon=0.15,
        move_temperature=1.1,
        move_temperature_plies=12,
        random_opening_plies=4,
        selfplay_concurrency=16,
        selfplay_workers=3,
        central_inference=True,
        device="cuda:0",
        net_blocks=8,
        net_filters=96,
        winrate_quick=0.55,
        gpu_util_pct=82.5,
        buffer_min_iter=3,
        buffer_max_iter=9,
    )


def test_exact_split_headers() -> None:
    assert TRAINING_METRICS_COLUMNS == EXPECTED_TRAINING_COLUMNS
    assert PERFORMANCE_METRICS_COLUMNS == EXPECTED_PERFORMANCE_COLUMNS


def test_historical_migration_derives_rates_and_preserves_backup(tmp_path) -> None:
    source = tmp_path / "metrics.csv"
    _write_legacy(source)
    original_source = source.read_bytes()
    existing_backup = tmp_path / "metrics_legacy.csv"
    existing_backup.write_text("do not overwrite\n", encoding="utf-8")

    retained = migrate_legacy_metrics(str(tmp_path))

    assert retained == str(tmp_path / "metrics_legacy_1.csv")
    assert existing_backup.read_text(encoding="utf-8") == "do not overwrite\n"
    assert not source.exists()
    assert (tmp_path / "metrics_legacy_1.csv").read_bytes() == original_source

    training_header, training = _read_csv(tmp_path / "metrics_training.csv")
    performance_header, performance = _read_csv(tmp_path / "metrics_performance.csv")
    assert training_header == list(EXPECTED_TRAINING_COLUMNS)
    assert performance_header == list(EXPECTED_PERFORMANCE_COLUMNS)
    assert training[0]["policy_loss"] == "1.250000"
    assert training[0]["value_target"] == ""
    assert training[0]["c_puct"] == ""

    assert performance[0]["overhead_seconds"] == "10.0"
    assert float(performance[0]["games_per_hour"]) == pytest.approx(36.0)
    assert float(performance[0]["samples_per_second"]) == pytest.approx(2.0)
    assert float(performance[0]["selfplay_games_per_hour"]) == pytest.approx(60.0)
    assert float(performance[0]["selfplay_samples_per_second"]) == pytest.approx(10 / 3)
    assert float(performance[0]["train_steps_per_second"]) == pytest.approx(1 / 3)
    assert performance[0]["device"] == ""

    assert performance[1]["selfplay_seconds"] == "na"
    assert performance[1]["train_seconds"] == ""
    assert performance[1]["overhead_seconds"] == ""
    for rate in (
        "games_per_hour",
        "samples_per_second",
        "selfplay_games_per_hour",
        "selfplay_samples_per_second",
        "train_steps_per_second",
    ):
        assert performance[1][rate] == ""


def test_migration_is_idempotent_after_split(tmp_path) -> None:
    _write_legacy(tmp_path / "metrics.csv")
    migrate_legacy_metrics(str(tmp_path))
    before = {
        path.name: path.read_bytes()
        for path in tmp_path.iterdir()
    }

    assert migrate_legacy_metrics(str(tmp_path)) is None
    after = {
        path.name: path.read_bytes()
        for path in tmp_path.iterdir()
    }
    assert after == before


def test_keep_source_leaves_legacy_metrics_in_place(tmp_path) -> None:
    source = tmp_path / "metrics.csv"
    _write_legacy(source)
    original = source.read_bytes()

    retained = migrate_legacy_metrics(str(tmp_path), keep_source=True)

    assert retained == str(source)
    assert source.read_bytes() == original
    assert not (tmp_path / "metrics_legacy.csv").exists()


def test_fresh_append_logs_recipe_runtime_and_rates(tmp_path) -> None:
    _log_fresh_metrics(str(tmp_path))

    training_header, training = _read_csv(tmp_path / "metrics_training.csv")
    performance_header, performance = _read_csv(tmp_path / "metrics_performance.csv")
    assert training_header == list(EXPECTED_TRAINING_COLUMNS)
    assert performance_header == list(EXPECTED_PERFORMANCE_COLUMNS)
    assert len(training) == len(performance) == 1

    train_row = training[0]
    assert train_row["iter"] == "9"
    assert train_row["games"] == "12"
    assert train_row["value_target"] == "q_z"
    assert train_row["value_q_ratio"] == "0.700000"
    assert train_row["value_coef"] == "1.500000"
    assert train_row["policy_surprise_data_weight"] == "0.250000"
    assert train_row["c_puct"] == "1.800000"
    assert train_row["dirichlet_alpha"] == "0.200000"
    assert train_row["dirichlet_epsilon"] == "0.150000"
    assert train_row["move_temperature"] == "1.100000"
    assert train_row["move_temperature_plies"] == "12"
    assert train_row["random_opening_plies"] == "4"
    assert train_row["terminations"] == "checkmate:10;draw:2"

    perf_row = performance[0]
    assert perf_row["overhead_seconds"] == "10.0"
    assert perf_row["selfplay_concurrency"] == "16"
    assert perf_row["selfplay_workers"] == "3"
    assert perf_row["central_inference"] == "true"
    assert perf_row["device"] == "cuda:0"
    assert perf_row["net_blocks"] == "8"
    assert perf_row["net_filters"] == "96"
    assert float(perf_row["games_per_hour"]) == pytest.approx(360.0)
    assert float(perf_row["samples_per_second"]) == pytest.approx(2.0)
    assert float(perf_row["selfplay_games_per_hour"]) == pytest.approx(480.0)
    assert float(perf_row["selfplay_samples_per_second"]) == pytest.approx(8 / 3)
    assert float(perf_row["train_steps_per_second"]) == pytest.approx(2.0)
