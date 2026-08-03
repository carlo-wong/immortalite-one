"""Pairing aggregation for masters-book gate matches."""

from __future__ import annotations

import csv
from pathlib import Path

from engine.train import (
    GATE_CSV_COLUMNS,
    GATE_CSV_HEADER,
    _log_gate_metrics,
    _pairing_field_values,
    _pairing_stats,
    backfill_gate_pairings,
)


def _row(game_idx: int, a_is_white: int, result: str) -> dict:
    return {
        "game_idx": game_idx,
        "a_is_white": a_is_white,
        "result": result,
        "opening_uci": "e2e4 e7e5",
        "termination": "checkmate",
        "plies": 40,
    }


def test_pairing_win_white_win_black_draw() -> None:
    # Win as White + draw as Black = 1.5 → pairing WIN
    stats = _pairing_stats([_row(0, 1, "W"), _row(1, 0, "D")])
    assert stats == {
        "pairings": 1,
        "pair_wins": 1,
        "pair_draws": 0,
        "pair_losses": 0,
        "pair_winrate": 1.0,
    }


def test_pairing_draw_and_loss_mix() -> None:
    openings = [
        _row(0, 1, "W"),
        _row(1, 0, "L"),  # 1.0 → draw
        _row(2, 1, "D"),
        _row(3, 0, "D"),  # 1.0 → draw
        _row(4, 1, "L"),
        _row(5, 0, "L"),  # 0.0 → loss
        _row(6, 1, "W"),
        _row(7, 0, "W"),  # 2.0 → win
    ]
    stats = _pairing_stats(openings)
    assert stats["pairings"] == 4
    assert stats["pair_wins"] == 1
    assert stats["pair_draws"] == 2
    assert stats["pair_losses"] == 1
    assert stats["pair_winrate"] == 0.5  # (1 + 0.5*2) / 4


def test_pairing_skips_incomplete_pair() -> None:
    stats = _pairing_stats([_row(0, 1, "W")])  # odd SPRT stop
    assert stats["pairings"] == 0
    assert _pairing_field_values([]) == {
        "pairings": "",
        "pair_wins": "",
        "pair_draws": "",
        "pair_losses": "",
        "pair_winrate": "",
    }
    assert _pairing_field_values([_row(0, 1, "W")]) == {
        "pairings": "0",
        "pair_wins": "0",
        "pair_draws": "0",
        "pair_losses": "0",
        "pair_winrate": "",
    }


def test_gate_csv_column_order() -> None:
    assert GATE_CSV_COLUMNS[:8] == (
        "iter",
        "prev_iter",
        "games",
        "games_played",
        "wins",
        "draws",
        "losses",
        "winrate",
    )
    assert GATE_CSV_COLUMNS[14:19] == (
        "pairings",
        "pair_wins",
        "pair_draws",
        "pair_losses",
        "pair_winrate",
    )
    assert GATE_CSV_COLUMNS[19:24] == (
        "elo",
        "elo_lower",
        "elo_upper",
        "los",
        "verdict",
    )
    assert GATE_CSV_COLUMNS[-2:] == ("mean_game_len", "terminations")


def test_log_gate_metrics_writes_pairing_columns(tmp_path: Path) -> None:
    metrics = {
        "wins_as_white": 1,
        "wins_as_black": 0,
        "losses_as_white": 0,
        "losses_as_black": 0,
        "draws_as_white": 0,
        "draws_as_black": 1,
        "winrate": 0.75,
        "mean_game_len": 40.0,
        "terminations": "checkmate:1;threefold_repetition:1",
        "games_played": 2,
        "openings": [_row(0, 1, "W"), _row(1, 0, "D")],
        "book_lines": 1,
    }
    _log_gate_metrics(str(tmp_path), 20, 0, metrics, games=2)
    with (tmp_path / "metrics_gates.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0].keys()) == list(GATE_CSV_COLUMNS)
    assert rows[0]["pairings"] == "1"
    assert rows[0]["pair_wins"] == "1"
    assert rows[0]["pair_draws"] == "0"
    assert rows[0]["pair_losses"] == "0"
    assert rows[0]["pair_winrate"] == "1.000000"
    assert rows[0]["wins"] == "1"
    assert rows[0]["verdict"] == "PASS" or rows[0]["verdict"] in {
        "PASS",
        "INCONCLUSIVE",
        "FAIL",
    }


def test_backfill_uses_trailing_openings_block(tmp_path: Path) -> None:
    gates = tmp_path / "metrics_gates.csv"
    openings = tmp_path / "metrics_gates_openings.csv"
    # Legacy header without pairing columns / old column order.
    gates.write_text(
        "iter,prev_iter,games,games_played,"
        "wins_as_white,wins_as_black,losses_as_white,losses_as_black,"
        "draws_as_white,draws_as_black,"
        "winrate,wins,draws,losses,mean_game_len,terminations,"
        "elo,elo_lower,elo_upper,los,verdict\n"
        "20,0,2,2,1,0,0,0,0,1,0.750000,1,1,0,40.00,checkmate:1,"
        "100.00,10.00,200.00,0.900000,PASS\n",
        encoding="utf-8",
    )
    # Stale first block (loss+loss) then trailing match matching the gate row.
    openings.write_text(
        "iter,prev_iter,game_idx,a_is_white,opening_uci,result,termination,plies\n"
        "20,0,0,1,e2e4,L,checkmate,10\n"
        "20,0,1,0,e2e4,L,checkmate,10\n"
        "20,0,0,1,e2e4,W,checkmate,40\n"
        "20,0,1,0,e2e4,D,threefold_repetition,40\n",
        encoding="utf-8",
    )
    filled, empty = backfill_gate_pairings(tmp_path)
    assert (filled, empty) == (1, 0)
    with gates.open(encoding="utf-8", newline="") as f:
        text = f.read()
        rows = list(csv.DictReader(text.splitlines()))
    assert text.startswith(GATE_CSV_HEADER.rstrip("\n"))
    assert list(rows[0].keys()) == list(GATE_CSV_COLUMNS)
    assert rows[0]["pair_wins"] == "1"
    assert rows[0]["pair_winrate"] == "1.000000"
    assert rows[0]["wins"] == "1"
    assert rows[0]["mean_game_len"] == "40.00"
