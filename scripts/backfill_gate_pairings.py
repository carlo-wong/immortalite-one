"""Backfill pairing columns on metrics_gates.csv from metrics_gates_openings.csv.

Pairings are color-swapped masters-book games (game_idx // 2). Pairing WIN if
the candidate's score across both colors is > 1.0 (e.g. win as White + draw as
Black). Rewrites metrics_gates.csv in place (atomic replace).
"""

from __future__ import annotations

import argparse
import os

from engine.train import backfill_gate_pairings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-dir",
        default="results",
        help="directory containing metrics_gates.csv (default: results)",
    )
    args = parser.parse_args()
    filled, empty = backfill_gate_pairings(args.checkpoint_dir)
    print(
        f"backfilled pairings in {os.path.abspath(args.checkpoint_dir)}/metrics_gates.csv: "
        f"filled={filled} empty={empty}"
    )


if __name__ == "__main__":
    main()
