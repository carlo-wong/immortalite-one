"""Split a legacy training metrics.csv into training and performance CSVs."""

from __future__ import annotations

import argparse
import os

from engine.train import migrate_legacy_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        help="directory containing the legacy metrics.csv",
    )
    parser.add_argument(
        "--keep-source",
        action="store_true",
        help="keep metrics.csv in place instead of rotating it to a legacy backup",
    )
    args = parser.parse_args()

    retained = migrate_legacy_metrics(
        args.checkpoint_dir,
        keep_source=args.keep_source,
    )
    if retained is None:
        print("metrics already split or no legacy metrics.csv found")
        return
    print(
        "split metrics.csv into metrics_training.csv and metrics_performance.csv; "
        f"legacy source retained at {os.path.abspath(retained)}"
    )


if __name__ == "__main__":
    main()
