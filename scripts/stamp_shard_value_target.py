"""Backfill ``value_target`` metadata onto self-play sample shards.

Unstamped shards still load under any expected target (legacy compat). After a
recipe cutover (e.g. root_q -> q_z), stamp older shards so warm-up skips them.

Example (Lightning sibling results/):

  python scripts/stamp_shard_value_target.py --checkpoint-dir ../results \\
      --value-target root_q

Dry-run first:

  python scripts/stamp_shard_value_target.py --checkpoint-dir ../results --dry-run
"""

from __future__ import annotations

import argparse
import os
import tempfile

import numpy as np

from engine.encoding import ENCODING_VERSION


_PREFIX = "samples_iter_"
_SUFFIX = ".npz"


def _shard_value_target(data: np.lib.npyio.NpzFile) -> str | None:
    if "value_target" not in data:
        return None
    raw = np.asarray(data["value_target"]).reshape(-1)
    if raw.size == 0:
        return None
    return str(raw[0])


def _list_shards(ckpt_dir: str) -> list[str]:
    if not os.path.isdir(ckpt_dir):
        return []
    names = [
        name
        for name in os.listdir(ckpt_dir)
        if name.startswith(_PREFIX) and name.endswith(_SUFFIX)
    ]
    names.sort()
    return [os.path.join(ckpt_dir, name) for name in names]


def _stamp_shard(path: str, value_target: str, *, dry_run: bool) -> str:
    """Return action: 'stamp' | 'skip_has' | 'skip_empty'."""
    with np.load(path) as data:
        existing = _shard_value_target(data)
        if existing is not None:
            return f"skip_has:{existing}"
        if "values" not in data and "value" not in data:
            return "skip_empty"
        payload = {key: data[key] for key in data.files}
        payload["value_target"] = np.array([value_target], dtype=np.str_)
        if "encoding_version" not in payload:
            payload["encoding_version"] = np.array([ENCODING_VERSION], dtype=np.int16)

    if dry_run:
        return "stamp"
    ckpt_dir = os.path.dirname(path) or "."
    fd, tmp_path = tempfile.mkstemp(dir=ckpt_dir, suffix=".npz")
    os.close(fd)
    try:
        np.savez_compressed(tmp_path, **payload)
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    return "stamp"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        help="directory containing samples_iter_XXXX.npz",
    )
    parser.add_argument(
        "--value-target",
        default="root_q",
        help="stamp to write on unstamped shards (default: root_q)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report actions without rewriting files",
    )
    args = parser.parse_args()

    shards = _list_shards(args.checkpoint_dir)
    if not shards:
        raise SystemExit(f"no sample shards in {args.checkpoint_dir}")

    stamped = 0
    skipped_has = 0
    skipped_empty = 0
    for path in shards:
        action = _stamp_shard(path, args.value_target, dry_run=args.dry_run)
        name = os.path.basename(path)
        if action == "stamp":
            stamped += 1
            print(f"{'would stamp' if args.dry_run else 'stamped'} {name} -> {args.value_target}")
        elif action.startswith("skip_has:"):
            skipped_has += 1
        elif action == "skip_empty":
            skipped_empty += 1
            print(f"skip empty {name}")

    mode = "dry-run" if args.dry_run else "wrote"
    print(
        f"{mode}: stamp={stamped} already_stamped={skipped_has} "
        f"empty={skipped_empty} total={len(shards)}"
    )


if __name__ == "__main__":
    main()
