"""Locate the sibling Immortalite Zero (Chess AI) repo for optional parity tests."""

from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CANDIDATE_NAMES = ("Chess AI", "chess-ai", "ChessAI", "immortalite-zero")


def find_zero_root() -> Path | None:
    """Return the Immortalite Zero repo root if it can be found, else None.

    Resolution order:
      1. ``IMMORTALITE_ZERO_ROOT`` environment variable
      2. Sibling directories under the parent of this repo (``../Chess AI``, etc.)
    """
    env = os.environ.get("IMMORTALITE_ZERO_ROOT", "").strip()
    if env:
        path = Path(env).expanduser().resolve()
        if path.is_dir():
            return path
        return None

    parent = _REPO_ROOT.parent
    for name in _CANDIDATE_NAMES:
        candidate = (parent / name).resolve()
        if candidate.is_dir() and (candidate / "engine").is_dir():
            return candidate
    return None


def require_zero_root() -> Path:
    """Like :func:`find_zero_root` but raise if the oracle repo is missing."""
    root = find_zero_root()
    if root is None:
        raise FileNotFoundError(
            "Immortalite Zero oracle repo not found. Set IMMORTALITE_ZERO_ROOT "
            "or place a sibling checkout named 'Chess AI' next to immortalite-one."
        )
    return root
