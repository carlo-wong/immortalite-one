"""Locate the archived pure-Python Immortalite Zero checkout for optional parity tests."""

from __future__ import annotations

import os
from pathlib import Path

from ._env import env_get

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CANDIDATE_NAMES = (
    "immortalite-zero-python",
    "Chess AI",
    "chess-ai",
    "ChessAI",
)


def find_python_oracle_root() -> Path | None:
    """Return the archived pure-Python Zero repo root if found, else None.

    Resolution order:
      1. ``IMMORTALITE_PYTHON_ORACLE_ROOT`` (preferred)
      2. ``IMMORTALITE_ZERO_ROOT`` (legacy alias from the Immortalite One era)
      3. Sibling directories under the parent of this repo
    """
    for env_name in ("IMMORTALITE_PYTHON_ORACLE_ROOT", "IMMORTALITE_ZERO_ROOT"):
        env = env_get(env_name)
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


def find_zero_root() -> Path | None:
    """Backward-compatible alias for :func:`find_python_oracle_root`."""
    return find_python_oracle_root()


def require_python_oracle_root() -> Path:
    """Like :func:`find_python_oracle_root` but raise if the oracle repo is missing."""
    root = find_python_oracle_root()
    if root is None:
        raise FileNotFoundError(
            "Archived pure-Python Immortalite Zero oracle not found. Set "
            "IMMORTALITE_PYTHON_ORACLE_ROOT or place a sibling checkout named "
            "'immortalite-zero-python' (or legacy 'Chess AI') next to immortalite-zero."
        )
    return root


def require_zero_root() -> Path:
    """Backward-compatible alias for :func:`require_python_oracle_root`."""
    return require_python_oracle_root()
