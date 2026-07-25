"""Environment helpers with Immortalite Zero names + One-release aliases."""

from __future__ import annotations

import os

_TRUTHY = frozenset({"1", "true", "yes", "on"})
_FALSY = frozenset({"0", "false", "no", "off"})


def env_get(*names: str, default: str | None = None) -> str | None:
    """Return the first non-empty env value among ``names``, else ``default``."""
    for name in names:
        raw = os.environ.get(name)
        if raw is not None and raw.strip() != "":
            return raw
    return default


def env_flag(*names: str, default: bool = False) -> bool:
    """Parse the first set env among ``names`` as a boolean flag."""
    raw = env_get(*names)
    if raw is None:
        return default
    val = raw.strip().lower()
    if val in _TRUTHY:
        return True
    if val in _FALSY:
        return False
    return default
