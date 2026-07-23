"""Immortalite One Python package (hybrid C++/Python).

The native extension ``engine._native`` is required. Build with::

    pip install -e .
"""

from __future__ import annotations

try:
    from . import _native
except ImportError as exc:  # pragma: no cover - install/build failure path
    raise ImportError(
        "Immortalite One requires the native extension `engine._native`, which failed "
        "to import. Install a C++ toolchain (MSVC on Windows, g++/clang on Linux) and "
        "run: pip install -e ."
    ) from exc

__all__ = ["_native"]
__version__ = "0.1.0"
