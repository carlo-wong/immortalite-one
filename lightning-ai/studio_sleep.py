"""Stop the current Lightning Studio after a job finishes.

Set SLEEP_STUDIO = False when debugging interactively on a studio.
Outside Lightning (no LIGHTNING_CLOUD_SPACE_ID), this is a no-op.
"""

from __future__ import annotations

import os

# --- edit here ---
SLEEP_STUDIO = True


def maybe_stop_studio() -> None:
    """Stop this studio so GPU billing ends (does not wait for idle auto-sleep)."""
    if not SLEEP_STUDIO:
        print("SLEEP_STUDIO=False — leaving studio running")
        return
    if not os.environ.get("LIGHTNING_CLOUD_SPACE_ID"):
        print("Not inside a Lightning Studio — skip studio stop")
        return
    try:
        from lightning_sdk import Studio
    except ImportError:
        print(
            "WARNING: lightning-sdk not installed; cannot stop studio. "
            "Install with: pip install lightning-sdk"
        )
        return
    print("Stopping Lightning Studio to end GPU billing...")
    Studio().stop()
