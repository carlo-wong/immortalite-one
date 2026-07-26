"""Stop the current Lightning Studio after a job finishes.

Set SLEEP_STUDIO = False when debugging interactively on a studio.
Outside Lightning (no LIGHTNING_CLOUD_SPACE_ID), this is a no-op.
"""

from __future__ import annotations

import os
import time

# --- edit here ---
SLEEP_STUDIO = True
# Delay before Studio().stop() so you can still grab logs / cancel if needed.
SLEEP_STUDIO_DELAY_SEC = 5 * 60


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
    delay = max(0, int(SLEEP_STUDIO_DELAY_SEC))
    if delay > 0:
        print(
            f"Sleeping {delay // 60}m {delay % 60}s before stopping "
            f"Lightning Studio (set SLEEP_STUDIO_DELAY_SEC=0 to stop immediately)..."
        )
        time.sleep(delay)
    print("Stopping Lightning Studio to end GPU billing...")
    Studio().stop()
