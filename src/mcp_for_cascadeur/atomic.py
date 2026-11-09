"""Bounded atomic file replacement while Windows readers release their handles."""

from __future__ import annotations

import time
from pathlib import Path


def replace_file(temporary: str | Path, destination: str | Path, *, timeout: float = 1.0) -> None:
    """Retry only a sharing-denied rename; never repeat the operation producing it."""
    source, target = Path(temporary), Path(destination)
    deadline, delay = time.monotonic() + max(0.0, timeout), 0.005
    while True:
        try:
            source.replace(target)
            return
        except PermissionError as error:
            remaining = deadline - time.monotonic()
            if getattr(error, "winerror", None) not in {5, 32, 33} or remaining <= 0:
                raise
            time.sleep(min(delay, remaining))
            delay = min(delay * 2, 0.05)
