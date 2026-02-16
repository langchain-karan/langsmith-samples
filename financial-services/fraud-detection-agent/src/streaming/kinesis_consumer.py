"""Local stand-in for Kinesis consumer."""

from __future__ import annotations

import json
from pathlib import Path


def read_local_events(path: Path) -> list[dict]:
    """Read event records from local JSON file for demos/tests."""
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))

