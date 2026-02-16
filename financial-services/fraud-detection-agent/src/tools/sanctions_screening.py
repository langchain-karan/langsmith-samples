"""Sanctions and PEP screening helper."""

from __future__ import annotations

import json
from pathlib import Path


def screen_sanctions(name: str, data_dir: Path) -> dict:
    path = data_dir / "sanctions_watchlist.json"
    if not path.exists():
        return {"sanctions_match": False, "pep_match": False, "matches": []}

    watchlist = json.loads(path.read_text(encoding="utf-8"))
    normalized = name.lower().strip()
    matches = [entry for entry in watchlist if entry["name"].lower() == normalized]
    return {
        "sanctions_match": any(m["type"] == "sanctions" for m in matches),
        "pep_match": any(m["type"] == "pep" for m in matches),
        "matches": matches,
    }

