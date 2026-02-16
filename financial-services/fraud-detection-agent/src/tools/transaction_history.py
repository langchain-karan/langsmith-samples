"""Transaction history access helpers."""

from __future__ import annotations

import json
from pathlib import Path


def get_recent_transactions(account_id: str, data_dir: Path) -> list[dict]:
    path = data_dir / "transaction_history.json"
    if not path.exists():
        return []
    all_rows = json.loads(path.read_text(encoding="utf-8"))
    return [row for row in all_rows if row.get("account_id") == account_id]

