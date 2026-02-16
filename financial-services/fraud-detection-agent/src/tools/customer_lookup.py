"""Customer profile retrieval for enrichment stage."""

from __future__ import annotations

import json
from pathlib import Path

from src.state import CustomerProfile


def load_customer_map(data_dir: Path) -> dict[str, dict]:
    path = data_dir / "customers.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def get_customer_profile(customer_id: str, data_dir: Path) -> CustomerProfile | None:
    data = load_customer_map(data_dir)
    raw = data.get(customer_id)
    if not raw:
        return None
    return CustomerProfile(**raw)

