"""Entity graph analysis helper for investigation stage."""

from __future__ import annotations

import json
from pathlib import Path


def analyze_entity_network(entity_id: str, data_dir: Path) -> dict:
    """Return local graph neighborhood and risk annotations."""
    path = data_dir / "entity_graph.json"
    if not path.exists():
        return {"connected_entities": [], "risk_indicators": [], "summary": "No graph data."}

    graph = json.loads(path.read_text(encoding="utf-8"))
    neighbors = graph.get(entity_id, [])
    risk_indicators: list[str] = []
    if len(neighbors) >= 3:
        risk_indicators.append("dense_counterparty_network")
    if any(n.get("flagged", False) for n in neighbors):
        risk_indicators.append("proximity_to_flagged_entity")

    return {
        "connected_entities": neighbors,
        "risk_indicators": risk_indicators,
        "summary": f"Entity {entity_id} has {len(neighbors)} direct counterparties.",
    }

