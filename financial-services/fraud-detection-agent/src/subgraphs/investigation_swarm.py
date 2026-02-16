"""Swarm-style investigation subgraph."""

from __future__ import annotations

from pydantic import BaseModel, Field

from langgraph.graph import END, START, StateGraph

from src.models.pattern_detector import detect_geo_velocity, detect_structuring
from src.state import InvestigationFinding, Transaction


class SwarmState(BaseModel):
    """State for subgraph branches."""

    transactions: list[Transaction] = Field(default_factory=list)
    findings: list[InvestigationFinding] = Field(default_factory=list)
    network_summary: str = ""


def _temporal_branch(state: SwarmState) -> SwarmState:
    state.findings.extend(detect_structuring(state.transactions))
    return state


def _geo_branch(state: SwarmState) -> SwarmState:
    state.findings.extend(detect_geo_velocity(state.transactions))
    return state


def _network_branch(state: SwarmState) -> SwarmState:
    counterparties = {t.counterparty_id for t in state.transactions if t.counterparty_id}
    if len(counterparties) >= 3:
        state.findings.append(
            InvestigationFinding(
                finding_type="network_anomaly",
                severity="high",
                description="Transactions span a dense counterparty graph.",
                evidence=sorted(c for c in counterparties if c),
                confidence=0.8,
            )
        )
    state.network_summary = f"Counterparty breadth={len(counterparties)}"
    return state


def _aggregate(state: SwarmState) -> SwarmState:
    # Branches update same list in demo mode; dedupe by description.
    seen: set[str] = set()
    unique: list[InvestigationFinding] = []
    for finding in state.findings:
        key = f"{finding.finding_type}:{finding.description}"
        if key in seen:
            continue
        seen.add(key)
        unique.append(finding)
    state.findings = unique
    return state


def build_investigation_swarm():
    """Build a simple multi-branch investigation graph."""
    g = StateGraph(SwarmState)
    g.add_node("temporal_branch", _temporal_branch)
    g.add_node("geo_branch", _geo_branch)
    g.add_node("network_branch", _network_branch)
    g.add_node("aggregate", _aggregate)

    g.add_edge(START, "temporal_branch")
    g.add_edge("temporal_branch", "geo_branch")
    g.add_edge("geo_branch", "network_branch")
    g.add_edge("network_branch", "aggregate")
    g.add_edge("aggregate", END)
    return g.compile()

