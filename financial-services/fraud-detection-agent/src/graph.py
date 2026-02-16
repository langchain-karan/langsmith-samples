"""Main sequential fraud detection workflow graph."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from src.agents.compliance import run_compliance
from src.agents.human_escalation import run_human_escalation
from src.agents.investigation import run_investigation
from src.agents.risk_enrichment import run_risk_enrichment
from src.agents.transaction_monitor import run_transaction_monitor
from src.models.risk_scorer import compute_risk_score, score_to_tier
from src.state import FraudDetectionState


def _ingest(state: FraudDetectionState) -> FraudDetectionState:
    state.stage_timestamps["ingestion"] = datetime.now(timezone.utc).isoformat()
    if not state.case_id:
        state.case_id = f"CASE-{uuid4().hex[:8].upper()}"
    if not state.batch_id:
        state.batch_id = f"BATCH-{uuid4().hex[:8].upper()}"
    state.decision_log.append(f"Ingested {len(state.transactions)} transactions.")
    return state


def _recompute_risk(state: FraudDetectionState) -> FraudDetectionState:
    score = compute_risk_score(state.anomaly_scores, state.customer_profile, state.investigation_findings)
    state.risk_score = score
    state.risk_tier = score_to_tier(score)
    state.decision_log.append(f"Risk recomputed after investigation: score={score:.2f} tier={state.risk_tier}")
    return state


def build_graph():
    graph = StateGraph(FraudDetectionState)
    graph.add_node("ingest", _ingest)
    graph.add_node("monitor", run_transaction_monitor)
    graph.add_node("enrich", run_risk_enrichment)
    graph.add_node("investigate", run_investigation)
    graph.add_node("risk_recompute", _recompute_risk)
    graph.add_node("comply", run_compliance)
    graph.add_node("escalate", run_human_escalation)

    graph.add_edge(START, "ingest")
    graph.add_edge("ingest", "monitor")
    graph.add_edge("monitor", "enrich")
    graph.add_edge("enrich", "investigate")
    graph.add_edge("investigate", "risk_recompute")
    graph.add_edge("risk_recompute", "comply")
    graph.add_edge("comply", "escalate")
    graph.add_edge("escalate", END)
    return graph.compile()


graph = build_graph()

