"""Stage 3: deep investigation using swarm subgraph and graph analysis."""

from __future__ import annotations

from datetime import datetime, timezone

from src.config import get_data_dir
from src.deepagents_adapter import deep_investigation_summary
from src.rag.typology_retriever import get_typology_hints
from src.state import FraudDetectionState, InvestigationFinding
from src.subgraphs.investigation_swarm import SwarmState, build_investigation_swarm
from src.tools.graph_analysis import analyze_entity_network


def run_investigation(state: FraudDetectionState) -> FraudDetectionState:
    state.stage_timestamps["investigation"] = datetime.now(timezone.utc).isoformat()
    swarm = build_investigation_swarm()
    swarm_result = swarm.invoke(SwarmState(transactions=state.transactions))
    swarm_dict = swarm_result.model_dump() if hasattr(swarm_result, "model_dump") else dict(swarm_result)

    state.investigation_findings.extend(swarm_dict.get("findings", []))
    state.network_analysis = swarm_dict.get("network_summary", "")

    if state.transactions:
        data_dir = get_data_dir()
        network = analyze_entity_network(state.transactions[0].account_id, data_dir=data_dir)
        if "proximity_to_flagged_entity" in network.get("risk_indicators", []):
            state.investigation_findings.append(
                InvestigationFinding(
                    finding_type="network_anomaly",
                    description="Proximity to flagged entity in network graph.",
                    severity="high",
                    evidence=[str(network.get("connected_entities", []))],
                    confidence=0.82,
                )
            )

    finding_types = [f.finding_type for f in state.investigation_findings]
    typology_hints = get_typology_hints(finding_types)
    state.investigation_summary = (
        f"Findings={len(state.investigation_findings)}. "
        f"Typology hints: {' | '.join(typology_hints)}"
    )
    deep_summary = deep_investigation_summary(
        case_id=state.case_id,
        risk_tier=state.risk_tier or "low",
        findings=state.investigation_findings,
        network_analysis=state.network_analysis,
    )
    if deep_summary:
        state.investigation_summary = deep_summary
        state.decision_log.append("DeepAgents: investigation summary applied.")
    state.decision_log.append(state.investigation_summary)
    return state

