"""Stage 4: compliance determination and SAR drafting."""

from __future__ import annotations

from datetime import datetime, timezone

from src.chains import build_sar_narrative_chain
from src.deepagents_adapter import deep_sar_refinement
from src.rag.regulatory_retriever import get_relevant_regulations
from src.state import FraudDetectionState
from src.tools.sar_generator import draft_sar


def run_compliance(state: FraudDetectionState) -> FraudDetectionState:
    state.stage_timestamps["compliance"] = datetime.now(timezone.utc).isoformat()
    finding_types = [f.finding_type for f in state.investigation_findings]
    state.regulatory_references = get_relevant_regulations(state.risk_tier or "low", finding_types)

    sar = draft_sar(state.transactions, state.investigation_findings, state.regulatory_references)

    chain = build_sar_narrative_chain()
    narrative = chain.invoke(
        {
            "case_id": state.case_id,
            "risk_tier": state.risk_tier,
            "findings": "; ".join(f.description for f in state.investigation_findings[:5]),
            "total_amount": sar.total_amount,
        }
    )
    refined = deep_sar_refinement(
        case_id=state.case_id,
        risk_tier=state.risk_tier or "low",
        sar_narrative=narrative,
        regulatory_references=state.regulatory_references,
    )
    if refined:
        narrative = refined
        state.decision_log.append("DeepAgents: SAR narrative refinement applied.")

    sar.narrative = narrative
    state.sar_draft = sar

    if sar.filing_recommendation == "file":
        state.compliance_determination = "SAR recommended for filing."
    else:
        state.compliance_determination = "Needs analyst review before filing decision."

    state.decision_log.append(state.compliance_determination)
    return state

