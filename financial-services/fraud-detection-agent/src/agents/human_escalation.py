"""Stage 5: escalation and final disposition."""

from __future__ import annotations

from datetime import datetime, timezone

from src.chains import build_escalation_chain
from src.config import get_settings
from src.state import FraudDetectionState
from src.tools.notification import send_alert


def run_human_escalation(state: FraudDetectionState) -> FraudDetectionState:
    state.stage_timestamps["escalation"] = datetime.now(timezone.utc).isoformat()
    chain = build_escalation_chain()
    decision = chain.invoke(
        {
            "risk_tier": state.risk_tier or "low",
            "filing_recommendation": state.sar_draft.filing_recommendation if state.sar_draft else "needs_review",
            "finding_count": len(state.investigation_findings),
        }
    )
    state.requires_human_review = decision.requires_human_review
    state.disposition = decision.disposition
    state.escalation_reason = decision.escalation_reason
    state.assigned_investigator = decision.assigned_investigator

    alert_body = (
        f"Case={state.case_id}, disposition={state.disposition}, "
        f"risk={state.risk_tier}, reason={state.escalation_reason}"
    )
    alert_id = send_alert(get_settings().alert_channel, "Fraud Case Update", alert_body)
    state.alerts_sent.append(alert_id)
    state.decision_log.append(f"Escalation completed: {state.disposition}")
    return state

