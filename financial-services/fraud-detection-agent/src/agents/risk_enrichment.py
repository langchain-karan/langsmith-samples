"""Stage 2: customer/counterparty enrichment and screening."""

from __future__ import annotations

from datetime import datetime, timezone

from src.config import get_data_dir
from src.state import FraudDetectionState
from src.tools.customer_lookup import get_customer_profile
from src.tools.sanctions_screening import screen_sanctions
from src.tools.transaction_history import get_recent_transactions


def run_risk_enrichment(state: FraudDetectionState) -> FraudDetectionState:
    state.stage_timestamps["enrichment"] = datetime.now(timezone.utc).isoformat()
    data_dir = get_data_dir()
    if not state.transactions:
        return state

    primary_account = state.transactions[0].account_id
    profile = get_customer_profile(primary_account, data_dir=data_dir)
    if profile:
        sanctions = screen_sanctions(profile.name, data_dir=data_dir)
        profile.sanctions_match = sanctions["sanctions_match"]
        profile.pep_match = sanctions["pep_match"]
        state.customer_profile = profile
        if profile.sanctions_match:
            state.enrichment_flags.append("sanctions_match")
        if profile.pep_match:
            state.enrichment_flags.append("pep_match")

    counterparties = sorted({tx.counterparty_id for tx in state.transactions if tx.counterparty_id})
    state.counterparty_profiles = [
        p
        for cp_id in counterparties
        if (p := get_customer_profile(cp_id, data_dir=data_dir)) is not None
    ]

    recent = get_recent_transactions(primary_account, data_dir=data_dir)
    if len(recent) >= 10:
        state.enrichment_flags.append("high_recent_activity")

    state.decision_log.append(
        f"Enrichment flags={state.enrichment_flags or ['none']}; counterparties={len(state.counterparty_profiles)}"
    )
    return state

