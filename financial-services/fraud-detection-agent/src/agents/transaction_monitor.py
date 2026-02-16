"""Stage 1: transaction monitoring and initial risk scoring."""

from __future__ import annotations

from datetime import datetime, timezone

from src.models.risk_scorer import score_to_tier
from src.state import FraudDetectionState
from src.tools.anomaly_detector import detect_anomaly


def run_transaction_monitor(state: FraudDetectionState) -> FraudDetectionState:
    state.stage_timestamps["monitor"] = datetime.now(timezone.utc).isoformat()
    state.anomaly_scores = [detect_anomaly(tx) for tx in state.transactions]

    peak_score = max((s.score for s in state.anomaly_scores), default=0.0)
    state.risk_score = peak_score
    state.risk_tier = score_to_tier(peak_score)
    state.initial_alert_type = "high_amount_anomaly" if peak_score >= 0.65 else "anomaly_watch"
    state.decision_log.append(
        f"Monitor scored {len(state.anomaly_scores)} events; peak={peak_score:.2f}, tier={state.risk_tier}"
    )
    return state

