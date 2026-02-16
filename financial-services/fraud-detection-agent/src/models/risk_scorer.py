"""Risk scoring model used by monitor and escalation stages."""

from __future__ import annotations

from src.state import AnomalyScore, CustomerProfile, InvestigationFinding


def compute_risk_score(
    anomaly_scores: list[AnomalyScore],
    customer_profile: CustomerProfile | None,
    findings: list[InvestigationFinding],
) -> float:
    """Compute normalized risk score from multiple evidence streams."""
    anomaly_component = max((a.score for a in anomaly_scores), default=0.0) * 0.45

    profile_component = 0.0
    if customer_profile:
        mapping = {"low": 0.05, "medium": 0.15, "high": 0.25, "pep": 0.35, "sanctioned": 0.5}
        profile_component = mapping.get(customer_profile.risk_rating, 0.1)
        if customer_profile.previous_sars > 0:
            profile_component += 0.1

    finding_component = 0.0
    for finding in findings:
        sev_weight = {"low": 0.03, "medium": 0.08, "high": 0.14, "critical": 0.22}[finding.severity]
        finding_component += sev_weight * max(0.4, finding.confidence)
    finding_component = min(0.5, finding_component)

    return round(min(1.0, anomaly_component + profile_component + finding_component), 4)


def score_to_tier(score: float) -> str:
    if score >= 0.85:
        return "critical"
    if score >= 0.65:
        return "high"
    if score >= 0.4:
        return "medium"
    return "low"

