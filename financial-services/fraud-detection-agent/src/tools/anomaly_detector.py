"""Anomaly scoring tool (local heuristic implementation)."""

from __future__ import annotations

from src.state import AnomalyScore, Transaction


def detect_anomaly(transaction: Transaction) -> AnomalyScore:
    """Score transaction anomaly using deterministic features."""
    score = 0.05
    features: list[str] = []

    if transaction.amount >= 100000:
        score += 0.45
        features.append("high_amount")
    elif transaction.amount >= 25000:
        score += 0.25
        features.append("elevated_amount")

    high_risk_locs = {"high-risk-jurisdiction", "offshore-hub", "unknown"}
    if transaction.location.lower() in high_risk_locs:
        score += 0.25
        features.append("high_risk_location")

    if transaction.transaction_type in {"wire", "crypto"}:
        score += 0.1
        features.append("high_risk_channel")

    if transaction.metadata.get("velocity_1h", 0) >= 5:
        score += 0.15
        features.append("high_velocity")

    if transaction.metadata.get("new_counterparty", False):
        score += 0.1
        features.append("new_counterparty")

    return AnomalyScore(
        transaction_id=transaction.transaction_id,
        score=min(1.0, round(score, 4)),
        model_version="heuristic-v1",
        features_contributing=features,
        confidence=0.9,
    )

