"""State models for fraud detection and AML workflow."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class Transaction(BaseModel):
    """A single transaction event under analysis."""

    transaction_id: str
    account_id: str
    counterparty_id: str | None = None
    amount: float
    currency: str = "USD"
    transaction_type: Literal["wire", "ach", "card", "check", "internal", "crypto"] = "wire"
    timestamp: datetime
    channel: str = ""
    location: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class AnomalyScore(BaseModel):
    """Anomaly model output for a transaction."""

    transaction_id: str
    score: float
    model_version: str = "heuristic-v1"
    features_contributing: list[str] = Field(default_factory=list)
    confidence: float = 0.9


class CustomerProfile(BaseModel):
    """Customer enrichment profile."""

    customer_id: str
    name: str
    account_type: str
    risk_rating: Literal["low", "medium", "high", "pep", "sanctioned"] = "low"
    kyc_status: str = "current"
    account_open_date: str = ""
    previous_alerts: int = 0
    previous_sars: int = 0
    sanctions_match: bool = False
    pep_match: bool = False


class InvestigationFinding(BaseModel):
    """A specific finding discovered in investigation."""

    finding_type: Literal[
        "network_anomaly",
        "temporal_pattern",
        "geographic_risk",
        "behavioral_deviation",
        "structuring",
        "layering",
        "sanctions_hit",
        "pep_association",
        "other",
    ]
    description: str
    severity: Literal["low", "medium", "high", "critical"] = "medium"
    evidence: list[str] = Field(default_factory=list)
    confidence: float = 0.8


class SARDraft(BaseModel):
    """Suspicious Activity Report narrative draft."""

    narrative: str
    suspicious_activity_type: list[str] = Field(default_factory=list)
    total_amount: float = 0.0
    date_range: str = ""
    filing_recommendation: Literal["file", "do_not_file", "needs_review"] = "needs_review"
    regulatory_citations: list[str] = Field(default_factory=list)


class FraudDetectionState(BaseModel):
    """State passed through sequential workflow and subgraph."""

    transactions: list[Transaction] = Field(default_factory=list)
    batch_id: str = ""
    case_id: str = ""

    anomaly_scores: list[AnomalyScore] = Field(default_factory=list)
    risk_tier: Literal["low", "medium", "high", "critical"] | None = None
    risk_score: float = 0.0
    initial_alert_type: str = ""

    customer_profile: CustomerProfile | None = None
    counterparty_profiles: list[CustomerProfile] = Field(default_factory=list)
    enrichment_flags: list[str] = Field(default_factory=list)

    investigation_findings: list[InvestigationFinding] = Field(default_factory=list)
    network_analysis: str = ""
    investigation_summary: str = ""

    sar_draft: SARDraft | None = None
    compliance_determination: str = ""
    regulatory_references: list[str] = Field(default_factory=list)

    disposition: Literal[
        "auto_cleared",
        "monitoring",
        "investigation_open",
        "sar_filed",
        "account_frozen",
        "law_enforcement_referral",
    ] | None = None
    escalation_reason: str = ""
    assigned_investigator: str = ""
    requires_human_review: bool = False

    stage_timestamps: dict[str, str] = Field(default_factory=dict)
    decision_log: list[str] = Field(default_factory=list)
    alerts_sent: list[str] = Field(default_factory=list)
    error: str = ""

