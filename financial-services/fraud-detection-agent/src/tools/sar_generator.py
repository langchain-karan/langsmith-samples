"""SAR generation helpers."""

from __future__ import annotations

from datetime import datetime

from src.state import InvestigationFinding, SARDraft, Transaction


def draft_sar(
    transactions: list[Transaction],
    findings: list[InvestigationFinding],
    regulatory_refs: list[str],
) -> SARDraft:
    """Create a structured SAR draft from findings."""
    total_amount = round(sum(t.amount for t in transactions), 2)
    activities = sorted({f.finding_type for f in findings}) or ["other"]

    timestamps = sorted(t.timestamp for t in transactions)
    if timestamps:
        date_range = f"{timestamps[0].date().isoformat()} to {timestamps[-1].date().isoformat()}"
    else:
        date_range = datetime.utcnow().date().isoformat()

    finding_text = "; ".join(f"{f.finding_type}: {f.description}" for f in findings[:5])
    narrative = (
        "Automated monitoring detected suspicious transaction behavior requiring AML review. "
        f"Observed indicators: {finding_text}. "
        f"Analyzed {len(transactions)} transactions totaling ${total_amount:,.2f}."
    )

    recommendation = "file" if any(f.severity in {"high", "critical"} for f in findings) else "needs_review"
    return SARDraft(
        narrative=narrative,
        suspicious_activity_type=activities,
        total_amount=total_amount,
        date_range=date_range,
        filing_recommendation=recommendation,
        regulatory_citations=regulatory_refs,
    )

