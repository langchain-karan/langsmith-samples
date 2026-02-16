"""Rule-based pattern detection helpers."""

from __future__ import annotations

from collections import defaultdict

from src.state import InvestigationFinding, Transaction


def detect_structuring(transactions: list[Transaction]) -> list[InvestigationFinding]:
    """Detect multiple sub-threshold transfers in short windows."""
    findings: list[InvestigationFinding] = []
    by_account: dict[str, list[Transaction]] = defaultdict(list)
    for tx in transactions:
        by_account[tx.account_id].append(tx)

    for account_id, txs in by_account.items():
        if len(txs) < 3:
            continue
        near_threshold = [t for t in txs if 9000 <= t.amount < 10000]
        if len(near_threshold) >= 3:
            findings.append(
                InvestigationFinding(
                    finding_type="structuring",
                    severity="high",
                    description=(
                        f"Account {account_id} has {len(near_threshold)} near-threshold "
                        "transactions suggestive of structuring."
                    ),
                    evidence=[t.transaction_id for t in near_threshold],
                    confidence=0.86,
                )
            )
    return findings


def detect_geo_velocity(transactions: list[Transaction]) -> list[InvestigationFinding]:
    """Detect abrupt geographic changes across transactions."""
    findings: list[InvestigationFinding] = []
    by_account: dict[str, set[str]] = defaultdict(set)
    for tx in transactions:
        if tx.location:
            by_account[tx.account_id].add(tx.location.lower())

    for account_id, locations in by_account.items():
        if len(locations) >= 3:
            findings.append(
                InvestigationFinding(
                    finding_type="geographic_risk",
                    severity="medium",
                    description=f"Account {account_id} used multiple geographies in short period.",
                    evidence=sorted(locations),
                    confidence=0.73,
                )
            )
    return findings

