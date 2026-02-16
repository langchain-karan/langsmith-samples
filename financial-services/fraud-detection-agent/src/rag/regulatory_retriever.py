"""Regulatory reference retriever."""

from __future__ import annotations


def get_relevant_regulations(risk_tier: str, finding_types: list[str]) -> list[str]:
    refs = ["31 CFR 1020.320 - SAR filing requirements", "BSA Recordkeeping Rule"]
    if "structuring" in finding_types:
        refs.append("31 USC 5324 - Structuring transactions to evade reporting")
    if risk_tier in {"high", "critical"}:
        refs.append("FFIEC BSA/AML Manual - Suspicious Activity Monitoring")
    return refs

