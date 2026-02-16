"""Fraud typology retriever."""

from __future__ import annotations


def get_typology_hints(finding_types: list[str]) -> list[str]:
    hints: list[str] = []
    if "structuring" in finding_types:
        hints.append("Structuring: repeated near-threshold cash/wire movement.")
    if "network_anomaly" in finding_types:
        hints.append("Layering: rapid transfer chains through multiple counterparties.")
    if "geographic_risk" in finding_types:
        hints.append("Geographic risk: activity across high-risk jurisdictions.")
    if not hints:
        hints.append("General anomaly: monitor for repeat behavior and escalation signals.")
    return hints

