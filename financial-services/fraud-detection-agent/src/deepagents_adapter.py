"""Optional Deep Agents integration for complex fraud tasks."""

from __future__ import annotations

from typing import Any

from src.config import get_settings
from src.state import InvestigationFinding

try:
    from deepagents import create_deep_agent
except Exception:  # pragma: no cover - optional dependency behavior.
    create_deep_agent = None  # type: ignore[assignment]


def _format_findings(findings: list[InvestigationFinding]) -> str:
    if not findings:
        return "No findings available."
    return "; ".join(f"{f.finding_type}({f.severity}): {f.description}" for f in findings[:8])


def _base_tools() -> list[Any]:
    def get_investigation_policy() -> str:
        """Get internal investigation policy."""
        return (
            "Investigations should prioritize objective evidence, typology mapping, and explicit caveats. "
            "Escalate when evidence indicates potential structuring, layering, sanctions risk, or repeated anomalies."
        )

    def get_regulatory_summary() -> str:
        """Get concise regulatory baseline summary."""
        return (
            "Relevant references include BSA suspicious activity monitoring and 31 CFR 1020.320 "
            "for SAR expectations."
        )

    return [get_investigation_policy, get_regulatory_summary]


def _extract_agent_text(response: Any) -> str:
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        messages = response.get("messages")
        if isinstance(messages, list) and messages:
            last = messages[-1]
            if isinstance(last, dict):
                content = last.get("content")
                if isinstance(content, str):
                    return content
        return str(response)
    return str(response)


def deep_investigation_summary(
    case_id: str,
    risk_tier: str,
    findings: list[InvestigationFinding],
    network_analysis: str,
) -> str | None:
    """Return Deep Agents-authored investigation summary, if enabled."""
    settings = get_settings()
    if not settings.enable_deepagents or not settings.llm_enabled or create_deep_agent is None:
        return None

    try:
        agent = create_deep_agent(
            tools=_base_tools(),
            model=settings.anthropic_sonnet_model,
            system_prompt=(
                "You are a financial crimes investigation specialist. "
                "Synthesize findings with clear risk reasoning and caveats."
            ),
        )
        response = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"Case ID: {case_id}\n"
                            f"Risk tier: {risk_tier}\n"
                            f"Findings: {_format_findings(findings)}\n"
                            f"Network analysis: {network_analysis}\n"
                            "Produce a concise 4-6 sentence investigation summary."
                        ),
                    }
                ]
            }
        )
        return _extract_agent_text(response)
    except Exception:
        return None


def deep_sar_refinement(
    case_id: str,
    risk_tier: str,
    sar_narrative: str,
    regulatory_references: list[str],
) -> str | None:
    """Return Deep Agents-refined SAR narrative, if enabled."""
    settings = get_settings()
    if not settings.enable_deepagents or not settings.llm_enabled or create_deep_agent is None:
        return None

    model_name = (
        settings.anthropic_opus_model if risk_tier.lower() in {"high", "critical"} else settings.anthropic_sonnet_model
    )
    try:
        agent = create_deep_agent(
            tools=_base_tools(),
            model=model_name,
            system_prompt=(
                "You are a senior AML compliance reviewer. "
                "Refine SAR narratives for clarity, factual grounding, and regulatory completeness."
            ),
        )
        response = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"Case ID: {case_id}\n"
                            f"Risk tier: {risk_tier}\n"
                            f"Regulatory references: {', '.join(regulatory_references)}\n"
                            f"Draft narrative: {sar_narrative}\n"
                            "Refine this into a concise, grounded SAR narrative with explicit caveats."
                        ),
                    }
                ]
            }
        )
        return _extract_agent_text(response)
    except Exception:
        return None

