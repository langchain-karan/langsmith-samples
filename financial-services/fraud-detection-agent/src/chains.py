"""LangChain LCEL chains for fraud workflow narratives."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda

from src.config import get_settings

try:
    from langchain_anthropic import ChatAnthropic
except Exception:  # pragma: no cover
    ChatAnthropic = None  # type: ignore[assignment]


class EscalationDecision(BaseModel):
    """Structured escalation decision output."""

    disposition: str = Field(
        description="One of auto_cleared, monitoring, investigation_open, sar_filed, account_frozen, law_enforcement_referral"
    )
    requires_human_review: bool
    escalation_reason: str
    assigned_investigator: str


def build_sar_narrative_chain() -> Runnable[dict[str, Any], str]:
    """Create an LCEL chain for SAR narrative drafting."""
    settings = get_settings()
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are an AML compliance analyst. Draft concise SAR narrative text from evidence only. "
                    "Do not hallucinate missing facts."
                ),
            ),
            (
                "human",
                (
                    "Case ID: {case_id}\n"
                    "Risk tier: {risk_tier}\n"
                    "Findings: {findings}\n"
                    "Total amount: {total_amount}\n"
                    "Return 4-7 sentences in formal compliance style."
                ),
            ),
        ]
    )

    if settings.llm_enabled and ChatAnthropic is not None:
        def _invoke(payload: dict[str, Any]) -> str:
            risk_tier = str(payload.get("risk_tier", "low")).lower()
            model_name = (
                settings.anthropic_opus_model if risk_tier in {"high", "critical"} else settings.anthropic_sonnet_model
            )
            try:
                llm = ChatAnthropic(model=model_name, temperature=0)
                base_chain = prompt | llm | StrOutputParser()
                return base_chain.invoke(payload)
            except Exception:
                return _fallback_narrative(payload)

        return RunnableLambda(_invoke)

    return RunnableLambda(_fallback_narrative)


def _fallback_narrative(payload: dict[str, Any]) -> str:
    findings = payload.get("findings", "No findings provided")
    return (
        f"Case {payload.get('case_id', 'N/A')} was escalated for AML review based on "
        f"{payload.get('risk_tier', 'unknown')} risk indicators. "
        f"Investigative evidence indicates: {findings}. "
        f"Observed activity totaled ${payload.get('total_amount', 0):,.2f}. "
        "The pattern is consistent with suspicious activity typologies and merits analyst validation. "
        "This draft should be reviewed and finalized by an authorized compliance officer."
    )


def build_escalation_chain() -> Runnable[dict[str, Any], EscalationDecision]:
    """Create Haiku-based fast escalation classifier with deterministic fallback."""
    settings = get_settings()
    parser = PydanticOutputParser(pydantic_object=EscalationDecision)

    if settings.llm_enabled and ChatAnthropic is not None:
        llm = ChatAnthropic(model=settings.anthropic_haiku_model, temperature=0)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "Classify AML case disposition from risk and recommendation. "
                        "Return JSON only.\n{format_instructions}"
                    ),
                ),
                (
                    "human",
                    (
                        "risk_tier={risk_tier}\n"
                        "filing_recommendation={filing_recommendation}\n"
                        "finding_count={finding_count}"
                    ),
                ),
            ]
        )
        chain = prompt.partial(format_instructions=parser.get_format_instructions()) | llm | parser

        def _invoke(payload: dict[str, Any]) -> EscalationDecision:
            try:
                return chain.invoke(payload)
            except Exception:
                return _fallback_escalation(payload)

        return RunnableLambda(_invoke)

    return RunnableLambda(_fallback_escalation)


def _fallback_escalation(payload: dict[str, Any]) -> EscalationDecision:
    risk_tier = str(payload.get("risk_tier", "low")).lower()
    filing = str(payload.get("filing_recommendation", "needs_review")).lower()
    if risk_tier in {"high", "critical"} or filing == "file":
        return EscalationDecision(
            disposition="sar_filed",
            requires_human_review=True,
            escalation_reason="High-risk case requires compliance officer approval.",
            assigned_investigator="aml_queue_tier2",
        )
    if risk_tier == "medium":
        return EscalationDecision(
            disposition="investigation_open",
            requires_human_review=True,
            escalation_reason="Medium-risk case routed for investigator triage.",
            assigned_investigator="aml_queue_tier1",
        )
    return EscalationDecision(
        disposition="monitoring",
        requires_human_review=False,
        escalation_reason="No immediate filing action required; continue monitoring.",
        assigned_investigator="",
    )

