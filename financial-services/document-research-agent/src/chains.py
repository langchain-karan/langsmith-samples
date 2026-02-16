"""LangChain v1 LCEL chains used by graph nodes."""

from __future__ import annotations

from typing import Any, Sequence

from pydantic import BaseModel, Field

from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda

from src.config import get_settings
from src.deepagents_adapter import deep_synthesis
from src.state import RetrievedChunk

try:
    from langchain_anthropic import ChatAnthropic
except Exception:  # pragma: no cover
    ChatAnthropic = None  # type: ignore[assignment]


class RouteDecision(BaseModel):
    """Structured route output from query classifier."""

    route: str = Field(
        description=(
            "One of: policy_lookup, regulatory_mapping, comparative_research, general_question"
        )
    )


def _keyword_route(payload: dict[str, Any]) -> RouteDecision:
    query = str(payload.get("query", "")).lower()
    if any(k in query for k in ["compare", "versus", "vs", "difference"]):
        return RouteDecision(route="comparative_research")
    if any(k in query for k in ["policy", "rule", "required", "requirement"]):
        return RouteDecision(route="policy_lookup")
    if any(k in query for k in ["regulation", "regulatory", "compliance"]):
        return RouteDecision(route="regulatory_mapping")
    return RouteDecision(route="general_question")


def build_route_chain() -> Runnable[dict[str, Any], RouteDecision]:
    """Build a v1 LCEL chain for route classification."""
    parser = PydanticOutputParser(pydantic_object=RouteDecision)
    settings = get_settings()

    if settings.llm_enabled and ChatAnthropic is not None:
        llm = ChatAnthropic(model=settings.anthropic_haiku_model, temperature=0)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Classify the query route. Return JSON only.\n{format_instructions}",
                ),
                ("human", "Query: {query}"),
            ]
        )
        return prompt.partial(format_instructions=parser.get_format_instructions()) | llm | parser

    return RunnableLambda(_keyword_route)


def _build_context(chunks: Sequence[RetrievedChunk]) -> str:
    return "\n\n".join(
        (
            f"[{idx}] {chunk.title} | {chunk.doc_type} | effective {chunk.effective_date}\n"
            f"Source: {chunk.source}\n"
            f"Excerpt: {chunk.text}"
        )
        for idx, chunk in enumerate(chunks, start=1)
    )


def _deterministic_synthesis(payload: dict[str, Any]) -> str:
    chunks: list[RetrievedChunk] = payload.get("chunks", [])
    if not chunks:
        return (
            "### Short Answer\n"
            "No relevant documents were retrieved for this query.\n\n"
            "### Supporting Details\n"
            "- Try adding more specific keywords (policy name, process step, threshold).\n\n"
            "### Caveats\n"
            "- The system cannot answer without matching source evidence."
        )

    top = chunks[:3]
    bullets = "\n".join(
        f"- [{idx}] {chunk.text[:220].rstrip()}..." for idx, chunk in enumerate(top, start=1)
    )
    return (
        "### Short Answer\n"
        "Based on the highest-relevance documents, the query is addressed by current internal policy guidance.\n\n"
        "### Supporting Details\n"
        f"{bullets}\n\n"
        "### Caveats\n"
        "- This fallback mode does not perform deep semantic reasoning.\n"
        "- Validate high-impact decisions through human review."
    )


def build_synthesis_chain() -> Runnable[dict[str, Any], str]:
    """Build LCEL chain for answer synthesis with robust fallback."""
    settings = get_settings()

    if settings.llm_enabled and ChatAnthropic is not None:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are a financial services document research assistant. "
                        "Only answer from provided context. Use concise markdown and citation markers [n]. "
                        "If evidence is incomplete, say so explicitly."
                    ),
                ),
                (
                    "human",
                    (
                        "Question: {query}\n\nContext:\n{context}\n\n"
                        "Return sections:\n"
                        "1) Short Answer\n2) Supporting Details\n3) Caveats\n"
                        "Use [n] citations tied to context blocks."
                    ),
                ),
            ]
        )
        def _invoke_with_fallback(payload: dict[str, Any]) -> str:
            chunks: list[RetrievedChunk] = payload.get("chunks", [])
            route = str(payload.get("route", "general_question"))
            # Use larger models for higher-complexity synthesis tasks.
            if route in {"comparative_research", "regulatory_mapping"} or len(chunks) >= 6:
                model_name = settings.anthropic_opus_model
            else:
                model_name = settings.anthropic_sonnet_model
            deep_output = deep_synthesis(
                query=str(payload.get("query", "")),
                route=route,
                chunks=chunks,
            )
            if deep_output:
                return deep_output
            try:
                llm = ChatAnthropic(model=model_name, temperature=0)
                core_chain = prompt | llm | StrOutputParser()
                return core_chain.invoke(
                    {"query": payload.get("query", ""), "context": _build_context(chunks)}
                )
            except Exception:
                return _deterministic_synthesis({"chunks": chunks})

        return RunnableLambda(_invoke_with_fallback)

    return RunnableLambda(_deterministic_synthesis)

