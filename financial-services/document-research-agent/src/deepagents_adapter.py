"""Optional Deep Agents integration for complex document research synthesis."""

from __future__ import annotations

from typing import Any

from src.config import get_settings
from src.state import RetrievedChunk

try:
    from deepagents import create_deep_agent
except Exception:  # pragma: no cover
    create_deep_agent = None  # type: ignore[assignment]


def _chunk_context(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(
        (
            f"[{idx}] {chunk.title} | {chunk.doc_type} | effective {chunk.effective_date}\n"
            f"Source: {chunk.source}\n"
            f"Excerpt: {chunk.text}"
        )
        for idx, chunk in enumerate(chunks, start=1)
    )


def _extract_text(response: Any) -> str:
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        messages = response.get("messages")
        if isinstance(messages, list) and messages:
            last = messages[-1]
            if isinstance(last, dict) and isinstance(last.get("content"), str):
                return last["content"]
        return str(response)
    return str(response)


def deep_synthesis(query: str, route: str, chunks: list[RetrievedChunk]) -> str | None:
    """Run optional deep-agents synthesis for high-context complex routes."""
    settings = get_settings()
    if not settings.enable_deepagents or not settings.llm_enabled or create_deep_agent is None:
        return None

    # Favor Opus for complex comparative/regulatory analysis, Sonnet otherwise.
    model = (
        settings.anthropic_opus_model
        if route in {"comparative_research", "regulatory_mapping"} or len(chunks) >= 6
        else settings.anthropic_sonnet_model
    )
    try:
        def get_answer_format() -> str:
            """Provide the required answer format."""
            return "Sections: Short Answer, Supporting Details, Caveats, with [n] citations."

        agent = create_deep_agent(
            tools=[get_answer_format],
            model=model,
            system_prompt=(
                "You are a financial services document research specialist. "
                "Use only provided context, provide grounded output, and cite evidence markers [n]."
            ),
        )
        response = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"Route: {route}\n"
                            f"Question: {query}\n\n"
                            f"Context:\n{_chunk_context(chunks)}\n\n"
                            "Return markdown with sections:\n"
                            "1) Short Answer\n2) Supporting Details\n3) Caveats\n"
                            "Use [n] citations tied to context blocks."
                        ),
                    }
                ]
            }
        )
        return _extract_text(response)
    except Exception:
        return None

