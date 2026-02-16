"""LangGraph workflow for the financial document research agent."""

from __future__ import annotations

from pathlib import Path

from langgraph.graph import END, START, StateGraph

from src.chains import build_route_chain, build_synthesis_chain
from src.retrieval import LocalDocumentRetriever
from src.state import Citation, ResearchState

def _intake_node_factory():
    route_chain = build_route_chain()

    def _intake_node(state: ResearchState) -> ResearchState:
        decision = route_chain.invoke({"query": state.query})
        state.route = decision.route
        return state

    return _intake_node


def _retrieval_node_factory(retriever: LocalDocumentRetriever):
    def _retrieval_node(state: ResearchState) -> ResearchState:
        state.retrieved_chunks = retriever.search(state.query, top_k=6)
        state.citations = [
            Citation(
                chunk_id=chunk.chunk_id,
                title=chunk.title,
                source=chunk.source,
                doc_type=chunk.doc_type,
                effective_date=chunk.effective_date,
                relevance_score=chunk.relevance_score,
            )
            for chunk in state.retrieved_chunks
        ]
        return state

    return _retrieval_node


def _synthesis_node_factory():
    synthesis_chain = build_synthesis_chain()

    def _synthesis_node(state: ResearchState) -> ResearchState:
        state.draft_answer = synthesis_chain.invoke(
            {"query": state.query, "chunks": state.retrieved_chunks, "route": state.route}
        )
        return state

    return _synthesis_node


def _quality_node(state: ResearchState) -> ResearchState:
    chunk_count = len(state.retrieved_chunks)
    avg_score = (
        sum(c.relevance_score for c in state.citations) / max(len(state.citations), 1)
        if state.citations
        else 0.0
    )

    # Simple confidence heuristic designed for transparency.
    confidence = min(1.0, round((chunk_count / 6) * 0.4 + avg_score * 0.6, 3))
    state.confidence = confidence
    state.requires_human_review = confidence < 0.55 or chunk_count < 2
    return state


def _review_actions_node(state: ResearchState) -> ResearchState:
    state.next_steps = [
        "Escalate to compliance/policy reviewer before operational use.",
        "Request narrower query context (product, region, effective date).",
        "Confirm source recency and version ownership.",
    ]
    return state


def _publish_actions_node(state: ResearchState) -> ResearchState:
    state.next_steps = [
        "Proceed with analyst validation for business sign-off.",
        "Log decision and citations in case management notes.",
    ]
    return state


def _post_quality_route(state: ResearchState) -> str:
    return "review_actions" if state.requires_human_review else "publish_actions"


def _finalize_node(state: ResearchState) -> ResearchState:
    citations_markdown = "\n".join(
        (
            f"- `{idx}` {c.title} ({c.doc_type}, effective {c.effective_date}) "
            f"- {c.source} [score={c.relevance_score:.2f}]"
        )
        for idx, c in enumerate(state.citations, start=1)
    )
    if not citations_markdown:
        citations_markdown = "- No citations available."

    next_steps_markdown = "\n".join(f"- {step}" for step in state.next_steps)

    state.final_answer = (
        f"# Document Research Response\n\n"
        f"**Query Type:** `{state.route}`\n"
        f"**Confidence:** `{state.confidence:.2f}`\n"
        f"**Human Review Required:** `{str(state.requires_human_review).lower()}`\n\n"
        f"{state.draft_answer}\n\n"
        f"## Citations\n"
        f"{citations_markdown}\n\n"
        f"## Recommended Next Steps\n"
        f"{next_steps_markdown}\n"
    )
    return state


def build_graph(docs_dir: Path):
    """Create and compile the research workflow graph."""
    retriever = LocalDocumentRetriever(docs_dir=docs_dir)
    graph = StateGraph(ResearchState)
    graph.add_node("intake", _intake_node_factory())
    graph.add_node("retrieve", _retrieval_node_factory(retriever))
    graph.add_node("synthesize", _synthesis_node_factory())
    graph.add_node("quality_gate", _quality_node)
    graph.add_node("review_actions", _review_actions_node)
    graph.add_node("publish_actions", _publish_actions_node)
    graph.add_node("finalize", _finalize_node)

    graph.add_edge(START, "intake")
    graph.add_edge("intake", "retrieve")
    graph.add_edge("retrieve", "synthesize")
    graph.add_edge("synthesize", "quality_gate")
    graph.add_conditional_edges(
        "quality_gate",
        _post_quality_route,
        {"review_actions": "review_actions", "publish_actions": "publish_actions"},
    )
    graph.add_edge("review_actions", "finalize")
    graph.add_edge("publish_actions", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile()

