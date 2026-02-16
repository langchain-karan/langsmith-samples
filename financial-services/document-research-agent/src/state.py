"""State and schema definitions for the document research graph."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Citation(BaseModel):
    """Citation metadata attached to generated responses."""

    chunk_id: str
    title: str
    source: str
    doc_type: str
    effective_date: str
    relevance_score: float


class RetrievedChunk(BaseModel):
    """A single retrieved chunk returned by the retriever."""

    chunk_id: str
    title: str
    source: str
    doc_type: str
    effective_date: str
    text: str
    relevance_score: float


class ResearchState(BaseModel):
    """Complete state passed between graph nodes."""

    query: str
    filters: dict[str, Any] = Field(default_factory=dict)
    route: str = "general_question"

    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)

    draft_answer: str = ""
    final_answer: str = ""
    confidence: float = 0.0
    requires_human_review: bool = False
    next_steps: list[str] = Field(default_factory=list)

    error: str = ""

