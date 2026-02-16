"""Backward-compatible wrappers around LCEL synthesis chains."""

from __future__ import annotations

from typing import Sequence

from src.chains import build_synthesis_chain
from src.state import RetrievedChunk


def generate_answer(query: str, chunks: Sequence[RetrievedChunk]) -> str:
    """Generate a cited answer through the v1 LCEL synthesis chain."""
    chain = build_synthesis_chain()
    return chain.invoke({"query": query, "chunks": list(chunks)})

