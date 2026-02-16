"""Document loading and retrieval utilities."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document as LCDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.state import RetrievedChunk


@dataclass(frozen=True)
class Document:
    """Represents one source document with metadata."""

    doc_id: str
    title: str
    source: str
    doc_type: str
    effective_date: str
    body: str


def _tokenize(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-zA-Z0-9]{3,}", text.lower())}


def _split_metadata(raw: str) -> tuple[dict[str, str], str]:
    header, _, body = raw.partition("\n---\n")
    meta: dict[str, str] = {}
    for line in header.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        meta[key.strip().lower()] = value.strip()
    return meta, body.strip()


class LocalDocumentRetriever:
    """LangChain-style local retriever over markdown policy documents."""

    def __init__(self, docs_dir: Path):
        self.docs_dir = docs_dir
        self.documents = self._load_documents()
        self.chunks = self._build_chunks()
        self.chunk_documents = self._build_langchain_documents()

    def _load_documents(self) -> list[Document]:
        documents: list[Document] = []
        for path in sorted(self.docs_dir.glob("*.md")):
            raw = path.read_text(encoding="utf-8")
            meta, body = _split_metadata(raw)
            documents.append(
                Document(
                    doc_id=path.stem,
                    title=meta.get("title", path.stem.replace("_", " ").title()),
                    source=meta.get("source", str(path.name)),
                    doc_type=meta.get("doctype", "other"),
                    effective_date=meta.get("effectivedate", "unknown"),
                    body=body,
                )
            )
        return documents

    def _build_chunks(self) -> list[RetrievedChunk]:
        chunks: list[RetrievedChunk] = []
        for doc in self.documents:
            paragraphs = [p.strip() for p in doc.body.split("\n\n") if p.strip()]
            for idx, paragraph in enumerate(paragraphs, start=1):
                chunks.append(
                    RetrievedChunk(
                        chunk_id=f"{doc.doc_id}-p{idx}",
                        title=doc.title,
                        source=doc.source,
                        doc_type=doc.doc_type,
                        effective_date=doc.effective_date,
                        text=paragraph,
                        relevance_score=0.0,
                    )
                )
        return chunks

    def _build_langchain_documents(self) -> list[LCDocument]:
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
        docs: list[LCDocument] = []

        for doc in self.documents:
            base_doc = LCDocument(
                page_content=doc.body,
                metadata={
                    "doc_id": doc.doc_id,
                    "title": doc.title,
                    "source": doc.source,
                    "doc_type": doc.doc_type,
                    "effective_date": doc.effective_date,
                },
            )
            split_docs = splitter.split_documents([base_doc])
            for idx, split_doc in enumerate(split_docs, start=1):
                split_doc.metadata["chunk_seq"] = idx
                docs.append(split_doc)

        return docs

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        scored: list[RetrievedChunk] = []
        for chunk_doc in self.chunk_documents:
            chunk_tokens = _tokenize(chunk_doc.page_content)
            overlap = len(query_tokens.intersection(chunk_tokens))
            lexical_score = overlap / max(len(query_tokens), 1)

            # Prefer policy/procedure chunks for procedural questions.
            boost = 0.0
            if any(word in query.lower() for word in ["policy", "procedure", "required", "rule"]):
                if chunk_doc.metadata.get("doc_type", "other") in {"policy", "procedure", "guidance"}:
                    boost += 0.15

            if lexical_score <= 0 and boost <= 0:
                continue

            scored_score = round(min(1.0, lexical_score + boost), 4)
            scored.append(
                RetrievedChunk(
                    chunk_id=f"{chunk_doc.metadata.get('doc_id', 'doc')}-c{chunk_doc.metadata.get('chunk_seq', 0)}",
                    title=str(chunk_doc.metadata.get("title", "Unknown Document")),
                    source=str(chunk_doc.metadata.get("source", "unknown")),
                    doc_type=str(chunk_doc.metadata.get("doc_type", "other")),
                    effective_date=str(chunk_doc.metadata.get("effective_date", "unknown")),
                    text=chunk_doc.page_content.strip(),
                    relevance_score=scored_score,
                )
            )

        scored.sort(key=lambda c: c.relevance_score, reverse=True)
        return scored[:top_k]

