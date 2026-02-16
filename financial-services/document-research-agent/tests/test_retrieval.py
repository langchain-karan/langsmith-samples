from pathlib import Path

from src.retrieval import LocalDocumentRetriever


def test_retrieval_returns_ranked_chunks():
    docs_dir = Path(__file__).resolve().parents[1] / "sample_data" / "docs"
    retriever = LocalDocumentRetriever(docs_dir=docs_dir)
    results = retriever.search("What is required before escalating an AML alert?", top_k=3)

    assert results
    assert all(results[i].relevance_score >= results[i + 1].relevance_score for i in range(len(results) - 1))

