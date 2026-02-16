from pathlib import Path

from src.graph import build_graph
from src.state import ResearchState


def test_graph_produces_final_answer():
    docs_dir = Path(__file__).resolve().parents[1] / "sample_data" / "docs"
    app = build_graph(docs_dir=docs_dir)

    result = app.invoke(ResearchState(query="What are wire transfer exception requirements?"))
    result_dict = result.model_dump() if hasattr(result, "model_dump") else dict(result)

    assert result_dict.get("final_answer")
    assert "Citations" in result_dict["final_answer"]
    assert isinstance(result_dict.get("requires_human_review"), bool)

