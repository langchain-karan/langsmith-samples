import json
from pathlib import Path

from src.graph import build_graph
from src.state import FraudDetectionState, Transaction


def test_pipeline_end_to_end_produces_disposition():
    input_path = Path(__file__).resolve().parents[2] / "sample_data" / "input_events.json"
    rows = json.loads(input_path.read_text(encoding="utf-8"))
    txs = [Transaction(**row) for row in rows]

    app = build_graph()
    result = app.invoke(FraudDetectionState(transactions=txs))
    result_dict = result.model_dump() if hasattr(result, "model_dump") else dict(result)

    assert result_dict.get("case_id")
    assert result_dict.get("risk_tier") in {"low", "medium", "high", "critical"}
    assert result_dict.get("disposition")
    assert result_dict.get("sar_draft") is not None

