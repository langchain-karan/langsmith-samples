"""CLI demo for fraud detection pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.graph import build_graph
from src.state import FraudDetectionState, Transaction


def _to_jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return _to_jsonable(value.model_dump())
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    return value


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fraud detection + AML workflow demo")
    p.add_argument(
        "--input",
        default="sample_data/input_events.json",
        help="Path to JSON array of transaction events",
    )
    p.add_argument("--json", action="store_true", help="Print full JSON state output")
    return p.parse_args()


def load_transactions(path: Path) -> list[Transaction]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [Transaction(**row) for row in rows]


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    txs = load_transactions(input_path)

    app = build_graph()
    result = app.invoke(FraudDetectionState(transactions=txs))
    result_dict = _to_jsonable(result.model_dump() if hasattr(result, "model_dump") else dict(result))

    if args.json:
        print(json.dumps(result_dict, indent=2))
        return

    print("# Fraud Detection Case Result")
    print(f"- case_id: {result_dict.get('case_id')}")
    print(f"- risk_tier: {result_dict.get('risk_tier')}")
    print(f"- risk_score: {result_dict.get('risk_score')}")
    print(f"- disposition: {result_dict.get('disposition')}")
    print(f"- requires_human_review: {result_dict.get('requires_human_review')}")
    print(f"- findings: {len(result_dict.get('investigation_findings', []))}")
    print("\n## Compliance determination")
    print(result_dict.get("compliance_determination", ""))
    if result_dict.get("sar_draft"):
        print("\n## SAR narrative")
        print(result_dict["sar_draft"]["narrative"])


if __name__ == "__main__":
    main()

