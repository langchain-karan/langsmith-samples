"""CLI entrypoint for the document research agent demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.graph import build_graph
from src.state import ResearchState


def _to_jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return _to_jsonable(value.model_dump())
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Financial Services Document Research Agent")
    parser.add_argument(
        "--query",
        required=True,
        help="Natural language question, e.g. 'What changed in KYC requirements in v3.2?'",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full final state JSON instead of markdown report only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    docs_dir = Path(__file__).parent / "sample_data" / "docs"
    app = build_graph(docs_dir=docs_dir)

    initial_state = ResearchState(query=args.query)
    result = app.invoke(initial_state)
    result_dict = _to_jsonable(result.model_dump() if hasattr(result, "model_dump") else dict(result))

    if args.json:
        print(json.dumps(result_dict, indent=2))
    else:
        print(result_dict.get("final_answer", "No final answer produced."))


if __name__ == "__main__":
    main()

