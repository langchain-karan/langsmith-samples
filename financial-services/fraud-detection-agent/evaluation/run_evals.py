"""Simple offline evaluation runner."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent))

from evaluators.detection_evaluator import evaluate_detection
from evaluators.grounding_evaluator import evaluate_grounding
from evaluators.reasoning_evaluator import evaluate_reasoning
from evaluators.sar_evaluator import evaluate_sar_quality


def main() -> None:
    base = Path(__file__).resolve().parent / "datasets"
    _ = json.loads((base / "fraud_detection.json").read_text(encoding="utf-8"))

    # Placeholder metrics for local quick checks.
    sample_result = {
        "disposition": "sar_filed",
        "investigation_findings": [{"finding_type": "structuring"}],
        "regulatory_references": ["31 CFR 1020.320 - SAR filing requirements"],
        "sar_narrative": "Suspicious activity requires review and filing consideration.",
    }
    metrics = {
        "detection_score": evaluate_detection(sample_result),
        "reasoning_score": evaluate_reasoning(sample_result["investigation_findings"]),
        "grounding_score": evaluate_grounding(sample_result["regulatory_references"]),
        "sar_quality_score": evaluate_sar_quality(sample_result["sar_narrative"]),
    }
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

