def evaluate_detection(results: dict) -> float:
    return 1.0 if results.get("disposition") in {"sar_filed", "investigation_open", "monitoring"} else 0.0
