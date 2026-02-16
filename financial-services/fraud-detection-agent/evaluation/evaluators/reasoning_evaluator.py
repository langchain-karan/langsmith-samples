def evaluate_reasoning(findings: list[dict]) -> float:
    return 1.0 if len(findings) >= 1 else 0.0
