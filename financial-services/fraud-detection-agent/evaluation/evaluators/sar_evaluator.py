def evaluate_sar_quality(narrative: str) -> float:
    checks = ["suspicious" in narrative.lower(), "review" in narrative.lower()]
    return sum(checks) / len(checks)
