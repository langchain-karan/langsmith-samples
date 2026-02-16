def evaluate_grounding(regulatory_refs: list[str]) -> float:
    return 1.0 if len(regulatory_refs) >= 1 else 0.0
