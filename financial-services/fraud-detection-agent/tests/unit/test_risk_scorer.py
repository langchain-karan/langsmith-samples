from src.models.risk_scorer import compute_risk_score, score_to_tier
from src.state import AnomalyScore, CustomerProfile, InvestigationFinding


def test_compute_risk_score_high_case():
    score = compute_risk_score(
        anomaly_scores=[AnomalyScore(transaction_id="t1", score=0.85)],
        customer_profile=CustomerProfile(
            customer_id="c1",
            name="Example",
            account_type="business",
            risk_rating="pep",
            previous_sars=1,
        ),
        findings=[
            InvestigationFinding(
                finding_type="structuring",
                description="Repeated near-threshold tx",
                severity="high",
                confidence=0.9,
            )
        ],
    )
    assert score >= 0.8
    assert score_to_tier(score) in {"high", "critical"}

