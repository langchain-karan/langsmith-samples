from datetime import datetime

from src.models.pattern_detector import detect_structuring
from src.state import Transaction


def test_detect_structuring_flags_multiple_near_threshold_transfers():
    txs = [
        Transaction(
            transaction_id="t1",
            account_id="A1",
            amount=9600,
            timestamp=datetime.fromisoformat("2026-02-14T09:00:00"),
        ),
        Transaction(
            transaction_id="t2",
            account_id="A1",
            amount=9700,
            timestamp=datetime.fromisoformat("2026-02-14T09:05:00"),
        ),
        Transaction(
            transaction_id="t3",
            account_id="A1",
            amount=9800,
            timestamp=datetime.fromisoformat("2026-02-14T09:10:00"),
        ),
    ]

    findings = detect_structuring(txs)
    assert findings
    assert findings[0].finding_type == "structuring"

