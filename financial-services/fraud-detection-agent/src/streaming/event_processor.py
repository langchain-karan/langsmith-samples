"""Event transformation helpers."""

from __future__ import annotations

from datetime import datetime

from src.state import Transaction


def event_to_transaction(event: dict) -> Transaction:
    """Transform dict event into typed transaction model."""
    return Transaction(
        transaction_id=event["transaction_id"],
        account_id=event["account_id"],
        counterparty_id=event.get("counterparty_id"),
        amount=float(event["amount"]),
        currency=event.get("currency", "USD"),
        transaction_type=event.get("transaction_type", "wire"),
        timestamp=datetime.fromisoformat(event["timestamp"]),
        channel=event.get("channel", ""),
        location=event.get("location", ""),
        metadata=event.get("metadata", {}),
    )

