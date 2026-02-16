"""Notification helpers for escalation actions."""

from __future__ import annotations


def send_alert(channel: str, title: str, body: str) -> str:
    """Return simulated alert id for demo mode."""
    alert_id = f"alert::{title.lower().replace(' ', '-')}"
    if channel == "console":
        print(f"[ALERT] {title}\n{body}\n")
    return alert_id

