# Sample Data Guide

This directory contains synthetic fraud/AML data used by the demo workflow.

## Core files used by default run

- `input_events.json`: baseline high-risk transaction batch for `run_demo.py`
- `customers.json`: customer and counterparty KYC/risk profiles
- `transaction_history.json`: recent history used for enrichment heuristics
- `entity_graph.json`: relationship graph used for network-risk signals
- `sanctions_watchlist.json`: sanctions and PEP reference list

## Additional scenario inputs

You can run alternative datasets with:

```bash
python run_demo.py --input sample_data/scenarios/<file>.json --json
```

Available scenario files:

- `input_events_low_risk.json`: benign, low-velocity activity
- `input_events_structuring_network.json`: near-threshold structuring with dense network links
- `input_events_sanctions_exposure.json`: repeated exposure to sanctioned/high-risk counterparty
- `input_events_crypto_layering.json`: mixed crypto/internal/wire pattern suggesting layering

All scenario files use the same transaction schema as `input_events.json`.
