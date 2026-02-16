# Supply Chain Samples

Production-style AI workflow samples for supply chain teams using LangSmith and LangChain.

## Available Sample

### Supply Chain Optimization

Path: `supply-chain-optimization`

This sample demonstrates a multi-agent workflow for:

- Demand forecasting
- Inventory optimization
- Logistics routing and disruption handling
- End-to-end orchestration with optional LangSmith tracing

## Run This Sample

```bash
cd supply-chain-optimization
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python example.py
```

For architecture and implementation details, see `supply-chain-optimization/README.md` and `supply-chain-optimization/ARCHITECTURE.md`.

