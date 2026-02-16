<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/LangSmith-logo-white.png">
    <source media="(prefers-color-scheme: light)" srcset="assets/LangSmith-logo.png">
    <img src="assets/LangSmith-logo.png" alt="LangSmith" height="36" style="vertical-align: middle;">
  </picture>
</h1>
<p align="center">Solution samples for industry verticals</p>

<p align="center">
  <a href="./LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://github.com/langchain-karan/langsmith-samples/stargazers"><img src="https://img.shields.io/github/stars/langchain-karan/langsmith-samples?style=social" alt="GitHub stars"></a>
  <a href="https://github.com/langchain-karan/langsmith-samples/commits/main"><img src="https://img.shields.io/github/commit-activity/m/langchain-karan/langsmith-samples" alt="Commit activity"></a>
  <a href="https://github.com/langchain-karan/langsmith-samples/pulls"><img src="https://img.shields.io/github/issues-pr/langchain-karan/langsmith-samples" alt="Pull requests"></a>
  <a href="https://x.com/LangChainAI"><img src="https://img.shields.io/twitter/follow/LangChainAI?style=social" alt="Follow LangChain"></a>
</p>

<p align="center">Production-style sample projects that show how to build, operate, and evaluate AI workflows with LangSmith and LangChain.</p>

This repository is built for teams moving from prototype to production. Each sample models a real business workflow and demonstrates implementation patterns you can adapt to your own stack.

## Documentation & SDK Links

- LangChain docs: https://docs.langchain.com/oss/python/langchain/
- LangChain Python API reference: https://reference.langchain.com/python/
- LangGraph docs: https://docs.langchain.com/oss/python/langgraph/
- Deep Agents overview: https://docs.langchain.com/oss/python/deepagents/overview
- Deep Agents Python API reference: https://reference.langchain.com/python/deepagents/
- LangSmith docs: https://docs.smith.langchain.com/

## Who This Is For

- **Engineering leaders** evaluating production patterns for AI systems
- **Product and platform teams** aligning AI features with business KPIs
- **Developers** who want runnable, domain-oriented reference implementations

## What You Will Find Here

- End-to-end sample applications by industry/domain
- LangGraph and LangChain orchestration patterns
- Optional LangSmith tracing and observability setup
- Runnable example code with local development instructions

## Why These Samples Matter

- Reduce time-to-first-production workflow with proven project scaffolding
- Improve reliability with traceable execution and debugging via LangSmith
- Demonstrate reusable patterns for multi-step and multi-agent systems
- Connect technical implementation to business outcomes (speed, quality, cost)

## Repository Layout

```text
langsmith-samples/
├── supply-chain/
│   └── supply-chain-optimization/   # Available now
├── financial-services/              # Available now
│   ├── document-research-agent/
│   └── fraud-detection-agent/
├── healthcare-life-sciences/        # Coming soon
└── retail-ecommerce/                # Coming soon
```

## Samples

| Domain | Sample | Status | Description |
| --- | --- | --- | --- |
| Supply Chain | `supply-chain/supply-chain-optimization` | Available | Multi-agent optimization for demand forecasting, inventory, and logistics coordination. |
| Financial Services | `financial-services/document-research-agent`, `financial-services/fraud-detection-agent` | Available | Document research plus fraud detection/AML workflows with AWS mapping, Deep Agents options, risk scoring, and escalation. |
| Healthcare & Life Sciences | - | Coming soon | Planned examples for care operations and decision support workflows. |
| Retail & Ecommerce | - | Coming soon | Planned examples for planning, merchandising, and fulfillment workflows. |

## Quick Start

### Prerequisites

- Python 3.10 or newer
- `pip`
- An LLM API key (current sample uses Anthropic)
- (Optional but recommended) LangSmith API key for traces

### Run the Supply Chain sample

```bash
cd supply-chain/supply-chain-optimization
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Add credentials to `.env`, then run:

```bash
python example.py
```

For sample-specific details, see `supply-chain/supply-chain-optimization/README.md`.

### Run Financial Services samples

```bash
# Use Case 1
cd financial-services/document-research-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python run_demo.py --query "What changed in KYC onboarding requirements between 3.1 and 3.2?"

# Use Case 2
cd ../fraud-detection-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python run_demo.py
```

## What You Can Learn from the Current Sample

From `supply-chain/supply-chain-optimization`, you can learn how to:

- Coordinate specialized agents through an orchestration layer
- Model domain state with typed schemas and structured outputs
- Incorporate external signals into planning and decisioning
- Stream workflow execution step-by-step for real-time visibility
- Instrument end-to-end traces for operational monitoring

## Configure LangSmith Tracing (Recommended)

Set these environment variables in your `.env` file to capture traces:

```env
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=your_project_name
```

Learn more:

- LangSmith docs: https://docs.smith.langchain.com/
- LangSmith product overview: https://www.langchain.com/langsmith

## How to Use These Samples

- Start with one sample and run it unmodified
- Map the sample architecture to your own system boundaries
- Review how state, tools, and orchestration are modeled
- Enable tracing and inspect execution runs in LangSmith
- Adapt schemas, prompts, and routing logic to your domain data
- Define success metrics before extending a sample in production

## Contributing

Issues and pull requests are welcome. If you are proposing a new sample, include:

- The business problem and target user
- Architecture overview and key design decisions
- Setup instructions and runnable example inputs
- Expected outputs and observability/tracing guidance

## License

This repository is licensed under the MIT License. See `LICENSE`.
