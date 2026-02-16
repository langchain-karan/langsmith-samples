# Financial Services Samples

Production-style AI workflow samples for banking, payments, insurance, and capital markets teams using LangSmith and LangChain.

## Status

This vertical now includes runnable samples for document intelligence and fraud/AML workflows.

## Shared Model Strategy

Both use cases follow the same model-sizing preference:

- Haiku for routing/fast classification
- Sonnet 4.5 as the default reasoning model
- Opus for highest-complexity/high-stakes synthesis
- Optional Deep Agents pattern for complex multi-step reasoning

## Available Sample

### Use Case 1: Document Research Agent

- Spec: `usecase1_document_research_agent.md`
- Implementation: `document-research-agent/`
- Run guide: `document-research-agent/README.md`

### Use Case 2: Fraud Detection & AML Agent

- Spec: `usecase2_fraud_detection_agent.md`
- Implementation: `fraud-detection-agent/`
- Run guide: `fraud-detection-agent/README.md`

## What to Expect

Additional planned solution patterns include:

- Customer operations copilots for service and case handling
- Risk and fraud investigation workflows with human-in-the-loop review
- Document and policy intelligence for underwriting and claims operations
- Multi-step decisioning pipelines with observability and auditability

## Design Goals

- Reliability and traceability for regulated environments
- Clear handoffs between automated actions and human approvals
- Reusable orchestration patterns that can be adapted by domain

## Getting Started

- Start with this vertical's samples:
  - `document-research-agent/README.md`
  - `fraud-detection-agent/README.md`
- Return to the repository root to view all domain samples: `../README.md`

