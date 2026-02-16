# Use Case 2: Autonomous Fraud Detection & AML Agent

## Overview

Build a multi-agent system for real-time transaction monitoring, fraud detection, anti-money laundering (AML) investigation, and Suspicious Activity Report (SAR) generation. The system uses a **sequential workflow pattern** for the primary detection-to-reporting pipeline, combined with a **swarm pattern** for complex multi-dimensional investigations. All workflows produce complete audit trails for BSA/AML regulatory compliance.

---

## Project Structure

```
fraud-detection-agent/
├── pyproject.toml
├── langgraph.json
├── .env.example
├── README.md
├── src/
│   ├── __init__.py
│   ├── graph.py                      # Main LangGraph workflow
│   ├── subgraphs/
│   │   ├── __init__.py
│   │   └── investigation_swarm.py    # Swarm sub-graph for complex cases
│   ├── state.py                      # GraphState schema
│   ├── config.py                     # Configuration and AWS clients
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── transaction_monitor.py    # Real-time anomaly detection
│   │   ├── risk_enrichment.py        # Customer/counterparty enrichment
│   │   ├── investigation.py          # Deep-dive investigation agent
│   │   ├── compliance.py             # Regulatory compliance & SAR drafting
│   │   └── human_escalation.py       # Case routing and approval workflows
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── anomaly_detector.py       # SageMaker anomaly model endpoint
│   │   ├── sanctions_screening.py    # OFAC/sanctions list check
│   │   ├── graph_analysis.py         # Neptune entity relationship queries
│   │   ├── customer_lookup.py        # DynamoDB customer profile retrieval
│   │   ├── transaction_history.py    # OpenSearch transaction search
│   │   ├── sar_generator.py          # SAR template and narrative tools
│   │   └── notification.py           # SNS alert and case management
│   ├── models/
│   │   ├── __init__.py
│   │   ├── risk_scorer.py            # Risk scoring logic
│   │   └── pattern_detector.py       # Transaction pattern detection
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── regulatory_retriever.py   # BSA/AML regulatory knowledge base
│   │   └── typology_retriever.py     # Known fraud typology retriever
│   ├── streaming/
│   │   ├── __init__.py
│   │   ├── kinesis_consumer.py       # Kinesis stream consumer
│   │   └── event_processor.py        # Event-to-state transformer
│   └── prompts/
│       ├── monitor.py
│       ├── enrichment.py
│       ├── investigation.py
│       ├── compliance.py
│       └── escalation.py
├── evaluation/
│   ├── datasets/
│   │   ├── fraud_detection.json      # Labeled fraud/non-fraud transactions
│   │   ├── sar_quality.json          # SAR narrative quality dataset
│   │   ├── investigation_logic.json  # Investigation reasoning dataset
│   │   └── adversarial.json          # Red team adversarial patterns
│   ├── evaluators/
│   │   ├── detection_evaluator.py    # Precision/recall evaluator
│   │   ├── sar_evaluator.py          # LLM-as-judge SAR completeness
│   │   ├── reasoning_evaluator.py    # Investigation logic evaluator
│   │   ├── grounding_evaluator.py    # Regulatory citation evaluator
│   │   └── adversarial_runner.py     # Adversarial pattern test runner
│   └── run_evals.py
├── infrastructure/
│   ├── cdk/
│   │   ├── app.py
│   │   ├── stacks/
│   │   │   ├── streaming_stack.py    # Kinesis, Lambda consumers
│   │   │   ├── data_stack.py         # DynamoDB, OpenSearch, Neptune
│   │   │   ├── ml_stack.py           # SageMaker endpoints
│   │   │   ├── bedrock_stack.py      # Bedrock, Guardrails config
│   │   │   ├── security_stack.py     # VPC, KMS, IAM roles
│   │   │   └── agent_stack.py        # AgentCore runtime
│   │   └── requirements.txt
│   └── docker/
│       └── Dockerfile
└── tests/
    ├── unit/
    │   ├── test_risk_scorer.py
    │   ├── test_pattern_detector.py
    │   └── test_tools.py
    └── integration/
        ├── test_pipeline.py
        └── test_kinesis_flow.py
```

---

## Dependencies

### pyproject.toml

```toml
[project]
name = "fraud-detection-agent"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "langgraph>=1.0.8",
    "langchain>=1.2.10",
    "deepagents>=0.4.1",
    "langchain-aws>=1.2.5",
    "langchain-community>=0.4.1",
    "langsmith>=0.2.0",
    "boto3>=1.35.0",
    "opensearch-py>=2.4.0",
    "gremlinpython>=3.7.0",
    "pydantic>=2.0",
    "python-dotenv>=1.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "moto[all]>=5.0",
    "mypy>=1.8",
    "ruff>=0.3",
]
```

### langgraph.json

```json
{
  "dependencies": [
    "boto3>=1.35.0",
    "langchain-aws>=1.2.5",
    "gremlinpython>=3.7.0",
    "."
  ],
  "graphs": {
    "fraud_detection": "src.graph:graph"
  },
  "env": ".env"
}
```

---

## Environment Configuration

### .env.example

```bash
# AWS
AWS_REGION=us-east-1
AWS_PROFILE=default

# Amazon Bedrock
BEDROCK_SONNET_MODEL_ID=us.anthropic.claude-sonnet-4-20250514
BEDROCK_HAIKU_MODEL_ID=us.anthropic.claude-haiku-4-5-20251001
BEDROCK_OPUS_MODEL_ID=us.anthropic.claude-opus-4-20250514
BEDROCK_EMBEDDING_MODEL_ID=amazon.titan-embed-text-v2:0
BEDROCK_GUARDRAIL_ID=xxxxxxxxxx
BEDROCK_GUARDRAIL_VERSION=1

# Amazon Kinesis
KINESIS_STREAM_NAME=transaction-events
KINESIS_SHARD_COUNT=4

# Amazon SageMaker
SAGEMAKER_ANOMALY_ENDPOINT=fraud-anomaly-detector-v2
SAGEMAKER_RISK_ENDPOINT=transaction-risk-scorer-v1

# Amazon Neptune
NEPTUNE_ENDPOINT=wss://<cluster>.neptune.amazonaws.com:8182/gremlin
NEPTUNE_IAM_AUTH=true

# Amazon OpenSearch Serverless
OPENSEARCH_ENDPOINT=https://<collection-id>.<region>.aoss.amazonaws.com
OPENSEARCH_TRANSACTIONS_INDEX=transaction-history
OPENSEARCH_REGULATORY_INDEX=regulatory-knowledge

# Amazon DynamoDB
DYNAMODB_CUSTOMER_TABLE=customer-profiles
DYNAMODB_CASES_TABLE=fraud-cases
DYNAMODB_ALERTS_TABLE=alert-history

# Amazon SNS
SNS_HIGH_RISK_TOPIC=arn:aws:sns:<region>:<account>:high-risk-alerts
SNS_CASE_TOPIC=arn:aws:sns:<region>:<account>:case-notifications

# AWS Step Functions
STEP_FUNCTIONS_APPROVAL_ARN=arn:aws:states:<region>:<account>:stateMachine:account-action-approval

# LangSmith
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_pt_xxxxxxxxxxxx
LANGSMITH_PROJECT=fraud-detection-agent
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
```

---

## Implementation

### 1. State Schema (`src/state.py`)

```python
"""GraphState for the fraud detection and AML workflow."""

from __future__ import annotations
from typing import Annotated, Literal
from datetime import datetime
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class Transaction(BaseModel):
    """A single financial transaction under analysis."""
    transaction_id: str
    account_id: str
    counterparty_id: str | None = None
    amount: float
    currency: str = "USD"
    transaction_type: Literal["wire", "ach", "card", "check", "internal", "crypto"]
    timestamp: datetime
    channel: str = ""
    location: str = ""
    metadata: dict = Field(default_factory=dict)


class AnomalyScore(BaseModel):
    """Output from ML anomaly detection model."""
    score: float  # 0.0 - 1.0, higher = more anomalous
    model_version: str
    features_contributing: list[str] = Field(default_factory=list)
    confidence: float = 1.0


class CustomerProfile(BaseModel):
    """Enriched customer profile data."""
    customer_id: str
    name: str
    account_type: str
    risk_rating: Literal["low", "medium", "high", "pep", "sanctioned"] = "low"
    kyc_status: str = "current"
    account_open_date: str = ""
    typical_transaction_pattern: str = ""
    previous_alerts: int = 0
    previous_sars: int = 0
    sanctions_match: bool = False
    pep_match: bool = False


class InvestigationFinding(BaseModel):
    """A single finding from the investigation process."""
    finding_type: Literal[
        "network_anomaly", "temporal_pattern", "geographic_risk",
        "behavioral_deviation", "structuring", "layering",
        "sanctions_hit", "pep_association", "other"
    ]
    description: str
    severity: Literal["low", "medium", "high", "critical"]
    evidence: list[str] = Field(default_factory=list)
    confidence: float = 0.8


class SARDraft(BaseModel):
    """Suspicious Activity Report draft."""
    narrative: str
    subject_information: dict = Field(default_factory=dict)
    suspicious_activity_type: list[str] = Field(default_factory=list)
    date_range: str = ""
    total_amount: float = 0.0
    filing_recommendation: Literal["file", "do_not_file", "needs_review"]
    regulatory_citations: list[str] = Field(default_factory=list)


class FraudDetectionState(BaseModel):
    """Complete state for the fraud detection workflow.

    Flows sequentially: Monitor → Enrich → Investigate → Comply → Escalate.
    State accumulates evidence at each stage for full audit trail.
    """
    # Conversation / trace
    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)

    # Input transaction(s)
    transactions: list[Transaction] = Field(default_factory=list)
    batch_id: str = ""

    # Stage 1: Transaction Monitor outputs
    anomaly_scores: list[AnomalyScore] = Field(default_factory=list)
    risk_tier: Literal["low", "medium", "high", "critical"] | None = None
    initial_alert_type: str = ""

    # Stage 2: Risk Enrichment outputs
    customer_profile: CustomerProfile | None = None
    counterparty_profiles: list[CustomerProfile] = Field(default_factory=list)
    historical_context: str = ""
    enrichment_flags: list[str] = Field(default_factory=list)

    # Stage 3: Investigation outputs
    investigation_findings: list[InvestigationFinding] = Field(default_factory=list)
    network_analysis: str = ""
    investigation_summary: str = ""

    # Stage 4: Compliance outputs
    sar_draft: SARDraft | None = None
    compliance_determination: str = ""
    regulatory_references: list[str] = Field(default_factory=list)

    # Stage 5: Escalation
    disposition: Literal[
        "auto_cleared", "monitoring", "investigation_open",
        "sar_filed", "account_frozen", "law_enforcement_referral"
    ] | None = None
    escalation_reason: str = ""
    assigned_investigator: str = ""

    # Control flow
    case_id: str = ""
    pipeline_stage: str = "ingestion"
    requires_human_review: bool = False
    error: str | None = None

    # Audit
    stage_timestamps: dict[str, str] = Field(default_factory=dict)
    decision_log: list[str] = Field(default_factory=list)

    # Tracing metadata
    trace_tags: list[str] = Field(default_factory=list)
```

### 2. Configuration (`src/config.py`)

```python
"""AWS clients and shared configuration."""

import os
import boto3
from dotenv import load_dotenv
from langchain_aws import ChatBedrockConverse, BedrockEmbeddings

load_dotenv()


def get_bedrock_llm(model: str = "sonnet", temperature: float = 0.0):
    """Get Bedrock LLM. Use 'haiku' for triage, 'sonnet' for analysis, and 'opus' for high-stakes SAR drafting."""
    model_map = {
        "haiku": os.getenv("BEDROCK_HAIKU_MODEL_ID"),
        "sonnet": os.getenv("BEDROCK_SONNET_MODEL_ID"),
        "opus": os.getenv("BEDROCK_OPUS_MODEL_ID"),
    }
    model_id = model_map.get(model, os.getenv("BEDROCK_SONNET_MODEL_ID"))
    client = boto3.client("bedrock-runtime", region_name=os.getenv("AWS_REGION"))

    guardrail_config = None
    if os.getenv("BEDROCK_GUARDRAIL_ID"):
        guardrail_config = {
            "guardrailIdentifier": os.getenv("BEDROCK_GUARDRAIL_ID"),
            "guardrailVersion": os.getenv("BEDROCK_GUARDRAIL_VERSION"),
        }

    return ChatBedrockConverse(
        model=model_id,
        temperature=temperature,
        max_tokens=4096,
        client=client,
        guardrail_config=guardrail_config,
    )


def get_embeddings():
    client = boto3.client("bedrock-runtime", region_name=os.getenv("AWS_REGION"))
    return BedrockEmbeddings(model_id=os.getenv("BEDROCK_EMBEDDING_MODEL_ID"), client=client)


def get_sagemaker_runtime():
    return boto3.client("sagemaker-runtime", region_name=os.getenv("AWS_REGION"))


def get_neptune_endpoint():
    return os.getenv("NEPTUNE_ENDPOINT")


def get_dynamodb_resource():
    return boto3.resource("dynamodb", region_name=os.getenv("AWS_REGION"))


def get_sns_client():
    return boto3.client("sns", region_name=os.getenv("AWS_REGION"))


def get_sfn_client():
    return boto3.client("stepfunctions", region_name=os.getenv("AWS_REGION"))
```

### 3. Tools (`src/tools/`)

#### `src/tools/anomaly_detector.py`

```python
"""SageMaker-hosted anomaly detection model."""

import os
import json
from langchain_core.tools import tool
from src.config import get_sagemaker_runtime


@tool
def detect_anomaly(transaction_features: dict) -> dict:
    """Run a transaction through the SageMaker-hosted anomaly detection model.

    Args:
        transaction_features: Dictionary of transaction features including:
            - amount, transaction_type, hour_of_day, day_of_week
            - account_age_days, avg_transaction_amount_30d, transaction_count_30d
            - distance_from_usual_location, counterparty_risk_score

    Returns:
        Dictionary with anomaly_score (0-1), contributing_features, and model_version.
    """
    runtime = get_sagemaker_runtime()
    endpoint = os.getenv("SAGEMAKER_ANOMALY_ENDPOINT")

    response = runtime.invoke_endpoint(
        EndpointName=endpoint,
        ContentType="application/json",
        Body=json.dumps(transaction_features),
    )

    result = json.loads(response["Body"].read().decode())
    return {
        "anomaly_score": result["score"],
        "contributing_features": result.get("feature_importance", []),
        "model_version": result.get("model_version", "unknown"),
        "threshold": result.get("threshold", 0.7),
    }
```

#### `src/tools/sanctions_screening.py`

```python
"""OFAC and sanctions list screening tool."""

import os
from langchain_core.tools import tool
from src.config import get_dynamodb_resource


@tool
def screen_sanctions(name: str, country: str = "", id_number: str = "") -> dict:
    """Screen a person or entity against OFAC sanctions lists and PEP databases.

    Args:
        name: Full name of the person or entity.
        country: Country of residence/incorporation.
        id_number: Government ID or tax ID if available.

    Returns:
        Dictionary with match status, match_score, matched_entries, and list_source.
    """
    # NOTE: In production, integrate with:
    # - OFAC SDN list (via Treasury API or data feed in DynamoDB/OpenSearch)
    # - Dow Jones Risk & Compliance
    # - World-Check (Refinitiv)
    # - Internal PEP database
    #
    # Implementation pattern:
    # 1. Fuzzy name matching against sanctions lists in DynamoDB/OpenSearch
    # 2. Score-based threshold for match determination
    # 3. Return all potential matches above threshold for human review
    raise NotImplementedError(
        "Connect to your sanctions screening provider. "
        "See infrastructure/cdk/stacks/data_stack.py for DynamoDB table schema."
    )
```

#### `src/tools/graph_analysis.py`

```python
"""Amazon Neptune graph analysis for entity relationship investigation."""

import os
from langchain_core.tools import tool
from gremlin_python.driver import client as gremlin_client
from gremlin_python.driver.serializer import GraphSONSerializersV2d0


@tool
def analyze_entity_network(entity_id: str, hops: int = 2) -> dict:
    """Analyze the transaction network around an entity using Neptune graph database.

    Args:
        entity_id: The account or customer ID to investigate.
        hops: Number of relationship hops to traverse (1-3).

    Returns:
        Dictionary with connected_entities, transaction_flows, risk_indicators,
        and network_visualization_data.
    """
    endpoint = os.getenv("NEPTUNE_ENDPOINT")
    neptune = gremlin_client.Client(
        endpoint,
        "g",
        message_serializer=GraphSONSerializersV2d0(),
    )

    # Find connected entities within N hops
    query = f"""
    g.V('{entity_id}')
     .repeat(both('transacted_with').simplePath())
     .times({min(hops, 3)})
     .path()
     .by(valueMap(true))
     .limit(100)
    """

    try:
        result = neptune.submit(query).all().result()

        # Analyze network for suspicious patterns
        entities = set()
        edges = []
        for path in result:
            for node in path:
                if isinstance(node, dict):
                    entities.add(node.get("id", [""])[0] if isinstance(node.get("id"), list) else str(node.get("id", "")))

        # Query for high-risk indicators
        risk_query = f"""
        g.V('{entity_id}')
         .repeat(both('transacted_with'))
         .times({min(hops, 3)})
         .has('risk_rating', within('high', 'pep', 'sanctioned'))
         .dedup()
         .valueMap(true)
        """
        high_risk = neptune.submit(risk_query).all().result()

        return {
            "entity_id": entity_id,
            "connected_entities_count": len(entities),
            "high_risk_connections": len(high_risk),
            "high_risk_entities": high_risk[:10],
            "network_depth": hops,
        }
    except Exception as e:
        return {"error": str(e), "entity_id": entity_id}
    finally:
        neptune.close()


@tool
def detect_structuring_pattern(account_id: str, lookback_days: int = 30) -> dict:
    """Detect potential structuring (smurfing) patterns in transaction history.

    Checks for patterns of transactions just below reporting thresholds ($10,000 CTR).

    Args:
        account_id: Account ID to analyze.
        lookback_days: Number of days to look back.

    Returns:
        Dictionary with structuring_score, suspicious_transactions, total_amount,
        and pattern_description.
    """
    endpoint = os.getenv("NEPTUNE_ENDPOINT")
    neptune = gremlin_client.Client(endpoint, "g",
                                     message_serializer=GraphSONSerializersV2d0())

    try:
        query = f"""
        g.V('{account_id}')
         .outE('sent')
         .has('amount', between(8000, 10000))
         .has('timestamp', gte(now().minus({lookback_days}, 'd')))
         .order().by('timestamp')
         .valueMap(true)
        """
        results = neptune.submit(query).all().result()

        suspicious_count = len(results)
        total = sum(r.get("amount", [0])[0] for r in results if isinstance(r.get("amount"), list))

        return {
            "account_id": account_id,
            "structuring_score": min(suspicious_count / 5.0, 1.0),  # Normalize
            "suspicious_transaction_count": suspicious_count,
            "total_amount": total,
            "pattern": "potential_structuring" if suspicious_count >= 3 else "normal",
            "lookback_days": lookback_days,
        }
    except Exception as e:
        return {"error": str(e)}
    finally:
        neptune.close()
```

#### `src/tools/notification.py`

```python
"""Notification and case management tools."""

import os
import json
from datetime import datetime
from langchain_core.tools import tool
from src.config import get_sns_client, get_sfn_client, get_dynamodb_resource


@tool
def send_high_risk_alert(case_id: str, risk_tier: str, summary: str) -> dict:
    """Send a high-risk alert via SNS to compliance team and case management system.

    Args:
        case_id: Unique case identifier.
        risk_tier: Risk classification ('high' or 'critical').
        summary: Brief summary of the alert for the notification.

    Returns:
        Dictionary with notification_id and status.
    """
    sns = get_sns_client()
    topic_arn = os.getenv("SNS_HIGH_RISK_TOPIC")

    message = {
        "case_id": case_id,
        "risk_tier": risk_tier,
        "summary": summary,
        "timestamp": datetime.utcnow().isoformat(),
        "action_required": "Immediate review required" if risk_tier == "critical" else "Review within 24 hours",
    }

    response = sns.publish(
        TopicArn=topic_arn,
        Subject=f"[{risk_tier.upper()}] Fraud Alert - Case {case_id}",
        Message=json.dumps(message),
        MessageAttributes={
            "risk_tier": {"DataType": "String", "StringValue": risk_tier},
            "case_id": {"DataType": "String", "StringValue": case_id},
        },
    )

    return {"notification_id": response["MessageId"], "status": "sent"}


@tool
def initiate_account_action_approval(
    case_id: str,
    account_id: str,
    action: str,
    justification: str,
) -> dict:
    """Start a Step Functions approval workflow for account actions (freeze, restrict, close).

    Args:
        case_id: Case identifier.
        account_id: Account to take action on.
        action: Proposed action ('freeze', 'restrict', 'close', 'enhanced_monitoring').
        justification: Justification for the proposed action.

    Returns:
        Dictionary with execution_arn and status.
    """
    sfn = get_sfn_client()
    state_machine_arn = os.getenv("STEP_FUNCTIONS_APPROVAL_ARN")

    response = sfn.start_execution(
        stateMachineArn=state_machine_arn,
        name=f"approval-{case_id}-{action}",
        input=json.dumps({
            "case_id": case_id,
            "account_id": account_id,
            "proposed_action": action,
            "justification": justification,
            "requested_at": datetime.utcnow().isoformat(),
            "requires_approval_from": "compliance_officer",
        }),
    )

    return {
        "execution_arn": response["executionArn"],
        "status": "approval_pending",
    }


@tool
def create_case_record(
    case_id: str,
    alert_type: str,
    risk_tier: str,
    transactions: list[dict],
    summary: str,
) -> dict:
    """Create or update a case record in the case management DynamoDB table.

    Args:
        case_id: Unique case ID.
        alert_type: Type of alert that triggered the case.
        risk_tier: Risk classification.
        transactions: List of transaction dicts involved.
        summary: Case summary.

    Returns:
        Confirmation of case creation with case_id.
    """
    dynamodb = get_dynamodb_resource()
    table = dynamodb.Table(os.getenv("DYNAMODB_CASES_TABLE"))

    table.put_item(Item={
        "case_id": case_id,
        "alert_type": alert_type,
        "risk_tier": risk_tier,
        "status": "open",
        "transactions": json.dumps(transactions),
        "summary": summary,
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
    })

    return {"case_id": case_id, "status": "created"}
```

### 4. Agent Implementations (`src/agents/`)

Deep Agents pattern recommendation for this use case:

- Use `deepagents` for complex, high-context investigation summarization and SAR narrative refinement.
- Keep transaction scoring, deterministic policy checks, and routing in the core LangGraph pipeline.
- Use model sizing by task: Haiku for triage, Sonnet for core analysis, Opus for high-stakes drafting/review.

#### `src/agents/transaction_monitor.py`

```python
"""Transaction Monitor Agent: ingests transactions and flags anomalies."""

import uuid
from datetime import datetime
from langchain_core.messages import AIMessage
from src.state import FraudDetectionState, AnomalyScore
from src.tools.anomaly_detector import detect_anomaly


def transaction_monitor_node(state: FraudDetectionState) -> dict:
    """Process incoming transactions through ML anomaly detection.

    This agent runs at high volume using the SageMaker endpoint for fast scoring.
    No LLM calls here — purely ML-based detection for speed.
    """
    anomaly_scores = []
    max_score = 0.0
    alert_types = []

    for txn in state.transactions:
        # Build feature vector for the ML model
        features = {
            "amount": txn.amount,
            "transaction_type": txn.transaction_type,
            "hour_of_day": txn.timestamp.hour,
            "day_of_week": txn.timestamp.weekday(),
            "channel": txn.channel,
            # Additional features would come from feature store in production
        }

        result = detect_anomaly.invoke({"transaction_features": features})

        score = AnomalyScore(
            score=result["anomaly_score"],
            model_version=result["model_version"],
            features_contributing=result.get("contributing_features", []),
        )
        anomaly_scores.append(score)
        max_score = max(max_score, score.score)

        # Classify alert type based on contributing features
        top_features = result.get("contributing_features", [])
        if "amount_spike" in top_features:
            alert_types.append("unusual_amount")
        if "velocity_anomaly" in top_features:
            alert_types.append("rapid_succession")
        if "geographic_anomaly" in top_features:
            alert_types.append("unusual_location")

    # Determine risk tier
    if max_score >= 0.9:
        risk_tier = "critical"
    elif max_score >= 0.7:
        risk_tier = "high"
    elif max_score >= 0.4:
        risk_tier = "medium"
    else:
        risk_tier = "low"

    case_id = f"CASE-{datetime.utcnow().strftime('%Y%m%d')}-{uuid.uuid4().hex[:8].upper()}"

    return {
        "anomaly_scores": anomaly_scores,
        "risk_tier": risk_tier,
        "initial_alert_type": ", ".join(set(alert_types)) or "general_anomaly",
        "case_id": case_id,
        "pipeline_stage": "monitor_complete",
        "stage_timestamps": {
            **state.stage_timestamps,
            "monitor_completed": datetime.utcnow().isoformat(),
        },
        "decision_log": state.decision_log + [
            f"Monitor: Max anomaly score {max_score:.3f}, risk tier: {risk_tier}, "
            f"alert types: {alert_types}"
        ],
        "messages": [AIMessage(content=f"Transaction monitoring complete. "
                               f"Risk tier: {risk_tier}, Case ID: {case_id}")],
    }
```

#### `src/agents/risk_enrichment.py`

```python
"""Risk Enrichment Agent: enriches alerts with customer and counterparty context."""

import os
from datetime import datetime
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from src.state import FraudDetectionState, CustomerProfile
from src.config import get_bedrock_llm, get_dynamodb_resource
from src.tools.sanctions_screening import screen_sanctions

ENRICHMENT_PROMPT = """You are a financial crime risk analyst performing customer due diligence
enrichment on a flagged transaction alert.

Given the customer profile, transaction details, and screening results, assess:
1. Does this transaction fit the customer's known behavior pattern?
2. Are there any KYC red flags (expired KYC, high-risk jurisdiction, PEP status)?
3. What is the overall risk assessment after enrichment?

Provide a structured assessment with specific risk flags and recommended next steps.
Always cite specific data points from the customer profile to support your assessment.
"""


def risk_enrichment_node(state: FraudDetectionState) -> dict:
    """Enrich the alert with customer profiles, historical data, and sanctions screening."""
    llm = get_bedrock_llm(model="haiku", temperature=0.0)
    dynamodb = get_dynamodb_resource()
    enrichment_flags = []

    # Retrieve customer profile
    customer_table = dynamodb.Table(os.getenv("DYNAMODB_CUSTOMER_TABLE"))
    primary_account = state.transactions[0].account_id if state.transactions else ""

    try:
        response = customer_table.get_item(Key={"account_id": primary_account})
        item = response.get("Item", {})
        customer = CustomerProfile(
            customer_id=item.get("customer_id", primary_account),
            name=item.get("name", "Unknown"),
            account_type=item.get("account_type", "unknown"),
            risk_rating=item.get("risk_rating", "medium"),
            kyc_status=item.get("kyc_status", "unknown"),
            account_open_date=item.get("account_open_date", ""),
            typical_transaction_pattern=item.get("typical_pattern", ""),
            previous_alerts=item.get("previous_alerts", 0),
            previous_sars=item.get("previous_sars", 0),
        )
    except Exception:
        customer = CustomerProfile(
            customer_id=primary_account,
            name="Profile Not Found",
            account_type="unknown",
        )
        enrichment_flags.append("customer_profile_missing")

    # Check for previous alerts/SARs
    if customer.previous_sars > 0:
        enrichment_flags.append("prior_sar_filed")
    if customer.previous_alerts >= 3:
        enrichment_flags.append("frequent_alerts")
    if customer.kyc_status != "current":
        enrichment_flags.append("kyc_expired_or_missing")
    if customer.risk_rating in ("high", "pep", "sanctioned"):
        enrichment_flags.append(f"customer_risk_rating_{customer.risk_rating}")

    # LLM assessment of behavioral fit
    txn_summary = "\n".join(
        f"- {t.transaction_type}: ${t.amount:,.2f} at {t.timestamp.isoformat()} via {t.channel}"
        for t in state.transactions
    )

    enrichment_response = llm.invoke([
        SystemMessage(content=ENRICHMENT_PROMPT),
        HumanMessage(content=f"Alert type: {state.initial_alert_type}\n"
                     f"Risk tier: {state.risk_tier}\n\n"
                     f"Customer profile:\n{customer.model_dump_json(indent=2)}\n\n"
                     f"Flagged transactions:\n{txn_summary}\n\n"
                     f"Anomaly scores: {[s.score for s in state.anomaly_scores]}\n\n"
                     "Provide your risk assessment and any additional flags."),
    ])

    return {
        "customer_profile": customer,
        "enrichment_flags": enrichment_flags,
        "historical_context": enrichment_response.content,
        "pipeline_stage": "enrichment_complete",
        "stage_timestamps": {
            **state.stage_timestamps,
            "enrichment_completed": datetime.utcnow().isoformat(),
        },
        "decision_log": state.decision_log + [
            f"Enrichment: Flags={enrichment_flags}, "
            f"Customer risk={customer.risk_rating}, Prior SARs={customer.previous_sars}"
        ],
        "messages": [AIMessage(content=f"Risk enrichment complete. Flags: {enrichment_flags}")],
    }
```

#### `src/agents/investigation.py`

```python
"""Investigation Agent: deep-dive analysis for medium and high-risk cases."""

from datetime import datetime
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from src.state import FraudDetectionState, InvestigationFinding
from src.config import get_bedrock_llm
from src.tools.graph_analysis import analyze_entity_network, detect_structuring_pattern
from src.rag.typology_retriever import get_typology_retriever

INVESTIGATION_PROMPT = """You are a senior financial crime investigator conducting a deep-dive
analysis on a flagged transaction alert.

You have access to:
1. Network analysis showing entity relationships and transaction flows
2. Pattern detection results for structuring and layering
3. Known fraud typology reference materials
4. Customer enrichment data from the previous stage

Your investigation should:
- Identify specific suspicious patterns (structuring, layering, round-tripping, etc.)
- Assess the strength of evidence for each finding
- Classify each finding by severity (low/medium/high/critical)
- Provide clear, evidence-based reasoning for each conclusion
- Recommend a disposition: clear, monitor, or escalate

Be thorough but precise. Every finding must be supported by specific evidence.
Avoid speculation — clearly label anything that requires further investigation.
"""


def investigation_node(state: FraudDetectionState) -> dict:
    """Perform deep investigation with network analysis and pattern detection."""
    llm = get_bedrock_llm(model="sonnet", temperature=0.0)
    findings = []

    primary_account = state.transactions[0].account_id if state.transactions else ""

    # Run network analysis
    network_result = analyze_entity_network.invoke({
        "entity_id": primary_account, "hops": 2
    })

    # Check for structuring patterns
    structuring_result = detect_structuring_pattern.invoke({
        "account_id": primary_account, "lookback_days": 30
    })

    if structuring_result.get("structuring_score", 0) > 0.5:
        findings.append(InvestigationFinding(
            finding_type="structuring",
            description=f"Detected potential structuring: {structuring_result['suspicious_transaction_count']} "
                        f"transactions between $8,000-$10,000 totaling ${structuring_result['total_amount']:,.2f}",
            severity="high" if structuring_result["structuring_score"] > 0.7 else "medium",
            evidence=[f"Transaction count: {structuring_result['suspicious_transaction_count']}",
                      f"Total amount: ${structuring_result['total_amount']:,.2f}"],
            confidence=structuring_result["structuring_score"],
        ))

    if network_result.get("high_risk_connections", 0) > 0:
        findings.append(InvestigationFinding(
            finding_type="network_anomaly",
            description=f"Account connected to {network_result['high_risk_connections']} "
                        f"high-risk entities within {network_result['network_depth']} hops",
            severity="high",
            evidence=[f"High risk connections: {network_result['high_risk_connections']}",
                      f"Total network size: {network_result['connected_entities_count']}"],
        ))

    # Retrieve known fraud typologies for context
    retriever = get_typology_retriever()
    typology_docs = retriever.invoke(
        f"{state.initial_alert_type} {state.risk_tier} "
        f"{'structuring' if structuring_result.get('structuring_score', 0) > 0.3 else ''}"
    )
    typology_context = "\n".join(d.page_content for d in typology_docs[:3])

    # LLM deep analysis
    investigation_response = llm.invoke([
        SystemMessage(content=INVESTIGATION_PROMPT),
        HumanMessage(content=f"Case ID: {state.case_id}\n"
                     f"Alert type: {state.initial_alert_type}\n"
                     f"Risk tier: {state.risk_tier}\n\n"
                     f"Customer: {state.customer_profile.model_dump_json(indent=2) if state.customer_profile else 'N/A'}\n\n"
                     f"Enrichment flags: {state.enrichment_flags}\n"
                     f"Enrichment assessment: {state.historical_context[:2000]}\n\n"
                     f"Network analysis: {network_result}\n\n"
                     f"Structuring detection: {structuring_result}\n\n"
                     f"Known typology references:\n{typology_context}\n\n"
                     f"Transactions:\n{[t.model_dump() for t in state.transactions[:10]]}\n\n"
                     "Conduct your investigation. List each finding with evidence and severity."),
    ])

    return {
        "investigation_findings": findings,  # ML-detected findings
        "investigation_summary": investigation_response.content,
        "network_analysis": str(network_result),
        "pipeline_stage": "investigation_complete",
        "stage_timestamps": {
            **state.stage_timestamps,
            "investigation_completed": datetime.utcnow().isoformat(),
        },
        "decision_log": state.decision_log + [
            f"Investigation: {len(findings)} ML findings, "
            f"network connections={network_result.get('connected_entities_count', 0)}, "
            f"high risk connections={network_result.get('high_risk_connections', 0)}"
        ],
        "messages": [AIMessage(content=f"Investigation complete. {len(findings)} findings identified.")],
    }
```

#### `src/agents/compliance.py`

```python
"""Compliance Agent: regulatory validation and SAR drafting."""

from datetime import datetime
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from src.state import FraudDetectionState, SARDraft
from src.config import get_bedrock_llm
from src.rag.regulatory_retriever import get_regulatory_retriever

COMPLIANCE_PROMPT = """You are a BSA/AML compliance officer reviewing an investigation case
and determining whether a Suspicious Activity Report (SAR) should be filed.

Your responsibilities:
1. Review all investigation findings and evidence
2. Determine if the activity meets SAR filing thresholds under BSA regulations
3. If SAR is warranted, draft the narrative section following FinCEN guidelines
4. Cite specific regulations that apply (31 CFR 1020.320, etc.)
5. Classify the suspicious activity type per FinCEN SAR form categories

SAR narrative requirements (per FinCEN guidance):
- Who: Subject information (name, account, relationship to institution)
- What: Description of the suspicious activity
- When: Dates and timeline of the activity
- Where: Location/channels involved
- Why: Why the activity is suspicious
- How: How the activity was conducted

Filing thresholds:
- Mandatory filing: Known/suspected violation >= $5,000 (if subject identified)
- Mandatory filing: Known/suspected violation >= $25,000 (regardless of subject ID)
- File within 30 days of initial detection (60 days if no subject identified)

Be precise with regulatory citations. Never recommend filing without clear evidence.
"""


def compliance_node(state: FraudDetectionState) -> dict:
    """Validate findings against regulations and draft SAR if warranted."""
    llm = get_bedrock_llm(model="sonnet", temperature=0.0)
    retriever = get_regulatory_retriever()

    # Retrieve relevant regulatory guidance
    reg_docs = retriever.invoke(
        f"SAR filing requirements {state.initial_alert_type} "
        f"{'structuring' if any(f.finding_type == 'structuring' for f in state.investigation_findings) else ''}"
    )
    regulatory_context = "\n".join(d.page_content for d in reg_docs[:5])

    total_amount = sum(t.amount for t in state.transactions)

    response = llm.invoke([
        SystemMessage(content=COMPLIANCE_PROMPT),
        HumanMessage(content=f"Case ID: {state.case_id}\n"
                     f"Risk tier: {state.risk_tier}\n"
                     f"Total transaction amount: ${total_amount:,.2f}\n\n"
                     f"Investigation findings:\n{state.investigation_summary}\n\n"
                     f"ML-detected findings:\n"
                     + "\n".join(f"- [{f.severity}] {f.finding_type}: {f.description}"
                               for f in state.investigation_findings) +
                     f"\n\nCustomer: {state.customer_profile.model_dump_json(indent=2) if state.customer_profile else 'N/A'}\n\n"
                     f"Enrichment flags: {state.enrichment_flags}\n\n"
                     f"Applicable regulations:\n{regulatory_context}\n\n"
                     "Determine SAR filing recommendation and draft narrative if warranted.\n"
                     "Respond with JSON: {\"filing_recommendation\": \"file|do_not_file|needs_review\", "
                     "\"narrative\": \"...\", \"suspicious_activity_types\": [...], "
                     "\"regulatory_citations\": [...], \"reasoning\": \"...\"}"),
    ])

    import json
    try:
        compliance_result = json.loads(response.content)
    except json.JSONDecodeError:
        compliance_result = {
            "filing_recommendation": "needs_review",
            "narrative": response.content,
            "suspicious_activity_types": [],
            "regulatory_citations": [],
        }

    sar_draft = SARDraft(
        narrative=compliance_result.get("narrative", ""),
        suspicious_activity_type=compliance_result.get("suspicious_activity_types", []),
        total_amount=total_amount,
        filing_recommendation=compliance_result.get("filing_recommendation", "needs_review"),
        regulatory_citations=compliance_result.get("regulatory_citations", []),
        subject_information=state.customer_profile.model_dump() if state.customer_profile else {},
    )

    requires_human = sar_draft.filing_recommendation != "do_not_file"

    return {
        "sar_draft": sar_draft,
        "compliance_determination": compliance_result.get("reasoning", ""),
        "regulatory_references": compliance_result.get("regulatory_citations", []),
        "requires_human_review": requires_human,
        "pipeline_stage": "compliance_complete",
        "stage_timestamps": {
            **state.stage_timestamps,
            "compliance_completed": datetime.utcnow().isoformat(),
        },
        "decision_log": state.decision_log + [
            f"Compliance: Recommendation={sar_draft.filing_recommendation}, "
            f"Amount=${total_amount:,.2f}, Citations={sar_draft.regulatory_citations}"
        ],
        "messages": [AIMessage(content=f"Compliance review complete. "
                               f"Recommendation: {sar_draft.filing_recommendation}")],
    }
```

### 5. Main Graph (`src/graph.py`)

```python
"""Main LangGraph workflow: sequential fraud detection pipeline."""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from src.state import FraudDetectionState
from src.agents.transaction_monitor import transaction_monitor_node
from src.agents.risk_enrichment import risk_enrichment_node
from src.agents.investigation import investigation_node
from src.agents.compliance import compliance_node
from src.agents.human_escalation import human_escalation_node


def route_after_monitor(state: FraudDetectionState) -> str:
    """After monitoring, route based on risk tier."""
    if state.risk_tier == "low":
        return "auto_clear"
    return "risk_enrichment"


def route_after_enrichment(state: FraudDetectionState) -> str:
    """After enrichment, decide if full investigation is needed."""
    high_risk_flags = {"prior_sar_filed", "customer_risk_rating_high",
                       "customer_risk_rating_sanctioned", "customer_risk_rating_pep"}
    if state.risk_tier in ("high", "critical") or high_risk_flags & set(state.enrichment_flags):
        return "investigation"
    if state.risk_tier == "medium":
        return "investigation"
    return "compliance"


def auto_clear_node(state: FraudDetectionState) -> dict:
    """Auto-clear low-risk alerts with audit logging."""
    from datetime import datetime
    from langchain_core.messages import AIMessage
    return {
        "disposition": "auto_cleared",
        "pipeline_stage": "closed",
        "stage_timestamps": {
            **state.stage_timestamps,
            "auto_cleared": datetime.utcnow().isoformat(),
        },
        "decision_log": state.decision_log + [
            f"Auto-cleared: risk_tier={state.risk_tier}, "
            f"max_anomaly={max((s.score for s in state.anomaly_scores), default=0):.3f}"
        ],
        "messages": [AIMessage(content=f"Case {state.case_id} auto-cleared (low risk).")],
    }


# Build the graph
workflow = StateGraph(FraudDetectionState)

# Add nodes
workflow.add_node("transaction_monitor", transaction_monitor_node)
workflow.add_node("risk_enrichment", risk_enrichment_node)
workflow.add_node("investigation", investigation_node)
workflow.add_node("compliance", compliance_node)
workflow.add_node("human_escalation", human_escalation_node)
workflow.add_node("auto_clear", auto_clear_node)

# Sequential pipeline with conditional routing
workflow.add_edge(START, "transaction_monitor")

workflow.add_conditional_edges("transaction_monitor", route_after_monitor, {
    "risk_enrichment": "risk_enrichment",
    "auto_clear": "auto_clear",
})

workflow.add_conditional_edges("risk_enrichment", route_after_enrichment, {
    "investigation": "investigation",
    "compliance": "compliance",
})

workflow.add_edge("investigation", "compliance")

# Human-in-the-loop checkpoint before any SAR filing or account action
workflow.add_edge("compliance", "human_escalation")

workflow.add_edge("human_escalation", END)
workflow.add_edge("auto_clear", END)

# Compile with persistence and human-in-the-loop
memory = MemorySaver()
graph = workflow.compile(
    checkpointer=memory,
    interrupt_before=["human_escalation"],  # MANDATORY human review before actions
)
```

---

## Evaluation Suite

### `evaluation/evaluators/detection_evaluator.py`

```python
"""Precision/recall evaluator for fraud detection accuracy."""

from langsmith.evaluation import EvaluationResult, run_evaluator


@run_evaluator
def detection_precision_evaluator(run, example) -> EvaluationResult:
    """Evaluate fraud detection precision against labeled data."""
    predicted_tier = run.outputs.get("risk_tier", "low")
    actual_label = example.outputs.get("is_fraud", False)
    actual_tier = example.outputs.get("expected_risk_tier", "low")

    # Score: did we correctly identify fraud?
    if actual_label and predicted_tier in ("high", "critical"):
        score = 1.0  # True positive
        comment = "Correctly identified as high risk"
    elif not actual_label and predicted_tier in ("low",):
        score = 1.0  # True negative
        comment = "Correctly identified as low risk"
    elif actual_label and predicted_tier in ("low",):
        score = 0.0  # False negative — worst case
        comment = "MISSED: Fraud classified as low risk"
    elif not actual_label and predicted_tier in ("high", "critical"):
        score = 0.3  # False positive — not great but safer
        comment = "False positive: non-fraud flagged as high risk"
    else:
        score = 0.5  # Medium tier for medium cases
        comment = f"Predicted: {predicted_tier}, Expected: {actual_tier}"

    return EvaluationResult(key="detection_precision", score=score, comment=comment)
```

### `evaluation/evaluators/sar_evaluator.py`

```python
"""LLM-as-Judge evaluator for SAR narrative quality."""

from langsmith.evaluation import EvaluationResult, run_evaluator


@run_evaluator
def sar_quality_evaluator(run, example) -> EvaluationResult:
    """Evaluate SAR narrative completeness and regulatory compliance."""
    from src.config import get_bedrock_llm
    import json

    sar = run.outputs.get("sar_draft", {})
    narrative = sar.get("narrative", "") if isinstance(sar, dict) else ""

    llm = get_bedrock_llm(model="sonnet", temperature=0.0)
    response = llm.invoke(
        f"""Evaluate this SAR narrative for FinCEN compliance.

SAR Narrative:
{narrative}

Check for the presence of these required elements (the "5 W's"):
1. WHO - Subject identified with account details
2. WHAT - Clear description of suspicious activity
3. WHEN - Specific dates and timeline
4. WHERE - Location/channel information
5. WHY - Clear explanation of why activity is suspicious
6. HOW - Method of conducting the suspicious activity
7. REGULATORY BASIS - Specific regulatory citations

Score 1-5:
5 = All elements present, clear and well-structured
4 = Most elements present, minor gaps
3 = Several elements present but notable gaps
2 = Major elements missing
1 = Narrative is inadequate or missing

Respond with JSON: {{"score": <1-5>, "missing_elements": [...], "reasoning": "..."}}"""
    )

    try:
        result = json.loads(response.content)
        return EvaluationResult(
            key="sar_quality",
            score=result["score"] / 5.0,
            comment=f"Missing: {result.get('missing_elements', [])}. {result.get('reasoning', '')}",
        )
    except (json.JSONDecodeError, KeyError):
        return EvaluationResult(key="sar_quality", score=0.0, comment="Eval parse error")
```

---

## LangSmith Configuration

### Tracing with fraud-specific metadata:

```python
import langsmith as ls

with ls.tracing_context(
    project_name="fraud-detection-agent",
    tags=["production", "aml-pipeline", "v1.3"],
    metadata={
        "case_id": state.case_id,
        "risk_tier": state.risk_tier,
        "alert_type": state.initial_alert_type,
        "disposition": state.disposition,
        "environment": "production",
    },
):
    result = graph.invoke(state, config={"configurable": {"thread_id": state.case_id}})
```

### Monitoring Dashboards

| Dashboard | Metrics | Alert Threshold |
|-----------|---------|-----------------|
| False Positive Rate | Auto-cleared / total alerts ratio | FPR > 95% (too many clears) |
| Detection Latency | E2E time: ingestion → disposition | P99 > 60s |
| SAR Quality | Online SAR evaluator scores | Mean score < 0.7 |
| Model Drift | Anomaly score distribution over time | Distribution shift > 2σ |
| Pipeline Throughput | Transactions/second processed | < 100 TPS |

---

## Deployment Checklist

1. Deploy streaming infrastructure (Kinesis, Lambda consumers) via CDK
2. Deploy ML models to SageMaker endpoints (anomaly detector, risk scorer)
3. Provision Neptune cluster and load entity relationship graph
4. Index regulatory documents into OpenSearch (BSA/AML regs, FinCEN guidance)
5. Create LangSmith eval datasets from historical labeled fraud cases
6. Run offline evals — target >85% precision, >90% recall on high-risk
7. Deploy agent to AgentCore Runtime with Step Functions approval workflow
8. Configure SNS topics and case management integrations
9. Run adversarial red team evaluation suite
10. Enable online SAR quality evaluations on production traffic
11. Set up compliance audit export from LangSmith traces
