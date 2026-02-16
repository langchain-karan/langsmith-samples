# Use Case 1: Intelligent Document Processing & Research Agent

## Overview

Build a multi-agent system that processes SEC filings (10-K, 10-Q, 8-K), earnings transcripts, and financial news to extract insights, perform quantitative analysis, and generate analyst-ready research reports. The system uses a **graph (hierarchical) pattern** with a Supervisor Agent orchestrating four specialist agents.

---

## Project Structure

```
document-research-agent/
├── pyproject.toml
├── langgraph.json
├── .env.example
├── README.md
├── src/
│   ├── __init__.py
│   ├── graph.py                  # Main LangGraph workflow definition
│   ├── state.py                  # GraphState schema
│   ├── config.py                 # Configuration and AWS clients
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── supervisor.py         # Supervisor/Router agent
│   │   ├── document_ingestion.py # Document parsing and extraction
│   │   ├── financial_analysis.py # Quantitative analysis agent
│   │   ├── market_research.py    # Market context and news agent
│   │   └── report_generation.py  # Report synthesis agent
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── textract_tool.py      # Amazon Textract integration
│   │   ├── s3_tools.py           # S3 document operations
│   │   ├── financial_data.py     # Financial data API tools
│   │   ├── opensearch_tools.py   # Vector search tools
│   │   └── code_interpreter.py   # Bedrock Code Interpreter
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── embeddings.py         # Amazon Titan Embeddings setup
│   │   ├── indexer.py            # Document indexing pipeline
│   │   └── retriever.py          # OpenSearch retriever
│   ├── memory/
│   │   ├── __init__.py
│   │   └── dynamodb_memory.py    # DynamoDB conversation memory
│   └── prompts/
│       ├── supervisor.py
│       ├── ingestion.py
│       ├── analysis.py
│       ├── research.py
│       └── report.py
├── evaluation/
│   ├── datasets/
│   │   ├── factual_accuracy.json     # Golden dataset for accuracy evals
│   │   ├── completeness.json         # Report completeness dataset
│   │   └── source_attribution.json   # Citation accuracy dataset
│   ├── evaluators/
│   │   ├── accuracy_evaluator.py     # LLM-as-judge for financial accuracy
│   │   ├── completeness_evaluator.py # Heuristic + LLM completeness check
│   │   ├── citation_evaluator.py     # Source attribution heuristic
│   │   └── regression_runner.py      # Offline regression test runner
│   └── run_evals.py                  # CLI entry point for evaluation suite
├── infrastructure/
│   ├── cdk/
│   │   ├── app.py
│   │   ├── stacks/
│   │   │   ├── data_stack.py         # S3, DynamoDB, OpenSearch
│   │   │   ├── bedrock_stack.py      # Bedrock config, Guardrails
│   │   │   ├── ingestion_stack.py    # EventBridge, Lambda triggers
│   │   │   └── agent_stack.py        # AgentCore runtime deployment
│   │   └── requirements.txt
│   └── docker/
│       └── Dockerfile
└── tests/
    ├── unit/
    │   ├── test_tools.py
    │   ├── test_agents.py
    │   └── test_state.py
    └── integration/
        ├── test_graph_workflow.py
        └── test_rag_pipeline.py
```

---

## Dependencies

### pyproject.toml

```toml
[project]
name = "document-research-agent"
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
    "pydantic>=2.0",
    "python-dotenv>=1.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
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
    "."
  ],
  "graphs": {
    "document_research": "src.graph:graph"
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
BEDROCK_MODEL_ID=us.anthropic.claude-sonnet-4-5-20250929
BEDROCK_HAIKU_MODEL_ID=us.anthropic.claude-haiku-4-5-20251001
BEDROCK_OPUS_MODEL_ID=us.anthropic.claude-opus-4-20250514
BEDROCK_EMBEDDING_MODEL_ID=amazon.titan-embed-text-v2:0

# Amazon OpenSearch Serverless
OPENSEARCH_ENDPOINT=https://<collection-id>.<region>.aoss.amazonaws.com
OPENSEARCH_INDEX_NAME=financial-documents

# Amazon DynamoDB
DYNAMODB_TABLE_NAME=research-agent-memory
DYNAMODB_METRICS_TABLE=extracted-financial-metrics

# Amazon S3
S3_DOCUMENTS_BUCKET=finserv-sec-filings
S3_REPORTS_BUCKET=finserv-generated-reports

# LangSmith
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_pt_xxxxxxxxxxxx
LANGSMITH_PROJECT=document-research-agent
LANGSMITH_ENDPOINT=https://api.smith.langchain.com

# Guardrails
BEDROCK_GUARDRAIL_ID=xxxxxxxxxx
BEDROCK_GUARDRAIL_VERSION=1
```

---

## Implementation

### 1. State Schema (`src/state.py`)

```python
"""GraphState definition for the document research workflow."""

from __future__ import annotations
from typing import Annotated, Literal
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class ExtractedMetric(BaseModel):
    """A single financial metric extracted from a document."""
    metric_name: str
    value: str
    period: str
    source_document: str
    page_number: int | None = None
    confidence: float = 1.0


class DocumentChunk(BaseModel):
    """A parsed chunk from a financial document."""
    content: str
    source: str
    doc_type: Literal["10-K", "10-Q", "8-K", "earnings_transcript", "news", "other"]
    ticker: str | None = None
    filing_date: str | None = None
    metadata: dict = Field(default_factory=dict)


class AnalysisResult(BaseModel):
    """Output from the financial analysis agent."""
    analysis_type: str
    findings: str
    metrics: list[ExtractedMetric] = Field(default_factory=list)
    charts_data: list[dict] = Field(default_factory=list)
    confidence: float = 1.0


class MarketContext(BaseModel):
    """Output from the market research agent."""
    news_summary: str
    peer_comparisons: list[dict] = Field(default_factory=list)
    industry_benchmarks: dict = Field(default_factory=dict)
    macro_context: str = ""
    sources: list[str] = Field(default_factory=list)


class ResearchState(BaseModel):
    """Complete state for the document research workflow.

    This is the central state object passed through all nodes in the LangGraph
    workflow. Each agent reads from and writes to specific fields.
    """
    # Conversation
    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)

    # User query metadata
    user_query: str = ""
    query_type: Literal[
        "single_company_analysis",
        "competitive_comparison",
        "sector_overview",
        "earnings_review",
        "filing_deep_dive",
        "general_question",
    ] | None = None
    tickers: list[str] = Field(default_factory=list)

    # Routing decisions
    next_agent: str = ""
    agents_completed: list[str] = Field(default_factory=list)
    requires_parallel: bool = False

    # Document ingestion outputs
    parsed_documents: list[DocumentChunk] = Field(default_factory=list)
    extracted_metrics: list[ExtractedMetric] = Field(default_factory=list)

    # Analysis outputs
    analysis_results: list[AnalysisResult] = Field(default_factory=list)

    # Market research outputs
    market_context: MarketContext | None = None

    # Final report
    report_markdown: str = ""
    report_s3_key: str = ""

    # Control flow
    iteration_count: int = 0
    max_iterations: int = 3
    needs_refinement: bool = False
    error: str | None = None

    # Tracing metadata
    session_id: str = ""
    trace_tags: list[str] = Field(default_factory=list)
```

### 2. Configuration (`src/config.py`)

```python
"""AWS client configuration and shared resources."""

import os
import boto3
from dotenv import load_dotenv
from langchain_aws import ChatBedrockConverse, BedrockEmbeddings

load_dotenv()


def get_bedrock_llm(model: str = "sonnet", temperature: float = 0.0):
    """Get a Bedrock LLM instance.

    Args:
        model: "haiku" for routing/classification, "sonnet" for standard reasoning, "opus" for complex synthesis.
        temperature: Sampling temperature. 0.0 for deterministic outputs.
    """
    model_map = {
        "haiku": os.getenv("BEDROCK_HAIKU_MODEL_ID"),
        "sonnet": os.getenv("BEDROCK_MODEL_ID"),
        "opus": os.getenv("BEDROCK_OPUS_MODEL_ID"),
    }
    model_id = model_map.get(model, os.getenv("BEDROCK_MODEL_ID"))
    client = boto3.client("bedrock-runtime", region_name=os.getenv("AWS_REGION"))
    return ChatBedrockConverse(
        model=model_id,
        temperature=temperature,
        max_tokens=4096,
        client=client,
        guardrail_config={
            "guardrailIdentifier": os.getenv("BEDROCK_GUARDRAIL_ID"),
            "guardrailVersion": os.getenv("BEDROCK_GUARDRAIL_VERSION"),
        }
        if os.getenv("BEDROCK_GUARDRAIL_ID")
        else None,
    )


def get_embeddings():
    """Get Amazon Titan Embeddings model."""
    client = boto3.client("bedrock-runtime", region_name=os.getenv("AWS_REGION"))
    return BedrockEmbeddings(
        model_id=os.getenv("BEDROCK_EMBEDDING_MODEL_ID"),
        client=client,
    )


def get_s3_client():
    return boto3.client("s3", region_name=os.getenv("AWS_REGION"))


def get_dynamodb_resource():
    return boto3.resource("dynamodb", region_name=os.getenv("AWS_REGION"))


def get_textract_client():
    return boto3.client("textract", region_name=os.getenv("AWS_REGION"))
```

### 3. Tools (`src/tools/`)

#### `src/tools/textract_tool.py`

```python
"""Amazon Textract tool for document parsing."""

import json
import time
from langchain_core.tools import tool
from src.config import get_textract_client, get_s3_client


@tool
def extract_document_text(s3_bucket: str, s3_key: str) -> dict:
    """Extract text and tables from a financial document stored in S3 using Amazon Textract.

    Args:
        s3_bucket: S3 bucket name containing the document.
        s3_key: S3 object key for the document (PDF, PNG, JPEG).

    Returns:
        Dictionary with 'text' (full extracted text), 'tables' (list of extracted tables),
        and 'pages' (page count).
    """
    textract = get_textract_client()

    # Start async job for multi-page documents
    response = textract.start_document_analysis(
        DocumentLocation={"S3Object": {"Bucket": s3_bucket, "Name": s3_key}},
        FeatureTypes=["TABLES", "FORMS"],
    )
    job_id = response["JobId"]

    # Poll for completion
    while True:
        result = textract.get_document_analysis(JobId=job_id)
        status = result["JobStatus"]
        if status == "SUCCEEDED":
            break
        elif status == "FAILED":
            return {"error": f"Textract job failed: {result.get('StatusMessage', 'Unknown')}"}
        time.sleep(2)

    # Collect all pages
    pages, blocks = [], result["Blocks"]
    next_token = result.get("NextToken")
    while next_token:
        result = textract.get_document_analysis(JobId=job_id, NextToken=next_token)
        blocks.extend(result["Blocks"])
        next_token = result.get("NextToken")

    # Extract text lines
    text_lines = [b["Text"] for b in blocks if b["BlockType"] == "LINE"]

    # Extract tables
    tables = _extract_tables(blocks)

    return {
        "text": "\n".join(text_lines),
        "tables": tables,
        "pages": max((b.get("Page", 1) for b in blocks), default=1),
        "source": f"s3://{s3_bucket}/{s3_key}",
    }


def _extract_tables(blocks: list[dict]) -> list[list[list[str]]]:
    """Parse Textract blocks into structured tables."""
    block_map = {b["Id"]: b for b in blocks}
    tables = []

    for block in blocks:
        if block["BlockType"] != "TABLE":
            continue
        table = {}
        for rel in block.get("Relationships", []):
            if rel["Type"] == "CHILD":
                for cell_id in rel["Ids"]:
                    cell = block_map[cell_id]
                    if cell["BlockType"] == "CELL":
                        row = cell["RowIndex"]
                        col = cell["ColumnIndex"]
                        # Get cell text
                        cell_text = ""
                        for c_rel in cell.get("Relationships", []):
                            if c_rel["Type"] == "CHILD":
                                for word_id in c_rel["Ids"]:
                                    word = block_map.get(word_id, {})
                                    if word.get("BlockType") == "WORD":
                                        cell_text += word.get("Text", "") + " "
                        table.setdefault(row, {})[col] = cell_text.strip()

        if table:
            max_row = max(table.keys())
            max_col = max(max(cols.keys()) for cols in table.values())
            structured = []
            for r in range(1, max_row + 1):
                row_data = []
                for c in range(1, max_col + 1):
                    row_data.append(table.get(r, {}).get(c, ""))
                structured.append(row_data)
            tables.append(structured)

    return tables
```

#### `src/tools/s3_tools.py`

```python
"""S3 tools for document storage and retrieval."""

import os
import json
from datetime import datetime
from langchain_core.tools import tool
from src.config import get_s3_client


@tool
def list_filings(ticker: str, filing_type: str = "10-K") -> list[dict]:
    """List available SEC filings for a given ticker symbol in the S3 document store.

    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'MSFT').
        filing_type: SEC filing type ('10-K', '10-Q', '8-K', 'earnings_transcript').

    Returns:
        List of available filings with s3_key, filing_date, and filing_type.
    """
    s3 = get_s3_client()
    bucket = os.getenv("S3_DOCUMENTS_BUCKET")
    prefix = f"filings/{ticker.upper()}/{filing_type}/"

    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    filings = []
    for obj in response.get("Contents", []):
        filings.append({
            "s3_key": obj["Key"],
            "s3_bucket": bucket,
            "size_mb": round(obj["Size"] / (1024 * 1024), 2),
            "last_modified": obj["LastModified"].isoformat(),
            "ticker": ticker.upper(),
            "filing_type": filing_type,
        })
    return filings


@tool
def save_report(report_content: str, ticker: str, report_type: str) -> str:
    """Save a generated research report to S3.

    Args:
        report_content: The full report content in markdown format.
        ticker: Primary ticker symbol for the report.
        report_type: Type of report ('equity_research', 'earnings_analysis', 'competitive').

    Returns:
        S3 URI of the saved report.
    """
    s3 = get_s3_client()
    bucket = os.getenv("S3_REPORTS_BUCKET")
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    key = f"reports/{ticker.upper()}/{report_type}/{timestamp}.md"

    s3.put_object(Bucket=bucket, Key=key, Body=report_content.encode("utf-8"),
                  ContentType="text/markdown")
    return f"s3://{bucket}/{key}"
```

#### `src/tools/financial_data.py`

```python
"""Financial data retrieval tools (via MCP or direct API)."""

from langchain_core.tools import tool


@tool
def get_financial_metrics(ticker: str, period: str = "annual", years: int = 3) -> dict:
    """Retrieve key financial metrics for a company from the financial data API.

    Args:
        ticker: Stock ticker symbol.
        period: 'annual' or 'quarterly'.
        years: Number of years of historical data.

    Returns:
        Dictionary with revenue, net_income, eps, margins, and other key metrics
        organized by period.
    """
    # NOTE: In production, this calls your financial data MCP server or API.
    # Example integration with an MCP server via AgentCore Gateway:
    #
    #   from mcp import ClientSession
    #   async with ClientSession(server_url) as session:
    #       result = await session.call_tool("get_financials", {
    #           "ticker": ticker, "period": period, "years": years
    #       })
    #
    # For the implementation, wire this to your preferred data source:
    # - Financial Modeling Prep API
    # - Alpha Vantage
    # - Bloomberg B-PIPE
    # - Refinitiv
    # - Internal data warehouse via Athena/Redshift
    raise NotImplementedError(
        "Connect to your financial data source. "
        "See MCP server setup in infrastructure/ for AgentCore Gateway integration."
    )


@tool
def get_peer_comparison(ticker: str, metric: str = "revenue_growth") -> dict:
    """Get peer comparison data for a given company and metric.

    Args:
        ticker: Stock ticker symbol.
        metric: Metric to compare ('revenue_growth', 'gross_margin', 'pe_ratio', etc.)

    Returns:
        Dictionary with company value, peer median, peer list, and percentile rank.
    """
    raise NotImplementedError("Connect to your financial data source.")


@tool
def calculate_financial_ratios(
    revenue: float,
    net_income: float,
    total_assets: float,
    total_equity: float,
    current_assets: float,
    current_liabilities: float,
) -> dict:
    """Calculate standard financial ratios from input values.

    Args:
        revenue: Total revenue.
        net_income: Net income.
        total_assets: Total assets.
        total_equity: Total stockholders equity.
        current_assets: Current assets.
        current_liabilities: Current liabilities.

    Returns:
        Dictionary of calculated ratios: profit_margin, roa, roe, current_ratio.
    """
    return {
        "profit_margin": round(net_income / revenue * 100, 2) if revenue else None,
        "return_on_assets": round(net_income / total_assets * 100, 2) if total_assets else None,
        "return_on_equity": round(net_income / total_equity * 100, 2) if total_equity else None,
        "current_ratio": round(current_assets / current_liabilities, 2) if current_liabilities else None,
    }
```

### 4. RAG Pipeline (`src/rag/`)

#### `src/rag/retriever.py`

```python
"""OpenSearch vector retriever for financial documents."""

import os
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_core.retrievers import BaseRetriever
from src.config import get_embeddings


def get_opensearch_retriever(
    index_name: str | None = None,
    k: int = 5,
    score_threshold: float = 0.7,
) -> BaseRetriever:
    """Create an OpenSearch retriever for financial document search.

    Args:
        index_name: OpenSearch index name. Defaults to env var.
        k: Number of documents to retrieve.
        score_threshold: Minimum similarity score threshold.

    Returns:
        A LangChain retriever backed by OpenSearch Serverless.
    """
    embeddings = get_embeddings()
    index = index_name or os.getenv("OPENSEARCH_INDEX_NAME")

    vectorstore = OpenSearchVectorSearch(
        index_name=index,
        embedding_function=embeddings,
        opensearch_url=os.getenv("OPENSEARCH_ENDPOINT"),
        http_auth=None,  # Uses IAM auth via boto3 session
        use_ssl=True,
        verify_certs=True,
        connection_class="opensearchpy.RequestsAWSV4SignerAuth",
    )

    return vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": k, "score_threshold": score_threshold},
    )
```

#### `src/rag/indexer.py`

```python
"""Document indexing pipeline for financial documents."""

import os
from langchain_community.document_loaders import S3DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import OpenSearchVectorSearch
from src.config import get_embeddings


def index_documents(
    s3_bucket: str,
    s3_prefix: str,
    index_name: str | None = None,
    chunk_size: int = 1500,
    chunk_overlap: int = 200,
):
    """Index documents from S3 into OpenSearch for RAG retrieval.

    Args:
        s3_bucket: Source S3 bucket.
        s3_prefix: S3 prefix to scan for documents.
        index_name: Target OpenSearch index.
        chunk_size: Text chunk size in characters.
        chunk_overlap: Overlap between chunks.
    """
    # Load documents from S3
    loader = S3DirectoryLoader(bucket=s3_bucket, prefix=s3_prefix)
    documents = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_documents(documents)

    # Enrich metadata
    for chunk in chunks:
        chunk.metadata["source_bucket"] = s3_bucket
        # Extract ticker from path if present (e.g., filings/AAPL/10-K/...)
        parts = chunk.metadata.get("source", "").split("/")
        if len(parts) >= 3:
            chunk.metadata["ticker"] = parts[-3] if parts[-3].isupper() else ""

    # Index into OpenSearch
    embeddings = get_embeddings()
    index = index_name or os.getenv("OPENSEARCH_INDEX_NAME")

    OpenSearchVectorSearch.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=index,
        opensearch_url=os.getenv("OPENSEARCH_ENDPOINT"),
        use_ssl=True,
        verify_certs=True,
    )
    print(f"Indexed {len(chunks)} chunks into {index}")
```

### 5. Agent Implementations (`src/agents/`)

Deep Agents pattern recommendation for this use case:

- Use `deepagents` for complex comparative/regulatory synthesis when context is large or multi-document reconciliation is required.
- Keep deterministic retrieval, ranking, and citation assembly in the core LangGraph pipeline.
- Use model sizing by task: Haiku for route classification, Sonnet 4.5 for standard synthesis, Opus for highest-complexity analysis.

#### `src/agents/supervisor.py`

```python
"""Supervisor agent: routes queries to specialist agents."""

from langchain_core.messages import SystemMessage, HumanMessage
from src.state import ResearchState
from src.config import get_bedrock_llm

SUPERVISOR_PROMPT = """You are a financial research supervisor agent. Your job is to analyze
the user's query and determine which specialist agents should handle it, and in what order.

Available specialist agents:
1. document_ingestion - Parses SEC filings and financial documents from S3, extracts text and tables
2. financial_analysis - Performs quantitative analysis: revenue trends, margin calculations, ratio analysis
3. market_research - Retrieves market context: news, peer comparisons, industry benchmarks
4. report_generation - Synthesizes all findings into a structured research report

Classification rules:
- "single_company_analysis": User asks about one company → ingestion → analysis → research → report
- "competitive_comparison": User compares companies → ingestion (all) → analysis → research → report
- "earnings_review": User asks about earnings → ingestion → analysis → report
- "filing_deep_dive": User asks about a specific filing → ingestion → analysis → report
- "sector_overview": User asks about an industry/sector → research → analysis → report
- "general_question": Simple factual question → answer directly, no specialist agents needed

Respond with a JSON object:
{
    "query_type": "<classification>",
    "tickers": ["<TICKER1>", ...],
    "agent_sequence": ["<agent1>", "<agent2>", ...],
    "requires_parallel": <true if agents can run concurrently, false for sequential>,
    "reasoning": "<brief explanation of your routing decision>"
}
"""


def supervisor_node(state: ResearchState) -> dict:
    """Analyze the user query and determine routing."""
    llm = get_bedrock_llm(model="haiku", temperature=0.0)

    response = llm.invoke([
        SystemMessage(content=SUPERVISOR_PROMPT),
        HumanMessage(content=f"User query: {state.user_query}\n\n"
                     f"Previously completed agents: {state.agents_completed}\n"
                     f"Current findings so far: {len(state.extracted_metrics)} metrics extracted, "
                     f"{len(state.analysis_results)} analyses completed."),
    ])

    # Parse the routing decision
    import json
    try:
        decision = json.loads(response.content)
    except json.JSONDecodeError:
        # Fallback: route through all agents
        decision = {
            "query_type": "single_company_analysis",
            "tickers": [],
            "agent_sequence": ["document_ingestion", "financial_analysis",
                               "market_research", "report_generation"],
            "requires_parallel": False,
        }

    # Determine next agent (first uncompleted agent in sequence)
    remaining = [a for a in decision.get("agent_sequence", [])
                 if a not in state.agents_completed]
    next_agent = remaining[0] if remaining else "report_generation"

    return {
        "query_type": decision.get("query_type"),
        "tickers": decision.get("tickers", []),
        "next_agent": next_agent,
        "requires_parallel": decision.get("requires_parallel", False),
    }
```

#### `src/agents/document_ingestion.py`

```python
"""Document Ingestion Agent: parses financial documents and extracts structured data."""

import os
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from src.state import ResearchState, DocumentChunk, ExtractedMetric
from src.config import get_bedrock_llm
from src.tools.textract_tool import extract_document_text
from src.tools.s3_tools import list_filings

INGESTION_PROMPT = """You are a financial document parsing specialist. Your job is to:
1. Identify which documents need to be retrieved based on the tickers and query
2. Extract key financial metrics from the parsed documents
3. Organize extracted data into structured format

When extracting metrics, focus on:
- Revenue, net income, EPS (basic and diluted)
- Gross margin, operating margin, net margin
- Cash flow from operations, free cash flow
- Total assets, total debt, shareholders' equity
- Any metrics specifically relevant to the user's query

For each metric, note the exact value, the period it covers, and the source page.
Always maintain the original precision of numbers from the source document.
"""


def document_ingestion_node(state: ResearchState) -> dict:
    """Parse documents and extract financial data for the specified tickers."""
    llm = get_bedrock_llm(model="sonnet", temperature=0.0)
    parsed_docs = []
    extracted_metrics = []
    bucket = os.getenv("S3_DOCUMENTS_BUCKET")

    for ticker in state.tickers:
        # Find available filings
        filings = list_filings.invoke({"ticker": ticker, "filing_type": "10-K"})

        if not filings:
            continue

        # Process the most recent filing
        latest = filings[0]
        textract_result = extract_document_text.invoke({
            "s3_bucket": latest["s3_bucket"],
            "s3_key": latest["s3_key"],
        })

        if "error" in textract_result:
            continue

        # Create document chunk
        parsed_docs.append(DocumentChunk(
            content=textract_result["text"][:10000],  # Truncate for context window
            source=textract_result["source"],
            doc_type="10-K",
            ticker=ticker,
            metadata={"tables": textract_result.get("tables", [])[:5]},
        ))

        # Use LLM to extract key metrics from the parsed text
        extraction_response = llm.invoke([
            SystemMessage(content=INGESTION_PROMPT),
            HumanMessage(content=f"Extract key financial metrics from this {ticker} 10-K filing:\n\n"
                         f"{textract_result['text'][:8000]}\n\n"
                         f"Tables found: {textract_result.get('tables', [])[:3]}\n\n"
                         f"User's original query for context: {state.user_query}\n\n"
                         "Respond with a JSON list of metrics, each with: "
                         "metric_name, value, period, confidence (0-1).")
        ])

        # Parse extracted metrics
        import json
        try:
            metrics_data = json.loads(extraction_response.content)
            for m in metrics_data:
                extracted_metrics.append(ExtractedMetric(
                    metric_name=m["metric_name"],
                    value=str(m["value"]),
                    period=m.get("period", "unknown"),
                    source_document=textract_result["source"],
                    confidence=m.get("confidence", 0.8),
                ))
        except (json.JSONDecodeError, KeyError):
            pass

    return {
        "parsed_documents": parsed_docs,
        "extracted_metrics": extracted_metrics,
        "agents_completed": state.agents_completed + ["document_ingestion"],
        "messages": [AIMessage(content=f"Document ingestion complete. "
                               f"Parsed {len(parsed_docs)} documents, "
                               f"extracted {len(extracted_metrics)} metrics.")],
    }
```

#### `src/agents/financial_analysis.py`

```python
"""Financial Analysis Agent: quantitative analysis and ratio calculations."""

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from src.state import ResearchState, AnalysisResult
from src.config import get_bedrock_llm
from src.tools.financial_data import calculate_financial_ratios

ANALYSIS_PROMPT = """You are a quantitative financial analyst. Given extracted financial metrics
and document data, perform rigorous quantitative analysis.

Your analysis should include:
1. Trend analysis: YoY and QoQ growth rates for key metrics
2. Margin analysis: Gross, operating, and net margin trends
3. Ratio analysis: Profitability, liquidity, leverage, and efficiency ratios
4. Peer benchmarking: Compare metrics against industry averages if available
5. Key drivers: Identify what is driving changes in financial performance

Be precise with numbers. Show your calculations. Flag any data quality concerns.
Structure your analysis clearly with sections and supporting data.
"""


def financial_analysis_node(state: ResearchState) -> dict:
    """Perform quantitative financial analysis on extracted data."""
    llm = get_bedrock_llm(model="sonnet", temperature=0.0)

    # Prepare metrics summary for the LLM
    metrics_text = "\n".join(
        f"- {m.metric_name}: {m.value} (Period: {m.period}, Source: {m.source_document})"
        for m in state.extracted_metrics
    )

    doc_context = "\n\n".join(
        f"[{d.ticker} - {d.doc_type}]\n{d.content[:3000]}"
        for d in state.parsed_documents
    )

    response = llm.invoke([
        SystemMessage(content=ANALYSIS_PROMPT),
        HumanMessage(content=f"User query: {state.user_query}\n\n"
                     f"Extracted metrics:\n{metrics_text}\n\n"
                     f"Document excerpts:\n{doc_context}\n\n"
                     "Perform a comprehensive quantitative analysis. "
                     "Structure your response as a detailed analysis with clear sections."),
    ])

    analysis = AnalysisResult(
        analysis_type=state.query_type or "general",
        findings=response.content,
        metrics=state.extracted_metrics,
    )

    return {
        "analysis_results": state.analysis_results + [analysis],
        "agents_completed": state.agents_completed + ["financial_analysis"],
        "messages": [AIMessage(content=f"Financial analysis complete for query type: {state.query_type}")],
    }
```

#### `src/agents/market_research.py`

```python
"""Market Research Agent: news, peer comparisons, and macro context."""

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from src.state import ResearchState, MarketContext
from src.config import get_bedrock_llm
from src.rag.retriever import get_opensearch_retriever

RESEARCH_PROMPT = """You are a financial market research analyst. Provide comprehensive
market context for the companies and topics being analyzed.

Your research should cover:
1. Recent news and events affecting the company/sector
2. Competitive landscape and peer positioning
3. Industry trends and growth drivers
4. Macroeconomic factors relevant to the analysis
5. Key risks and catalysts to watch

Cite your sources. Distinguish between confirmed facts and analyst opinions.
Focus on information that is material to investment decision-making.
"""


def market_research_node(state: ResearchState) -> dict:
    """Gather market context, news, and peer comparisons."""
    llm = get_bedrock_llm(model="sonnet", temperature=0.0)
    retriever = get_opensearch_retriever(k=8)

    # Build search queries based on tickers and query
    search_queries = []
    for ticker in state.tickers:
        search_queries.append(f"{ticker} recent financial performance news")
        search_queries.append(f"{ticker} competitive landscape industry position")

    if state.query_type == "sector_overview":
        search_queries.append(state.user_query)

    # Retrieve relevant documents from knowledge base
    retrieved_docs = []
    for query in search_queries[:4]:  # Limit queries
        docs = retriever.invoke(query)
        retrieved_docs.extend(docs)

    # Deduplicate
    seen = set()
    unique_docs = []
    for doc in retrieved_docs:
        key = doc.page_content[:100]
        if key not in seen:
            seen.add(key)
            unique_docs.append(doc)

    context = "\n\n".join(
        f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}"
        for d in unique_docs[:10]
    )

    response = llm.invoke([
        SystemMessage(content=RESEARCH_PROMPT),
        HumanMessage(content=f"User query: {state.user_query}\n"
                     f"Tickers: {state.tickers}\n\n"
                     f"Retrieved context:\n{context}\n\n"
                     "Provide comprehensive market context. "
                     "Structure your response with clear sections for "
                     "news summary, peer comparison, industry context, and key risks."),
    ])

    market_context = MarketContext(
        news_summary=response.content,
        sources=[d.metadata.get("source", "") for d in unique_docs],
    )

    return {
        "market_context": market_context,
        "agents_completed": state.agents_completed + ["market_research"],
        "messages": [AIMessage(content="Market research complete.")],
    }
```

#### `src/agents/report_generation.py`

```python
"""Report Generation Agent: synthesizes findings into structured research reports."""

import os
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from src.state import ResearchState
from src.config import get_bedrock_llm
from src.tools.s3_tools import save_report

REPORT_PROMPT = """You are a senior equity research analyst preparing a research report.
Synthesize all findings from the document analysis, quantitative analysis, and market research
into a comprehensive, professionally structured report.

Report structure:
1. **Executive Summary** - Key findings in 3-5 bullet points
2. **Company Overview** - Brief description of the company/companies analyzed
3. **Financial Performance** - Key metrics, trends, and ratio analysis with data tables
4. **Competitive Positioning** - Peer comparison and market position
5. **Industry & Macro Context** - Market environment and sector dynamics
6. **Key Risks & Catalysts** - Material risks and potential catalysts
7. **Conclusion** - Summary assessment and key takeaways

Formatting rules:
- Use markdown formatting with clear headers
- Include data tables for financial comparisons
- Cite all sources with [Source: <document>] notation
- Be specific with numbers - never round without noting it
- Distinguish between facts and interpretations
- Flag any data gaps or low-confidence extractions
"""


def report_generation_node(state: ResearchState) -> dict:
    """Generate the final research report from all collected data."""
    llm = get_bedrock_llm(model="sonnet", temperature=0.1)

    # Compile all findings
    metrics_summary = "\n".join(
        f"- {m.metric_name}: {m.value} ({m.period}) [Source: {m.source_document}]"
        for m in state.extracted_metrics
    )

    analysis_summary = "\n\n".join(
        f"### Analysis: {a.analysis_type}\n{a.findings}"
        for a in state.analysis_results
    )

    market_summary = state.market_context.news_summary if state.market_context else "No market context gathered."

    response = llm.invoke([
        SystemMessage(content=REPORT_PROMPT),
        HumanMessage(content=f"Original query: {state.user_query}\n"
                     f"Tickers: {state.tickers}\n"
                     f"Query type: {state.query_type}\n\n"
                     f"--- EXTRACTED METRICS ---\n{metrics_summary}\n\n"
                     f"--- QUANTITATIVE ANALYSIS ---\n{analysis_summary}\n\n"
                     f"--- MARKET CONTEXT ---\n{market_summary}\n\n"
                     "Generate a comprehensive research report following the specified structure."),
    ])

    # Save report to S3
    primary_ticker = state.tickers[0] if state.tickers else "MULTI"
    report_s3_key = save_report.invoke({
        "report_content": response.content,
        "ticker": primary_ticker,
        "report_type": state.query_type or "general",
    })

    return {
        "report_markdown": response.content,
        "report_s3_key": report_s3_key,
        "agents_completed": state.agents_completed + ["report_generation"],
        "messages": [AIMessage(content=response.content)],
    }
```

### 6. Main Graph (`src/graph.py`)

```python
"""Main LangGraph workflow definition for the document research agent."""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from src.state import ResearchState
from src.agents.supervisor import supervisor_node
from src.agents.document_ingestion import document_ingestion_node
from src.agents.financial_analysis import financial_analysis_node
from src.agents.market_research import market_research_node
from src.agents.report_generation import report_generation_node


def route_to_agent(state: ResearchState) -> str:
    """Conditional edge: route to the next agent based on supervisor decision."""
    if state.error:
        return "report_generation"  # Generate report with whatever we have
    if state.next_agent in ("document_ingestion", "financial_analysis",
                            "market_research", "report_generation"):
        return state.next_agent
    return "report_generation"


def should_continue(state: ResearchState) -> str:
    """After each agent, decide whether to continue to next agent or generate report."""
    if state.error:
        return "report_generation"
    if "report_generation" in state.agents_completed:
        return END
    # Return to supervisor to route to next agent
    return "supervisor"


# Build the graph
workflow = StateGraph(ResearchState)

# Add nodes
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("document_ingestion", document_ingestion_node)
workflow.add_node("financial_analysis", financial_analysis_node)
workflow.add_node("market_research", market_research_node)
workflow.add_node("report_generation", report_generation_node)

# Entry point
workflow.add_edge(START, "supervisor")

# Supervisor routes to agents
workflow.add_conditional_edges("supervisor", route_to_agent, {
    "document_ingestion": "document_ingestion",
    "financial_analysis": "financial_analysis",
    "market_research": "market_research",
    "report_generation": "report_generation",
})

# After each agent, go back to supervisor or end
workflow.add_conditional_edges("document_ingestion", should_continue, {
    "supervisor": "supervisor",
    "report_generation": "report_generation",
    END: END,
})
workflow.add_conditional_edges("financial_analysis", should_continue, {
    "supervisor": "supervisor",
    "report_generation": "report_generation",
    END: END,
})
workflow.add_conditional_edges("market_research", should_continue, {
    "supervisor": "supervisor",
    "report_generation": "report_generation",
    END: END,
})
workflow.add_edge("report_generation", END)

# Compile with checkpointing
memory = MemorySaver()
graph = workflow.compile(
    checkpointer=memory,
    interrupt_before=["report_generation"],  # Human-in-the-loop before final report
)
```

---

## Evaluation Suite

### `evaluation/evaluators/accuracy_evaluator.py`

```python
"""LLM-as-Judge evaluator for financial metric accuracy."""

from langsmith.evaluation import EvaluationResult, run_evaluator


@run_evaluator
def financial_accuracy_evaluator(run, example) -> EvaluationResult:
    """Evaluate whether extracted financial metrics match the ground truth.

    Scores 1-5:
    5 = All metrics exactly correct
    4 = Minor rounding differences only
    3 = Most metrics correct, 1-2 errors
    2 = Several metrics incorrect
    1 = Fundamentally wrong or hallucinated values
    """
    from src.config import get_bedrock_llm

    llm = get_bedrock_llm(model="sonnet", temperature=0.0)

    prediction = run.outputs.get("extracted_metrics", [])
    reference = example.outputs.get("expected_metrics", [])

    prompt = f"""You are evaluating the accuracy of financial metrics extracted by an AI agent.

Ground truth metrics:
{reference}

Agent-extracted metrics:
{prediction}

Score the extraction accuracy on a 1-5 scale:
5 = All metrics exactly match ground truth
4 = Minor differences (rounding, formatting) but values are correct
3 = Most metrics correct, 1-2 substantive errors
2 = Several metrics are wrong or missing
1 = Fundamentally incorrect, hallucinated values, or mostly missing

Respond with JSON: {{"score": <1-5>, "reasoning": "<explanation>"}}"""

    import json
    response = llm.invoke(prompt)
    try:
        result = json.loads(response.content)
        return EvaluationResult(
            key="financial_accuracy",
            score=result["score"] / 5.0,  # Normalize to 0-1
            comment=result.get("reasoning", ""),
        )
    except (json.JSONDecodeError, KeyError):
        return EvaluationResult(key="financial_accuracy", score=0.0, comment="Eval parse error")
```

### `evaluation/evaluators/completeness_evaluator.py`

```python
"""Heuristic + LLM evaluator for report completeness."""

from langsmith.evaluation import EvaluationResult, run_evaluator


@run_evaluator
def report_completeness_evaluator(run, example) -> EvaluationResult:
    """Check whether the generated report contains all required sections."""
    report = run.outputs.get("report_markdown", "")

    required_sections = [
        "Executive Summary",
        "Financial Performance",
        "Competitive",
        "Risk",
        "Conclusion",
    ]

    found = sum(1 for section in required_sections if section.lower() in report.lower())
    score = found / len(required_sections)

    missing = [s for s in required_sections if s.lower() not in report.lower()]

    return EvaluationResult(
        key="report_completeness",
        score=score,
        comment=f"Missing sections: {missing}" if missing else "All sections present",
    )
```

### `evaluation/run_evals.py`

```python
"""CLI entry point for running the evaluation suite."""

import argparse
from langsmith import Client
from langsmith.evaluation import evaluate
from src.graph import graph
from evaluation.evaluators.accuracy_evaluator import financial_accuracy_evaluator
from evaluation.evaluators.completeness_evaluator import report_completeness_evaluator


def run_target(inputs: dict) -> dict:
    """Invoke the graph for evaluation."""
    result = graph.invoke(
        {"user_query": inputs["query"], "tickers": inputs.get("tickers", [])},
        config={"configurable": {"thread_id": f"eval-{inputs.get('id', 'test')}"}},
    )
    return {
        "report_markdown": result.get("report_markdown", ""),
        "extracted_metrics": [m.model_dump() for m in result.get("extracted_metrics", [])],
    }


def main():
    parser = argparse.ArgumentParser(description="Run evaluation suite")
    parser.add_argument("--dataset", required=True, help="LangSmith dataset name")
    parser.add_argument("--experiment-prefix", default="doc-research", help="Experiment name prefix")
    args = parser.parse_args()

    results = evaluate(
        run_target,
        data=args.dataset,
        evaluators=[
            financial_accuracy_evaluator,
            report_completeness_evaluator,
        ],
        experiment_prefix=args.experiment_prefix,
        max_concurrency=2,
    )
    print(f"Evaluation complete. Results: {results}")


if __name__ == "__main__":
    main()
```

---

## LangSmith Tracing Configuration

### Tracing is automatic with LangGraph. Ensure these environment variables are set:

```bash
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY=lsv2_pt_xxxxxxxxxxxx
export LANGSMITH_PROJECT=document-research-agent
```

### Custom metadata tagging per invocation:

```python
import langsmith as ls

with ls.tracing_context(
    project_name="document-research-agent",
    tags=["production", "10-K-analysis", "v2.1"],
    metadata={
        "user_id": "analyst_42",
        "tickers": ["AAPL", "MSFT"],
        "document_type": "10-K",
        "environment": "production",
    },
):
    result = graph.invoke(
        {"user_query": "Compare AAPL and MSFT revenue growth over the past 3 years",
         "tickers": ["AAPL", "MSFT"]},
        config={"configurable": {"thread_id": "session-abc123"}},
    )
```

### LangSmith Monitoring Dashboard Configuration

Set up the following custom dashboards in LangSmith:

| Dashboard | Metrics | Alert Threshold |
|-----------|---------|-----------------|
| Agent Latency | P50, P99 latency per agent node | P99 > 30s |
| Token Cost | Token usage by agent, model breakdown | Daily cost > $X |
| Error Rate | Tool failures, LLM errors, parse failures | Error rate > 5% |
| Quality Score | Rolling average of eval scores from online evals | Score < 0.7 |

---

## AWS Infrastructure (CDK)

### Key resources to provision:

```python
# infrastructure/cdk/stacks/data_stack.py — key resources

# S3 buckets for documents and reports (with versioning, encryption)
# DynamoDB table for conversation memory (PAY_PER_REQUEST billing)
# DynamoDB table for extracted metrics cache
# OpenSearch Serverless collection for vector search
# KMS key for encryption (customer-managed)
# VPC with private subnets for agent runtime
# VPC endpoints for Bedrock, S3, DynamoDB, OpenSearch, Textract
```

### EventBridge rule for automated processing:

```python
# Trigger agent workflow when new SEC filing arrives in S3
# S3 → EventBridge → Lambda → LangGraph workflow invocation
```

---

## Testing Strategy

| Test Type | What to Test | Tool |
|-----------|-------------|------|
| Unit | Tool functions, state transitions, prompt templates | pytest |
| Integration | End-to-end graph with mock LLM, RAG pipeline | pytest + moto |
| Eval (Offline) | Golden dataset regression tests | LangSmith evaluate() |
| Eval (Online) | Production traffic quality monitoring | LangSmith online evals |
| Load | Concurrent agent invocations, scaling behavior | locust |

---

## Deployment Checklist

1. Provision AWS infrastructure via CDK (`cdk deploy --all`)
2. Index initial document corpus into OpenSearch (`python -m src.rag.indexer`)
3. Create LangSmith evaluation datasets from golden examples
4. Run offline evaluation suite and verify >80% quality threshold
5. Deploy agent to AgentCore Runtime (`agentcore deploy`)
6. Configure EventBridge rules for automated filing processing
7. Set up LangSmith monitoring dashboards and PagerDuty alerts
8. Run integration tests against deployed endpoint
9. Enable online evaluations on production traffic
10. Document runbook for common failure modes and escalation procedures
