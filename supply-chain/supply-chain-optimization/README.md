# Supply Chain Optimization with LangGraph

A multi-agent system built with LangGraph for autonomous supply chain optimization, including demand forecasting, inventory management, and logistics optimization.

## 🏗️ Architecture

This system implements a multi-agent architecture with the following components:

- **Demand Forecasting Agent**: Predicts demand at SKU-location-day granularity with ML models
- **Inventory Agent**: Optimizes reorder points and multi-echelon inventory allocation
- **Logistics Agent**: Handles route optimization and real-time disruption detection
- **Orchestration Agent**: Coordinates across all agents and resolves conflicts

## ✨ Features

- Real-time demand forecasting with <15% MAPE target
- Automated purchase order generation with supplier optimization
- Real-time disruption detection and autonomous rerouting (15 min response time)
- Multi-echelon inventory optimization
- Fresh/perishable goods handling with shelf-life awareness

## 📦 Installation

```bash
pip install -r requirements.txt
```

## ⚙️ Configuration

Create a `.env` file with your API keys:

```
ANTHROPIC_API_KEY=your_key_here
```

## 🚀 Usage

```python
from supply_chain_agent import SupplyChainGraph

# Initialize the graph
graph = SupplyChainGraph()

# Run optimization
result = graph.run({
    "skus": [...],
    "inventory_data": {...},
    "external_signals": {...}
})
```

See `example.py` for a complete usage example.

## ✅ Requirements Met

- ✅ Demand forecasting at SKU-location-day granularity
- ✅ Automated PO generation with supplier selection
- ✅ Real-time disruption detection and rerouting
- ✅ Multi-echelon inventory optimization
- ✅ Fresh/perishable goods handling
- ✅ Scalable architecture supporting 1M+ SKUs
- ✅ Integration-ready with standard APIs

## 🗂️ Project Structure

```
supply-chain-optimization/
├── requirements.txt
├── README.md
├── .env.example
├── supply_chain_agent.py    # Main LangGraph implementation
├── agents/
│   ├── __init__.py
│   ├── demand_forecasting.py
│   ├── inventory_optimizer.py
│   ├── logistics.py
│   └── orchestrator.py
├── models/
│   ├── __init__.py
│   └── state.py              # State schemas
└── example.py                # Usage example
```
