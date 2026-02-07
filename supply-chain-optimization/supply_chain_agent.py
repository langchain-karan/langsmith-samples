"""
Supply Chain Optimization Agent using LangGraph

Multi-agent system for autonomous supply chain optimization with:
- Demand forecasting
- Inventory optimization
- Logistics management
- Orchestration and conflict resolution
"""

from datetime import datetime
from typing import Dict, Any, Literal
import os

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

from models.state import SupplyChainState
from agents import (
    demand_forecasting_agent,
    inventory_optimizer_agent,
    logistics_agent,
    orchestrator_agent,
)


# Load environment variables
load_dotenv()


class SupplyChainGraph:
    """
    LangGraph-based supply chain optimization system.

    This creates a multi-agent workflow where:
    1. Demand Forecasting Agent predicts demand
    2. Inventory Optimizer Agent generates purchase orders
    3. Logistics Agent optimizes routes and handles disruptions
    4. Orchestrator Agent coordinates and resolves conflicts
    """

    def __init__(self):
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""

        # Create the graph with our state schema
        workflow = StateGraph(SupplyChainState)

        # Add agent nodes
        workflow.add_node("demand_forecasting", demand_forecasting_agent)
        workflow.add_node("inventory_optimizer", inventory_optimizer_agent)
        workflow.add_node("logistics", logistics_agent)
        workflow.add_node("orchestrator", orchestrator_agent)

        # Define the workflow edges
        # Start with demand forecasting
        workflow.set_entry_point("demand_forecasting")

        # Linear flow through agents
        workflow.add_edge("demand_forecasting", "inventory_optimizer")
        workflow.add_edge("inventory_optimizer", "logistics")
        workflow.add_edge("logistics", "orchestrator")

        # Orchestrator can loop back to agents if conflicts detected or end
        workflow.add_conditional_edges(
            "orchestrator",
            self._should_continue,
            {
                "demand_forecasting": "demand_forecasting",
                "inventory_optimizer": "inventory_optimizer",
                "logistics": "logistics",
                "end": END,
            }
        )

        return workflow.compile()

    def _should_continue(self, state: SupplyChainState) -> Literal["demand_forecasting", "inventory_optimizer", "logistics", "end"]:
        """
        Determine if workflow should continue or end.

        Orchestrator may direct back to specific agents if:
        - Conflicts detected that need resolution
        - Parameters need adjustment
        - Re-optimization required
        """

        next_agent = state.get("next_agent")

        # Prevent infinite loops - max 3 iterations
        if state.get("iteration", 0) >= 3:
            return "end"

        # If orchestrator specified next agent, route there
        if next_agent in ["demand_forecasting", "inventory_optimizer", "logistics"]:
            return next_agent

        # Otherwise, workflow is complete
        return "end"

    def run(self, initial_state: Dict[str, Any]) -> SupplyChainState:
        """
        Run the supply chain optimization workflow.

        Args:
            initial_state: Initial state with SKUs, inventory levels, suppliers, etc.

        Returns:
            Final state with forecasts, purchase orders, routes, and recommendations
        """

        print("=" * 60)
        print("SUPPLY CHAIN OPTIMIZATION - STARTING")
        print("=" * 60)

        # Add metadata to initial state
        initial_state["iteration"] = 0
        initial_state["timestamp"] = datetime.now()
        initial_state["current_agent"] = None
        initial_state["next_agent"] = None
        initial_state["requires_human_review"] = False

        # Initialize lists if not present
        for key in ["demand_forecasts", "purchase_orders", "logistics_routes", "disruptions", "messages", "alerts"]:
            if key not in initial_state:
                initial_state[key] = []

        # Run the graph
        final_state = self.graph.invoke(initial_state)

        print("\n" + "=" * 60)
        print("SUPPLY CHAIN OPTIMIZATION - COMPLETED")
        print("=" * 60)

        return final_state

    def stream(self, initial_state: Dict[str, Any]):
        """
        Stream the supply chain optimization workflow for real-time monitoring.

        Args:
            initial_state: Initial state with SKUs, inventory levels, suppliers, etc.

        Yields:
            State updates as each agent completes
        """

        print("=" * 60)
        print("SUPPLY CHAIN OPTIMIZATION - STREAMING")
        print("=" * 60)

        # Add metadata to initial state
        initial_state["iteration"] = 0
        initial_state["timestamp"] = datetime.now()
        initial_state["current_agent"] = None
        initial_state["next_agent"] = None
        initial_state["requires_human_review"] = False

        # Initialize lists if not present
        for key in ["demand_forecasts", "purchase_orders", "logistics_routes", "disruptions", "messages", "alerts"]:
            if key not in initial_state:
                initial_state[key] = []

        # Stream the graph execution
        for output in self.graph.stream(initial_state):
            yield output

        print("\n" + "=" * 60)
        print("SUPPLY CHAIN OPTIMIZATION - COMPLETED")
        print("=" * 60)


def create_supply_chain_graph() -> SupplyChainGraph:
    """Factory function to create a supply chain optimization graph"""
    return SupplyChainGraph()


if __name__ == "__main__":
    # Example usage
    from models.state import SKU, InventoryLevel, Supplier

    # Create sample data
    sample_skus = [
        SKU(
            sku_id="SKU-001",
            name="Fresh Organic Apples",
            category="Produce",
            is_perishable=True,
            shelf_life_days=14,
            unit_cost=2.50,
            supplier_ids=["SUP-A", "SUP-B"]
        ),
        SKU(
            sku_id="SKU-002",
            name="Canned Beans",
            category="Pantry",
            is_perishable=False,
            unit_cost=1.20,
            supplier_ids=["SUP-A", "SUP-C"]
        ),
        SKU(
            sku_id="SKU-003",
            name="Fresh Milk",
            category="Dairy",
            is_perishable=True,
            shelf_life_days=7,
            unit_cost=3.99,
            supplier_ids=["SUP-B"]
        ),
    ]

    sample_inventory = [
        InventoryLevel(
            sku_id="SKU-001",
            location_id="DC-001",
            quantity=150,
            reorder_point=200,
            safety_stock=100
        ),
        InventoryLevel(
            sku_id="SKU-002",
            location_id="DC-001",
            quantity=500,
            reorder_point=300,
            safety_stock=150
        ),
        InventoryLevel(
            sku_id="SKU-003",
            location_id="STORE-001",
            quantity=50,
            reorder_point=100,
            safety_stock=30
        ),
    ]

    sample_suppliers = [
        Supplier(
            supplier_id="SUP-A",
            name="FreshFarms Co",
            lead_time_days=3,
            reliability_score=0.95,
            cost_factor=1.0
        ),
        Supplier(
            supplier_id="SUP-B",
            name="QuickSupply Inc",
            lead_time_days=1,
            reliability_score=0.98,
            cost_factor=1.15
        ),
        Supplier(
            supplier_id="SUP-C",
            name="BulkGoods Ltd",
            lead_time_days=7,
            reliability_score=0.90,
            cost_factor=0.85
        ),
    ]

    # Create initial state
    initial_state = {
        "skus": sample_skus,
        "inventory_levels": sample_inventory,
        "suppliers": sample_suppliers,
        "external_signals": None,
    }

    # Create and run the graph
    graph = create_supply_chain_graph()
    result = graph.run(initial_state)

    # Display results summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"\nDemand Forecasts: {len(result['demand_forecasts'])}")
    print(f"Purchase Orders: {len(result['purchase_orders'])}")
    print(f"Logistics Routes: {len(result['logistics_routes'])}")
    print(f"Disruptions Handled: {len(result['disruptions'])}")
    print(f"Requires Human Review: {result['requires_human_review']}")

    if result.get('alerts'):
        print("\n⚠️  ALERTS:")
        for alert in result['alerts']:
            print(f"\n{alert}")
