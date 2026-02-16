"""
Example usage of the Supply Chain Optimization system

This demonstrates how to use the LangGraph-based multi-agent system
for supply chain optimization tasks.
"""

from datetime import datetime
from supply_chain_agent import create_supply_chain_graph
from models.state import SKU, InventoryLevel, Supplier, ExternalSignals


def basic_example():
    """Basic example with minimal data"""

    print("=" * 70)
    print("BASIC EXAMPLE: Simple Supply Chain Optimization")
    print("=" * 70)

    # Define SKUs
    skus = [
        SKU(
            sku_id="PROD-001",
            name="Premium Coffee Beans",
            category="Beverages",
            is_perishable=False,
            unit_cost=12.99,
            supplier_ids=["SUPPLIER-001"]
        ),
        SKU(
            sku_id="PROD-002",
            name="Fresh Strawberries",
            category="Produce",
            is_perishable=True,
            shelf_life_days=5,
            unit_cost=4.50,
            supplier_ids=["SUPPLIER-002", "SUPPLIER-003"]
        ),
    ]

    # Define current inventory
    inventory = [
        InventoryLevel(
            sku_id="PROD-001",
            location_id="WAREHOUSE-CENTRAL",
            quantity=50,
            reorder_point=100,
            safety_stock=75
        ),
        InventoryLevel(
            sku_id="PROD-002",
            location_id="STORE-DOWNTOWN",
            quantity=20,
            reorder_point=50,
            safety_stock=30
        ),
    ]

    # Define suppliers
    suppliers = [
        Supplier(
            supplier_id="SUPPLIER-001",
            name="Global Coffee Importers",
            lead_time_days=5,
            reliability_score=0.95,
            cost_factor=1.0
        ),
        Supplier(
            supplier_id="SUPPLIER-002",
            name="Local Farms Fresh",
            lead_time_days=1,
            reliability_score=0.98,
            cost_factor=1.2
        ),
        Supplier(
            supplier_id="SUPPLIER-003",
            name="Regional Produce Co",
            lead_time_days=2,
            reliability_score=0.92,
            cost_factor=0.95
        ),
    ]

    # Create initial state
    initial_state = {
        "skus": skus,
        "inventory_levels": inventory,
        "suppliers": suppliers,
    }

    # Run optimization
    graph = create_supply_chain_graph()
    result = graph.run(initial_state)

    # Display key metrics
    print("\n" + "=" * 70)
    print("OPTIMIZATION RESULTS")
    print("=" * 70)

    print(f"\n📊 Demand Forecasts Generated: {len(result['demand_forecasts'])}")
    print(f"📦 Purchase Orders Created: {len(result['purchase_orders'])}")
    print(f"🚚 Logistics Routes Optimized: {len(result['logistics_routes'])}")

    # Show purchase orders
    if result['purchase_orders']:
        print("\n📋 PURCHASE ORDERS:")
        for po in result['purchase_orders'][:3]:
            print(f"  • {po.po_id}: {po.quantity} units of {po.sku_id}")
            print(f"    Supplier: {po.supplier_id} | Priority: {po.priority}")
            print(f"    Cost: ${po.total_cost:.2f} | Delivery: {po.expected_delivery.strftime('%Y-%m-%d')}")

    # Show routes
    if result['logistics_routes']:
        print("\n🗺️  LOGISTICS ROUTES:")
        for route in result['logistics_routes'][:3]:
            print(f"  • {route.route_id}: {len(route.po_ids)} orders")
            print(f"    {route.origin} → {route.destination} via {route.carrier}")
            print(f"    Duration: {route.estimated_duration_hours:.1f}h | Cost: ${route.estimated_cost:.2f}")

    return result


def advanced_example_with_external_signals():
    """Advanced example with external data signals"""

    print("\n\n" + "=" * 70)
    print("ADVANCED EXAMPLE: With External Signals & Multiple Locations")
    print("=" * 70)

    # Create more SKUs
    skus = [
        SKU(
            sku_id="ELEC-001",
            name="Smartphone Model X",
            category="Electronics",
            is_perishable=False,
            unit_cost=599.99,
            supplier_ids=["TECH-SUPPLIER-A"]
        ),
        SKU(
            sku_id="FOOD-001",
            name="Organic Salmon Fillet",
            category="Seafood",
            is_perishable=True,
            shelf_life_days=3,
            unit_cost=15.99,
            supplier_ids=["SEAFOOD-DIRECT", "OCEAN-HARVEST"]
        ),
        SKU(
            sku_id="CLOTH-001",
            name="Winter Jacket Premium",
            category="Apparel",
            is_perishable=False,
            unit_cost=89.99,
            supplier_ids=["FASHION-SUPPLY"]
        ),
    ]

    # Multiple locations
    inventory = [
        InventoryLevel(sku_id="ELEC-001", location_id="DC-EAST", quantity=150, reorder_point=200, safety_stock=100),
        InventoryLevel(sku_id="ELEC-001", location_id="DC-WEST", quantity=80, reorder_point=150, safety_stock=75),
        InventoryLevel(sku_id="FOOD-001", location_id="STORE-NYC", quantity=30, reorder_point=50, safety_stock=20),
        InventoryLevel(sku_id="FOOD-001", location_id="STORE-LA", quantity=25, reorder_point=50, safety_stock=20),
        InventoryLevel(sku_id="CLOTH-001", location_id="DC-CENTRAL", quantity=200, reorder_point=150, safety_stock=100),
    ]

    # Multiple suppliers
    suppliers = [
        Supplier(supplier_id="TECH-SUPPLIER-A", name="TechGlobal Inc", lead_time_days=10, reliability_score=0.97, cost_factor=1.0),
        Supplier(supplier_id="SEAFOOD-DIRECT", name="Direct Seafood", lead_time_days=1, reliability_score=0.99, cost_factor=1.3),
        Supplier(supplier_id="OCEAN-HARVEST", name="Ocean Harvest Co", lead_time_days=2, reliability_score=0.93, cost_factor=1.1),
        Supplier(supplier_id="FASHION-SUPPLY", name="Fashion Supply Chain", lead_time_days=7, reliability_score=0.90, cost_factor=0.95),
    ]

    # Add external signals
    external_signals = ExternalSignals(
        weather_data={
            "NYC": {"temp": 32, "condition": "snow", "severity": "medium"},
            "LA": {"temp": 75, "condition": "clear", "severity": "none"}
        },
        social_trends={
            "smartphone_demand_spike": 1.25,  # 25% increase
            "winter_wear_trend": 1.5  # 50% increase due to cold weather
        },
        economic_indicators={
            "consumer_confidence": 98.5,
            "retail_sales_growth": 0.03
        },
        port_congestion={
            "port_of_la": {"delay_days": 3, "severity": "high"},
            "port_of_ny": {"delay_days": 1, "severity": "low"}
        }
    )

    # Create initial state with external signals
    initial_state = {
        "skus": skus,
        "inventory_levels": inventory,
        "suppliers": suppliers,
        "external_signals": external_signals,
    }

    # Run optimization
    graph = create_supply_chain_graph()
    result = graph.run(initial_state)

    # Display comprehensive results
    print("\n" + "=" * 70)
    print("OPTIMIZATION RESULTS")
    print("=" * 70)

    print(f"\n📊 Total Demand Forecasts: {len(result['demand_forecasts'])}")
    print(f"📦 Total Purchase Orders: {len(result['purchase_orders'])}")
    print(f"🚚 Total Logistics Routes: {len(result['logistics_routes'])}")
    print(f"⚠️  Disruptions Detected: {len(result['disruptions'])}")

    # Show critical orders
    critical_orders = [po for po in result['purchase_orders'] if po.priority == 'critical']
    if critical_orders:
        print(f"\n🔴 CRITICAL PRIORITY ORDERS: {len(critical_orders)}")
        for po in critical_orders[:3]:
            print(f"  • {po.po_id}: {po.sku_id} - ${po.total_cost:,.2f}")

    # Show disruptions
    if result['disruptions']:
        print(f"\n⚠️  SUPPLY CHAIN DISRUPTIONS:")
        for disruption in result['disruptions']:
            print(f"  • {disruption.type.upper()} - Severity: {disruption.severity}")
            print(f"    Affected Routes: {len(disruption.affected_routes)}")
            if disruption.mitigation_action:
                print(f"    Mitigation: {disruption.mitigation_action}")

    # Show total costs
    total_po_value = sum(po.total_cost for po in result['purchase_orders'])
    total_logistics_cost = sum(r.estimated_cost for r in result['logistics_routes'])

    print(f"\n💰 FINANCIAL SUMMARY:")
    print(f"  • Total Purchase Orders Value: ${total_po_value:,.2f}")
    print(f"  • Total Logistics Cost: ${total_logistics_cost:,.2f}")
    print(f"  • Combined Total: ${(total_po_value + total_logistics_cost):,.2f}")

    # Show alerts
    if result.get('alerts'):
        print(f"\n📢 SYSTEM ALERTS:")
        for alert in result['alerts']:
            print(f"{alert}")

    print(f"\n🔍 Human Review Required: {'YES' if result['requires_human_review'] else 'NO'}")

    return result


def streaming_example():
    """Example using streaming for real-time monitoring"""

    print("\n\n" + "=" * 70)
    print("STREAMING EXAMPLE: Real-time Monitoring")
    print("=" * 70)

    # Simple setup
    skus = [
        SKU(sku_id="TEST-001", name="Test Product", category="Test",
            is_perishable=False, unit_cost=10.0, supplier_ids=["SUP-001"])
    ]

    inventory = [
        InventoryLevel(sku_id="TEST-001", location_id="TEST-DC",
                      quantity=50, reorder_point=100, safety_stock=50)
    ]

    suppliers = [
        Supplier(supplier_id="SUP-001", name="Test Supplier",
                lead_time_days=3, reliability_score=0.95, cost_factor=1.0)
    ]

    initial_state = {
        "skus": skus,
        "inventory_levels": inventory,
        "suppliers": suppliers,
    }

    # Stream the execution
    graph = create_supply_chain_graph()

    print("\n📡 Starting streaming execution...\n")

    for step_output in graph.stream(initial_state):
        for node_name, node_output in step_output.items():
            print(f"\n{'='*60}")
            print(f"✓ Completed: {node_name.upper()}")
            print(f"{'='*60}")

            # Show what this node produced
            if 'demand_forecasts' in node_output and node_output['demand_forecasts']:
                print(f"  Forecasts: {len(node_output['demand_forecasts'])} generated")

            if 'purchase_orders' in node_output and node_output['purchase_orders']:
                print(f"  Purchase Orders: {len(node_output['purchase_orders'])} created")

            if 'logistics_routes' in node_output and node_output['logistics_routes']:
                print(f"  Routes: {len(node_output['logistics_routes'])} optimized")

    print("\n✅ Streaming complete!\n")


if __name__ == "__main__":
    # Run examples
    print("\n🚀 Supply Chain Optimization - Examples\n")

    # Example 1: Basic
    basic_example()

    # Example 2: Advanced with external signals
    advanced_example_with_external_signals()

    # Example 3: Streaming
    streaming_example()

    print("\n" + "=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)
