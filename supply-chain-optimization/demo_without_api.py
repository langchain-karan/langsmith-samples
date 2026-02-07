"""
Demo of Supply Chain Optimization structure without API calls

This demonstrates the data flow and structure without requiring API keys.
"""

from datetime import datetime, timedelta
from models.state import (
    SKU, InventoryLevel, Supplier, DemandForecast,
    PurchaseOrder, LogisticsRoute, ExternalSignals
)

print("=" * 70)
print("SUPPLY CHAIN OPTIMIZATION - STRUCTURE DEMO")
print("=" * 70)

# 1. Define SKUs
print("\n1. PRODUCT CATALOG (SKUs)")
print("-" * 70)

skus = [
    SKU(
        sku_id="PROD-001",
        name="Premium Coffee Beans",
        category="Beverages",
        is_perishable=False,
        unit_cost=12.99,
        supplier_ids=["SUP-001", "SUP-002"]
    ),
    SKU(
        sku_id="PROD-002",
        name="Fresh Strawberries",
        category="Produce",
        is_perishable=True,
        shelf_life_days=5,
        unit_cost=4.50,
        supplier_ids=["SUP-003"]
    ),
    SKU(
        sku_id="PROD-003",
        name="Organic Milk",
        category="Dairy",
        is_perishable=True,
        shelf_life_days=7,
        unit_cost=3.99,
        supplier_ids=["SUP-002", "SUP-003"]
    ),
]

for sku in skus:
    print(f"  • {sku.sku_id}: {sku.name}")
    print(f"    Category: {sku.category} | Perishable: {sku.is_perishable}")
    print(f"    Cost: ${sku.unit_cost} | Suppliers: {len(sku.supplier_ids)}")

# 2. Current Inventory
print("\n2. CURRENT INVENTORY STATUS")
print("-" * 70)

inventory = [
    InventoryLevel(
        sku_id="PROD-001",
        location_id="DC-EAST",
        quantity=50,
        reorder_point=100,
        safety_stock=75
    ),
    InventoryLevel(
        sku_id="PROD-002",
        location_id="STORE-NYC",
        quantity=20,
        reorder_point=50,
        safety_stock=30
    ),
    InventoryLevel(
        sku_id="PROD-003",
        location_id="STORE-LA",
        quantity=35,
        reorder_point=60,
        safety_stock=40
    ),
]

for inv in inventory:
    status = "⚠️ LOW" if inv.quantity < inv.reorder_point else "✓ OK"
    print(f"  {status} {inv.sku_id} @ {inv.location_id}")
    print(f"     Current: {inv.quantity} | Reorder Point: {inv.reorder_point}")

# 3. Suppliers
print("\n3. SUPPLIER NETWORK")
print("-" * 70)

suppliers = [
    Supplier(
        supplier_id="SUP-001",
        name="Global Coffee Importers",
        lead_time_days=5,
        reliability_score=0.95,
        cost_factor=1.0
    ),
    Supplier(
        supplier_id="SUP-002",
        name="Local Farms Fresh",
        lead_time_days=1,
        reliability_score=0.98,
        cost_factor=1.2
    ),
    Supplier(
        supplier_id="SUP-003",
        name="Regional Produce Co",
        lead_time_days=2,
        reliability_score=0.92,
        cost_factor=0.95
    ),
]

for sup in suppliers:
    print(f"  • {sup.supplier_id}: {sup.name}")
    print(f"    Lead Time: {sup.lead_time_days} days | Reliability: {sup.reliability_score:.0%}")
    print(f"    Cost Factor: {sup.cost_factor}x")

# 4. External Signals
print("\n4. EXTERNAL SIGNALS")
print("-" * 70)

external_signals = ExternalSignals(
    weather_data={
        "NYC": {"temp": 32, "condition": "snow", "impact": "high"},
        "LA": {"temp": 75, "condition": "clear", "impact": "none"}
    },
    social_trends={
        "coffee_demand": 1.15,  # 15% increase
        "organic_trend": 1.25   # 25% increase
    },
    economic_indicators={
        "consumer_confidence": 98.5,
        "retail_growth": 0.03
    },
    port_congestion={
        "port_la": {"delay_days": 3, "severity": "medium"}
    }
)

print(f"  Weather: {len(external_signals.weather_data)} locations tracked")
print(f"  Social Trends: {len(external_signals.social_trends)} signals")
print(f"  Economic: {len(external_signals.economic_indicators)} indicators")
print(f"  Ports: {len(external_signals.port_congestion)} monitored")

# 5. Simulated Agent Outputs
print("\n" + "=" * 70)
print("SIMULATED AGENT OUTPUTS")
print("=" * 70)

# Demand Forecasts
print("\n5. DEMAND FORECASTING AGENT OUTPUT")
print("-" * 70)

forecasts = [
    DemandForecast(
        sku_id="PROD-001",
        location_id="DC-EAST",
        date=datetime.now() + timedelta(days=1),
        predicted_demand=150.0,
        confidence_lower=135.0,
        confidence_upper=165.0,
        mape=0.12,
        factors={"seasonality": 1.1, "trend": 1.05}
    ),
    DemandForecast(
        sku_id="PROD-002",
        location_id="STORE-NYC",
        date=datetime.now() + timedelta(days=1),
        predicted_demand=75.0,
        confidence_lower=60.0,
        confidence_upper=90.0,
        mape=0.14,
        factors={"weather_impact": 0.8, "perishable_variance": 1.3}
    ),
]

print(f"  Total Forecasts: {len(forecasts)}")
for fc in forecasts:
    print(f"  • {fc.sku_id} @ {fc.location_id}")
    print(f"    Predicted: {fc.predicted_demand:.0f} units (MAPE: {fc.mape:.1%})")
    print(f"    Range: {fc.confidence_lower:.0f} - {fc.confidence_upper:.0f}")

# Purchase Orders
print("\n6. INVENTORY OPTIMIZER AGENT OUTPUT")
print("-" * 70)

purchase_orders = [
    PurchaseOrder(
        po_id="PO-20260207-0001",
        sku_id="PROD-001",
        supplier_id="SUP-001",
        quantity=200,
        destination_location="DC-EAST",
        expected_delivery=datetime.now() + timedelta(days=5),
        total_cost=2598.00,
        priority="urgent",
        status="pending"
    ),
    PurchaseOrder(
        po_id="PO-20260207-0002",
        sku_id="PROD-002",
        supplier_id="SUP-003",
        quantity=100,
        destination_location="STORE-NYC",
        expected_delivery=datetime.now() + timedelta(days=2),
        total_cost=427.50,
        priority="critical",
        status="pending"
    ),
]

print(f"  Total Purchase Orders: {len(purchase_orders)}")
total_value = sum(po.total_cost for po in purchase_orders)
print(f"  Total Value: ${total_value:,.2f}")
print()
for po in purchase_orders:
    priority_icon = "🔴" if po.priority == "critical" else "🟡" if po.priority == "urgent" else "🟢"
    print(f"  {priority_icon} {po.po_id} - {po.priority.upper()}")
    print(f"     SKU: {po.sku_id} | Qty: {po.quantity} units")
    print(f"     Supplier: {po.supplier_id} | Cost: ${po.total_cost:,.2f}")
    print(f"     Delivery: {po.expected_delivery.strftime('%Y-%m-%d')}")

# Logistics Routes
print("\n7. LOGISTICS AGENT OUTPUT")
print("-" * 70)

routes = [
    LogisticsRoute(
        route_id="ROUTE-20260207-0001",
        po_ids=["PO-20260207-0001"],
        carrier="FastShip Logistics",
        origin="SUPPLIER-WAREHOUSE-A",
        destination="DC-EAST",
        waypoints=[],
        estimated_duration_hours=48.0,
        estimated_cost=350.00,
        status="planned"
    ),
    LogisticsRoute(
        route_id="ROUTE-20260207-0002",
        po_ids=["PO-20260207-0002"],
        carrier="ExpressFreight",
        origin="SUPPLIER-WAREHOUSE-B",
        destination="STORE-NYC",
        waypoints=[],
        estimated_duration_hours=12.0,
        estimated_cost=185.00,
        status="planned"
    ),
]

print(f"  Total Routes: {len(routes)}")
total_logistics_cost = sum(r.estimated_cost for r in routes)
print(f"  Total Logistics Cost: ${total_logistics_cost:,.2f}")
print()
for route in routes:
    print(f"  • {route.route_id}")
    print(f"    {route.origin} → {route.destination}")
    print(f"    Carrier: {route.carrier} | Duration: {route.estimated_duration_hours:.0f}h")
    print(f"    Cost: ${route.estimated_cost:,.2f} | Orders: {len(route.po_ids)}")

# Summary
print("\n" + "=" * 70)
print("OPTIMIZATION SUMMARY")
print("=" * 70)

print(f"""
📊 Forecasts Generated: {len(forecasts)}
📦 Purchase Orders: {len(purchase_orders)} (${total_value:,.2f})
🚚 Routes Optimized: {len(routes)} (${total_logistics_cost:,.2f})
💰 Total Supply Chain Cost: ${(total_value + total_logistics_cost):,.2f}

✅ Low inventory items identified and orders generated
✅ Suppliers optimized for cost, reliability, and lead time
✅ Routes consolidated for efficiency
✅ Perishable goods handled with appropriate lead times
✅ External signals (weather, trends) incorporated

This demonstrates the complete data flow through the LangGraph system!
""")

print("=" * 70)
print("To run with actual AI agents, set ANTHROPIC_API_KEY in .env")
print("=" * 70)
