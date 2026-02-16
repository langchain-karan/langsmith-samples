"""
Inventory Optimizer Agent

Autonomous reorder point optimization and multi-echelon inventory allocation.
Generates purchase orders with supplier selection optimization.
"""

from datetime import datetime, timedelta
from typing import List, Dict
import random

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from models.state import SupplyChainState, PurchaseOrder, AgentMessage, InventoryLevel


def inventory_optimizer_agent(state: SupplyChainState) -> SupplyChainState:
    """
    Agent that optimizes inventory levels and generates purchase orders.

    - Analyzes demand forecasts to identify reorder needs
    - Optimizes supplier selection based on cost, lead time, and reliability
    - Handles multi-echelon inventory allocation
    - Special handling for perishable goods with shelf-life awareness
    """

    print("\n=== Inventory Optimizer Agent ===")

    # Initialize LLM
    llm = ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0.2)

    # Get data from state
    skus = state["skus"]
    inventory_levels = state.get("inventory_levels", [])
    demand_forecasts = state.get("demand_forecasts", [])
    suppliers = state.get("suppliers", [])

    # Analyze messages from demand forecasting agent
    messages = state.get("messages", [])
    forecast_message = next(
        (m for m in messages if m.from_agent == "demand_forecasting"),
        None
    )

    print(f"Received forecast data: {forecast_message.content if forecast_message else 'None'}")

    # Organize forecasts by SKU and location
    forecast_by_sku_loc: Dict[tuple, List] = {}
    for forecast in demand_forecasts:
        key = (forecast.sku_id, forecast.location_id)
        if key not in forecast_by_sku_loc:
            forecast_by_sku_loc[key] = []
        forecast_by_sku_loc[key].append(forecast)

    # Create optimization prompt
    system_prompt = """You are an expert inventory optimization agent.
Your role is to ensure optimal stock levels while minimizing costs and preventing stockouts.

Analyze inventory levels against demand forecasts and determine:
1. Which SKUs need reordering at which locations
2. Optimal order quantities considering lead times and demand uncertainty
3. Best supplier selection based on cost, reliability, and lead time
4. Special considerations for perishable goods (shelf-life constraints)
5. Multi-echelon allocation strategy (warehouse vs stores)

Generate purchase orders that balance:
- Service level (avoid stockouts)
- Inventory carrying costs
- Supplier lead times and reliability
- Perishable goods waste minimization"""

    # Prepare context
    context = f"""
Current Inventory Status:
- Total SKUs: {len(skus)}
- Inventory locations tracked: {len(set([inv.location_id for inv in inventory_levels]))}
- Demand forecasts available: {len(demand_forecasts)}
- Available suppliers: {len(suppliers)}

SKUs requiring attention (low stock or high demand):
"""

    # Identify SKUs needing reorder
    reorder_needed = []
    for inv in inventory_levels:
        # Get forecasts for this SKU-location
        key = (inv.sku_id, inv.location_id)
        forecasts = forecast_by_sku_loc.get(key, [])

        if forecasts:
            # Sum demand for next 7 days
            total_demand = sum(f.predicted_demand for f in forecasts)
            days_of_stock = inv.quantity / (total_demand / 7) if total_demand > 0 else 999

            if days_of_stock < 14 or inv.quantity < inv.reorder_point:
                sku = next((s for s in skus if s.sku_id == inv.sku_id), None)
                reorder_needed.append({
                    "sku_id": inv.sku_id,
                    "location": inv.location_id,
                    "current_qty": inv.quantity,
                    "reorder_point": inv.reorder_point,
                    "days_of_stock": days_of_stock,
                    "weekly_demand": total_demand,
                    "is_perishable": sku.is_perishable if sku else False
                })

    context += "\n".join([
        f"- {item['sku_id']} at {item['location']}: {item['current_qty']} units, "
        f"{item['days_of_stock']:.1f} days of stock, weekly demand: {item['weekly_demand']:.0f}"
        for item in reorder_needed[:10]
    ])

    context += f"\n\nTotal SKUs requiring reorder: {len(reorder_needed)}"

    messages_llm = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=context)
    ]

    # Get optimization recommendations
    response = llm.invoke(messages_llm)
    analysis = response.content

    print(f"Optimization Analysis:\n{analysis[:500]}...")
    print(f"SKUs requiring reorder: {len(reorder_needed)}")

    # Generate purchase orders
    purchase_orders = []

    for item in reorder_needed:
        sku = next((s for s in skus if s.sku_id == item['sku_id']), None)
        if not sku:
            continue

        # Select best supplier
        available_suppliers = [s for s in suppliers if s.supplier_id in sku.supplier_ids]
        if not available_suppliers and suppliers:
            available_suppliers = [suppliers[0]]  # Fallback

        if available_suppliers:
            # Score suppliers: balance cost, lead time, and reliability
            best_supplier = max(
                available_suppliers,
                key=lambda s: s.reliability_score / (s.cost_factor * (1 + s.lead_time_days / 30))
            )

            # Calculate order quantity
            # For perishable goods, order conservatively based on shelf life
            if item['is_perishable'] and sku.shelf_life_days:
                # Order enough for shelf_life period, not more
                max_days = min(sku.shelf_life_days, 14)
                order_qty = int(item['weekly_demand'] * (max_days / 7))
            else:
                # Order for 30 days + safety stock
                order_qty = int(item['weekly_demand'] * 4.3 + item['reorder_point'] * 0.5)

            # Determine priority based on days of stock
            if item['days_of_stock'] < 3:
                priority = "critical"
            elif item['days_of_stock'] < 7:
                priority = "urgent"
            else:
                priority = "normal"

            po = PurchaseOrder(
                po_id=f"PO-{datetime.now().strftime('%Y%m%d')}-{len(purchase_orders) + 1:04d}",
                sku_id=item['sku_id'],
                supplier_id=best_supplier.supplier_id,
                quantity=order_qty,
                destination_location=item['location'],
                expected_delivery=datetime.now() + timedelta(days=best_supplier.lead_time_days),
                total_cost=order_qty * sku.unit_cost * best_supplier.cost_factor,
                priority=priority,
                status="pending"
            )
            purchase_orders.append(po)

    print(f"Generated {len(purchase_orders)} purchase orders")
    print(f"Critical: {sum(1 for po in purchase_orders if po.priority == 'critical')}")
    print(f"Urgent: {sum(1 for po in purchase_orders if po.priority == 'urgent')}")

    # Send message to logistics agent
    message = AgentMessage(
        from_agent="inventory_optimizer",
        to_agent="logistics",
        message_type="orders_created",
        content={
            "num_orders": len(purchase_orders),
            "critical_orders": [po.po_id for po in purchase_orders if po.priority == "critical"],
            "total_value": sum(po.total_cost for po in purchase_orders),
        }
    )

    # Check if any critical orders require human review
    requires_review = any(po.priority == "critical" and po.total_cost > 100000 for po in purchase_orders)

    alerts = []
    if requires_review:
        alerts.append(f"High-value critical orders detected - human review recommended")

    return {
        **state,
        "purchase_orders": purchase_orders,
        "messages": [message],
        "current_agent": "inventory_optimizer",
        "next_agent": "logistics",
        "requires_human_review": requires_review,
        "alerts": alerts,
    }
