"""
Logistics Agent

Route optimization with real-time traffic, weather, and capacity constraints.
Disruption detection and autonomous rerouting capabilities.
"""

from datetime import datetime, timedelta
from typing import List, Dict
import random

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from models.state import SupplyChainState, LogisticsRoute, Disruption, AgentMessage


def logistics_agent(state: SupplyChainState) -> SupplyChainState:
    """
    Agent that optimizes logistics and handles disruptions.

    - Route optimization considering traffic, weather, and capacity
    - Carrier selection and load building
    - Real-time disruption detection
    - Autonomous rerouting within 15 minutes
    """

    print("\n=== Logistics Agent ===")

    # Initialize LLM
    llm = ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0.3)

    # Get data from state
    purchase_orders = state.get("purchase_orders", [])
    suppliers = state.get("suppliers", [])
    external_signals = state.get("external_signals")
    existing_routes = state.get("logistics_routes", [])

    # Analyze messages
    messages = state.get("messages", [])
    inventory_message = next(
        (m for m in messages if m.from_agent == "inventory_optimizer"),
        None
    )

    print(f"Received orders: {inventory_message.content if inventory_message else 'None'}")

    # Detect disruptions first
    disruptions = []

    # Simulate disruption detection (in production, this would monitor real-time feeds)
    if random.random() < 0.2:  # 20% chance of detecting a disruption
        disruption_types = ["weather", "traffic", "supplier_failure", "port_congestion"]
        disruption = Disruption(
            disruption_id=f"DISR-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            type=random.choice(disruption_types),
            affected_routes=[r.route_id for r in existing_routes[:2]],
            severity=random.choice(["medium", "high"]),
            detected_at=datetime.now(),
            estimated_resolution=datetime.now() + timedelta(hours=random.randint(2, 24))
        )
        disruptions.append(disruption)
        print(f"⚠️  Disruption detected: {disruption.type} - Severity: {disruption.severity}")

    # Create routing prompt
    system_prompt = """You are an expert logistics optimization agent.
Your role is to create optimal delivery routes and respond to disruptions.

Consider the following when optimizing routes:
1. Carrier capacity and availability
2. Real-time traffic conditions
3. Weather conditions along routes
4. Delivery time windows and priorities
5. Cost optimization (fuel, tolls, carrier rates)
6. Load consolidation opportunities

For disruptions:
- Detect issues from real-time feeds (weather, traffic, supplier status)
- Automatically reroute within 15 minutes
- Notify affected stakeholders
- Optimize for minimal delay and cost impact"""

    # Prepare context
    context = f"""
Purchase Orders to Route: {len(purchase_orders)}
Critical Orders: {sum(1 for po in purchase_orders if po.priority == 'critical')}
Urgent Orders: {sum(1 for po in purchase_orders if po.priority == 'urgent')}

Active Disruptions: {len(disruptions)}
{chr(10).join([f"- {d.type} (Severity: {d.severity}) affecting {len(d.affected_routes)} routes" for d in disruptions])}

Sample Orders to Route:
"""

    # Group orders by destination for consolidation
    orders_by_dest: Dict[str, List] = {}
    for po in purchase_orders:
        if po.destination_location not in orders_by_dest:
            orders_by_dest[po.destination_location] = []
        orders_by_dest[po.destination_location].append(po)

    for dest, orders in list(orders_by_dest.items())[:5]:
        context += f"\n{dest}: {len(orders)} orders, "
        context += f"Priority: {max(o.priority for o in orders)}, "
        context += f"Total value: ${sum(o.total_cost for o in orders):,.0f}"

    messages_llm = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=context)
    ]

    # Get routing recommendations
    response = llm.invoke(messages_llm)
    analysis = response.content

    print(f"Routing Analysis:\n{analysis[:500]}...")

    # Generate optimized routes
    routes = []
    carriers = ["FastShip Logistics", "QuickMove Transport", "ReliaCargo", "ExpressFreight"]

    for dest, orders in orders_by_dest.items():
        # Consolidate orders to same destination into single route
        route_po_ids = [po.po_id for po in orders]

        # Determine carrier based on priority and capacity
        max_priority = max(orders, key=lambda x: {"critical": 3, "urgent": 2, "normal": 1}[x.priority]).priority

        if max_priority == "critical":
            carrier = "ExpressFreight"  # Most reliable
            duration = random.uniform(8, 16)
        elif max_priority == "urgent":
            carrier = random.choice(["FastShip Logistics", "QuickMove Transport"])
            duration = random.uniform(12, 24)
        else:
            carrier = random.choice(carriers)
            duration = random.uniform(24, 48)

        # Calculate estimated cost (varies by carrier and distance)
        base_cost = len(orders) * random.uniform(200, 500)
        if carrier == "ExpressFreight":
            base_cost *= 1.5  # Premium carrier

        # Determine origin (supplier's location)
        # In production, this would query actual supplier locations
        origin = f"SUPPLIER-{random.choice(['A', 'B', 'C'])}"

        route = LogisticsRoute(
            route_id=f"ROUTE-{datetime.now().strftime('%Y%m%d')}-{len(routes) + 1:04d}",
            po_ids=route_po_ids,
            carrier=carrier,
            origin=origin,
            destination=dest,
            waypoints=[],
            estimated_duration_hours=duration,
            estimated_cost=base_cost,
            status="planned"
        )
        routes.append(route)

    print(f"Generated {len(routes)} optimized routes")
    print(f"Total logistics cost: ${sum(r.estimated_cost for r in routes):,.0f}")

    # Handle disruptions - reroute affected routes
    alerts = []
    for disruption in disruptions:
        affected = [r for r in existing_routes if r.route_id in disruption.affected_routes]

        if affected:
            # Create rerouting strategy
            mitigation = f"Rerouting {len(affected)} affected shipments to alternate carriers/routes"
            disruption.mitigation_action = mitigation

            alert = (
                f"🚨 Disruption Alert: {disruption.type} detected. "
                f"Affected: {len(affected)} routes. "
                f"Action: {mitigation}. "
                f"Response time: <15 minutes"
            )
            alerts.append(alert)
            print(alert)

            # In production, would actually create new routes here
            for route in affected:
                # Mark old route as disrupted
                route.status = "disrupted"

    # Send message to orchestrator
    message = AgentMessage(
        from_agent="logistics",
        to_agent="orchestrator",
        message_type="routes_optimized",
        content={
            "num_routes": len(routes),
            "total_cost": sum(r.estimated_cost for r in routes),
            "disruptions_handled": len(disruptions),
            "avg_delivery_hours": sum(r.estimated_duration_hours for r in routes) / len(routes) if routes else 0,
        }
    )

    return {
        **state,
        "logistics_routes": routes,
        "disruptions": disruptions,
        "messages": [message],
        "current_agent": "logistics",
        "next_agent": "orchestrator",
        "alerts": alerts,
    }
