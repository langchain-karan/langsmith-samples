"""
Orchestrator Agent

Coordinates across demand, inventory, and logistics agents.
Resolves conflicts and manages overall supply chain optimization workflow.
"""

from datetime import datetime
from typing import Dict, List

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from models.state import SupplyChainState, AgentMessage


def orchestrator_agent(state: SupplyChainState) -> SupplyChainState:
    """
    Orchestrator agent that coordinates all other agents.

    - Monitors overall system health
    - Resolves conflicts between agents
    - Determines next actions
    - Communicates with stakeholders
    - Decides when human intervention is needed
    """

    print("\n=== Orchestrator Agent ===")

    # Initialize LLM
    llm = ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0.2)

    # Gather status from all agents
    messages = state.get("messages", [])
    demand_forecasts = state.get("demand_forecasts", [])
    purchase_orders = state.get("purchase_orders", [])
    logistics_routes = state.get("logistics_routes", [])
    disruptions = state.get("disruptions", [])
    alerts = state.get("alerts", [])

    # Build comprehensive status report
    system_prompt = """You are the orchestrator agent for a supply chain optimization system.
Your role is to:
1. Monitor the overall health of the supply chain
2. Coordinate between demand forecasting, inventory, and logistics agents
3. Identify conflicts or issues requiring resolution
4. Determine if human intervention is needed
5. Provide executive summary of system status

Analyze the current state and determine:
- Are there any conflicts between agent outputs?
- Is the system operating efficiently?
- What risks or opportunities exist?
- Should any agent be re-run with adjusted parameters?
- Does anything require immediate human attention?"""

    # Prepare comprehensive status
    context = f"""
=== SUPPLY CHAIN OPTIMIZATION STATUS ===
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Iteration: {state.get('iteration', 0)}

DEMAND FORECASTING:
- Forecasts generated: {len(demand_forecasts)}
- Forecast locations: {len(set([f.location_id for f in demand_forecasts]))}
- Avg MAPE: {sum(f.mape for f in demand_forecasts if f.mape) / len([f for f in demand_forecasts if f.mape]):.2%} (target: <15%)

INVENTORY OPTIMIZATION:
- Purchase orders created: {len(purchase_orders)}
- Critical priority: {sum(1 for po in purchase_orders if po.priority == 'critical')}
- Urgent priority: {sum(1 for po in purchase_orders if po.priority == 'urgent')}
- Total PO value: ${sum(po.total_cost for po in purchase_orders):,.0f}
- Avg delivery time: {sum((po.expected_delivery - datetime.now()).days for po in purchase_orders) / len(purchase_orders) if purchase_orders else 0:.1f} days

LOGISTICS:
- Routes optimized: {len(logistics_routes)}
- Total logistics cost: ${sum(r.estimated_cost for r in logistics_routes):,.0f}
- Disruptions detected: {len(disruptions)}
- Active disruptions: {sum(1 for d in disruptions if not d.mitigation_action)}

ALERTS:
{chr(10).join([f'- {alert}' for alert in alerts]) if alerts else '- No critical alerts'}

INTER-AGENT MESSAGES:
{chr(10).join([f'- {m.from_agent} → {m.to_agent}: {m.message_type}' for m in messages[-5:]])}

HUMAN REVIEW REQUIRED: {state.get('requires_human_review', False)}
"""

    messages_llm = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=context)
    ]

    # Get orchestration analysis
    response = llm.invoke(messages_llm)
    analysis = response.content

    print(f"Orchestration Analysis:\n{analysis}")

    # Determine next actions
    next_agent = None
    requires_review = state.get('requires_human_review', False)

    # Check for conflicts or issues
    conflicts = []

    # Example conflict: High demand forecast but no PO generated
    high_demand_skus = set(f.sku_id for f in demand_forecasts if f.predicted_demand > 400)
    po_skus = set(po.sku_id for po in purchase_orders)
    missing_pos = high_demand_skus - po_skus

    if missing_pos and len(missing_pos) > 5:
        conflicts.append(f"High demand forecasted for {len(missing_pos)} SKUs but no POs generated")
        next_agent = "inventory_optimizer"  # Re-run inventory optimizer

    # Check for unmitigated disruptions
    unmitigated_disruptions = [d for d in disruptions if not d.mitigation_action]
    if unmitigated_disruptions:
        conflicts.append(f"{len(unmitigated_disruptions)} disruptions without mitigation plans")
        next_agent = "logistics"  # Re-run logistics to handle

    # Check for high-cost routes
    expensive_routes = [r for r in logistics_routes if r.estimated_cost > 5000]
    if expensive_routes:
        conflicts.append(f"{len(expensive_routes)} routes with high costs - review recommended")

    # Create summary message
    summary_message = AgentMessage(
        from_agent="orchestrator",
        to_agent="system",
        message_type="status_summary",
        content={
            "forecasts": len(demand_forecasts),
            "purchase_orders": len(purchase_orders),
            "routes": len(logistics_routes),
            "disruptions": len(disruptions),
            "conflicts": conflicts,
            "requires_human_review": requires_review or len(conflicts) > 0,
            "system_health": "healthy" if not conflicts else "needs_attention",
        }
    )

    # Generate executive summary alert
    executive_summary = f"""
📊 SUPPLY CHAIN OPTIMIZATION SUMMARY

✅ Demand Forecasting: {len(demand_forecasts)} forecasts generated
✅ Inventory: {len(purchase_orders)} purchase orders (${sum(po.total_cost for po in purchase_orders):,.0f})
✅ Logistics: {len(logistics_routes)} routes optimized (${sum(r.estimated_cost for r in logistics_routes):,.0f})

"""

    if disruptions:
        executive_summary += f"⚠️  Active Disruptions: {len(disruptions)}\n"

    if conflicts:
        executive_summary += f"⚠️  Issues Detected: {len(conflicts)}\n"
        for conflict in conflicts:
            executive_summary += f"   - {conflict}\n"

    if state.get('requires_human_review'):
        executive_summary += "\n🔍 Human review required for high-value critical orders\n"

    print(executive_summary)

    # Update alerts with executive summary
    new_alerts = alerts + [executive_summary] if executive_summary.strip() else alerts

    return {
        **state,
        "messages": [summary_message],
        "current_agent": "orchestrator",
        "next_agent": next_agent,  # None if workflow complete
        "requires_human_review": requires_review or len(conflicts) > 0,
        "alerts": new_alerts,
        "iteration": state.get("iteration", 0) + 1,
    }
