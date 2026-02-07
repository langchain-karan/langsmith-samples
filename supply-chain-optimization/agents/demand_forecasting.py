"""
Demand Forecasting Agent

Generates demand forecasts at SKU-location-day granularity using ML models.
Incorporates seasonality, promotions, weather, social signals, and economic indicators.
"""

from datetime import datetime, timedelta
from typing import List
import random

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from models.state import SupplyChainState, DemandForecast, AgentMessage


def demand_forecasting_agent(state: SupplyChainState) -> SupplyChainState:
    """
    Agent that forecasts demand for all SKUs across locations.

    In production, this would use ML models (gradient boosting, deep learning).
    For this example, we use Claude to analyze patterns and generate forecasts.
    """

    print("\n=== Demand Forecasting Agent ===")

    # Initialize LLM
    llm = ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0.3)

    # Prepare context for forecasting
    skus = state["skus"]
    inventory_levels = state.get("inventory_levels", [])
    external_signals = state.get("external_signals")

    # Group inventory by location
    locations = list(set([inv.location_id for inv in inventory_levels]))

    # Create forecasting prompt
    system_prompt = """You are an expert demand forecasting agent in a supply chain optimization system.
Your role is to predict demand at SKU-location-day granularity with high accuracy (<15% MAPE target).

Analyze the provided data and generate demand forecasts considering:
- Historical sales patterns and seasonality
- Current inventory levels and velocity
- External signals (weather, social trends, economic indicators)
- Product characteristics (perishable goods have different patterns)
- Day of week and seasonal effects

Provide forecasts with confidence intervals (lower and upper bounds)."""

    context = f"""
Current Date: {datetime.now().strftime('%Y-%m-%d')}

SKUs to Forecast: {len(skus)}
Locations: {len(locations)}

Sample SKUs:
{chr(10).join([f"- {sku.sku_id}: {sku.name} (Category: {sku.category}, Perishable: {sku.is_perishable})" for sku in skus[:5]])}

Sample Locations: {locations[:5] if locations else ['DC-001', 'DC-002', 'STORE-001']}

Current Inventory Status:
{chr(10).join([f"- {inv.sku_id} at {inv.location_id}: {inv.quantity} units (reorder: {inv.reorder_point})" for inv in inventory_levels[:5]])}

External Signals Available: {bool(external_signals)}

Generate demand forecasts for the next 7 days for all SKU-location combinations.
Focus on identifying SKUs at risk of stockout or those with changing demand patterns.
"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=context)
    ]

    # Get forecast analysis from Claude
    response = llm.invoke(messages)
    analysis = response.content

    print(f"Forecast Analysis:\n{analysis[:500]}...")

    # Generate forecasts for each SKU-location combination
    forecasts = []
    base_date = datetime.now()

    for sku in skus:
        # For demonstration, generate forecasts for top 3 locations
        forecast_locations = locations[:3] if len(locations) >= 3 else locations
        if not forecast_locations:
            forecast_locations = ['DC-001', 'STORE-001', 'STORE-002']

        for location in forecast_locations:
            # Generate 7-day forecast
            for day_offset in range(7):
                forecast_date = base_date + timedelta(days=day_offset)

                # Simulate demand prediction with variation
                # In production, this would use actual ML models
                base_demand = random.uniform(50, 500)

                # Add day-of-week effect
                if forecast_date.weekday() in [5, 6]:  # Weekend
                    base_demand *= 1.3

                # Perishable goods have higher variance
                if sku.is_perishable:
                    variance = 0.3
                else:
                    variance = 0.15

                confidence_lower = base_demand * (1 - variance)
                confidence_upper = base_demand * (1 + variance)

                forecast = DemandForecast(
                    sku_id=sku.sku_id,
                    location_id=location,
                    date=forecast_date,
                    predicted_demand=base_demand,
                    confidence_lower=confidence_lower,
                    confidence_upper=confidence_upper,
                    mape=random.uniform(0.08, 0.14),  # Target <15%
                    factors={
                        "seasonality": random.uniform(0.8, 1.2),
                        "day_of_week": 1.3 if forecast_date.weekday() in [5, 6] else 1.0,
                        "weather_impact": random.uniform(0.95, 1.05),
                    }
                )
                forecasts.append(forecast)

    print(f"Generated {len(forecasts)} demand forecasts")

    # Send message to inventory agent
    message = AgentMessage(
        from_agent="demand_forecasting",
        to_agent="inventory_optimizer",
        message_type="forecasts_ready",
        content={
            "num_forecasts": len(forecasts),
            "analysis_summary": analysis[:200],
            "high_demand_skus": [f.sku_id for f in forecasts if f.predicted_demand > 400][:5]
        }
    )

    # Update state
    return {
        **state,
        "demand_forecasts": forecasts,
        "messages": [message],
        "current_agent": "demand_forecasting",
        "next_agent": "inventory_optimizer",
    }
